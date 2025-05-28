import requests
import argparse
import asyncio
import json
import logging
import websockets
import os.path
from collections import namedtuple
from game import Game

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
wslogger = logging.getLogger('websockets')
wslogger.setLevel(logging.WARN)

logger = logging.getLogger('Server')
logger.setLevel(logging.INFO)

Player = namedtuple('Player', ['name', 'ws']) 

class Game_server:
    def __init__(self, mapfile, ghosts, level_ghosts, lives, timeout, grading=None):
        self.game = Game(mapfile, ghosts, level_ghosts, lives, timeout)
        self.game_properties = {'map': mapfile,
                                'n_ghosts': ghosts,
                                'l_ghosts': level_ghosts}
        self.players = asyncio.Queue()
        self.viewers = set()
        self.current_player = None 
        self.grading = grading

    def is_connection_closed(self, websocket):
        """Check if websocket connection is closed - compatible with different websocket versions"""
        try:
            # Try different possible attributes/methods
            if hasattr(websocket, 'closed'):
                return websocket.closed
            elif hasattr(websocket, 'close_code'):
                return websocket.close_code is not None
            elif hasattr(websocket, 'state'):
                return websocket.state.name == 'CLOSED'
            else:
                # If we can't determine, assume it's open
                return False
        except:
            # If any error occurs, assume connection is closed
            return True

    async def handle_connection(self, websocket):
        """Handle both player and viewer connections"""
        path = "/player"  # Default to player
        
        logger.info(f"New connection (assuming {path})")
        
        try:
            async for message in websocket:
                logger.debug(f"Received message: {message[:100]}...")
                
                try:
                    data = json.loads(message)
                    logger.debug(f"Parsed message: {data}")
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    continue
                
                if data["cmd"] == "join":
                    logger.info(f"Processing join command for {data.get('name', 'Unknown')}")
                    
                    try:
                        # Test game.info() before sending
                        logger.debug("Calling game.info()...")
                        map_info = self.game.info()
                        logger.debug(f"game.info() returned {len(map_info)} characters")
                        
                        # Verify it's valid JSON
                        json.loads(map_info)  # This will throw if invalid
                        logger.debug("game.info() JSON is valid")
                        
                        # Send the info
                        logger.debug("Sending game info to client...")
                        await websocket.send(map_info)
                        logger.info("Game info sent successfully")
                        
                        # Determine if this should be a player or viewer based on name
                        name = data.get("name", "")
                        if "viewer" in name.lower():
                            logger.info("Adding viewer to set")
                            self.viewers.add(websocket)
                        else:
                            logger.info("Adding player <%s> to queue", data["name"])
                            await self.players.put(Player(data["name"], websocket))
                            
                    except Exception as e:
                        logger.error(f"Error processing join command: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        # Send error response
                        try:
                            error_response = json.dumps({"error": f"Join failed: {str(e)}"})
                            await websocket.send(error_response)
                        except:
                            pass
                        continue

                elif data["cmd"] == "key":
                    logger.debug(f"Processing key command: {data.get('key')}")
                    if self.current_player and self.current_player.ws == websocket:
                        logger.debug(f"Sending keypress to game: {data['key']}")
                        self.game.keypress(data["key"][0])
                    else:
                        logger.warning("Key command from non-current player")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed")
            if websocket in self.viewers:
                self.viewers.remove(websocket)
        except Exception as e:
            logger.error(f"Unexpected error in connection handler: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def mainloop(self):
        while True:
            logger.info("Waiting for players")
            self.current_player = await self.players.get()
            
            # FIXED: Use our compatibility function instead of .closed
            if self.is_connection_closed(self.current_player.ws):
                logger.error("<{}> disconnect while waiting".format(self.current_player.name))
                continue
           
            try:
                logger.info("Starting game for <{}>".format(self.current_player.name))
                self.game.start(self.current_player.name)
                
                if self.grading:
                    game_rec = dict(self.game_properties)
                    game_rec['player'] = self.current_player.name
            
                step_count = 0
                while self.game.running:
                    await self.game.next_frame()
                    
                    try:
                        game_state = self.game.state
                        await self.current_player.ws.send(game_state)
                        
                        # Send to viewers
                        if self.viewers:
                            viewers_to_remove = set()
                            for viewer in self.viewers:
                                try:
                                    await viewer.send(game_state)
                                except websockets.exceptions.ConnectionClosed:
                                    viewers_to_remove.add(viewer)
                            
                            # Remove closed viewers
                            for viewer in viewers_to_remove:
                                self.viewers.discard(viewer)
                        
                        step_count += 1
                        if step_count % 100 == 0:
                            logger.debug(f"Game step {step_count}")
                            
                    except websockets.exceptions.ConnectionClosed:
                        logger.info("Player disconnected during game")
                        break
                    except Exception as e:
                        logger.error(f"Error sending game state: {e}")
                        break
                        
                # Send final score
                try:
                    final_message = json.dumps({"score": self.game.score})
                    await self.current_player.ws.send(final_message)
                    logger.info("Final score sent")
                except Exception as e:
                    logger.debug(f"Could not send final score (player may have disconnected): {e}")

                logger.info("Game finished for <{}> with score {}".format(self.current_player.name, self.game.score))
                
            except websockets.exceptions.ConnectionClosed:
                logger.info("Player disconnected during game")
            except Exception as e:
                logger.error(f"Game loop error: {e}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                if self.grading and 'game_rec' in locals():
                   game_rec['score'] = self.game.score
                   try:
                       r = requests.post(self.grading, json=game_rec)
                   except:
                       pass
                if self.current_player:
                    try:
                        await self.current_player.ws.close()
                    except:
                        pass
                self.current_player = None

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", help="IP address to bind to", default="")
    parser.add_argument("--port", help="TCP port", type=int, default=8000)
    parser.add_argument("--ghosts", help="Number of ghosts", type=int, default=1)
    parser.add_argument("--level", help="difficulty level of ghosts", choices=['0','1','2','3'], default='1')
    parser.add_argument("--lives", help="Number of lives", type=int, default=3)
    parser.add_argument("--timeout", help="Timeout after this amount steps", type=int, default=3000)
    parser.add_argument("--map", help="path to the map bmp", default="data/fixed_classic.bmp")
    parser.add_argument("--grading-server", help="url of grading server", default=None)
    args = parser.parse_args()

    g = Game_server(args.map, args.ghosts, int(args.level), args.lives, args.timeout, args.grading_server)

    async def main():
        game_loop_task = asyncio.create_task(g.mainloop())
        
        # Simplified server setup - no path handling
        async with websockets.serve(g.handle_connection, args.bind, args.port):
            logger.info(f"🎮 Pac-Man Server started on {args.bind if args.bind else 'localhost'}:{args.port}")
            logger.info("All connections will be treated as players")
            logger.info("Connect with: ws://localhost:8000/")
            logger.info("Ready for players!")
            await game_loop_task

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        traceback.print_exc()