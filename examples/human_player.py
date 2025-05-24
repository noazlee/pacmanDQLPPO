#!/usr/bin/env python3
"""
Human player client for Pac-Man
Allows manual control using WASD keys
"""

import os
import asyncio
import json
import websockets
import logging
import sys
import termios
import tty
import threading
from queue import Queue

logger = logging.getLogger('HumanPlayer')
logger.setLevel(logging.INFO)

class KeyboardInput:
    """Non-blocking keyboard input handler"""
    
    def __init__(self):
        self.key_queue = Queue()
        self.running = True
        
    def get_key(self):
        """Get a single keypress"""
        if sys.stdin.isatty():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.cbreak(fd)
                ch = sys.stdin.read(1)
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return input()
    
    def keyboard_thread(self):
        """Background thread to capture keystrokes"""
        print("Use WASD keys to control Pac-Man. Press 'q' to quit.")
        print("   W = Up, A = Left, S = Down, D = Right")
        
        while self.running:
            try:
                key = self.get_key().lower()
                if key in ['w', 'a', 's', 'd', 'q']:
                    self.key_queue.put(key)
                    if key == 'q':
                        break
            except:
                break
    
    def start(self):
        """Start the keyboard input thread"""
        self.thread = threading.Thread(target=self.keyboard_thread, daemon=True)
        self.thread.start()
    
    def get_next_key(self):
        """Get the next key from the queue (non-blocking)"""
        try:
            return self.key_queue.get_nowait()
        except:
            return None
    
    def stop(self):
        """Stop the keyboard input"""
        self.running = False

async def human_player_loop(player_name="HumanPlayer", server_address="localhost:8000"):
    """Main human player loop"""
    
    # Setup keyboard input
    keyboard = KeyboardInput()
    keyboard.start()
    
    current_action = 'd'  # Default action (right)
    
    try:
        async with websockets.connect(f"ws://{server_address}/player") as websocket:
            # Join the game
            await websocket.send(json.dumps({"cmd": "join", "name": player_name}))
            msg = await websocket.recv()
            
            game_properties = json.loads(msg)
            print(f"\nGame Started!")
            print(f"   Map: {game_properties['map']}")
            print(f"   Ghosts: {game_properties['ghosts']}")
            print(f"   Player: {player_name}")
            print("=" * 40)
            
            # Game loop
            while True:
                # Check for new keypress
                key = keyboard.get_next_key()
                if key == 'q':
                    print("Quitting game...")
                    break
                elif key in ['w', 'a', 's', 'd']:
                    current_action = key
                
                # Receive game state
                try:
                    state_msg = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    state = json.loads(state_msg)
                    
                    # Check if game over
                    if 'lives' in state:
                        if not state['lives']:
                            print(f"\nGAME OVER!")
                            print(f"   Final Score: {state['score']}")
                            print(f"   Steps: {state.get('step', 0)}")
                            break
                    else:
                        print(f"\nVICTORY!")
                        print(f"   Final Score: {state['score']}")
                        print(f"   Steps: {state.get('step', 0)}")
                        break
                    
                    # Send current action
                    await websocket.send(json.dumps({"cmd": "key", "key": current_action}))
                    
                    # Display game info
                    step = state.get('step', 0)
                    score = state.get('score', 0)
                    lives = state.get('lives', 0)
                    pacman_pos = state.get('pacman', (0, 0))
                    
                    if step % 20 == 0:  # Update display every 20 steps
                        action_names = {'w': '↑', 'a': '←', 's': '↓', 'd': '→'}
                        print(f"\rStep: {step:4d} | Score: {score:4d} | Lives: {lives} | Pos: {pacman_pos} | Action: {action_names.get(current_action, '?')}", end='', flush=True)
                    
                except asyncio.TimeoutError:
                    # No message received, continue with current action
                    pass
                except websockets.exceptions.ConnectionClosed:
                    print("\n Connection lost")
                    break
                    
    except KeyboardInterrupt:
        print("\n  Game interrupted")
    except Exception as e:
        print(f"\n Error: {e}")
    finally:
        keyboard.stop()

def main():
    """Run the human player"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Human Pac-Man Player')
    parser.add_argument('--name', default='Human', help='Player name')
    parser.add_argument('--server', default='localhost', help='Server address')
    parser.add_argument('--port', default='8000', help='Server port')
    args = parser.parse_args()
    
    server_address = f"{args.server}:{args.port}"
    
    print("Human Pac-Man Player")
    print("=" * 30)
    print("Make sure the server and viewer are running:")
    print("  Terminal 1: python server.py")
    print("  Terminal 2: python viewer.py")
    print("  Terminal 3: python examples/human_player.py")
    print()
    
    try:
        asyncio.run(human_player_loop(args.name, server_address))
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()