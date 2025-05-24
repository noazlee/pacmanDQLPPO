import os
import asyncio
import json
import random
import websockets
import logging

logger = logging.getLogger('Client')
logger.setLevel(logging.INFO)

class SimpleAgent:
    """Simple agent that avoids getting stuck and chases food"""
    
    def __init__(self):
        self.last_pos = None
        self.stuck_count = 0
        self.last_action = 'd'  # Start moving right
        
    def get_action(self, state):
        """Choose action based on simple rules"""
        if 'pacman' not in state:
            return 'd'
            
        pacman_pos = state['pacman']
        
        # Check if stuck in same position
        if pacman_pos == self.last_pos:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
            
        self.last_pos = pacman_pos
        
        # If stuck, try different direction
        if self.stuck_count > 2:
            actions = ['w', 'a', 's', 'd']
            actions.remove(self.last_action)  # Don't repeat same action
            self.last_action = random.choice(actions)
            self.stuck_count = 0
            return self.last_action
            
        # Simple strategy: prefer horizontal movement, change on collision
        px, py = pacman_pos
        
        # Look for nearby energy/boost
        if 'energy' in state and state['energy']:
            closest_energy = min(state['energy'], key=lambda e: abs(e[0]-px) + abs(e[1]-py))
            ex, ey = closest_energy
            
            # Move towards closest energy
            if abs(ex - px) > abs(ey - py):  # Prioritize horizontal movement
                action = 'd' if ex > px else 'a'
            else:
                action = 's' if ey > py else 'w'
        else:
            # Default movement pattern
            if self.last_action in ['a', 'd']:  # If moving horizontally
                action = random.choice(['a', 'd'])
            else:  # If moving vertically  
                action = random.choice(['w', 's'])
                
        self.last_action = action
        return action

async def agent_loop(agent_name="SimpleAI", server_address="localhost:8000"):
    """Main agent loop"""
    agent = SimpleAgent()
    
    async with websockets.connect(f"ws://{server_address}/player") as websocket:
        # Join the game
        await websocket.send(json.dumps({"cmd": "join", "name": agent_name}))
        msg = await websocket.recv()
        
        logger.info(f"Agent {agent_name} joined the game")
        game_properties = json.loads(msg)
        logger.info(f"Map: {game_properties['map']}, Ghosts: {game_properties['ghosts']}")
        
        # Game loop
        while True:
            try:
                # Receive game state
                state_msg = await websocket.recv()
                state = json.loads(state_msg)
                
                # Check if game over
                if 'lives' in state:
                    if not state['lives']:
                        logger.info(f"GAME OVER - Final Score: {state['score']}")
                        break
                else:
                    logger.info(f"VICTORY! - Final Score: {state['score']}")
                    break
                
                # Choose action
                action = agent.get_action(state)
                
                # Send action to server
                await websocket.send(json.dumps({"cmd": "key", "key": action}))
                
                # Optional: Log progress occasionally
                if state.get('step', 0) % 100 == 0:
                    logger.info(f"Step: {state.get('step', 0)}, Score: {state.get('score', 0)}, Lives: {state.get('lives', 0)}")
                    
            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed")
                break
            except KeyboardInterrupt:
                logger.info("Agent stopped by user")
                break

def main():
    """Run the agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Pac-Man AI Agent')
    parser.add_argument('--name', default='SimpleAI', help='Agent name')
    parser.add_argument('--server', default='localhost', help='Server address')
    parser.add_argument('--port', default='8000', help='Server port')
    args = parser.parse_args()
    
    server_address = f"{args.server}:{args.port}"
    
    try:
        asyncio.run(agent_loop(args.name, server_address))
    except KeyboardInterrupt:
        print("\nAgent stopped by user")

if __name__ == "__main__":
    main()