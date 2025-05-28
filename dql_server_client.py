#!/usr/bin/env python3
"""
DQL Agent that connects to the Pacman server like a regular client
This allows using the existing viewer.py for real-time visualization
"""

import asyncio
import websockets
import json
import torch
import numpy as np
from dql_model import create_dqn
from gym_observations import MultiChannelObs
from gym_pacman import PacmanEnv
import os


class DQLServerClient:
    """DQL Agent that connects to Pacman server for real-time gameplay"""
    
    def __init__(self, model_path="pacman_dqn_final.pt", server_address="localhost:8000"):
        self.model_path = model_path
        self.server_address = server_address
        self.device = torch.device("cpu")
        
        # Load the trained model
        self.policy_dqn = None
        self.load_model()
        
        # Create a dummy environment to get observation processor
        self.setup_observation_processor()
        
        print(f"🤖 DQL Agent ready to connect to server at {server_address}")
    
    def load_model(self):
        """Load the trained DQL model"""
        if not os.path.exists(self.model_path):
            print(f"❌ Model file {self.model_path} not found!")
            print("The agent will play randomly.")
            return
        
        try:
            # Create a dummy environment to get observation space
            dummy_env = PacmanEnv(
                obs_type=MultiChannelObs,
                positive_rewards=True,
                agent_name="Dummy",
                ghosts=2,
                level_ghosts=1,
                lives=3,
                timeout=3000,
                training=False
            )
            
            # Create and load model
            self.policy_dqn = create_dqn(dummy_env.observation_space, use_cnn=True).to(self.device)
            self.policy_dqn.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.policy_dqn.eval()
            
            print(f"✅ Loaded trained model from {self.model_path}")
            dummy_env.close()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("The agent will play randomly.")
            self.policy_dqn = None
    
    def setup_observation_processor(self):
        """Set up observation processing"""
        # Create observation processor
        dummy_env = PacmanEnv(
            obs_type=MultiChannelObs,
            positive_rewards=True,
            agent_name="ObsDummy",
            ghosts=2,
            level_ghosts=1,
            lives=3,
            timeout=3000,
            training=False
        )
        
        # Get the observation processor
        self.obs_processor = dummy_env.pacman_obs
        self.game_map = dummy_env._game.map
        dummy_env.close()
    
    def get_action_from_game_state(self, game_state):
        """Convert game state to action using trained model"""
        if self.policy_dqn is None:
            # Random action if no model
            return np.random.choice(['w', 'a', 's', 'd'])
        
        try:
            # Convert game state to observation
            obs = self.obs_processor.get_obs(game_state, self.game_map)
            
            # Convert to tensor
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get Q-values and select action
            with torch.no_grad():
                q_values = self.policy_dqn(state_tensor)
                action_idx = q_values.argmax().item()
            
            # Convert action index to key
            actions = ['w', 'a', 's', 'd']
            return actions[action_idx]
            
        except Exception as e:
            print(f"Error getting action: {e}")
            return np.random.choice(['w', 'a', 's', 'd'])
    
    async def play_on_server(self, agent_name="DQL_Agent"):
        """Connect to server and play using trained model"""
        print(f"🔗 Connecting to server at ws://{self.server_address}/")
        
        try:
            async with websockets.connect(f"ws://{self.server_address}/") as websocket:
                print("✅ Connected to server!")
                
                # Join the game
                join_msg = json.dumps({"cmd": "join", "name": agent_name})
                await websocket.send(join_msg)
                
                # Receive game info
                game_info_msg = await websocket.recv()
                game_info = json.loads(game_info_msg)
                print(f"🎮 Game info received: {game_info.get('map', 'unknown map')}")
                
                step_count = 0
                last_score = 0
                
                # Game loop
                while True:
                    try:
                        # Receive game state
                        state_msg = await websocket.recv()
                        game_state = json.loads(state_msg)
                        
                        # Check if game is over
                        if 'lives' in game_state:
                            if game_state['lives'] == 0:
                                final_score = game_state.get('score', 0)
                                print(f"🎯 GAME OVER! Final Score: {final_score}")
                                break
                        else:
                            # Game completed successfully
                            final_score = game_state.get('score', 0)
                            print(f"🏆 VICTORY! Final Score: {final_score}")
                            break
                        
                        # Get action from trained model
                        action = self.get_action_from_game_state(game_state)
                        
                        # Send action to server
                        action_msg = json.dumps({"cmd": "key", "key": action})
                        await websocket.send(action_msg)
                        
                        step_count += 1
                        current_score = game_state.get('score', 0)
                        
                        # Print progress occasionally
                        if step_count % 50 == 0 or current_score > last_score:
                            lives = game_state.get('lives', 0)
                            print(f"Step {step_count:3d} | Score: {current_score:3d} | Lives: {lives} | Action: {action}")
                            last_score = current_score
                        
                    except websockets.exceptions.ConnectionClosed:
                        print("🔌 Connection closed by server")
                        break
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON decode error: {e}")
                        continue
                    except KeyboardInterrupt:
                        print("🛑 Interrupted by user")
                        break
                    except Exception as e:
                        print(f"❌ Unexpected error: {e}")
                        continue
                        
        except Exception as e:
            print(f"❌ Connection error: {e}")
            print("Make sure the server is running:")
            print("  python server.py --map data/fixed_classic.bmp --ghosts 2")


async def main():
    """Main function to run DQL agent on server"""
    print("🤖 DQL Agent Server Client")
    print("="*50)
    
    # You can specify different model files
    model_path = "pacman_dqn_final.pt"
    
    # Check if model exists
    if os.path.exists(model_path):
        print(f"✅ Using model: {model_path}")
    else:
        print(f"⚠️  Model {model_path} not found - agent will play randomly")
    
    # Create agent
    agent = DQLServerClient(model_path=model_path)
    
    # Play on server
    await agent.play_on_server("DQL_Agent_v1")


if __name__ == "__main__":
    print("Instructions:")
    print("1. Start the server: python server.py --map data/fixed_classic.bmp --ghosts 2")
    print("2. Start the viewer: python viewer.py")
    print("3. Run this agent: python dql_server_client.py")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Agent stopped by user")