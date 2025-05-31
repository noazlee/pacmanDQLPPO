#!/usr/bin/env python3
"""
PPO Agent that connects to the Pacman server like a regular client
This allows using the existing viewer.py for real-time visualization
"""

import asyncio
import websockets
import json
import torch
import numpy as np
from gym_observations import MultiChannelObs
from gym_pacman import PacmanEnv
import os
import argparse
import sys


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PPO Pacman Agent')
    parser.add_argument('--model', default='ppo_model_ep2000.pt',
                       help='Path to the trained model file')
    return parser.parse_args()


# Import your actual model
try:
    from ppo_model import PacmanActorCritic, create_actor_critic
    PPO_MODEL_AVAILABLE = True
except ImportError:
    print("Warning: ppo_model.py not found, using fallback implementation")
    PPO_MODEL_AVAILABLE = False


class FallbackPPOPolicy(torch.nn.Module):
    """Fallback PPO Policy Network if ppo_model.py is not available"""
    def __init__(self, observation_space, action_space, use_cnn=True):
        super(FallbackPPOPolicy, self).__init__()
        
        self.use_cnn = use_cnn
        observation_shape = observation_space.shape
        channels, height, width = observation_shape
        
        # Your actual architecture from ppo_model.py
        self.conv1 = torch.nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2)
        
        # Calculate conv output size
        conv_output_size = self._get_conv_output_size(observation_shape)
        
        # Shared feature layer
        self.shared_fc = torch.nn.Linear(conv_output_size, 512)
        
        # Actor network (policy)
        self.actor_fc1 = torch.nn.Linear(512, 256)
        self.actor_head = torch.nn.Linear(256, 4)  # 4 actions
        
        # Value network (critic) 
        self.critic_fc1 = torch.nn.Linear(512, 256)
        self.critic_head = torch.nn.Linear(256, 1)
    
    def _get_conv_output_size(self, shape):
        """Calculate the output size after conv layers"""
        dummy_input = torch.zeros(1, *shape)
        dummy_output = self._forward_conv(dummy_input)
        return dummy_output.view(1, -1).size(1)
    
    def _forward_conv(self, x):
        """Forward pass through convolutional layers only"""
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        return x
    
    def forward(self, x):
        # Handle single observation (add batch dimension)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        # Normalize input to [0, 1] range if needed
        if x.max() > 1.0:
            x = x.float() / 255.0
        
        # Shared CNN feature extraction
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        shared_features = torch.nn.functional.relu(self.shared_fc(x))
        
        # Actor network (policy)
        actor_features = torch.nn.functional.relu(self.actor_fc1(shared_features))
        action_logits = self.actor_head(actor_features)
        
        # Critic network (value function)
        critic_features = torch.nn.functional.relu(self.critic_fc1(shared_features))
        values = self.critic_head(critic_features)
        
        return action_logits, values.squeeze(-1)
    
    def get_action(self, x, deterministic=False):
        action_logits, value = self.forward(x)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=1)
        else:
            action_probs = torch.softmax(action_logits, dim=1)
            action = torch.multinomial(action_probs, 1).squeeze(1)
        
        return action, value


class PPOServerClient:
    """PPO Agent that connects to Pacman server for real-time gameplay"""
    
    def __init__(self, model_path="ppo_model_ep2000.pt", server_address="localhost:8000"):
        self.model_path = model_path
        self.server_address = server_address
        self.device = torch.device("cpu")
        
        # Load the trained model
        self.policy = None
        self.load_model()
        
        # Create a dummy environment to get observation processor
        self.setup_observation_processor()
        
        print(f"PPO Agent ready to connect to server at {server_address}")
    
    def load_model(self):
        """Load the trained PPO model"""
        # Check for environment variable first (for evaluation script)
        model_path = os.environ.get('PPO_MODEL_PATH', self.model_path)
        
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found!")
            print("The agent will play randomly.")
            return
        
        try:
            # Create a dummy environment to get spaces
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
            
            # Create your actual PPO model
            if PPO_MODEL_AVAILABLE:
                print("Using actual PacmanActorCritic model")
                self.policy = create_actor_critic(dummy_env.observation_space, use_cnn=True).to(self.device)
            else:
                print("Using fallback PPO implementation")
                # Fallback to recreated implementation
                self.policy = FallbackPPOPolicy(dummy_env.observation_space, dummy_env.action_space, use_cnn=True).to(self.device)
            
            # Load the model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Your model saves just the state dict directly
            self.policy.load_state_dict(checkpoint)
            self.policy.eval()
            
            print(f"✅ Loaded trained PPO model from {model_path}")
            dummy_env.close()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("The agent will play randomly.")
            self.policy = None
    
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
        if self.policy is None:
            # Random action if no model
            return np.random.choice(['w', 'a', 's', 'd'])
        
        try:
            # Convert game state to observation
            obs = self.obs_processor.get_obs(game_state, self.game_map)
            
            # Convert to tensor
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                if PPO_MODEL_AVAILABLE:
                    # Use your actual PacmanActorCritic model
                    action_logits, _ = self.policy(state_tensor)
                    action_idx = torch.argmax(action_logits, dim=1).item()
                else:
                    # Use fallback implementation
                    action_logits, _ = self.policy(state_tensor)
                    action_idx = torch.argmax(action_logits, dim=1).item()
            
            # Convert action index to key
            actions = ['w', 'a', 's', 'd']
            return actions[action_idx]
            
        except Exception as e:
            print(f"Error getting action: {e}")
            return np.random.choice(['w', 'a', 's', 'd'])
    
    async def play_on_server(self, agent_name="PPO_Agent"):
        """Connect to server and play using trained model"""
        print(f"🔗 Connecting to server at ws://{self.server_address}/")
        
        try:
            async with websockets.connect(f"ws://{self.server_address}/") as websocket:
                print("Connected to server!")
                
                # Join the game
                join_msg = json.dumps({"cmd": "join", "name": agent_name})
                await websocket.send(join_msg)
                
                # Receive game info
                game_info_msg = await websocket.recv()
                game_info = json.loads(game_info_msg)
                print(f"Game info received: {game_info.get('map', 'unknown map')}")
                
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
                                print(f"GAME OVER! Final Score: {final_score}")
                                break
                        else:
                            # Game completed successfully
                            final_score = game_state.get('score', 0)
                            print(f"VICTORY! Final Score: {final_score}")
                            break
                        
                        # Get action from trained model
                        action = self.get_action_from_game_state(game_state)
                        
                        # Send action to server
                        action_msg = json.dumps({"cmd": "key", "key": action})
                        await websocket.send(action_msg)
                        
                        step_count += 1
                        current_score = game_state.get('score', 0)
                        
                        # Print progress occasionally
                        if step_count % 4 == 0 or current_score > last_score:
                            lives = game_state.get('lives', 0)
                            print(f"Step {step_count:3d} | Score: {current_score:3d} | Lives: {lives} | Action: {action}")
                            last_score = current_score
                        
                    except websockets.exceptions.ConnectionClosed:
                        print("Connection closed by server")
                        break
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        continue
                    except KeyboardInterrupt:
                        print("Interrupted by user")
                        break
                    except Exception as e:
                        print(f"Unexpected error: {e}")
                        continue
                        
        except Exception as e:
            print(f"Connection error: {e}")
            print("Make sure the server is running:")
            print("  python server.py --map data/fixed_classic.bmp --ghosts 2")


async def main():
    """Main function to run PPO agent on server"""
    print("PPO Agent Server Client")
    print("="*50)
    
    # Parse command line arguments
    args = parse_arguments()
    model_path = args.model
    
    # Check for environment variable (used by evaluation script)
    if 'PPO_MODEL_PATH' in os.environ:
        model_path = os.environ['PPO_MODEL_PATH']
        print(f"Using model from environment: {model_path}")
    
    # Check if model exists
    if os.path.exists(model_path):
        print(f"Using model: {model_path}")
    else:
        print(f"⚠️  Model {model_path} not found - agent will play randomly")
    
    # Create agent
    agent = PPOServerClient(model_path=model_path)
    
    # Play on server
    await agent.play_on_server("PPO_Agent_v1")


if __name__ == "__main__":
    print("Instructions:")
    print("1. Start the server: python server.py --map data/fixed_classic.bmp --ghosts 2")
    print("2. Start the viewer: python viewer.py")
    print("3. Run this agent: python ppo_server_client.py")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Agent stopped by user")