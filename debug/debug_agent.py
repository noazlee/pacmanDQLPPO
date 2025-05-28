#!/usr/bin/env python3
"""
Debug script to check what's wrong with the trained agent
"""

import torch
import numpy as np
from dql_agent import DQLAgent
from gym_observations import MultiChannelObs

def debug_agent_behavior():
    """Check if the agent is actually learning"""
    print("Debugging Agent Behavior...")
    
    # Create agent
    agent = DQLAgent(obs_type=MultiChannelObs, use_cnn=True)
    
    # Create environment (same as testing)
    env = agent.test.__defaults__  # This won't work, let's create manually
    from gym_pacman import PacmanEnv
    
    env = PacmanEnv(
        obs_type=MultiChannelObs,
        positive_rewards=True,
        agent_name="DebugAgent",
        ghosts=2,
        level_ghosts=1,
        lives=3,
        timeout=3000,
        training=False
    )
    
    print(f"Environment created: {env.observation_space.shape}")
    
    # Load the model
    from dql_model import create_dqn
    policy_dqn = create_dqn(env.observation_space, use_cnn=True).to(agent.device)
    
    try:
        policy_dqn.load_state_dict(torch.load("pacman_dqn_final.pt", map_location=agent.device))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    policy_dqn.eval()
    
    # Test a single step
    print("\nTesting Agent Decision Making...")
    
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"State min/max: {state.min():.2f}/{state.max():.2f}")
    
    # Convert to tensor
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    print(f"State tensor shape: {state_tensor.shape}")
    
    # Get Q-values
    with torch.no_grad():
        q_values = policy_dqn(state_tensor)
        print(f"Q-values: {q_values.squeeze().cpu().numpy()}")
        
        # Check if all Q-values are the same (indicating untrained network)
        q_vals = q_values.squeeze().cpu().numpy()
        if np.allclose(q_vals, q_vals[0], atol=0.1):
            print("WARNING: All Q-values are nearly identical - network may be untrained!")
        
        action = q_values.argmax().item()
        print(f"Selected action: {action} ({agent.ACTIONS[action]})")
    
    # Test multiple states
    print("\nTesting 10 random states...")
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for i in range(10):
        state = env.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        
        with torch.no_grad():
            q_values = policy_dqn(state_tensor)
            action = q_values.argmax().item()
            action_counts[action] += 1
            
        print(f"State {i+1}: Action {action} ({agent.ACTIONS[action]}) | Q-vals: {q_values.squeeze().cpu().numpy()}")
    
    print(f"\nAction distribution: {action_counts}")
    
    # Check if the agent always chooses the same action
    most_common_action = max(action_counts, key=action_counts.get)
    if action_counts[most_common_action] >= 8:
        print(f"PROBLEM: Agent heavily biased toward action {most_common_action} ({agent.ACTIONS[most_common_action]})")
        print("This suggests the network hasn't learned properly")
    
    env.close()

def check_model_file():
    """Check if the model file exists and is valid"""
    print("\nchecking Model File...")
    
    import os
    if not os.path.exists("pacman_dqn_final.pt"):
        print("Model file 'pacman_dqn_final.pt' not found!")
        return False
    
    try:
        model_data = torch.load("pacman_dqn_final.pt", map_location='cpu')
        print(f"Model file loaded successfully")
        print(f"Model contains {len(model_data)} parameter tensors")
        
        # Check if parameters are reasonable (not all zeros)
        total_params = 0
        zero_params = 0
        for param_name, param_tensor in model_data.items():
            total_params += param_tensor.numel()
            zero_params += (param_tensor == 0).sum().item()
        
        zero_ratio = zero_params / total_params
        print(f"Zero parameters: {zero_ratio:.2%} ({zero_params}/{total_params})")
        
        if zero_ratio > 0.8:
            print("⚠️  WARNING: Too many zero parameters - model may not be trained")
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("DQL Agent Debug Session")
    print("=" * 60)
    
    # Check model file first
    if check_model_file():
        # Then debug behavior
        debug_agent_behavior()
    
    print("\n" + "=" * 60)
    print("Debug session completed!")