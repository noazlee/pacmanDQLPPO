"""
Test script for the Pac-Man RL environment
Tests basic functionality and runs a short episode with random actions
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_pacman import PacmanEnv
from gym_observations import MultiChannelObs, SingleChannelObs

def test_environment_creation():
    """Test that we can create the environment"""
    print("Testing environment creation...")
    
    try:
        env = PacmanEnv(
            obs_type=MultiChannelObs,
            positive_rewards=True,
            agent_name="TestAgent",
            ghosts=2,
            level_ghosts=1,
            lives=3,
            timeout=1000,
            training=False
        )
        print("✓ Environment created successfully")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Observation space: {env.observation_space}")
        return env
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return None

def test_reset_and_step(env):
    """Test basic environment functionality"""
    print("\nTesting reset and step...")
    
    try:
        # Reset environment
        obs = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Observation type: {obs.dtype}")
        
        # Take a few random steps
        total_reward = 0
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                print(f"  - Episode ended at step {i+1}")
                break
                
        print(f"✓ Environment stepping successful")
        print(f"  - Total reward: {total_reward}")
        print(f"  - Final info: {info}")
        return True
        
    except Exception as e:
        print(f"✗ Environment testing failed: {e}")
        return False

def run_short_episode(env, max_steps=100):
    """Run a short episode with random actions"""
    print(f"\nRunning short episode ({max_steps} steps max)...")
    
    obs = env.reset()
    total_reward = 0
    step_count = 0
    
    print("Episode progress:")
    
    while step_count < max_steps:
        # Random action
        action = env.action_space.sample()
        action_names = ['Up', 'Left', 'Down', 'Right']
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Print progress every 10 steps
        if step_count % 10 == 0 or reward != 0 or done:
            print(f"  Step {step_count:3d}: {action_names[action]:5s} -> Reward: {reward:6.1f}, Score: {info.get('score', 0):3d}, Lives: {info.get('lives', 0)}")
        
        if done:
            break
    
    print(f"\nEpisode finished!")
    print(f"  - Steps taken: {step_count}")
    print(f"  - Total reward: {total_reward:.1f}")
    print(f"  - Final score: {info.get('score', 0)}")
    print(f"  - Lives remaining: {info.get('lives', 0)}")
    
    if step_count < max_steps:
        print(f"  - Episode ended naturally (done=True)")
    else:
        print(f"  - Episode ended due to step limit")

def test_different_observation_types():
    """Test both observation types"""
    print("\nTesting different observation types...")
    
    obs_types = [
        ("MultiChannel", MultiChannelObs),
        ("SingleChannel", SingleChannelObs)
    ]
    
    for name, obs_type in obs_types:
        try:
            env = PacmanEnv(
                obs_type=obs_type,
                positive_rewards=True,
                agent_name=f"Test{name}",
                ghosts=1,
                level_ghosts=0,
                lives=1,
                timeout=100,
                training=False
            )
            
            obs = env.reset()
            obs, _, _, _ = env.step(0)  # Take one step
            
            print(f"✓ {name} observation works")
            print(f"  - Shape: {obs.shape}")
            print(f"  - Range: [{obs.min():.1f}, {obs.max():.1f}]")
            
        except Exception as e:
            print(f"✗ {name} observation failed: {e}")

def main():
    """Run all tests"""
    print("=" * 50)
    print("Pac-Man RL Environment Test")
    print("=" * 50)
    
    # Test 1: Environment creation
    env = test_environment_creation()
    if not env:
        print("\n❌ Basic environment creation failed. Check dependencies and files.")
        return
    
    # Test 2: Basic functionality  
    if not test_reset_and_step(env):
        print("\n❌ Basic environment functionality failed.")
        return
    
    # Test 3: Short episode
    run_short_episode(env, max_steps=50)
    
    # Test 4: Different observation types
    test_different_observation_types()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed! Environment is working.")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run server: python server.py")
    print("2. Run viewer: python viewer.py") 
    print("3. Run client: python client.py")
    print("4. Implement your RL algorithm using gym_pacman.PacmanEnv")

if __name__ == "__main__":
    main()