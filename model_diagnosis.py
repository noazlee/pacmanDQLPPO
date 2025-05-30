#!/usr/bin/env python3
"""
DETAILED MODEL DEBUG ANALYSIS
This script will examine exactly what the trained DQL model is doing step-by-step
to understand why it only takes one action despite showing score increases during training.
"""

import torch
import numpy as np
import json
from gym_pacman import PacmanEnv
from gym_observations import MultiChannelObs
from dql_model import create_dqn
import os

def analyze_model_behavior(model_path="fixed_dql_ep10000.pt", max_steps=100):
    """
    Detailed analysis of what the model is actually doing
    """
    print("🔍 DETAILED MODEL BEHAVIOR ANALYSIS")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model {model_path} not found!")
        return
    
    print(f"📁 Model: {model_path}")
    
    # Create environment (same as evaluation)
    env = PacmanEnv(
        obs_type=MultiChannelObs,
        positive_rewards=True,
        agent_name="DebugAnalysis",
        ghosts=2,
        level_ghosts=1,
        lives=3,
        timeout=3000,
        training=False
    )
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_dqn = create_dqn(env.observation_space, use_cnn=True).to(device)
    
    try:
        policy_dqn.load_state_dict(torch.load(model_path, map_location=device))
        policy_dqn.eval()
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print(f"\n🎮 Starting detailed episode analysis...")
    print("="*60)
    
    # Reset environment and get initial state
    obs = env.reset()
    game_state = json.loads(env._game.state)
    
    print(f"INITIAL STATE:")
    print(f"  Map size: {env._game.map.size}")
    print(f"  Pacman position: {game_state['pacman']}")
    print(f"  Lives: {game_state['lives']}")
    print(f"  Score: {game_state['score']}")
    print(f"  Total energy dots: {len(game_state['energy'])}")
    print(f"  Total power pellets: {len(game_state['boost'])}")
    print(f"  Ghost positions: {[g[0] for g in game_state['ghosts']]}")
    print(f"  Ghost zombie status: {[g[1] for g in game_state['ghosts']]}")
    
    # Show first few energy positions
    if len(game_state['energy']) > 0:
        print(f"  First 10 energy positions: {game_state['energy'][:10]}")
    
    print(f"\n📊 STEP-BY-STEP ANALYSIS:")
    print("-"*60)
    
    actions = ['w', 'a', 's', 'd']
    action_counts = [0, 0, 0, 0]
    q_value_history = []
    position_history = []
    
    step = 0
    done = False
    
    while not done and step < max_steps:
        # Get current game state
        current_state = json.loads(env._game.state)
        
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        # Get Q-values from model
        with torch.no_grad():
            q_values = policy_dqn(obs_tensor)
            q_vals = q_values.squeeze().cpu().numpy()
            action = q_values.argmax().item()
        
        # Record data
        action_counts[action] += 1
        q_value_history.append(q_vals.copy())
        position_history.append(current_state['pacman'])
        
        # Print detailed step info
        if step < 20 or step % 10 == 0:  # Print first 20 steps, then every 10th
            print(f"Step {step:3d}: Pos={current_state['pacman']} | "
                  f"Q=[{q_vals[0]:6.2f}, {q_vals[1]:6.2f}, {q_vals[2]:6.2f}, {q_vals[3]:6.2f}] | "
                  f"Action={action}({actions[action]}) | "
                  f"Score={current_state['score']} | Lives={current_state['lives']}")
            
            # Check if Q-values are all the same (indicating collapsed policy)
            if np.allclose(q_vals, q_vals[0], atol=0.001):
                print(f"      ⚠️  WARNING: All Q-values are nearly identical!")
            
            # Check the Q-value spread
            q_range = q_vals.max() - q_vals.min()
            if q_range < 0.01:
                print(f"      ⚠️  WARNING: Q-values have very small range: {q_range:.6f}")
        
        # Take action
        prev_score = current_state['score']
        prev_lives = current_state['lives']
        prev_energy_count = len(current_state['energy'])
        
        obs, reward, done, info = env.step(action)
        
        new_state = json.loads(env._game.state)
        
        # Check what happened
        score_change = new_state['score'] - prev_score
        lives_change = new_state['lives'] - prev_lives
        energy_change = len(new_state['energy']) - prev_energy_count
        
        if step < 20 or step % 10 == 0:
            print(f"      → Reward={reward:6.1f} | Score Δ={score_change} | Lives Δ={lives_change} | Energy Δ={energy_change}")
            
            if score_change > 0:
                print(f"      🎉 SCORED! Energy collected or bonus gained")
            if lives_change < 0:
                print(f"      💀 DIED! Lost {abs(lives_change)} life/lives")
            if new_state['pacman'] == current_state['pacman']:
                print(f"      🧱 BLOCKED! Tried to move into wall")
        
        step += 1
        
        if done:
            print(f"\n🏁 Episode ended at step {step}")
            print(f"   Final score: {new_state['score']}")
            print(f"   Final lives: {new_state['lives']}")
            print(f"   Win status: {info.get('win', 0)}")
            break
    
    print(f"\n📈 EPISODE SUMMARY:")
    print("="*60)
    print(f"Total steps: {step}")
    print(f"Final score: {json.loads(env._game.state)['score']}")
    
    # Action distribution analysis
    total_actions = sum(action_counts)
    print(f"\nAction distribution:")
    for i, count in enumerate(action_counts):
        percentage = (count / total_actions) * 100 if total_actions > 0 else 0
        print(f"  {actions[i]} (action {i}): {count:3d} times ({percentage:5.1f}%)")
    
    # Check for policy collapse
    max_action_percentage = max(action_counts) / total_actions * 100 if total_actions > 0 else 0
    if max_action_percentage > 90:
        print(f"\n🚨 SEVERE POLICY COLLAPSE: {max_action_percentage:.1f}% of actions are the same!")
    elif max_action_percentage > 70:
        print(f"\n⚠️  POLICY BIAS: {max_action_percentage:.1f}% of actions are the same")
    else:
        print(f"\n✅ Action distribution seems reasonable")
    
    # Q-value analysis
    if q_value_history:
        q_array = np.array(q_value_history)
        print(f"\nQ-value statistics:")
        print(f"  Mean Q-values: [{q_array[:, 0].mean():.3f}, {q_array[:, 1].mean():.3f}, {q_array[:, 2].mean():.3f}, {q_array[:, 3].mean():.3f}]")
        print(f"  Q-value std:   [{q_array[:, 0].std():.3f}, {q_array[:, 1].std():.3f}, {q_array[:, 2].std():.3f}, {q_array[:, 3].std():.3f}]")
        print(f"  Q-value range: {q_array.min():.3f} to {q_array.max():.3f}")
        
        # Check if Q-values are collapsed
        overall_std = q_array.std()
        if overall_std < 0.01:
            print(f"  🚨 Q-values are collapsed! Overall std: {overall_std:.6f}")
        elif overall_std < 0.1:
            print(f"  ⚠️  Q-values have low variance: {overall_std:.6f}")
        else:
            print(f"  ✅ Q-values show reasonable variance: {overall_std:.6f}")
    
    # Movement analysis
    if len(position_history) > 1:
        print(f"\nMovement analysis:")
        unique_positions = len(set(position_history))
        print(f"  Unique positions visited: {unique_positions}")
        print(f"  Position variety: {unique_positions / len(position_history) * 100:.1f}%")
        
        # Check if stuck in one position
        if unique_positions == 1:
            print(f"  🚨 STUCK! Agent never moved from starting position!")
        elif unique_positions < len(position_history) * 0.1:
            print(f"  ⚠️  Very limited movement - mostly stuck")
        else:
            print(f"  ✅ Agent moved around the map")
        
        print(f"  Starting position: {position_history[0]}")
        print(f"  Ending position: {position_history[-1]}")
        
        # Show position pattern for first 20 steps
        if len(position_history) >= 20:
            print(f"  First 20 positions: {position_history[:20]}")
    
    env.close()
    
    print(f"\n🔍 ANALYSIS COMPLETE")
    print("="*60)

def compare_q_values_across_states(model_path="fixed_dql_ep10000.pt"):
    """
    Test Q-values on multiple different game states to see if the model 
    always predicts the same action regardless of state
    """
    print(f"\n🧪 TESTING Q-VALUES ACROSS DIFFERENT STATES")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model {model_path} not found!")
        return
    
    env = PacmanEnv(
        obs_type=MultiChannelObs,
        positive_rewards=True,
        agent_name="QValueTest",
        ghosts=0,  # No ghosts for cleaner testing
        level_ghosts=0,
        lives=3,
        timeout=3000,
        training=False
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_dqn = create_dqn(env.observation_space, use_cnn=True).to(device)
    policy_dqn.load_state_dict(torch.load(model_path, map_location=device))
    policy_dqn.eval()
    
    print("Testing Q-values on 10 different random states...")
    
    all_q_values = []
    all_actions = []
    
    for test_state in range(10):
        obs = env.reset()
        
        # Take a few random actions to get to different states
        for _ in range(test_state * 3):
            random_action = np.random.randint(0, 4)
            obs, _, done, _ = env.step(random_action)
            if done:
                obs = env.reset()
                break
        
        # Get Q-values for this state
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            q_values = policy_dqn(obs_tensor)
            q_vals = q_values.squeeze().cpu().numpy()
            action = q_values.argmax().item()
        
        all_q_values.append(q_vals.copy())
        all_actions.append(action)
        
        game_state = json.loads(env._game.state)
        print(f"State {test_state + 1:2d}: Pos={game_state['pacman']} | "
              f"Q=[{q_vals[0]:6.2f}, {q_vals[1]:6.2f}, {q_vals[2]:6.2f}, {q_vals[3]:6.2f}] | "
              f"Action={action}")
    
    # Analysis
    actions_array = np.array(all_actions)
    q_values_array = np.array(all_q_values)
    
    print(f"\nSummary:")
    print(f"  Actions taken: {actions_array.tolist()}")
    print(f"  Unique actions: {len(np.unique(actions_array))}")
    print(f"  Most common action: {np.bincount(actions_array).argmax()} (taken {np.bincount(actions_array).max()}/10 times)")
    
    if len(np.unique(actions_array)) == 1:
        print(f"  🚨 POLICY COMPLETELY COLLAPSED! Always takes action {actions_array[0]}")
    elif np.bincount(actions_array).max() >= 8:
        print(f"  🔴 SEVERE POLICY COLLAPSE! Heavily biased toward one action")
    elif np.bincount(actions_array).max() >= 6:
        print(f"  🟡 MODERATE POLICY BIAS")
    else:
        print(f"  ✅ Actions show some diversity")
    
    # Check Q-value variance
    q_std = q_values_array.std()
    print(f"  Overall Q-value std: {q_std:.6f}")
    if q_std < 0.001:
        print(f"  🚨 Q-values are essentially identical across all states!")
    elif q_std < 0.01:
        print(f"  ⚠️  Q-values show very little variation")
    else:
        print(f"  ✅ Q-values show some variation")
    
    env.close()

def main():
    print("🔧 COMPREHENSIVE MODEL ANALYSIS")
    print("="*70)
    print("This will analyze exactly what your trained DQL model is doing")
    print("and why it might be showing the reward hacking behavior you observed.")
    print("="*70)
    
    # Check available models
    pt_files = [f for f in os.listdir('.') if f.endswith('.pt')]
    if pt_files:
        print(f"\nAvailable model files:")
        for f in pt_files:
            print(f"  - {f}")
    
    # Analyze the specific model that was failing
    model_path = "fixed_dql_ep10000.pt"
    if not os.path.exists(model_path):
        # Try other models
        if pt_files:
            model_path = pt_files[0]
            print(f"\nUsing {model_path} instead")
        else:
            print("❌ No model files found!")
            return
    
    # Run comprehensive analysis
    analyze_model_behavior(model_path, max_steps=100)
    compare_q_values_across_states(model_path)
    
    print(f"\n🎯 CONCLUSION:")
    print("="*60)
    print("If the analysis shows:")
    print("• All Q-values are nearly identical → Model weights collapsed")
    print("• One action taken 90%+ of time → Severe policy collapse") 
    print("• Agent gets stuck at starting position → Reward hacking")
    print("• Score increases during training but poor evaluation → Exploiting reward bugs")
    print("\nThe most likely causes are:")
    print("1. Reward structure allows gaining points without real progress")
    print("2. Learning rate too high causing Q-value explosion then collapse")
    print("3. Insufficient exploration leading to local optimum")
    print("4. Bug in action selection or environment step function")

if __name__ == "__main__":
    main()