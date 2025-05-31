#!/usr/bin/env python3
"""
Test PPO model with no ghosts to see basic functionality
"""

import torch
import numpy as np
import os
import json
from gym_pacman import PacmanEnv
from gym_observations import MultiChannelObs

def test_ppo_no_ghosts(model_path="ppo_model_ep4000.pt", episodes=5):
    """Test PPO with no ghosts - much easier environment"""
    print("🧪 TESTING PPO MODEL - NO GHOSTS ENVIRONMENT")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print("Environment: 0 ghosts, extended timeout")
    print("="*60)
    
    # Check model exists
    if not os.path.exists(model_path):
        print(f"❌ Model {model_path} not found!")
        ppo_files = [f for f in os.listdir('.') if f.endswith('.pt') and 'ppo' in f]
        if ppo_files:
            model_path = ppo_files[0]
            print(f"🔧 Using {model_path} instead")
        else:
            print("No PPO models found!")
            return
    
    # Create EASY environment
    env = PacmanEnv(
        obs_type=MultiChannelObs,
        positive_rewards=True,
        agent_name="PPONoGhosts",
        ghosts=0,                # NO GHOSTS AT ALL
        level_ghosts=0,
        lives=5,                 # Extra lives
        timeout=2000,            # Longer timeout
        training=False
    )
    
    print(f"Environment created:")
    print(f"  - Ghosts: 0")
    print(f"  - Lives: 5") 
    print(f"  - Timeout: 2000 steps")
    print(f"  - Map size: {env._game.map.size}")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Device: {device}")
    
    try:
        # Import model creation - handling potential import issues
        try:
            from paste import create_actor_critic
        except ImportError:
            print("❌ Could not import from 'paste' - creating model manually")
            # Fallback: create model architecture manually
            from ppo_model import create_actor_critic
        
        model = create_actor_critic(env.observation_space, use_cnn=True).to(device)
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Model loaded successfully")
        
        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  - Parameters: {total_params:,}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Creating untrained model for comparison...")
        try:
            from paste import create_actor_critic
        except ImportError:
            from ppo_model import create_actor_critic
        model = create_actor_critic(env.observation_space, use_cnn=True).to(device)
    
    model.eval()
    
    # Test the model
    results = []
    action_names = ['w', 'a', 's', 'd']
    
    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1} ---")
        
        state = env.reset()
        episode_reward = 0
        episode_score = 0
        steps = 0
        max_steps = 1000
        
        # Track movement and actions
        positions = []
        actions = []
        scores_over_time = []
        
        # Get initial state info
        game_state = json.loads(env._game.state)
        initial_energy = len(game_state['energy'])
        print(f"  Initial energy dots: {initial_energy}")
        print(f"  Starting position: {game_state['pacman']}")
        
        while steps < max_steps:
            # Get action from model (deterministic for testing)
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                action_logits, value = model(state_tensor)
                action = torch.argmax(action_logits, dim=-1).item()
                
                # Get probabilities for analysis
                probs = torch.softmax(action_logits, dim=-1).squeeze().cpu().numpy()
            
            # Record before step
            prev_game_state = json.loads(env._game.state)
            prev_pos = prev_game_state['pacman']
            prev_score = prev_game_state['score']
            
            # Take action
            state, reward, done, info = env.step(action)
            
            # Record after step
            new_game_state = json.loads(env._game.state)
            new_pos = new_game_state['pacman']
            new_score = new_game_state['score']
            
            positions.append(tuple(new_pos))  # Convert to tuple for hashing
            actions.append(action_names[action])
            scores_over_time.append(new_score)
            
            episode_reward += reward
            episode_score = new_score
            steps += 1
            
            # Check movement and scoring
            moved = prev_pos != new_pos
            scored = new_score > prev_score
            
            # Print first 10 steps for debugging
            if steps <= 10:
                # Check if there's an energy dot at current position
                energy_at_pos = tuple(new_pos) in [(tuple(e) if isinstance(e, list) else e) for e in new_game_state['energy']]
                boost_at_pos = tuple(new_pos) in [(tuple(b) if isinstance(b, list) else b) for b in new_game_state['boost']]
                
                print(f"    Step {steps:2d}: {action_names[action]} | "
                      f"{prev_pos} -> {new_pos} | "
                      f"Score: {prev_score} -> {new_score} | "
                      f"Reward: {reward:6.1f} | "
                      f"{'✅ MOVED' if moved else '🧱 STUCK'} | "
                      f"{'🎉 SCORED!' if scored else ''} | "
                      f"{'🟡 ENERGY HERE' if energy_at_pos else '⚫ NO ENERGY'}")
                
                # Show action probabilities for first few steps
                if steps <= 3:
                    print(f"         Probs: w={probs[0]:.3f}, a={probs[1]:.3f}, s={probs[2]:.3f}, d={probs[3]:.3f}")
                
                # Show nearby energy dots for first few steps
                if steps <= 5:
                    nearby_energy = []
                    for energy_pos in new_game_state['energy'][:5]:  # Show first 5 energy positions
                        if isinstance(energy_pos, list):
                            energy_pos = tuple(energy_pos)
                        distance = abs(energy_pos[0] - new_pos[0]) + abs(energy_pos[1] - new_pos[1])
                        if distance <= 3:  # Within 3 steps
                            nearby_energy.append(f"{energy_pos}(d={distance})")
                    
                    if nearby_energy:
                        print(f"         Nearby energy: {', '.join(nearby_energy)}")
                    else:
                        print(f"         No energy within 3 steps")
            
            if done:
                print(f"    🏁 Episode ended at step {steps}")
                break
        
        # Episode analysis
        unique_positions = len(set(positions))
        unique_actions = len(set(actions))
        energy_remaining = len(new_game_state['energy'])
        energy_collected = initial_energy - energy_remaining
        
        # Action distribution
        action_counts = {name: actions.count(name) for name in action_names}
        most_common_action = max(action_counts, key=action_counts.get)
        action_bias = action_counts[most_common_action] / len(actions) * 100
        
        result = {
            'episode': episode + 1,
            'score': episode_score,
            'reward': episode_reward,
            'steps': steps,
            'unique_positions': unique_positions,
            'unique_actions': unique_actions,
            'energy_collected': energy_collected,
            'energy_remaining': energy_remaining,
            'action_bias': action_bias,
            'most_common_action': most_common_action,
            'win': info.get('win', 0) == 1,
            'action_distribution': action_counts
        }
        
        results.append(result)
        
        # Episode summary
        print(f"  📊 Episode Summary:")
        print(f"     Score: {episode_score:3d} | Reward: {episode_reward:7.1f} | Steps: {steps:3d}")
        print(f"     Positions visited: {unique_positions:2d} | Actions used: {unique_actions}")
        print(f"     Energy collected: {energy_collected}/{initial_energy}")
        print(f"     Action bias: {action_bias:.1f}% toward '{most_common_action}'")
        print(f"     Result: {'🏆 WIN' if result['win'] else '💀 LOSE/TIMEOUT'}")
        
        # Identify problems
        if episode_score == 0:
            print(f"     🔴 PROBLEM: No score - agent not collecting energy")
        if unique_actions == 1:
            print(f"     🔴 PROBLEM: Only one action type - policy collapsed")
        if unique_positions <= 3:
            print(f"     🔴 PROBLEM: Barely moving - stuck or hitting walls")
        if action_bias >= 80:
            print(f"     🔴 PROBLEM: Severe action bias - policy likely collapsed")
        if energy_collected == 0:
            print(f"     🔴 PROBLEM: No energy collected - not finding/reaching dots")
    
    # Overall analysis
    print(f"\n📊 OVERALL RESULTS ANALYSIS:")
    print("="*50)
    
    avg_score = np.mean([r['score'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    total_wins = sum(r['win'] for r in results)
    avg_positions = np.mean([r['unique_positions'] for r in results])
    avg_actions = np.mean([r['unique_actions'] for r in results])
    avg_energy = np.mean([r['energy_collected'] for r in results])
    
    print(f"Average Score: {avg_score:.1f}")
    print(f"Average Reward: {avg_reward:.1f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Total Wins: {total_wins}/{episodes}")
    print(f"Average Positions Visited: {avg_positions:.1f}")
    print(f"Average Action Types Used: {avg_actions:.1f}")
    print(f"Average Energy Collected: {avg_energy:.1f}")
    
    # Overall action distribution
    all_actions = []
    for r in results:
        for action, count in r['action_distribution'].items():
            all_actions.extend([action] * count)
    
    if all_actions:
        overall_action_counts = {name: all_actions.count(name) for name in action_names}
        total_actions = len(all_actions)
        
        print(f"\nOverall Action Distribution:")
        for name in action_names:
            pct = overall_action_counts[name] / total_actions * 100
            print(f"  {name}: {overall_action_counts[name]:4d} ({pct:5.1f}%)")
        
        max_action_pct = max(overall_action_counts[name] / total_actions * 100 for name in action_names)
        if max_action_pct >= 70:
            print(f"🔴 SEVERE POLICY COLLAPSE: {max_action_pct:.1f}% bias")
        elif max_action_pct >= 50:
            print(f"🟡 POLICY BIAS: {max_action_pct:.1f}% bias")
        else:
            print(f"✅ BALANCED POLICY: No severe bias")
    
    # Final diagnosis
    print(f"\n🩺 DIAGNOSIS:")
    print("="*30)
    
    if avg_score == 0:
        print("🔴 CRITICAL FAILURE: Model never scores")
        print("   - Policy likely collapsed to always choosing same action")
        print("   - Or model is moving but not toward energy dots")
        
    elif avg_score < 5:
        print("🔴 SEVERE PROBLEMS: Very low performance")
        print("   - Model moves but ineffectively")
        print("   - Poor exploration or decision making")
        
    elif avg_score < 20:
        print("🟡 MODERATE ISSUES: Some learning but needs work")
        print("   - Model shows basic functionality")
        print("   - Need better training or hyperparameters")
        
    else:
        print("✅ REASONABLE PERFORMANCE: Model learned basics")
        print("   - Can navigate and collect some energy")
        print("   - Ready to test with ghosts")
    
    # Specific recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if avg_actions < 2:
        print("   - Policy collapsed - retrain with lower learning rate")
        print("   - Check KL divergence and early stopping")
    
    if avg_positions < 10:
        print("   - Agent not exploring - check reward structure")
        print("   - Ensure positive rewards for movement")
    
    if avg_energy < 5:
        print("   - Agent not collecting energy - check observation encoding")
        print("   - Verify energy dots are visible in observations")
    
    env.close()
    return results

def main():
    """Run the no-ghosts test"""
    print("🎮 PPO MODEL TESTING - NO GHOSTS VERSION")
    
    # Find PPO model files
    ppo_files = [f for f in os.listdir('.') if f.endswith('.pt') and 'ppo' in f.lower()]
    
    if not ppo_files:
        print("❌ No PPO model files found!")
        print("Looking for files containing 'ppo'...")
        all_pt_files = [f for f in os.listdir('.') if f.endswith('.pt')]
        print(f"Available .pt files: {all_pt_files}")
        return
    
    print(f"Found PPO models: {ppo_files}")
    
    # Test each model or just the first one
    model_to_test = ppo_files[0]
    print(f"Testing: {model_to_test}")
    
    results = test_ppo_no_ghosts(model_to_test, episodes=5)
    
    print(f"\n✅ Testing complete!")

if __name__ == "__main__":
    main()