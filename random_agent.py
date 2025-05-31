#!/usr/bin/env python3
"""
Test random agent to verify environment works correctly
"""

import random
import json
from gym_pacman import PacmanEnv
from gym_observations import MultiChannelObs

def test_random_agent(episodes=3):
    """Test random agent to see if environment rewards work"""
    print("🎲 TESTING RANDOM AGENT")
    print("="*40)
    print("This will show if the environment can reward energy collection")
    print("A random agent should score some points just by chance")
    print("="*40)
    
    env = PacmanEnv(
        obs_type=MultiChannelObs,
        positive_rewards=True,
        agent_name="RandomAgent",
        ghosts=0,               # No ghosts for clean test
        level_ghosts=0,
        lives=3,
        timeout=1000,           # Shorter episodes
        training=False
    )
    
    action_names = ['w', 'a', 's', 'd']
    results = []
    
    for episode in range(episodes):
        print(f"\n--- Random Episode {episode + 1} ---")
        
        state = env.reset()
        episode_reward = 0
        episode_score = 0
        steps = 0
        positions_visited = set()
        actions_taken = [0, 0, 0, 0]
        
        while steps < 500:  # Limit steps
            # Random action
            action = random.randint(0, 3)
            actions_taken[action] += 1
            
            # Get position before action
            game_state = json.loads(env._game.state)
            prev_pos = tuple(game_state['pacman'])
            prev_score = game_state['score']
            
            # Take action
            state, reward, done, info = env.step(action)
            
            # Get position after action
            game_state = json.loads(env._game.state)
            new_pos = tuple(game_state['pacman'])
            new_score = game_state['score']
            
            positions_visited.add(new_pos)
            episode_reward += reward
            episode_score = new_score
            steps += 1
            
            # Show energy collection
            if new_score > prev_score:
                score_increase = new_score - prev_score
                print(f"  Step {steps:3d}: {action_names[action]} | "
                      f"{prev_pos} -> {new_pos} | "
                      f"🎉 SCORED {score_increase} points! | "
                      f"Total score: {new_score}")
            
            # Show first few steps
            elif steps <= 10:
                moved = prev_pos != new_pos
                print(f"  Step {steps:3d}: {action_names[action]} | "
                      f"{prev_pos} -> {new_pos} | "
                      f"{'✅' if moved else '🧱'} | "
                      f"Reward: {reward:5.2f}")
            
            if done:
                print(f"  Episode ended after {steps} steps")
                break
        
        # Episode summary
        unique_positions = len(positions_visited)
        total_actions = sum(actions_taken)
        action_diversity = 1.0 - max(actions_taken) / total_actions if total_actions > 0 else 0
        
        results.append({
            'episode': episode + 1,
            'score': episode_score,
            'reward': episode_reward,
            'steps': steps,
            'positions': unique_positions,
            'diversity': action_diversity
        })
        
        print(f"  📊 Episode Summary:")
        print(f"     Final Score: {episode_score}")
        print(f"     Total Reward: {episode_reward:.1f}")
        print(f"     Steps: {steps}")
        print(f"     Positions visited: {unique_positions}")
        print(f"     Action diversity: {action_diversity:.3f}")
        
        # Action distribution
        print(f"     Actions: w={actions_taken[0]} a={actions_taken[1]} s={actions_taken[2]} d={actions_taken[3]}")
    
    # Overall results
    print(f"\n📈 RANDOM AGENT RESULTS:")
    print("="*35)
    
    avg_score = sum(r['score'] for r in results) / len(results)
    avg_reward = sum(r['reward'] for r in results) / len(results)
    avg_positions = sum(r['positions'] for r in results) / len(results)
    avg_diversity = sum(r['diversity'] for r in results) / len(results)
    total_scores = [r['score'] for r in results]
    
    print(f"Average Score: {avg_score:.1f}")
    print(f"Average Reward: {avg_reward:.1f}")
    print(f"Average Positions Visited: {avg_positions:.1f}")
    print(f"Average Action Diversity: {avg_diversity:.3f}")
    print(f"Score Range: {min(total_scores)} to {max(total_scores)}")
    
    # Interpretation
    print(f"\n🧠 INTERPRETATION:")
    if avg_score > 5:
        print("✅ ENVIRONMENT WORKS: Random agent can score points")
        print("   Problem is definitely in PPO training, not environment")
    elif avg_score > 0:
        print("🟡 ENVIRONMENT WORKS: Some scoring possible")
        print("   PPO should be able to do much better than random")
    else:
        print("❌ ENVIRONMENT ISSUE: Even random agent can't score")
        print("   There might be a fundamental problem")
    
    if avg_diversity > 0.6:
        print("✅ RANDOM DIVERSITY: Good action distribution as expected")
    else:
        print("❌ RANDOM BIAS: Unexpected action bias (check random number generator)")
    
    print(f"\n💡 CONCLUSION:")
    print(f"   Your PPO model (68.3% bias toward 'd') is much worse than random")
    print(f"   Random agent diversity: {avg_diversity:.3f} vs PPO: ~0.17")
    print(f"   This confirms severe PPO policy collapse")
    
    env.close()
    return results

if __name__ == "__main__":
    test_random_agent(episodes=3)