#!/usr/bin/env python3
"""
Test the environment's reward structure to see if it's encouraging exploration
"""

import json
from gym_pacman import PacmanEnv
from gym_observations import MultiChannelObs

def test_reward_structure():
    """Test if the environment gives proper rewards for energy collection"""
    print("🔍 TESTING ENVIRONMENT REWARD STRUCTURE")
    print("="*50)
    
    # Create environment  
    env = PacmanEnv(
        obs_type=MultiChannelObs,
        positive_rewards=True,
        agent_name="RewardTest",
        ghosts=0,
        level_ghosts=0,
        lives=3,
        timeout=1000,
        training=False
    )
    
    state = env.reset()
    game_state = json.loads(env._game.state)
    
    print(f"Initial state:")
    print(f"  Position: {game_state['pacman']}")
    print(f"  Score: {game_state['score']}")
    print(f"  Energy dots: {len(game_state['energy'])}")
    
    # Test moving to collect the closest energy
    print(f"\n🎯 Manual pathfinding to collect energy...")
    
    # We know from earlier that closest energy is at (15,13) - just left and up
    actions = ['a', 'w']  # left, then up
    action_indices = [1, 0]  # a=1, w=0
    
    for i, (action_name, action_idx) in enumerate(zip(actions, action_indices)):
        prev_game_state = json.loads(env._game.state)
        prev_pos = prev_game_state['pacman']
        prev_score = prev_game_state['score']
        
        print(f"\nStep {i+1}: Taking action '{action_name}' from {prev_pos}")
        
        state, reward, done, info = env.step(action_idx)
        
        new_game_state = json.loads(env._game.state)
        new_pos = new_game_state['pacman']
        new_score = new_game_state['score']
        
        print(f"  Result: {prev_pos} -> {new_pos}")
        print(f"  Score: {prev_score} -> {new_score}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        
        if new_score > prev_score:
            print(f"  🎉 SUCCESS! Energy collected!")
            print(f"  Environment IS working correctly for energy collection")
            break
        elif new_pos == prev_pos:
            print(f"  🧱 Hit wall - couldn't move")
        else:
            print(f"  ✅ Moved successfully, no energy at this position")
    
    # Test a few random actions to see reward patterns
    print(f"\n🎲 Testing random actions to see reward patterns...")
    
    for i in range(5):
        prev_game_state = json.loads(env._game.state)
        prev_pos = prev_game_state['pacman']
        prev_score = prev_game_state['score']
        
        action = i % 4  # cycle through all actions
        action_name = ['w', 'a', 's', 'd'][action]
        
        state, reward, done, info = env.step(action)
        
        new_game_state = json.loads(env._game.state)
        new_pos = new_game_state['pacman']
        new_score = new_game_state['score']
        
        moved = new_pos != prev_pos
        scored = new_score > prev_score
        
        print(f"  Action {action_name}: {prev_pos} -> {new_pos}, "
              f"Score: {prev_score}->{new_score}, "
              f"Reward: {reward:6.2f}, "
              f"{'MOVED' if moved else 'STUCK'}, "
              f"{'SCORED!' if scored else ''}")
        
        if done:
            break
    
    env.close()

def test_ppo_vs_environment():
    """Compare PPO model behavior with what the environment actually rewards"""
    print(f"\n🤖 TESTING PPO MODEL VS ENVIRONMENT INCENTIVES")
    print("="*55)
    
    # Create environment
    env = PacmanEnv(
        obs_type=MultiChannelObs,
        positive_rewards=True,
        agent_name="ComparisonTest",
        ghosts=0,
        level_ghosts=0,
        lives=3,
        timeout=1000,
        training=False
    )
    
    # Load PPO model
    import torch
    try:
        from paste import create_actor_critic
    except ImportError:
        from ppo_model import create_actor_critic
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_actor_critic(env.observation_space, use_cnn=True).to(device)
    
    try:
        model.load_state_dict(torch.load("ppo_model_ep4000.pt", map_location=device))
        print("✅ PPO model loaded")
    except:
        print("❌ Could not load PPO model - using untrained")
    
    model.eval()
    
    state = env.reset()
    action_names = ['w', 'a', 's', 'd']
    
    print(f"\n📊 Action preferences from PPO model vs environment rewards:")
    
    # Test what PPO model wants to do vs what environment rewards
    for i in range(4):
        # Reset to same starting position
        state = env.reset()
        game_state = json.loads(env._game.state)
        start_pos = game_state['pacman']
        
        # Get PPO model's action preference
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            action_logits, value = model(state_tensor)
            probs = torch.softmax(action_logits, dim=-1).squeeze().cpu().numpy()
            preferred_action = torch.argmax(action_logits, dim=-1).item()
        
        # Test each action to see environment reward
        action_rewards = {}
        for action in range(4):
            env_state = env.reset()  # Reset for clean test
            test_state, reward, done, info = env.step(action)
            action_rewards[action] = reward
        
        print(f"\nFrom position {start_pos}:")
        print(f"  PPO model prefers: {action_names[preferred_action]} (prob: {probs[preferred_action]:.3f})")
        print(f"  Action probabilities: w={probs[0]:.3f}, a={probs[1]:.3f}, s={probs[2]:.3f}, d={probs[3]:.3f}")
        print(f"  Environment rewards:")
        for action in range(4):
            print(f"    {action_names[action]}: {action_rewards[action]:6.2f}")
        
        # Check if PPO prefers the action with highest environment reward
        best_env_action = max(action_rewards, key=action_rewards.get)
        alignment = preferred_action == best_env_action
        
        print(f"  Best environment action: {action_names[best_env_action]}")
        print(f"  PPO-Environment alignment: {'✅ YES' if alignment else '❌ NO'}")
        
        break  # Just test once from starting position
    
    env.close()

def main():
    """Run reward structure tests"""
    test_reward_structure()
    test_ppo_vs_environment()

if __name__ == "__main__":
    main()