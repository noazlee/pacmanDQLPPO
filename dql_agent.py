#!/usr/bin/env python3
"""
FIXED DQL Agent - Addresses the complete policy collapse issue
Key fixes: Better initialization, stable training, proper reward structure
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import os

from dql_model import create_dqn
from gym_pacman import PacmanEnv
from gym_observations import MultiChannelObs

class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class ImprovedDQLAgent():
    """FIXED DQL Agent - addresses policy collapse and training instability"""
    
    # FIXED: Much more conservative and stable hyperparameters
    learning_rate = 0.0001          # Much lower - was causing Q-value explosion
    discount_factor = 0.99
    network_sync_rate = 2000        # Less frequent updates
    replay_memory_size = 100000     # Larger memory for stability
    mini_batch_size = 64            # Larger batch for stable gradients
    
    # FIXED: Much longer and more conservative exploration
    epsilon_start = 1.0             
    epsilon_end = 0.05              # Higher minimum exploration to prevent collapse
    epsilon_decay_episodes = 15000  # Much longer exploration phase
    
    # FIXED: More conservative training
    gradient_clip = 0.5             # Stricter gradient clipping
    warmup_episodes = 1000          # Longer warmup period
    train_frequency = 4             # Train less frequently initially
    
    loss_fn = nn.SmoothL1Loss()     
    optimizer = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ACTIONS = ['w', 'a', 's', 'd']
    
    def __init__(self, obs_type=MultiChannelObs, use_cnn=True):
        self.obs_type = obs_type
        self.use_cnn = use_cnn
    
    def select_action_with_tiebreaking(self, q_values_tensor, epsilon=0.0):
        """Proper action selection that prevents policy collapse"""
        if random.random() < epsilon:
            return random.randint(0, 3)
        
        q_values = q_values_tensor.squeeze().cpu().numpy()
        
        # Add tiny random noise to break ties and prevent always choosing same action
        noise = np.random.normal(0, 1e-6, size=q_values.shape)
        q_values_noisy = q_values + noise
        
        # Select action with highest Q-value (with tiebreaking)
        return np.argmax(q_values_noisy)
    
    def train(self, episodes=20000, render_freq=0, save_freq=2000):
        print("Starting FIXED DQL training with policy collapse prevention")
        print("="*70)
        
        # Create environment with simpler start
        env = PacmanEnv(
            obs_type=self.obs_type,
            positive_rewards=True,  
            agent_name="fixed_dql",
            ghosts=0,               # Start with no ghosts
            level_ghosts=0,         
            lives=5,                # More lives for stable learning
            timeout=3000,           
            training=True           
        )
        
        print(f"Environment: Starting with simplified setup for stable learning")
        
        # Create networks with MUCH better initialization
        policy_dqn = create_dqn(env.observation_space, use_cnn=self.use_cnn).to(self.device)
        target_dqn = create_dqn(env.observation_space, use_cnn=self.use_cnn).to(self.device)
        
        # CRITICAL: Proper weight initialization to prevent Q-value explosion
        def safe_init(m):
            if isinstance(m, nn.Linear):
                if m.out_features == 4:  # Output layer - CRITICAL
                    # Initialize output layer to small values near zero
                    nn.init.uniform_(m.weight, -0.001, 0.001)
                    nn.init.constant_(m.bias, 0.0)
                    print(f"Output layer initialized safely: {m.weight.data.abs().max().item():.6f}")
                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        policy_dqn.apply(safe_init)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        
        # FIXED: Much more conservative optimizer
        self.optimizer = torch.optim.Adam(
            policy_dqn.parameters(), 
            lr=self.learning_rate,
            eps=1e-8,
            weight_decay=1e-5  # Add weight decay for stability
        )
        
        memory = ReplayMemory(self.replay_memory_size)
        
        # Initialize tracking variables
        rewards_per_episode = []
        scores_per_episode = []
        running_avg_scores = []
        epsilon_history = []
        loss_history = []
        q_value_history = []
        episode_lengths = []
        
        step_count = 0
        best_score = -float('inf')
        episodes_since_improvement = 0
        
        # Check initial Q-values
        print("\nChecking initial Q-values...")
        with torch.no_grad():
            dummy_state = env.reset()
            dummy_tensor = torch.FloatTensor(dummy_state).unsqueeze(0).to(self.device)
            initial_q = policy_dqn(dummy_tensor).squeeze().cpu().numpy()
            print(f"Initial Q-values: {initial_q}")
            print(f"Initial Q-value range: [{initial_q.min():.6f}, {initial_q.max():.6f}]")
            print(f"Initial Q-value std: {initial_q.std():.6f}")
        
        for episode in range(episodes):
            # Progressive curriculum - start simple
            if episode == 5000:
                print("🎓 Adding 1 ghost at episode 5000")
                env._game._n_ghosts = 1
                env._game._l_ghosts = 0
            elif episode == 10000:
                print("🎓 Adding 2nd ghost at episode 10000")
                env._game._n_ghosts = 2
                env._game._l_ghosts = 0
            elif episode == 15000:
                print("🎓 Increasing ghost intelligence at episode 15000")
                env._game._l_ghosts = 1
            
            state = env.reset()
            episode_reward = 0
            episode_score = 0
            episode_steps = 0
            terminated = False
            episode_q_values = []
            
            # Track action distribution to detect collapse
            episode_actions = [0, 0, 0, 0]

            # Conservative epsilon decay
            if episode < self.epsilon_decay_episodes:
                epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (episode / self.epsilon_decay_episodes)
            else:
                epsilon = self.epsilon_end
            
            while not terminated and episode_steps < 2000:
                # Action selection with proper tie-breaking
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        q_values = policy_dqn(state_tensor)
                        action = self.select_action_with_tiebreaking(q_values, epsilon=0.0)
                        
                        # Track Q-values
                        q_vals = q_values.squeeze().cpu().numpy()
                        episode_q_values.append(q_vals.copy())
                
                episode_actions[action] += 1
                
                # Take action
                next_state, reward, terminated, info = env.step(action)
                
                episode_reward += reward
                episode_score = info.get('score', 0)
                episode_steps += 1
                
                memory.append((state, action, next_state, reward, terminated))
                
                # CONSERVATIVE training - only after sufficient warmup
                if (episode >= self.warmup_episodes and 
                    len(memory) > self.mini_batch_size * 4 and 
                    step_count % self.train_frequency == 0):
                    
                    loss = self.optimize_stable(memory, policy_dqn, target_dqn)
                    if loss is not None:
                        loss_history.append(loss)
                        
                        # Check for training instability
                        if len(loss_history) > 100:
                            recent_loss = np.mean(loss_history[-100:])
                            if recent_loss > 100:  # Loss explosion
                                print(f"Training instability detected at episode {episode}, loss: {recent_loss:.2f}")
                
                state = next_state
                step_count += 1
                
                # Conservative target network updates
                if step_count % self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
            
            # Record statistics
            rewards_per_episode.append(episode_reward)
            scores_per_episode.append(episode_score)
            epsilon_history.append(epsilon)
            episode_lengths.append(episode_steps)
            
            # Calculate running average for smooth plotting
            window_size = min(100, len(scores_per_episode))
            running_avg = np.mean(scores_per_episode[-window_size:])
            running_avg_scores.append(running_avg)
            
            if episode_q_values:
                avg_q_values = np.mean(episode_q_values, axis=0)
                q_value_history.append(avg_q_values)
            
            # CRITICAL: Check for policy collapse
            if episode_steps > 10:  # Only check if episode was long enough
                max_action_pct = max(episode_actions) / sum(episode_actions)
                if max_action_pct > 0.8 and episode > 100:
                    print(f"Policy collapse warning at episode {episode}: {max_action_pct:.1%} bias toward one action")
                    if max_action_pct > 0.95:
                        print(f"STOPPING: Complete policy collapse detected!")
                        break
            
            # Progress reporting 
            if episode % 500 == 0 or episode < 10:
                recent_window = min(100, episode + 1)
                recent_rewards = rewards_per_episode[-recent_window:]
                recent_scores = scores_per_episode[-recent_window:]
                
                avg_reward = np.mean(recent_rewards)
                avg_score = np.mean(recent_scores)
                max_score = max(recent_scores) if recent_scores else 0
                
                # Check Q-value health
                if episode_q_values:
                    recent_q = np.mean(episode_q_values, axis=0)
                    q_range = recent_q.max() - recent_q.min()
                    q_mean = recent_q.mean()
                    
                    print(f"Ep {episode:5d} | "
                          f"Score: {avg_score:5.1f} (max: {max_score:3.0f}) | "
                          f"Reward: {avg_reward:7.1f} | "
                          f"ε: {epsilon:.3f} | "
                          f"Q-range: {q_range:.3f} | Q-mean: {q_mean:.1f}")
                    
                    # Detect Q-value explosion
                    if abs(q_mean) > 1000:
                        print(f"Q-value explosion detected! Mean Q-value: {q_mean:.1f}")
                        print("Consider restarting with lower learning rate")
                        break
                else:
                    print(f"Ep {episode:5d} | Score: {avg_score:5.1f} | Reward: {avg_reward:7.1f} | ε: {epsilon:.3f}")
                
                # Check for improvement
                if max_score > best_score:
                    best_score = max_score
                    episodes_since_improvement = 0
                    if max_score > 20:  # Save promising models
                        self.save_model(policy_dqn, f"fixed_dql_score_{int(max_score)}_ep{episode}.pt", verbose=False)
                else:
                    episodes_since_improvement += 1
                
                # Early stopping if no progress
                if episodes_since_improvement > 3000 and episode > 5000:
                    print(f"\nEarly stopping: No improvement for {episodes_since_improvement} episodes")
                    break
            
            # Periodic saves
            if save_freq > 0 and episode % save_freq == 0 and episode > 0:
                self.save_model(policy_dqn, f"fixed_dql_ep{episode}.pt", verbose=False)
        
        # Save final model
        self.save_model(policy_dqn, "improved_dql_FINAL.pt")
        
        # Create training plots
        self.plot_training_results(scores_per_episode, running_avg_scores, 
                                 rewards_per_episode, epsilon_history, loss_history)
        
        print(f"\nTraining completed!")
        print(f"Best score achieved: {best_score}")
        
        # Final Q-value check
        print("\nFinal Q-value analysis...")
        with torch.no_grad():
            dummy_state = env.reset()
            dummy_tensor = torch.FloatTensor(dummy_state).unsqueeze(0).to(self.device)
            final_q = policy_dqn(dummy_tensor).squeeze().cpu().numpy()
            print(f"Final Q-values: {final_q}")
            print(f"Final Q-value range: [{final_q.min():.3f}, {final_q.max():.3f}]")
            print(f"Final Q-value std: {final_q.std():.3f}")
            
            if final_q.std() < 0.01:
                print("Final Q-values are too similar - model may not have learned properly")
            elif abs(final_q.mean()) > 500:
                print("Q-values are very large - possible instability")
            else:
                print("Final Q-values look reasonable")
        
        env.close()
        
        return policy_dqn, {
            'rewards': rewards_per_episode,
            'scores': scores_per_episode,
            'running_avg_scores': running_avg_scores,
            'epsilon': epsilon_history,
            'loss': loss_history,
            'q_values': q_value_history,
            'best_score': best_score
        }
    
    def optimize_stable(self, memory, policy_dqn, target_dqn):
        """More stable optimization to prevent Q-value explosion"""
        if len(memory) < self.mini_batch_size:
            return None
        
        batch = memory.sample(self.mini_batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        actions = torch.LongTensor([t[1] for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[2] for t in batch])).to(self.device)
        rewards = torch.FloatTensor([t[3] for t in batch]).to(self.device)
        dones = torch.BoolTensor([t[4] for t in batch]).to(self.device)
        
        # Clamp rewards to prevent explosion
        rewards = torch.clamp(rewards, -100, 100)
        
        # Current Q values
        current_q_values = policy_dqn(states).gather(1, actions.unsqueeze(1))
        
        # STABLE target computation
        with torch.no_grad():
            next_q_values = target_dqn(next_states).max(1)[0]
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
            
            # Clamp target Q-values to prevent explosion
            target_q_values = torch.clamp(target_q_values, -200, 200)
        
        # Compute loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        # Optimization with strict gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        
        # Very strict gradient clipping
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), self.gradient_clip)
        
        self.optimizer.step()
        
        return loss.item()
    
    def plot_training_results(self, scores, running_avg_scores, rewards, epsilon, losses):
        """Create comprehensive training plots"""
        print("\nCreating training plots...")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQL Training Results', fontsize=16, fontweight='bold')
        
        episodes = range(len(scores))
        
        # Plot 1: Scores with running average
        ax1.plot(episodes, scores, alpha=0.3, color='lightblue', label='Episode Scores')
        ax1.plot(episodes, running_avg_scores, color='darkblue', linewidth=2, label='Running Average (100 episodes)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.set_title('Training Scores Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add curriculum phase markers
        curriculum_episodes = [5000, 10000, 15000]
        curriculum_labels = ['Add 1 Ghost', 'Add 2nd Ghost', 'Smart Ghosts']
        for ep, label in zip(curriculum_episodes, curriculum_labels):
            if ep < len(scores):
                ax1.axvline(x=ep, color='red', linestyle='--', alpha=0.7)
                ax1.text(ep, max(running_avg_scores) * 0.9, label, rotation=90, 
                        verticalalignment='bottom', fontsize=8)
        
        # Plot 2: Rewards
        if rewards:
            reward_episodes = range(len(rewards))
            running_avg_rewards = []
            for i in range(len(rewards)):
                window_start = max(0, i - 99)
                running_avg_rewards.append(np.mean(rewards[window_start:i+1]))
            
            ax2.plot(reward_episodes, rewards, alpha=0.3, color='lightgreen', label='Episode Rewards')
            ax2.plot(reward_episodes, running_avg_rewards, color='darkgreen', linewidth=2, label='Running Average')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Reward')
            ax2.set_title('Training Rewards Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Epsilon decay
        if epsilon:
            ax3.plot(range(len(epsilon)), epsilon, color='orange', linewidth=2)
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Epsilon')
            ax3.set_title('Exploration Rate (Epsilon) Decay')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training loss
        if losses:
            # Smooth the loss for better visualization
            if len(losses) > 100:
                loss_smooth = []
                for i in range(len(losses)):
                    window_start = max(0, i - 49)
                    loss_smooth.append(np.mean(losses[window_start:i+1]))
                ax4.plot(range(len(losses)), losses, alpha=0.2, color='lightcoral', label='Raw Loss')
                ax4.plot(range(len(loss_smooth)), loss_smooth, color='darkred', linewidth=2, label='Smoothed Loss')
                ax4.legend()
            else:
                ax4.plot(range(len(losses)), losses, color='red', linewidth=2)
            
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Loss')
            ax4.set_title('Training Loss Over Time')
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')  # Log scale for loss
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = 'dql_training_results.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📈 Training plots saved as: {plot_filename}")
        
        # Also create a focused score plot
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, scores, alpha=0.2, color='lightblue', label='Individual Episodes')
        plt.plot(episodes, running_avg_scores, color='darkblue', linewidth=3, label='Running Average (100 episodes)')
        
        # Add best score line
        best_score = max(scores) if scores else 0
        plt.axhline(y=best_score, color='green', linestyle=':', alpha=0.7, label=f'Best Score: {best_score:.1f}')
        
        # Add curriculum markers
        for ep, label in zip(curriculum_episodes, curriculum_labels):
            if ep < len(scores):
                plt.axvline(x=ep, color='red', linestyle='--', alpha=0.7)
                plt.text(ep, max(running_avg_scores) * 0.95, label, rotation=90, 
                        verticalalignment='bottom', fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('DQL Training Progress: Score vs Episode', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add performance zones
        if max(running_avg_scores) > 0:
            plt.axhspan(0, 10, alpha=0.1, color='red', label='Learning Phase')
            plt.axhspan(10, 50, alpha=0.1, color='yellow', label='Improvement Phase')
            if max(running_avg_scores) > 50:
                plt.axhspan(50, max(running_avg_scores), alpha=0.1, color='green', label='Mastery Phase')
        
        score_plot_filename = 'dql_score_progress.png'
        plt.savefig(score_plot_filename, dpi=300, bbox_inches='tight')
        print(f"🎯 Score progress plot saved as: {score_plot_filename}")
        
        # Print final statistics
        if running_avg_scores:
            final_avg = running_avg_scores[-1]
            print(f"\n📊 FINAL TRAINING STATISTICS:")
            print(f"   Final running average score: {final_avg:.2f}")
            print(f"   Best single episode score: {max(scores):.1f}")
            print(f"   Total episodes completed: {len(scores)}")
            print(f"   Training improvement: {final_avg - running_avg_scores[0]:.2f} points" if len(running_avg_scores) > 1 else "")
        
        plt.show()
    
    def save_model(self, model, filepath, verbose=True):
        """Save model state dict"""
        torch.save(model.state_dict(), filepath)
        if verbose:
            print(f"Model saved to {filepath}")
    
    def test(self, episodes=5, model_path="improved_dql_FINAL.pt", render=False):
        """Test the trained agent"""
        print(f"\nTesting FIXED DQL agent from {model_path}")
        
        env = PacmanEnv(
            obs_type=self.obs_type,
            positive_rewards=True,
            agent_name="FixedDQLAgent_Test",
            ghosts=2,           
            level_ghosts=1,     
            lives=3,
            timeout=2000,
            training=False
        )
        
        # Load trained model
        policy_dqn = create_dqn(env.observation_space, use_cnn=self.use_cnn).to(self.device)
        
        if os.path.exists(model_path):
            policy_dqn.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded model from {model_path}")
        else:
            print(f"❌ Model file {model_path} not found! Using untrained network.")
        
        policy_dqn.eval()
        
        test_scores = []
        test_rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_score = 0
            steps = 0
            
            print(f"\nTest Episode {episode + 1}:")
            
            while steps < 1000:
                # Select best action (no exploration)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = policy_dqn(state_tensor)
                    action = self.select_action_with_tiebreaking(q_values, epsilon=0.0)
                    
                    # Print Q-values for first few steps
                    if steps < 5:
                        q_vals = q_values.squeeze().cpu().numpy()
                        print(f"  Step {steps}: Q-values: {q_vals}, Action: {self.ACTIONS[action]}")
                
                state, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_score = info.get('score', 0)
                steps += 1
                
                if done:
                    break
            
            test_scores.append(episode_score)
            test_rewards.append(episode_reward)
            
            win_status = "WIN" if info.get('win', 0) == 1 else "LOSE"
            print(f"  Result: {win_status} | Score: {episode_score:3.0f} | "
                  f"Reward: {episode_reward:6.1f} | Steps: {steps:3d}")
        
        # Final test results
        avg_score = np.mean(test_scores)
        avg_reward = np.mean(test_rewards)
        
        print(f"\nTest Results Summary:")
        print(f"  Average Score: {avg_score:.1f}")
        print(f"  Average Reward: {avg_reward:.1f}")
        print(f"  Best Score: {max(test_scores)}")
        print(f"  Worst Score: {min(test_scores)}")
        
        if avg_score > 50:
            print("EXCELLENT: Model learned successfully!")
        elif avg_score > 10:
            print("GOOD: Model shows learning progress")
        elif avg_score > 0:
            print("PROGRESS: Some learning detected")
        else:
            print("POOR: Model failed to learn")
        
        env.close()
        return test_scores, test_rewards


def main():
    """Run the FIXED DQL training"""
    
    print("FIXED DQL TRAINING - Preventing Policy Collapse")
    print("="*60)
    print("Key fixes applied:")
    print("• Much lower learning rate (0.0001)")
    print("• Proper weight initialization")
    print("• Longer exploration phase")
    print("• Stricter gradient clipping")
    print("• Q-value explosion detection")
    print("• Tie-breaking in action selection")
    print("="*60)
    
    agent = ImprovedDQLAgent(
        obs_type=MultiChannelObs,
        use_cnn=True
    )
    
    trained_model, stats = agent.train(
        episodes=20000,
        render_freq=0,
        save_freq=2000
    )
    
    print(f"\nTraining completed!")
    print(f"Best score achieved: {stats['best_score']}")
    
    # Test the trained agent
    print("\n" + "="*50)
    test_scores, test_rewards = agent.test(episodes=5, render=False)
    
    print(f"\nFixed training complete!")
    print(f"If this model still fails, the issue is likely in the reward structure.")


if __name__ == "__main__":
    main()