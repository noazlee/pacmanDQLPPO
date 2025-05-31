#!/usr/bin/env python3
"""
FIXED PPO Agent Implementation - Prevents Policy Collapse
Key fixes: Conservative hyperparameters, action diversity monitoring, proper initialization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import os
from typing import List, Tuple, Dict, Optional

from ppo_model import create_actor_critic
from gym_pacman import PacmanEnv
from gym_observations import MultiChannelObs


class RolloutBuffer:
    """
    Buffer for storing rollout data for PPO training
    FIXED: Handles variable episode lengths correctly
    """
    
    def __init__(self, buffer_size: int, observation_shape: Tuple, device: torch.device):
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Initialize buffers
        self.observations = torch.zeros((buffer_size, *observation_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        
        # For advantage calculation
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32, device=device)
    
    def add(self, obs: np.ndarray, action: int, reward: float, value: float, 
            log_prob: float, done: bool):
        """Add a single step to the buffer"""
        self.observations[self.ptr] = torch.from_numpy(obs).to(self.device)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_advantages_and_returns(self, last_value: float, gamma: float = 0.99, 
                                     gae_lambda: float = 0.95):
        """
        Compute advantages using GAE - FIXED to handle variable episode lengths
        """
        if self.size == 0:
            return
            
        with torch.no_grad():
            # Only compute for actual data, not the full buffer
            advantages = torch.zeros(self.size, device=self.device)
            last_gae_lam = 0
            
            for step in reversed(range(self.size)):
                if step == self.size - 1:
                    next_non_terminal = 1.0 - self.dones[step].float()
                    next_values = last_value
                else:
                    next_non_terminal = 1.0 - self.dones[step].float()
                    next_values = self.values[step + 1]
                
                delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
                advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            
            # FIXED: Only assign to actual size, not full buffer
            self.advantages[:self.size] = advantages
            self.returns[:self.size] = advantages + self.values[:self.size]
    
    def get_batch(self, batch_size: int):
        """Get a random batch from the buffer"""
        if self.size < batch_size:
            batch_size = self.size
            
        indices = torch.randperm(self.size, device=self.device)[:batch_size]
        
        return (
            self.observations[indices],
            self.actions[indices],
            self.log_probs[indices],
            self.advantages[indices],
            self.returns[indices],
            self.values[indices]
        )
    
    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0


class PPOAgent:
    """
    FIXED PPO Agent - Conservative hyperparameters to prevent policy collapse
    Based on successful PPO implementations with anti-collapse measures
    """
    
    # FIXED: Much more conservative hyperparameters
    learning_rate = 3e-4           # Conservative learning rate
    gamma = 0.99                   # Standard discount factor
    gae_lambda = 0.95              # GAE lambda
    clip_epsilon = 0.2             # Standard clipping range
    value_loss_coef = 0.5          # Value loss coefficient  
    entropy_coef = 0.01            # Entropy bonus for exploration
    max_grad_norm = 0.5            # Gradient clipping
    
    # FIXED: Conservative training schedule
    steps_per_update = 2048        # Collect more data before updates
    batch_size = 64                # Smaller batches for stability
    ppo_epochs = 10                # Standard number of epochs
    target_kl = 0.01               # KL threshold for early stopping
    
    def __init__(self, obs_type=None, use_cnn=True, device=None):
        self.obs_type = obs_type or MultiChannelObs
        self.use_cnn = use_cnn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Fixed PPO Agent using device: {self.device}")
        print(f"Anti-collapse measures: Conservative LR, batch norm, action monitoring")
        
        # Will be initialized in train()
        self.actor_critic = None
        self.optimizer = None
        self.rollout_buffer = None
    
    def compute_ppo_loss(self, observations, actions, old_log_probs, advantages, returns, old_values):
        """
        Compute PPO loss with additional safeguards against collapse
        """
        # Get current policy outputs
        _, new_log_probs, entropy, new_values = self.actor_critic.get_action_and_value(
            observations, actions
        )
        
        # Normalize advantages (important for stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate ratio
        log_ratio = new_log_probs - old_log_probs
        ratio = log_ratio.exp()
        
        # FIXED: Clamp ratios to prevent extreme updates
        ratio = torch.clamp(ratio, 0.5, 2.0)
        
        # PPO clipped surrogate objective
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        # Value function loss with clipping
        value_pred_clipped = old_values + torch.clamp(
            new_values - old_values, -self.clip_epsilon, self.clip_epsilon
        )
        value_loss_1 = (new_values - returns) ** 2
        value_loss_2 = (value_pred_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (
            policy_loss + 
            self.value_loss_coef * value_loss + 
            self.entropy_coef * entropy_loss
        )
        
        # Calculate KL divergence for monitoring
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
        
        return total_loss, policy_loss, value_loss, entropy_loss, approx_kl
    
    def train(self, episodes=10000, save_freq=2000):
        """Train the FIXED PPO agent with anti-collapse measures"""
        print("🚀 Starting FIXED PPO training")
        print("="*70)
        
        # Create environment
        env = PacmanEnv(
            obs_type=self.obs_type,
            positive_rewards=True,
            agent_name="fixed_ppo",
            ghosts=0,               # Start simple
            level_ghosts=0,
            lives=3,
            timeout=1000,           # Reasonable timeout
            training=True
        )
        
        print(f"Environment: Starting with 0 ghosts for stable learning")
        
        # Create actor-critic network
        self.actor_critic = create_actor_critic(
            env.observation_space, 
            use_cnn=self.use_cnn
        ).to(self.device)
        
        # FIXED: Conservative optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.learning_rate,
            eps=1e-5
        )
        
        # Create rollout buffer
        self.rollout_buffer = RolloutBuffer(
            self.steps_per_update,
            env.observation_space.shape,
            self.device
        )
        
        # Training tracking
        episode_rewards = []
        episode_scores = []
        episode_lengths = []
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divergences = []
        action_distributions = []  # Track action diversity
        
        # Training state
        global_step = 0
        episode_count = 0
        best_score = -float('inf')
        consecutive_good_episodes = 0
        
        # Get initial observation
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_actions = [0, 0, 0, 0]  # Track action counts per episode
        
        print("Starting training with anti-collapse monitoring...")
        print("Will upgrade to ghosts after achieving consistent scores > 5")
        print("-" * 80)
        
        while episode_count < episodes:
            # Collect rollouts
            for step in range(self.steps_per_update):
                global_step += 1
                episode_length += 1
                
                # Get action from policy
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                    action, log_prob, entropy, value = self.actor_critic.get_action_and_value(obs_tensor)
                    action = action.item()
                    log_prob = log_prob.item()
                    value = value.item()
                
                # Track action for diversity monitoring
                episode_actions[action] += 1
                
                # Take action in environment
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Store transition in buffer
                self.rollout_buffer.add(obs, action, reward, value, log_prob, done)
                
                obs = next_obs
                
                # Handle episode termination
                if done:
                    episode_rewards.append(episode_reward)
                    episode_scores.append(info.get('score', 0))
                    episode_lengths.append(episode_length)
                    
                    # Track action diversity
                    total_actions = sum(episode_actions)
                    if total_actions > 0:
                        action_dist = [count/total_actions for count in episode_actions]
                        action_distributions.append(action_dist)
                        
                        # Check for policy collapse
                        max_action_pct = max(action_dist)
                        if max_action_pct > 0.7 and episode_count > 100:
                            print(f"⚠️  Policy collapse warning ep {episode_count}: {max_action_pct:.1%} bias toward action {action_dist.index(max_action_pct)}")
                    
                    # Smart curriculum progression based on performance
                    current_score = info.get('score', 0)
                    if current_score > 5:
                        consecutive_good_episodes += 1
                    else:
                        consecutive_good_episodes = 0
                    
                    # Add ghosts when agent is performing well consistently
                    if consecutive_good_episodes >= 50 and env._game._n_ghosts == 0:
                        print(f"🎓 Adding 1 ghost at episode {episode_count} (50 consecutive scores > 5)")
                        env._game._n_ghosts = 1
                        env._game._l_ghosts = 0
                        consecutive_good_episodes = 0
                    elif consecutive_good_episodes >= 100 and env._game._n_ghosts == 1:
                        print(f"🎓 Adding 2nd ghost at episode {episode_count}")
                        env._game._n_ghosts = 2
                        env._game._l_ghosts = 0
                        consecutive_good_episodes = 0
                    elif consecutive_good_episodes >= 150 and env._game._l_ghosts == 0:
                        print(f"🎓 Increasing ghost intelligence at episode {episode_count}")
                        env._game._l_ghosts = 1
                        consecutive_good_episodes = 0
                    
                    # Reset for next episode
                    obs = env.reset()
                    episode_reward = 0
                    episode_length = 0
                    episode_actions = [0, 0, 0, 0]
                    episode_count += 1
                    
                    # Log progress
                    if episode_count % 250 == 0 or episode_count < 10:
                        recent_window = min(100, episode_count)
                        recent_rewards = episode_rewards[-recent_window:]
                        recent_scores = episode_scores[-recent_window:]
                        
                        avg_reward = np.mean(recent_rewards)
                        avg_score = np.mean(recent_scores)
                        max_score = max(recent_scores) if recent_scores else 0
                        
                        # Action diversity check
                        if action_distributions:
                            recent_dists = action_distributions[-min(10, len(action_distributions)):]
                            avg_max_action = np.mean([max(dist) for dist in recent_dists])
                            diversity_score = 1.0 - avg_max_action  # Higher = more diverse
                        else:
                            diversity_score = 0.0
                        
                        # Recent metrics
                        recent_entropy = np.mean(entropy_losses[-50:]) if len(entropy_losses) >= 50 else 0
                        recent_kl = np.mean(kl_divergences[-50:]) if len(kl_divergences) >= 50 else 0
                        
                        print(f"Ep {episode_count:5d} | "
                              f"Score: {avg_score:5.1f} (max: {max_score:3.0f}) | "
                              f"Reward: {avg_reward:7.1f} | "
                              f"Diversity: {diversity_score:.3f} | "
                              f"Entropy: {-recent_entropy:.3f} | "
                              f"KL: {recent_kl:.4f} | "
                              f"Ghosts: {env._game._n_ghosts}")
                        
                        # Check for critical issues
                        if diversity_score < 0.4 and episode_count > 500:
                            print(f"🔴 CRITICAL: Low action diversity ({diversity_score:.3f})")
                        if -recent_entropy < 0.8 and episode_count > 500:
                            print(f"🔴 CRITICAL: Low entropy ({-recent_entropy:.3f})")
                        if avg_score > best_score:
                            best_score = avg_score
                            if max_score > 15:
                                self.save_model(f"fixed_ppo_score_{int(max_score)}_ep{episode_count}.pt")
                    
                    if episode_count >= episodes:
                        break
            
            # Compute advantages and returns
            if not done:
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                    _, _, _, last_value = self.actor_critic.get_action_and_value(obs_tensor)
                    last_value = last_value.item()
            else:
                last_value = 0.0
            
            self.rollout_buffer.compute_advantages_and_returns(
                last_value, self.gamma, self.gae_lambda
            )
            
            # PPO update
            for epoch in range(self.ppo_epochs):
                # Get batch data
                (batch_obs, batch_actions, batch_old_log_probs, 
                 batch_advantages, batch_returns, batch_old_values) = self.rollout_buffer.get_batch(
                    self.batch_size
                )
                
                # Compute loss
                total_loss, policy_loss, value_loss, entropy_loss, approx_kl = self.compute_ppo_loss(
                    batch_obs, batch_actions, batch_old_log_probs,
                    batch_advantages, batch_returns, batch_old_values
                )
                
                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                kl_divergences.append(approx_kl.item())
                
                # Early stopping on high KL divergence
                if approx_kl > self.target_kl:
                    break
            
            # Clear buffer for next rollout
            self.rollout_buffer.clear()
            
            # Save model periodically
            if save_freq > 0 and episode_count % save_freq == 0 and episode_count > 0:
                self.save_model(f"fixed_ppo_ep{episode_count}.pt")
        
        # Save final model
        self.save_model("fixed_ppo_FINAL.pt")
        
        # Create training plots
        self.plot_training_results(
            episode_scores, episode_rewards, episode_lengths,
            policy_losses, value_losses, entropy_losses, kl_divergences
        )
        
        print(f"\n🏁 Training completed!")
        print(f"Episodes trained: {episode_count}")
        print(f"Best average score: {best_score:.1f}")
        
        # Final action diversity check
        if action_distributions:
            final_dists = action_distributions[-100:]
            final_diversity = 1.0 - np.mean([max(dist) for dist in final_dists])
            print(f"Final action diversity: {final_diversity:.3f}")
            
            if final_diversity < 0.4:
                print("🔴 WARNING: Final policy shows low diversity - may need more entropy")
            else:
                print("✅ Final policy maintains good action diversity")
        
        env.close()
        return self.actor_critic, {
            'episode_rewards': episode_rewards,
            'episode_scores': episode_scores,
            'episode_lengths': episode_lengths,
            'policy_losses': policy_losses,
            'value_losses': value_losses,
            'entropy_losses': entropy_losses,
            'kl_divergences': kl_divergences,
            'action_distributions': action_distributions,
            'best_score': best_score
        }
    
    def plot_training_results(self, scores, rewards, lengths, policy_losses, 
                            value_losses, entropy_losses, kl_divergences):
        """Create comprehensive training plots"""
        print("\n📊 Creating training plots...")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Fixed PPO Training Results', fontsize=16, fontweight='bold')
        
        episodes = range(len(scores))
        
        # Plot 1: Scores with running average
        if scores:
            running_avg_scores = []
            for i in range(len(scores)):
                window_start = max(0, i - 99)
                running_avg_scores.append(np.mean(scores[window_start:i+1]))
            
            ax1.plot(episodes, scores, alpha=0.3, color='lightblue', label='Episode Scores')
            ax1.plot(episodes, running_avg_scores, color='darkblue', linewidth=2, label='Running Average (100 episodes)')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Score')
            ax1.set_title('Training Scores Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add baseline from random agent
            ax1.axhline(y=14.3, color='red', linestyle='--', alpha=0.7, label='Random Agent Baseline')
        
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
        
        # Plot 3: Policy Loss and Entropy
        if policy_losses and entropy_losses:
            # Smooth the losses
            if len(policy_losses) > 100:
                policy_smooth = []
                entropy_smooth = []
                for i in range(len(policy_losses)):
                    window_start = max(0, i - 49)
                    policy_smooth.append(np.mean(policy_losses[window_start:i+1]))
                    entropy_smooth.append(np.mean(entropy_losses[window_start:i+1]))
                
                ax3.plot(range(len(policy_smooth)), policy_smooth, color='red', linewidth=2, label='Policy Loss')
                ax3_twin = ax3.twinx()
                ax3_twin.plot(range(len(entropy_smooth)), entropy_smooth, color='orange', linewidth=2, label='Entropy Loss')
                
                ax3.set_ylabel('Policy Loss', color='red')
                ax3_twin.set_ylabel('Entropy Loss', color='orange')
                ax3.set_xlabel('Training Update')
                ax3.set_title('Policy Loss and Entropy')
                ax3.grid(True, alpha=0.3)
        
        # Plot 4: KL Divergence
        if kl_divergences:
            if len(kl_divergences) > 100:
                kl_smooth = []
                for i in range(len(kl_divergences)):
                    window_start = max(0, i - 49)
                    kl_smooth.append(np.mean(kl_divergences[window_start:i+1]))
                ax4.plot(range(len(kl_smooth)), kl_smooth, color='purple', linewidth=2)
            else:
                ax4.plot(range(len(kl_divergences)), kl_divergences, color='purple', linewidth=2)
            
            ax4.axhline(y=self.target_kl, color='red', linestyle='--', alpha=0.7, label=f'Target KL: {self.target_kl}')
            ax4.set_xlabel('Training Update')
            ax4.set_ylabel('KL Divergence')
            ax4.set_title('KL Divergence Over Time')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = 'fixed_ppo_training_results.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📈 Training plots saved as: {plot_filename}")
        
        # Print final statistics
        if running_avg_scores:
            final_avg = running_avg_scores[-1]
            print(f"\n📊 FINAL TRAINING STATISTICS:")
            print(f"   Final running average score: {final_avg:.2f}")
            print(f"   Best single episode score: {max(scores):.1f}")
            print(f"   Total episodes completed: {len(scores)}")
            print(f"   Improvement over random (14.3): {final_avg - 14.3:.2f} points")
            
            if final_avg > 20:
                print("🌟 EXCELLENT: Agent significantly outperforms random!")
            elif final_avg > 14.3:
                print("✅ SUCCESS: Agent beats random baseline!")
            else:
                print("❌ NEEDS WORK: Agent underperforms random baseline")
        
        plt.show()
    
    def save_model(self, filepath):
        """Save model state dict"""
        if self.actor_critic is not None:
            torch.save(self.actor_critic.state_dict(), filepath)
            print(f"✅ Model saved to {filepath}")
    
    def test(self, episodes=5, model_path="fixed_ppo_FINAL.pt", render=False):
        """Test the trained agent with action diversity analysis"""
        print(f"\n🧪 Testing Fixed PPO agent from {model_path}")
        
        # Create test environment with NO GHOSTS first
        env = PacmanEnv(
            obs_type=self.obs_type,
            positive_rewards=True,
            agent_name="FixedPPO_Test",
            ghosts=0,               # Start with no ghosts for testing
            level_ghosts=0,
            lives=3,
            timeout=1000,
            training=False
        )
        
        # Initialize actor-critic if needed
        if self.actor_critic is None:
            self.actor_critic = create_actor_critic(
                env.observation_space, 
                use_cnn=self.use_cnn
            ).to(self.device)
        
        # Load trained model
        if os.path.exists(model_path):
            self.actor_critic.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded model from {model_path}")
        else:
            print(f"❌ Model file {model_path} not found! Using untrained network.")
        
        self.actor_critic.eval()
        
        test_scores = []
        test_rewards = []
        action_counts = [0, 0, 0, 0]
        action_names = ['w', 'a', 's', 'd']
        
        for episode in range(episodes):
            obs = env.reset()
            episode_reward = 0
            episode_score = 0
            steps = 0
            episode_actions = [0, 0, 0, 0]
            
            print(f"\nTest Episode {episode + 1}:")
            
            while steps < 500:  # Reasonable limit for testing
                # Select action (deterministic for testing)
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                    action_logits, _ = self.actor_critic(obs_tensor)
                    probs = torch.softmax(action_logits, dim=-1).squeeze().cpu().numpy()
                    action = torch.argmax(action_logits, dim=-1).item()
                
                episode_actions[action] += 1
                action_counts[action] += 1
                
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_score = info.get('score', 0)
                steps += 1
                
                # Show first few steps with action probabilities
                if steps <= 5:
                    print(f"  Step {steps}: {action_names[action]} | Score: {episode_score} | "
                          f"Probs: w={probs[0]:.3f} a={probs[1]:.3f} s={probs[2]:.3f} d={probs[3]:.3f}")
                
                if done:
                    break
            
            test_scores.append(episode_score)
            test_rewards.append(episode_reward)
            
            # Action diversity for this episode
            total_ep_actions = sum(episode_actions)
            ep_diversity = 1.0 - max(episode_actions) / total_ep_actions if total_ep_actions > 0 else 0
            
            win_status = "WIN" if info.get('win', 0) == 1 else "LOSE"
            print(f"  Result: {win_status} | Score: {episode_score:3.0f} | "
                  f"Reward: {episode_reward:6.1f} | Steps: {steps:3d} | "
                  f"Diversity: {ep_diversity:.3f}")
        
        # Overall results
        avg_score = np.mean(test_scores)
        avg_reward = np.mean(test_rewards)
        
        # Overall action diversity
        total_actions = sum(action_counts)
        overall_diversity = 1.0 - max(action_counts) / total_actions if total_actions > 0 else 0
        
        print(f"\n📊 Test Results Summary:")
        print(f"  Average Score: {avg_score:.1f}")
        print(f"  Average Reward: {avg_reward:.1f}")
        print(f"  Best Score: {max(test_scores)}")
        print(f"  Action diversity: {overall_diversity:.3f}")
        print(f"  Comparison to random baseline (14.3): {avg_score - 14.3:+.1f}")
        
        print(f"\n🎯 Action Distribution:")
        for i, (name, count) in enumerate(zip(action_names, action_counts)):
            pct = count / total_actions * 100 if total_actions > 0 else 0
            print(f"    {name}: {count:3d} ({pct:5.1f}%)")
        
        # Diagnosis
        print(f"\n🩺 DIAGNOSIS:")
        if overall_diversity < 0.3:
            print("🔴 POLICY COLLAPSED: Very low action diversity")
        elif overall_diversity < 0.5:
            print("🟡 POLICY BIASED: Moderate action diversity")
        else:
            print("✅ HEALTHY POLICY: Good action diversity")
        
        if avg_score > 20:
            print("🌟 EXCELLENT: Model significantly outperforms random!")
        elif avg_score > 14.3:
            print("✅ SUCCESS: Model beats random baseline!")
        elif avg_score > 5:
            print("📈 PROGRESS: Some learning detected")
        else:
            print("❌ POOR: Model failed to learn properly")
        
        env.close()
        return test_scores, test_rewards


def main():
    """Run the FIXED PPO training"""
    print("🔧 FIXED PPO TRAINING - Anti-Policy-Collapse Version")
    print("="*60)
    print("Key improvements over original:")
    print("• Proper weight initialization with BatchNorm")
    print("• Conservative hyperparameters")  
    print("• Action diversity monitoring")
    print("• Smart curriculum based on performance")
    print("• Early collapse detection")
    print("• Comparison to random baseline (14.3 points)")
    print("="*60)
    
    agent = PPOAgent(
        obs_type=MultiChannelObs,
        use_cnn=True
    )
    
    trained_model, stats = agent.train(
        episodes=10000,
        save_freq=2000
    )
    
    print(f"\nTraining completed!")
    print(f"Best average score achieved: {stats['best_score']:.1f}")
    
    # Test the trained agent
    print("\n" + "="*50)
    test_scores, test_rewards = agent.test(episodes=5, render=False)
    
    print(f"\n✅ Fixed PPO training complete!")
    print(f"🎯 Target: Beat random agent's 14.3 average score")
    if np.mean(test_scores) > 14.3:
        print(f"🏆 SUCCESS: {np.mean(test_scores):.1f} > 14.3!")
    else:
        print(f"📈 PROGRESS: {np.mean(test_scores):.1f} (needs improvement)")


if __name__ == "__main__":
    main()