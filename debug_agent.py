"""
DEBUG VERSION - Find out what's happening with training
"""

import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import os
import time
import sys

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

class DebugDQLAgent():
    """DEBUG VERSION - Shows exactly what's happening"""
    
    learning_rate = 0.0005
    discount_factor = 0.99
    network_sync_rate = 1000
    replay_memory_size = 50000
    mini_batch_size = 32
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay_episodes = 20000
    gradient_clip = 1.0
    warmup_episodes = 500
    train_frequency = 1
    
    loss_fn = nn.SmoothL1Loss()
    optimizer = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ACTIONS = ['w', 'a', 's', 'd']
    
    def __init__(self, obs_type=MultiChannelObs, use_cnn=True):
        self.obs_type = obs_type
        self.use_cnn = use_cnn
        print(f"DEBUG DQL Agent initialized", flush=True)
    
    def unbiased_action_selection(self, q_values_tensor, add_noise=True):
        q_values = q_values_tensor.squeeze()
        if add_noise:
            noise = torch.randn_like(q_values) * 1e-8
            q_values = q_values + noise
        action = q_values.argmax().item()
        return action
    
    def train(self, episodes=25000, render_freq=0, save_freq=2000):
        print(f"DEBUG: Starting training for {episodes} episodes", flush=True)
        
        # Track timing
        start_time = time.time()
        last_log_time = start_time
        
        env = PacmanEnv(
            obs_type=self.obs_type,
            positive_rewards=True,
            agent_name="debug_dql",
            ghosts=0,
            level_ghosts=0,
            lives=3,
            timeout=2000,
            training=True
        )
        
        print(f"DEBUG: Environment created", flush=True)
        
        # Create networks
        policy_dqn = create_dqn(env.observation_space, use_cnn=self.use_cnn).to(self.device)
        target_dqn = create_dqn(env.observation_space, use_cnn=self.use_cnn).to(self.device)
        
        # Simple initialization
        target_dqn.load_state_dict(policy_dqn.state_dict())
        
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)
        memory = ReplayMemory(self.replay_memory_size)
        
        print(f"DEBUG: Networks and optimizer ready", flush=True)
        print(f"DEBUG: Starting episode loop...", flush=True)
        
        # Episode tracking
        step_count = 0
        episode_times = []
        
        for episode in range(episodes):
            episode_start_time = time.time()
            
            # FORCE FREQUENT LOGGING FOR DEBUG
            if episode % 100 == 0:
                current_time = time.time()
                elapsed = current_time - start_time
                since_last = current_time - last_log_time
                
                print(f"DEBUG Episode {episode:5d} | "
                      f"Total time: {elapsed/60:.1f}min | "
                      f"Since last: {since_last:.1f}s | "
                      f"Avg ep time: {np.mean(episode_times[-50:]):.2f}s" if episode_times else "🐛 DEBUG Episode 0", 
                      flush=True)
                last_log_time = current_time
            
            # Apply curriculum with debug
            curriculum_updated = env.apply_curriculum(episode)
            if curriculum_updated:
                print(f"DEBUG: Curriculum updated at episode {episode}", flush=True)
            
            # Episode execution
            state = env.reset()
            episode_reward = 0
            episode_score = 0
            episode_steps = 0
            terminated = False
            
            # Epsilon calculation
            if episode < self.epsilon_decay_episodes:
                epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (episode / self.epsilon_decay_episodes)
            else:
                epsilon = self.epsilon_end
            
            # Episode loop with timeout detection
            max_steps_per_episode = 2500  # Safety limit
            step_start_time = time.time()
            
            while not terminated and episode_steps < max_steps_per_episode:
                # Action selection
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        q_values = policy_dqn(state_tensor)
                        action = self.unbiased_action_selection(q_values, add_noise=True)
                
                # Take action
                next_state, reward, terminated, info = env.step(action)
                
                episode_reward += reward
                episode_score = info.get('score', 0)
                episode_steps += 1
                
                # Add to memory
                memory.append((state, action, next_state, reward, terminated))
                
                # Training
                if (episode >= self.warmup_episodes and 
                    len(memory) > self.mini_batch_size * 2 and 
                    step_count % self.train_frequency == 0):
                    
                    loss = self.optimize_simple(memory, policy_dqn, target_dqn)
                
                state = next_state
                step_count += 1
                
                # Target network sync
                if step_count % self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                
                # Check for stuck episodes
                step_time = time.time() - step_start_time
                if step_time > 30:  # Episode taking more than 30 seconds
                    print(f"DEBUG: Episode {episode} stuck! {episode_steps} steps, {step_time:.1f}s", flush=True)
                    terminated = True
                    break
            
            # Record episode timing
            episode_time = time.time() - episode_start_time
            episode_times.append(episode_time)
            
            # Check for slow episodes
            if episode_time > 5.0:  # Episode took more than 5 seconds
                print(f"DEBUG: Slow episode {episode}: {episode_time:.2f}s, {episode_steps} steps", flush=True)
            
            # Regular logging condition (but more frequent for debug)
            if episode % 1000 == 0 or (episode < 100 and episode % 10 == 0):
                print(f"Ep {episode:5d} | "
                      f"Score: {episode_score:3.0f} | "
                      f"Reward: {episode_reward:6.1f} | "
                      f"ε: {epsilon:.3f} | "
                      f"Steps: {episode_steps} | "
                      f"Time: {episode_time:.2f}s", flush=True)
            
            # Emergency break for testing
            if episode >= 500:  # Stop after 500 episodes for debug
                print(f"DEBUG: Stopping at episode {episode} for analysis", flush=True)
                break
        
        print(f"DEBUG: Training loop completed", flush=True)
        env.close()
        
        return policy_dqn, {}
    
    def optimize_simple(self, memory, policy_dqn, target_dqn):
        """Simplified optimization to avoid issues"""
        if len(memory) < self.mini_batch_size:
            return None
        
        try:
            batch = memory.sample(self.mini_batch_size)
            
            states = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
            actions = torch.LongTensor([t[1] for t in batch]).to(self.device)
            next_states = torch.FloatTensor(np.array([t[2] for t in batch])).to(self.device)
            rewards = torch.FloatTensor([t[3] for t in batch]).to(self.device)
            dones = torch.BoolTensor([t[4] for t in batch]).to(self.device)
            
            # Simple Q-learning update
            current_q_values = policy_dqn(states).gather(1, actions.unsqueeze(1))
            
            with torch.no_grad():
                next_q_values = target_dqn(next_states).max(1)[0]
                target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
            
            loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), self.gradient_clip)
            self.optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            print(f"DEBUG: Optimization error: {e}", flush=True)
            return None

def main():
    
    agent = DebugDQLAgent(
        obs_type=MultiChannelObs,
        use_cnn=True
    )
    
    try:
        trained_model, stats = agent.train(episodes=500)  # Short debug run
        print("DEBUG: Training completed successfully", flush=True)
    except Exception as e:
        print(f"DEBUG: Training failed with error: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()