import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import json
import os

from dql_model import create_dqn
from gym_pacman import PacmanEnv, EnvParams
from gym_observations import MultiChannelObs, SingleChannelObs


class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class DQLAgent():
    """Deep Q-Learning Agent for Pacman"""
    
    # Hyperparameters
    learning_rate = 0.001           # Increased from 0.0001
    discount_factor = 0.99
    network_sync_rate = 100         # steps before syncing target network
    replay_memory_size = 10000      # replay memory size
    mini_batch_size = 32            # training batch size
    epsilon_start = 1.0             # initial exploration rate
    epsilon_end = 0.01              # final exploration rate
    epsilon_decay_steps = 300000     # Longer decay for more training
    
    # Neural Network
    loss_fn = nn.SmoothL1Loss()     # Huber loss, more stable than MSE
    optimizer = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ACTIONS = ['w', 'a', 's', 'd']  # up, left, down, right
    
    def __init__(self, obs_type=MultiChannelObs, use_cnn=True):
        """
        Initialize DQL Agent
        
        Args:
            obs_type: MultiChannelObs or SingleChannelObs
            use_cnn: Whether to use CNN architecture
        """
        self.obs_type = obs_type
        self.use_cnn = use_cnn
        print(f"Initialized DQL Agent with {obs_type.__name__} and {'CNN' if use_cnn else 'FC'} architecture")
        print(f"Using device: {self.device}")
    
    def train(self, episodes=1000, render_freq=100, save_freq=500):
        """Train the agent on Pacman environment"""
        print(f"Starting training for {episodes} episodes...")
        
        # Create environment
        env = PacmanEnv(
            obs_type=self.obs_type,
            positive_rewards=True,  # Use positive reward shaping
            agent_name="DQLAgent",
            ghosts=2,               # Start with 2 ghosts
            level_ghosts=1,         # Medium difficulty
            lives=3,
            timeout=3000,
            training=True
        )
        
        print(f"Environment created with observation shape: {env.observation_space.shape}")
        
        # Create networks
        policy_dqn = create_dqn(env.observation_space, use_cnn=self.use_cnn).to(self.device)
        target_dqn = create_dqn(env.observation_space, use_cnn=self.use_cnn).to(self.device)
        
        # Copy policy network to target network
        target_dqn.load_state_dict(policy_dqn.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)
        
        # Memory
        memory = ReplayMemory(self.replay_memory_size)
        
        # Tracking variables
        rewards_per_episode = []
        scores_per_episode = []
        epsilon_history = []
        loss_history = []
        running_reward = []
        running_score = []
        
        step_count = 0
        epsilon = self.epsilon_start
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_score = 0
            episode_steps = 0
            terminated = False
            
            while not terminated:
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = policy_dqn(state_tensor)
                        action = q_values.argmax().item()
                
                # Take action
                next_state, reward, terminated, info = env.step(action)
                
                episode_reward += reward
                episode_score = info.get('score', 0)
                episode_steps += 1
                
                # Store transition in memory
                memory.append((state, action, next_state, reward, terminated))
                
                # Train if enough samples
                if len(memory) > self.mini_batch_size:
                    loss = self.optimize(memory, policy_dqn, target_dqn)
                    if loss is not None:
                        loss_history.append(loss)
                
                # Update state
                state = next_state
                step_count += 1
                
                # Sync networks
                if step_count % self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                
                # Decay epsilon
                if step_count < self.epsilon_decay_steps:
                    epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (step_count / self.epsilon_decay_steps)
                else:
                    epsilon = self.epsilon_end
            
            # Record episode statistics
            rewards_per_episode.append(episode_reward)
            scores_per_episode.append(episode_score)
            epsilon_history.append(epsilon)
            
            # Calculate running averages (last 100 episodes)
            window_size = min(100, episode + 1)
            running_reward.append(np.mean(rewards_per_episode[-window_size:]))
            running_score.append(np.mean(scores_per_episode[-window_size:]))
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = running_reward[-1] if running_reward else 0
                avg_score = running_score[-1] if running_score else 0
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:6.1f} | "
                      f"Score: {episode_score:4.0f} | "
                      f"Avg Reward: {avg_reward:6.1f} | "
                      f"Avg Score: {avg_score:4.0f} | "
                      f"Epsilon: {epsilon:.3f} | "
                      f"Steps: {episode_steps:3d}")
            
            # Render occasionally
            if render_freq > 0 and episode % render_freq == 0:
                print(f"\nRendering episode {episode}:")
                env.render()
                print()
            
            # Save model periodically
            if save_freq > 0 and episode % save_freq == 0 and episode > 0:
                self.save_model(policy_dqn, f"pacman_dqn_ep{episode}.pt")
        
        # Save final model with explicit success check
        self.save_model(policy_dqn, "pacman_dqn_final.pt")
        
        # Verify the model was saved and can be loaded
        try:
            test_model = create_dqn(env.observation_space, use_cnn=self.use_cnn).to(self.device)
            test_model.load_state_dict(torch.load("pacman_dqn_final.pt", map_location=self.device))
            print("✅ Model save/load verification successful")
        except Exception as e:
            print(f"❌ Model save/load verification failed: {e}")
        
        # Create training plots
        self.plot_training_results(
            rewards_per_episode, scores_per_episode, 
            running_reward, running_score, 
            epsilon_history, loss_history
        )
        
        env.close()
        return policy_dqn, {
            'rewards': rewards_per_episode,
            'scores': scores_per_episode,
            'running_reward': running_reward,
            'running_score': running_score,
            'epsilon': epsilon_history,
            'loss': loss_history
        }
    
    def optimize(self, memory, policy_dqn, target_dqn):
        """Optimize the policy network using a batch from replay memory"""
        if len(memory) < self.mini_batch_size:
            return None
        
        # Sample batch from memory
        batch = memory.sample(self.mini_batch_size)
        
        # Separate batch components
        states = torch.FloatTensor(np.array([transition[0] for transition in batch])).to(self.device)
        actions = torch.LongTensor([transition[1] for transition in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([transition[2] for transition in batch])).to(self.device)
        rewards = torch.FloatTensor([transition[3] for transition in batch]).to(self.device)
        dones = torch.BoolTensor([transition[4] for transition in batch]).to(self.device)
        
        # Current Q values
        current_q_values = policy_dqn(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = target_dqn(next_states).max(1)[0]
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        # Compute loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def test(self, episodes=10, model_path="pacman_dqn_final.pt", render=True):
        """Test the trained agent"""
        print(f"Testing agent for {episodes} episodes...")
        
        # Create environment
        env = PacmanEnv(
            obs_type=self.obs_type,
            positive_rewards=True,
            agent_name="DQLAgent_Test",
            ghosts=2,
            level_ghosts=1,
            lives=3,
            timeout=3000,
            training=False
        )
        
        # Load trained model
        policy_dqn = create_dqn(env.observation_space, use_cnn=self.use_cnn).to(self.device)
        
        if os.path.exists(model_path):
            policy_dqn.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model file {model_path} not found! Using untrained network.")
        
        policy_dqn.eval()
        
        # Initialize visual renderer if requested
        visual_renderer = None
        if render:
            try:
                visual_renderer = self._initialize_visual_renderer(env)
                print("✅ Visual renderer initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize visual renderer: {e}")
                print("Falling back to text rendering")
                visual_renderer = None
        
        test_scores = []
        test_rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_score = 0
            episode_steps = 0
            terminated = False
            
            if render and not visual_renderer:
                print(f"\n=== Test Episode {episode + 1} ===")
            
            while not terminated:
                # Render visual game state
                if render and visual_renderer:
                    game_state = json.loads(env._game.state)
                    self._render_visual_frame(visual_renderer, game_state, env._game.map, episode_steps)
                
                # Select best action (no exploration)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = policy_dqn(state_tensor)
                    action = q_values.argmax().item()
                
                # Take action
                state, reward, terminated, info = env.step(action)
                
                episode_reward += reward
                episode_score = info.get('score', 0)
                episode_steps += 1
                
                if render and not visual_renderer and episode_steps % 50 == 0:
                    print(f"  Step {episode_steps:3d} | Action: {self.ACTIONS[action]} | "
                          f"Reward: {reward:4.1f} | Score: {episode_score:4.0f}")
            
            test_scores.append(episode_score)
            test_rewards.append(episode_reward)
            
            if render:
                win_status = "WIN" if info.get('win', 0) == 1 else "LOSE"
                if visual_renderer:
                    print(f"Episode {episode + 1} - {win_status} | Score: {episode_score:4.0f} | "
                          f"Total Reward: {episode_reward:6.1f} | Steps: {episode_steps:3d}")
                    input("Press Enter to continue to next episode...")
                else:
                    print(f"  Final - {win_status} | Score: {episode_score:4.0f} | "
                          f"Total Reward: {episode_reward:6.1f} | Steps: {episode_steps:3d}")
                    if episode < episodes - 1:
                        env.render()
        
        # Clean up visual renderer
        if visual_renderer:
            self._cleanup_visual_renderer(visual_renderer)
        
        # Print test results
        avg_score = np.mean(test_scores)
        avg_reward = np.mean(test_rewards)
        
        print(f"\n=== Test Results ===")
        print(f"Average Score: {avg_score:.1f}")
        print(f"Average Reward: {avg_reward:.1f}")
        print(f"Best Score: {max(test_scores)}")
        print(f"Worst Score: {min(test_scores)}")
        
        env.close()
        return test_scores, test_rewards
    
    def _initialize_visual_renderer(self, env):
        """Initialize pygame-based visual renderer"""
        import pygame
        import time
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Constants from viewer.py
        CHAR_LENGTH = 26
        SCALE = 1
        
        # Calculate display size
        map_width, map_height = env._game.map.size
        display_width = int(map_width * CHAR_LENGTH / SCALE)
        display_height = int(map_height * CHAR_LENGTH / SCALE)
        
        # Create display
        screen = pygame.display.set_mode((display_width, display_height))
        pygame.display.set_caption("DQL Agent Playing Pacman")
        
        # Try to load sprites
        sprites = None
        try:
            sprites = pygame.image.load("data/sprites/spritemap.png")
        except:
            print("⚠️ Could not load sprites, using colored rectangles")
        
        return {
            'screen': screen,
            'sprites': sprites,
            'scale': SCALE,
            'char_length': CHAR_LENGTH,
            'map_size': (map_width, map_height),
            'last_render_time': time.time()
        }
    
    def _render_visual_frame(self, renderer, game_state, game_map, step_count):
        """Render a single frame using pygame"""
        import pygame
        import time
        
        screen = renderer['screen']
        scale = renderer['scale']
        char_length = renderer['char_length']
        
        # Limit rendering frequency
        current_time = time.time()
        if current_time - renderer['last_render_time'] < 0.1:  # 10 FPS max
            return
        renderer['last_render_time'] = current_time
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Helper function to scale coordinates
        def scale_pos(x, y):
            return int(x * char_length / scale), int(y * char_length / scale)
        
        # Draw walls
        for x in range(game_map.hor_tiles):
            for y in range(game_map.ver_tiles):
                if game_map.is_wall((x, y)):
                    wx, wy = scale_pos(x, y)
                    wall_rect = pygame.Rect(wx, wy, *scale_pos(1, 1))
                    pygame.draw.rect(screen, (100, 100, 100), wall_rect)
        
        # Draw energy dots
        if 'energy' in game_state:
            for x, y in game_state['energy']:
                ex, ey = scale_pos(x, y)
                center_x = ex + int(char_length / scale / 2)
                center_y = ey + int(char_length / scale / 2)
                pygame.draw.circle(screen, (255, 255, 255), (center_x, center_y), 2)
        
        # Draw boost items
        if 'boost' in game_state:
            for x, y in game_state['boost']:
                bx, by = scale_pos(x, y)
                center_x = bx + int(char_length / scale / 2)
                center_y = by + int(char_length / scale / 2)
                pygame.draw.circle(screen, (255, 255, 0), (center_x, center_y), 6)
        
        # Draw ghosts
        if 'ghosts' in game_state:
            ghost_colors = [(255, 0, 0), (255, 105, 180), (255, 165, 0), (0, 255, 0)]
            for i, (pos, zombie, timeout) in enumerate(game_state['ghosts']):
                x, y = pos
                gx, gy = scale_pos(x, y)
                color = (0, 0, 255) if zombie else ghost_colors[i % len(ghost_colors)]
                ghost_rect = pygame.Rect(gx + 2, gy + 2, char_length - 4, char_length - 4)
                pygame.draw.rect(screen, color, ghost_rect)
        
        # Draw Pacman
        if 'pacman' in game_state:
            x, y = game_state['pacman']
            px, py = scale_pos(x, y)
            pacman_rect = pygame.Rect(px + 3, py + 3, char_length - 6, char_length - 6)
            pygame.draw.rect(screen, (255, 255, 0), pacman_rect)
        
        # Draw game info
        font = pygame.font.Font(None, 24)
        if 'score' in game_state:
            score_text = font.render(f"Score: {game_state['score']}", True, (255, 255, 255))
            screen.blit(score_text, (10, 10))
        if 'lives' in game_state:
            lives_text = font.render(f"Lives: {game_state['lives']}", True, (255, 255, 255))
            screen.blit(lives_text, (10, 35))
        
        step_text = font.render(f"Step: {step_count}", True, (255, 255, 255))
        screen.blit(step_text, (10, 60))
        
        # Update display
        pygame.display.flip()
        
        # Handle pygame events to prevent window from becoming unresponsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt("User closed window")
    
    def _cleanup_visual_renderer(self, renderer):
        """Clean up pygame resources"""
        import pygame
        pygame.quit()
    
    def plot_training_results(self, rewards, scores, running_reward, running_score, epsilon, losses):
        """Create comprehensive training plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DQL Agent Training Results', fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(rewards, alpha=0.6, label='Episode Reward')
        axes[0, 0].plot(running_reward, label='Running Average (100ep)', linewidth=2)
        axes[0, 0].set_title('Rewards per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Episode scores
        axes[0, 1].plot(scores, alpha=0.6, label='Episode Score')
        axes[0, 1].plot(running_score, label='Running Average (100ep)', linewidth=2)
        axes[0, 1].set_title('Scores per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Game Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Epsilon decay
        axes[0, 2].plot(epsilon)
        axes[0, 2].set_title('Epsilon Decay')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Epsilon')
        axes[0, 2].grid(True)
        
        # Training loss
        if losses:
            axes[1, 0].plot(losses)
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        # Score distribution
        axes[1, 1].hist(scores, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Score Distribution')
        axes[1, 1].set_xlabel('Game Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
        
        # Running average comparison
        axes[1, 2].plot(running_reward, label='Avg Reward', linewidth=2)
        axes[1, 2].plot(running_score, label='Avg Score', linewidth=2)
        axes[1, 2].set_title('Running Averages')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig('pacman_dql_training.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Training plots saved as 'pacman_dql_training.png'")
    
    def save_model(self, model, filepath):
        """Save model state dict"""
        torch.save(model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, model, filepath):
        """Load model state dict"""
        model.load_state_dict(torch.load(filepath, map_location=self.device))
        model.eval()
        print(f"Model loaded from {filepath}")
        return model


def main():
    """Main function to train and test the DQL agent"""
    print("=== Pacman DQL Agent ===")
    
    # Create agent - you can experiment with different configurations
    agent = DQLAgent(
        obs_type=MultiChannelObs,  # or SingleChannelObs
        use_cnn=True               # or False for fully connected
    )
    
    # Train the agent
    # print("\n1. Training Phase")
    # trained_model, training_stats = agent.train(
    #     episodes=1000,      # Much longer training
    #     render_freq=0,      # Set to 0 to disable rendering during training
    #     save_freq=100       # Save model every 100 episodes
    # )
    
    # Test the agent
    print("\n2. Testing Phase")
    test_scores, test_rewards = agent.test(
        episodes=5,
        model_path="pacman_dqn_final.pt",
        render=True
    )
    
    print("\nTraining and testing completed!")
    print(f"Final training average reward: {training_stats['running_reward'][-1]:.2f}")
    print(f"Final training average score: {training_stats['running_score'][-1]:.2f}")
    print(f"Test average score: {np.mean(test_scores):.2f}")


if __name__ == "__main__":
    main()