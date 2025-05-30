from collections import namedtuple
import random
import json
import numpy as np
import gym as gym
from gym import spaces
from game import Game, POINT_ENERGY, TIME_BONUS_STEPS, POINT_BOOST
from mapa import Map
from gym_observations import SingleChannelObs, MultiChannelObs


class EnvParams:
    def __init__(self, ghosts, level, mapa, test_runs):
        self._ghosts = ghosts
        self._level = level
        self._map = mapa
        self._test_runs = test_runs

    @property
    def ghosts(self):
        return self._ghosts

    @property
    def level(self):
        return self._level

    @property
    def map(self):
        return self._map

    @property
    def test_runs(self):
        return self._test_runs


# Simplified ENV_PARAMS - using only one map to avoid constant reloading
ENV_PARAMS = [
    EnvParams(0, 0, 'data/fixed_classic.bmp', 1), 
    # EnvParams(1, 1, 'data/fixed_classic.bmp', 10),
    # EnvParams(2, 2, 'data/fixed_classic.bmp', 10), 
    # EnvParams(4, 0, 'data/fixed_classic.bmp', 10),
    # EnvParams(4, 1, 'data/fixed_classic.bmp', 11), 
    # EnvParams(4, 2, 'data/fixed_classic.bmp', 9)
]


class PacmanEnv(gym.Env):
    """Optimized Pacman environment for reinforcement learning"""

    metadata = {'render.modes': ['human']}
    keys = {0: 'w', 1: 'a', 2: 's', 3: 'd'}
    info_keywords = ('step', 'score', 'lives')

    MAX_ENERGY_REWARD = 60
    MIN_ENERGY_REWARD = 10
    INITIAL_DIFFICULTY = 0
    MIN_WINS = 200

    # Class-level map cache to avoid reloading
    _map_cache = {}

    def __init__(self, obs_type, positive_rewards, agent_name,
                 ghosts, level_ghosts, lives, timeout, map_files=None, training=True):

        self.positive_rewards = positive_rewards
        self.agent_name = agent_name
        self.training = training

        # Filter training parameters
        self.train_env_params = [p for p in ENV_PARAMS if p.test_runs > 1]

        # Initialize maps with caching
        self._initialize_maps(map_files)

        # Use first map as default
        mapfile = list(self._map_cache.keys())[0]
        self._current_map_file = mapfile

        # Create game instance
        self._game = Game(mapfile, ghosts, level_ghosts, lives, timeout)

        # Initialize observation system
        maps = list(self._map_cache.values())
        self.pacman_obs = obs_type(maps, lives)

        # Set up spaces
        self.observation_space = self.pacman_obs.space
        self.action_space = spaces.Discrete(len(self.keys))

        # Initialize state tracking
        self._current_score = 0
        self.current_lives = self._game._initial_lives
        self._last_pos = None
        self._current_params = EnvParams(ghosts, level_ghosts, mapfile, 10)
        self.idle_steps = 0

        # Energy reward system
        self.total_energy = len(self._game.map.energy)
        self.energy_reward_increment = (self.MAX_ENERGY_REWARD - self.MIN_ENERGY_REWARD) / max(1, self.total_energy - 1)
        self._current_energy_reward = self.MIN_ENERGY_REWARD

        # Training progression
        self.difficulty = self.INITIAL_DIFFICULTY
        self.wins_count = 0

        print(f"PacmanEnv initialized with {len(self._map_cache)} cached maps")

    def _initialize_maps(self, map_files):
        """Initialize and cache map objects to avoid repeated loading"""
        maps_to_load = set()
        
        if map_files:
            maps_to_load.update(map_files)
        else:
            # Get unique map files from ENV_PARAMS
            maps_to_load.update(param.map for param in ENV_PARAMS)

        for map_file in maps_to_load:
            if map_file not in self._map_cache:
                print(f"Loading and caching map: {map_file}")
                self._map_cache[map_file] = Map(map_file)


    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        # Execute action
        self._game.keypress(self.keys[action])
        self._game.compute_next_frame()
        game_state = json.loads(self._game.state)

        # Info dict
        info = {k: game_state[k] for k in self.info_keywords if k in game_state}
        info.update({
            'ghosts': self._current_params.ghosts,
            'level': self._current_params.level,
            'map': self._current_params.map,
            'win': 0,
            'd': self.difficulty
        })

        done = not self._game.running
        
        # *** COMPLETELY NEW REWARD STRUCTURE ***
        reward = 0
        
        # 1. ONLY reward actual game progress
        if game_state['score'] > self._current_score:
            score_increase = game_state['score'] - self._current_score
            if score_increase == 1:      # Energy dot
                reward += 10
                print(f"Energy collected! +10")
            elif score_increase == 10:   # Power pellet  
                reward += 50
                print(f"Power pellet! +50")
            elif score_increase == 50:   # Ghost eaten
                reward += 25
                print(f"Ghost eaten! +25")
            else:
                reward += score_increase * 2
        
        # 2. Small penalty for time (encourages efficiency)
        reward -= 0.01
        
        # 3. Death penalty
        if game_state['lives'] < self.current_lives:
            reward -= 150
            print(f"Death penalty: -150")
            self.current_lives = game_state['lives']
        
        # 4. End game rewards
        if done:
            items_remaining = len(game_state['energy']) + len(game_state['boost'])
            if items_remaining == 0:
                reward += 500
                info['win'] = 1
                print(f"🏆 WIN BONUS: +500")
            else:
                reward -= 50
                print(f"📉 Game over penalty: -50")
        
        
        # Update tracking
        self._current_score = game_state['score']
        self._last_pos = game_state['pacman']
        
        # Get observation
        obs = self.pacman_obs.get_obs(game_state, self._map_cache[self._current_map_file])
        
        return obs, reward, done, info

    def reset(self):
        """Reset environment for new episode with curriculum learning"""
        # Reset tracking variables
        self._current_score = 0
        self._last_pos = None
        self.idle_steps = 0
        self._current_energy_reward = self.MIN_ENERGY_REWARD

        # Start new game episode
        self._game.start(self.agent_name)
        self._game.compute_next_frame()
        self.current_lives = self._game._initial_lives

        # Return initial observation
        game_state = json.loads(self._game.state)
        return self.pacman_obs.get_obs(game_state, self._map_cache[self._current_map_file])

    def _set_env_params(self, env_params):
        """Update environment parameters efficiently"""
        self._current_params = env_params
        
        # Update game parameters
        self._game._n_ghosts = env_params.ghosts
        self._game._l_ghosts = env_params.level
        
        # Only reload map if it's different
        if env_params.map != self._current_map_file:
            self._current_map_file = env_params.map
            # Use cached map instead of creating new one
            self._game.map = self._map_cache[env_params.map]

    def set_env_params(self, env_params):
        """Public interface for setting environment parameters"""
        self._set_env_params(env_params)

    def render(self, mode='human'):
        """Render current observation"""
        self.pacman_obs.render()

    def get_curriculum_params(self, episode):
        """
        Progressive difficulty curriculum - gradually introduce ghosts and increase difficulty
        
        Args:
            episode: Current training episode number
            
        Returns:
            dict: Parameters for current difficulty level
        """
        
        # Phase 1: Pure exploration and energy collection (0-10,000 episodes)

        if episode < 5000:
            return {
                'ghosts': 0,
                'level': 0,
                'lives': 3,
                'timeout': 400,  
                'description': 'Phase 1: Learning efficient movement under time pressure'
            }

        if episode < 10000:
            return {
                'ghosts': 0,
                'level': 0,
                'lives': 5,
                'timeout': 800,
                'description': 'Phase 1: Learning basic movement and energy collection'
            }
        
        # Phase 2: Introduction of 1 easy ghost (10,000-15,000 episodes)
        elif episode < 15000:
            return {
                'ghosts': 1,
                'level': 0,  # Level 0 = easiest ghost AI
                'lives': 5,
                'timeout': 2000,
                'description': 'Phase 2: Learning ghost avoidance with 1 easy ghost'
            }
        
        # Phase 3: 1 medium difficulty ghost (15,000-20,000 episodes)
        elif episode < 20000:
            return {
                'ghosts': 1,
                'level': 1,  # Level 1 = medium ghost AI
                'lives': 4,
                'timeout': 1800,
                'description': 'Phase 3: Improving ghost avoidance with smarter ghost'
            }
        
        # Phase 4: 2 easy ghosts (20,000-25,000 episodes)
        elif episode < 25000:
            return {
                'ghosts': 2,
                'level': 0,  # 2 easy ghosts
                'lives': 4,
                'timeout': 1800,
                'description': 'Phase 4: Learning multi-ghost avoidance'
            }
        
        # Phase 5: 2 medium difficulty ghosts (25,000+ episodes)
        else:
            return {
                'ghosts': 2,
                'level': 1,  # 2 medium ghosts
                'lives': 3,  # Standard lives
                'timeout': 1500,  # Shorter episodes for efficiency
                'description': 'Phase 5: Final difficulty - 2 smart ghosts'
            }

    def apply_curriculum(self, episode):
        """
        Apply curriculum learning parameters at the start of training phases
        
        Args:
            episode: Current episode number
        """
        # Only update curriculum at specific episode milestones
        phase_boundaries = [0, 10000, 15000, 20000, 25000]
        
        if episode in phase_boundaries:
            curriculum = self.get_curriculum_params(episode)
            
            print(f"\n{'='*60}")
            print(f"CURRICULUM UPDATE AT EPISODE {episode}")
            print(f"{curriculum['description']}")
            print(f"Ghosts: {curriculum['ghosts']} (level {curriculum['level']})")
            print(f" Lives: {curriculum['lives']}")
            print(f"Timeout: {curriculum['timeout']}")
            print(f"{'='*60}\n")
            
            # Update environment parameters
            new_params = EnvParams(
                ghosts=curriculum['ghosts'],
                level=curriculum['level'], 
                mapa=self._current_map_file,
                test_runs=10
            )
            
            # Update current parameters
            self._current_params = new_params
            
            # Update game settings
            self._game._n_ghosts = curriculum['ghosts']
            self._game._l_ghosts = curriculum['level']
            self._game._initial_lives = curriculum['lives']
            self._game._timeout = curriculum['timeout']
            
            return True  # Indicates curriculum was updated
        
        return False  # No curriculum update needed


def main():
    """Test the optimized Pacman environment"""
    print("Testing Optimized PacmanEnv...")
    
    env = PacmanEnv(
        obs_type=MultiChannelObs,
        positive_rewards=True,
        agent_name="TestAgent",
        ghosts=2,
        level_ghosts=1,
        lives=3,
        timeout=3000,
        training=False
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test multiple episodes quickly
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 20:  # Limited steps for testing
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            steps += 1
            print(f"Step {steps}: Action={env.keys[action]}, Reward={reward:.1f}, Score={info.get('score', 0)}")

    print("\nTest completed - no excessive map loading!")
    env.close() if hasattr(env, 'close') else None


if __name__ == "__main__":
    main()