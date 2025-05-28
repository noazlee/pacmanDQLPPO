from collections import namedtuple
import random
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
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
    EnvParams(0, 1, 'data/fixed_classic.bmp', 1), 
    EnvParams(1, 1, 'data/fixed_classic.bmp', 10),
    EnvParams(2, 2, 'data/fixed_classic.bmp', 10), 
    EnvParams(4, 0, 'data/fixed_classic.bmp', 10),
    EnvParams(4, 1, 'data/fixed_classic.bmp', 11), 
    EnvParams(4, 2, 'data/fixed_classic.bmp', 9)
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
        """Execute one environment step"""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        # Execute action in game
        self._game.keypress(self.keys[action])
        self._game.compute_next_frame()

        # Get game state
        game_state = json.loads(self._game.state)

        # Prepare info dictionary
        info = {k: game_state[k] for k in self.info_keywords if k in game_state}
        info.update({
            'ghosts': self._current_params.ghosts,
            'level': self._current_params.level,
            'map': self._current_params.map,
            'win': 0,
            'd': self.difficulty
        })

        done = not self._game.running
        reward = game_state['score'] - self._current_score
        self._current_score = game_state['score']

        # Apply reward shaping
        if not self.positive_rewards:
            if game_state['lives'] == 0:
                reward -= (self._game._timeout - game_state['step'] + 1) * (1.0 / TIME_BONUS_STEPS)
            reward -= 1.0 / TIME_BONUS_STEPS

        # Check for win condition
        if done:
            if self._game._timeout != game_state['step'] and game_state['lives'] > 0:
                info['win'] = 1
                if not self.positive_rewards:
                    reward -= 1.0 / TIME_BONUS_STEPS

            if info['ghosts'] == self.difficulty:
                self.wins_count += info['win']

        # Penalty for losing lives
        if game_state['lives'] < self.current_lives:
            reward -= 50
            self.current_lives = game_state['lives']

        # Penalty for staying in same position
        if self._last_pos and game_state['pacman'] == self._last_pos:
            reward -= 0.5
            self.idle_steps += 1

        self._last_pos = game_state['pacman']
        info['idle'] = self.idle_steps

        # Get observation using cached map
        obs = self.pacman_obs.get_obs(game_state, self._map_cache[self._current_map_file])
        
        return obs, reward, done, info

    def reset(self):
        """Reset environment for new episode"""
        # Reset tracking variables
        self._current_score = 0
        self._last_pos = None
        self.idle_steps = 0
        self._current_energy_reward = self.MIN_ENERGY_REWARD

        # Optionally change environment parameters during training
        if self.training and len(self.train_env_params) > 1:
            new_params = random.choice(self.train_env_params)
            # Only update if different to avoid unnecessary operations
            if (new_params.ghosts != self._current_params.ghosts or 
                new_params.level != self._current_params.level or
                new_params.map != self._current_params.map):
                self._set_env_params(new_params)

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