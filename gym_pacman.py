from collections import namedtuple
import random
import json
import gym
import numpy as np

# from stable_baselines.common.env_checker import check_env

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


ENV_PARAMS = [
    EnvParams(0, 1, 'data/map1.bmp', 1), EnvParams(1, 1, 'data/map1.bmp', 10),
    EnvParams(2, 2, 'data/map1.bmp', 10), EnvParams(4, 0, 'data/map1.bmp', 10),
    EnvParams(4, 1, 'data/map1.bmp', 11), EnvParams(4, 2, 'data/map1.bmp', 9),
    EnvParams(1, 1, 'data/map2.bmp', 10), EnvParams(2, 1, 'data/map2.bmp', 12),
    EnvParams(4, 0, 'data/map2.bmp', 12), EnvParams(4, 1, 'data/map2.bmp', 10),
    EnvParams(4, 2, 'data/map2.bmp', 10),
    # EnvParams(1, 1, 'data/map1_1.bmp', 10), EnvParams(2, 1, 'data/map1_2.bmp', 10),
    # EnvParams(4, 0, 'data/map1_3.bmp', 10), EnvParams(4, 1, 'data/map1_4.bmp', 10),
    # EnvParams(4, 2, 'data/map1_5.bmp', 10), EnvParams(4, 2, 'data/map1_6.bmp', 10),
    # EnvParams(1, 1, 'data/map1_7.bmp', 10), EnvParams(1, 1, 'data/map1_8.bmp', 10)
]


class PacmanEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    keys = {0: 'w', 1: 'a', 2: 's', 3: 'd'}

    info_keywords = ('step', 'score', 'lives')

    MAX_ENERGY_REWARD = 60
    MIN_ENERGY_REWARD = 10

    INITIAL_DIFFICULTY = 0
    MIN_WINS = 200

    def __init__(self, obs_type, positive_rewards, agent_name,
                 ghosts, level_ghosts, lives, timeout, map_files=None, training=True):

        self.positive_rewards = positive_rewards

        self.agent_name = agent_name

        self.training = training

        self.train_env_params = [p for p in ENV_PARAMS if p.test_runs > 1]

        maps = []
        if not map_files:
            for mf in {param.map for param in ENV_PARAMS}:
                maps.append(Map(mf))
        else:
            for mf in map_files:
                maps.append(Map(mf))

        mapfile = maps[0].filename

        self._game = Game(mapfile, ghosts, level_ghosts, lives, timeout)

        self.pacman_obs = obs_type(maps, lives)

        self.observation_space = self.pacman_obs.space

        self.action_space = gym.spaces.Discrete(len(self.keys))

        self._current_score = 0

        self.current_lives = self._game._initial_lives

        self._last_pos = None

        self._current_params = EnvParams(ghosts, level_ghosts, mapfile, 10)

        self.idle_steps = 0

        self.total_energy = len(self._game.map.energy)

        self.energy_reward_increment = (self.MAX_ENERGY_REWARD - self.MIN_ENERGY_REWARD) / (self.total_energy - 1)

        self._current_energy_reward = self.MIN_ENERGY_REWARD

        self.difficulty = self.INITIAL_DIFFICULTY

        self.wins_count = 0

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Received invalid action={} which is not part of the action space.".format(action))

        self._game.keypress(self.keys[action])

        self._game.compute_next_frame()

        game_state = json.loads(self._game.state)

        info = {k: game_state[k] for k in self.info_keywords if k in game_state}
        info['ghosts'] = self._current_params.ghosts
        info['level'] = self._current_params.level
        info['map'] = self._current_params.map
        info['win'] = 0
        info['d'] = self.difficulty

        done = not self._game.running

        reward = game_state['score'] - self._current_score

        # if reward == POINT_BOOST:
        #     reward = self.MAX_ENERGY_REWARD
        # elif reward == POINT_ENERGY:
        #     reward = self._current_energy_reward
        #     self._current_energy_reward += self.energy_reward_increment
        #     # print(self.total_energy, self.energy_reward_increment)
        #     # print(" --- Energy ID:", self.total_energy - len(game_state['energy']), "Reward:", reward)

        self._current_score = game_state['score']

        if not self.positive_rewards:

            if game_state['lives'] == 0:
                reward -= (self._game._timeout - game_state['step'] + 1) * (1.0 / TIME_BONUS_STEPS)

            reward -= 1.0 / TIME_BONUS_STEPS

        if done:
            if self._game._timeout != game_state['step'] and game_state['lives'] > 0:
                info['win'] = 1

                if not self.positive_rewards:
                    # reward = self._current_energy_reward
                    reward -= 1.0 / TIME_BONUS_STEPS

            if info['ghosts'] == self.difficulty:
                self.wins_count += info['win']

        if game_state['lives'] < self.current_lives:
            reward -= 50
            self.current_lives = game_state['lives']

        if self._last_pos:
            if game_state['pacman'] == self._last_pos:
                reward -= 0.5
                self.idle_steps += 1

        self._last_pos = game_state['pacman']
        info['idle'] = self.idle_steps

        # if not done and info['ghosts'] == 0 and game_state['step'] >= 500:
        #     self._game.stop()
        #     done = True

        #     if not self.positive_rewards:
        #         reward -= (self._game._timeout - game_state['step'] + 1) * (1.0 / TIME_BONUS_STEPS)

        #     if info['ghosts'] == self.difficulty:
        #         self.wins_count += info['win']

        return self.pacman_obs.get_obs(game_state, self._game.map), reward, done, info

    def reset(self):
        self._current_score = 0
        self._last_pos = None
        self.idle_steps = 0
        self._current_energy_reward = self.MIN_ENERGY_REWARD

        if self.training:

            self._current_params = random.choice(self.train_env_params)

            self.set_env_params(self._current_params)

            # if self.difficulty < self.max_ghosts and self.wins_count >= self.MIN_WINS:
            #     self.difficulty += 1
            #     self.wins_count = 0

            # easier_episode_prob = self.difficulty / 20

            # if random.random() < easier_episode_prob:
            #     if self.difficulty - 1 > self.INITIAL_DIFFICULTY:
            #         n_ghosts = random.randint(self.INITIAL_DIFFICULTY, self.difficulty - 1)
            #     else:
            #         n_ghosts = self.INITIAL_DIFFICULTY
            # else:
            #     n_ghosts = self.difficulty

            # self.set_n_ghosts(n_ghosts)

        self._game.start(self.agent_name)
        self._game.compute_next_frame()
        self.current_lives = self._game._initial_lives
        # self.total_energy = len(self._game.map.energy)
        # self.energy_reward_increment = (self.MAX_ENERGY_REWARD - self.MIN_ENERGY_REWARD) / (self.total_energy - 1)
        game_state = json.loads(self._game.state)
        return self.pacman_obs.get_obs(game_state, self._game.map)

    def set_env_params(self, env_params):
        self._current_params = env_params
        self._game._n_ghosts = env_params.ghosts
        self._game._l_ghosts = env_params.level
        self._game.map = Map(env_params.map)

    def render(self, mode='human'):
        self.pacman_obs.render()


def main():
    """
    Testing gym pacman enviorment.
    """

    agent_name = "GymEnvTestAgent"
    ghosts = 4
    level_ghosts = 1
    lives = 3
    timeout = 3000

    obs_type = MultiChannelObs

    positive_rewards = True

    env = PacmanEnv(obs_type, positive_rewards, agent_name, ghosts, level_ghosts, lives, timeout, training=False)
    env.set_env_params(EnvParams(1, 1, 'data/map2.bmp', 10))
    # print("Checking environment...")
    # check_env(env, warn=True)

    print("Environment created.")

    print("\nObservation space:", env.observation_space)
    print("Shape:", env.observation_space.shape)
    # print("Observation space high:", env.observation_space.high)
    # print("Observation space low:", env.observation_space.low)

    print("Action space:", env.action_space)

    obs = env.reset()
    done = False

    sum_rewards = 0
    action = 1  # a
    cur_x, cur_y = None, None

    # ADDED these lines to prevent infinite testing:
    step_count = 0
    max_steps = 100  # Limit for testing

    while not done and step_count < max_steps:
        env.render()

        x, y = env._game._pacman

        # Using agent from client example
        if x == cur_x and y == cur_y:
            if action in [1, 3]:    # ad
                action = random.choice([0, 2])
            elif action in [0, 2]:  # ws
                action = random.choice([1, 3])
        cur_x, cur_y = x, y

        print("key:", PacmanEnv.keys[action])

        obs, reward, done, info = env.step(action)

        sum_rewards += reward
        step_count += 1

        print("reward:", reward)
        print("sum_rewards:", sum_rewards)
        print("info:", info)
        print()

        # # Stop game for debugging

        # if reward > 0:
        #     env.render()
        #     print("Received positive reward.")
        #     break

        # if (obs_type == SingleChannelObs and np.isin(SingleChannelObs.GHOST_ZOMBIE, obs[0])) \
        #         or (obs_type == MultiChannelObs and
        #             np.isin(MultiChannelObs.PIXEL_IN, obs[MultiChannelObs.ZOMBIE_CH])):
        #     env.render()
        #     print("Zombie")
        #     break

        # if info['lives'] == 1:
        #     print("Lives: 1")
        #     break

    # print("score:", sum_rewards + (env._game._timeout / TIME_BONUS_STEPS))


if __name__ == "__main__":
    main()