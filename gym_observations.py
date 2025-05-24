from abc import ABC, abstractmethod
import numpy as np
import gym

from colorama import Back, Style


class PacmanObservation(ABC):

    def __init__(self, maps_list, max_lives):
        self._max_lives = max_lives
        self._shape = None

        self.center_x, self.center_y = None, None

        self.pac_x, self.pac_y = None, None

        self.walls = {}

        self.width = max(mapa.hor_tiles for mapa in maps_list)
        self.height = max(mapa.ver_tiles for mapa in maps_list)

        for y in range(self.height):
            for x in range(self.width):
                for mapa in maps_list:
                    if x >= mapa.hor_tiles or y >= mapa.ver_tiles:
                        continue
                    if mapa.is_wall((x, y)):
                        if mapa.filename in self.walls:
                            self.walls[mapa.filename].append((x, y))
                        else:
                            self.walls[mapa.filename] = [(x, y)]

    def _new_points(self, x, y, mapa):
        new_points = []
        new_x = ((x - self.pac_x) + self.center_x) % mapa.hor_tiles
        new_y = ((y - self.pac_y) + self.center_y) % mapa.ver_tiles
        new_points.append((new_x, new_y))

        # if y == mapa.ver_tiles - 1 and mapa.ver_tiles < self.height:
        #     for i in range(self.height - mapa.ver_tiles + 1):
        #         new_points.append((new_x, (new_y + i) % self.height))

        # if x == mapa.hor_tiles - 1 and mapa.hor_tiles < self.width:
        #     for i in range(self.width - mapa.hor_tiles + 1):
        #         new_points.append(((new_x + i) % self.width, new_y))

        x = new_x
        y = new_y
        new_x = None
        new_y = None
        if x + mapa.hor_tiles < self.width:
            new_x = x + mapa.hor_tiles
            new_points.append((new_x, y))

        if y + mapa.ver_tiles < self.height:
            new_y = y + mapa.ver_tiles
            new_points.append((x, new_y))

        if new_x is not None and new_y is not None:
            new_points.append((new_x, new_y))

        return new_points

    @property
    def space(self):
        return gym.spaces.Box(low=0, high=255, shape=self._shape, dtype=np.uint8)

    @abstractmethod
    def get_obs(self, game_state, mapa):
        pass

    @abstractmethod
    def render(self):
        pass


class MultiChannelObs(PacmanObservation):

    PIXEL_IN = 255
    PIXEL_EMPTY = 0

    ENERGY_IN = 64
    BOOST_IN = 255
    ENERGY_EMPTY = 0

    # Channels Index
    WALL_CH = 0
    EMPTY_CH = 1
    ENERGY_CH = 2
    GHOST_CH = 3
    ZOMBIE_CH = 4
    LIVES_CH = 5

    def __init__(self, game_map, max_lives):
        super().__init__(game_map, max_lives)

        self._shape = (6, self.height, self.width)

        self._obs = np.full(self._shape, self.PIXEL_EMPTY, dtype=np.uint8)

    def get_obs(self, game_state, mapa):

        self.center_x, self.center_y = int(self.width / 2), int(self.height / 2)

        # Reset channels
        self._obs[self.EMPTY_CH][...] = self.PIXEL_IN
        self._obs[self.EMPTY_CH][self.center_y][self.center_x] = self.PIXEL_EMPTY
        self._obs[self.WALL_CH][...] = self.PIXEL_EMPTY
        self._obs[self.ENERGY_CH][...] = self.ENERGY_EMPTY
        self._obs[self.GHOST_CH][...] = self.PIXEL_EMPTY
        self._obs[self.ZOMBIE_CH][...] = self.PIXEL_EMPTY
        self._obs[self.LIVES_CH][...] = self.PIXEL_EMPTY

        self.pac_x, self.pac_y = game_state['pacman']

        for x, y in self.walls[mapa.filename]:
            for x, y in self._new_points(x, y, mapa):
                self._obs[self.WALL_CH][y][x] = self.PIXEL_IN
                self._obs[self.EMPTY_CH][y][x] = self.PIXEL_EMPTY

        for x, y in game_state['energy']:
            for x, y in self._new_points(x, y, mapa):
                self._obs[self.ENERGY_CH][y][x] = self.ENERGY_IN
                self._obs[self.EMPTY_CH][y][x] = self.PIXEL_EMPTY

        for x, y in game_state['boost']:
            for x, y in self._new_points(x, y, mapa):
                self._obs[self.ENERGY_CH][y][x] = self.BOOST_IN
                self._obs[self.EMPTY_CH][y][x] = self.PIXEL_EMPTY

        for ghost in game_state['ghosts']:
            x, y = ghost[0]

            for x, y in self._new_points(x, y, mapa):
                if ghost[1]:
                    self._obs[self.ZOMBIE_CH][y][x] = self.PIXEL_IN
                else:
                    self._obs[self.GHOST_CH][y][x] = self.PIXEL_IN

            self._obs[self.EMPTY_CH][y][x] = self.PIXEL_EMPTY

        lives_y_fill = int(game_state['lives'] * (self.height / self._max_lives))
        self._obs[self.LIVES_CH][:lives_y_fill][...] = self.PIXEL_IN

        return self._obs

    def render(self):
        for y in range(self.height):
            for x in range(self.width):
                color = None

                if self._obs[self.GHOST_CH][y][x] == self.PIXEL_IN:
                    color = Back.MAGENTA
                elif self._obs[self.ZOMBIE_CH][y][x] == self.PIXEL_IN:
                    color = Back.BLUE
                elif self._obs[self.ENERGY_CH][y][x] == self.BOOST_IN:
                    color = Back.CYAN
                elif self._obs[self.ENERGY_CH][y][x] == self.ENERGY_IN:
                    color = Back.RED
                elif self._obs[self.WALL_CH][y][x] == self.PIXEL_IN:
                    color = Back.WHITE
                elif self._obs[self.EMPTY_CH][y][x] == self.PIXEL_IN:
                    color = Back.BLACK
                else:
                    color = Back.YELLOW

                print(color, ' ', end='')
            print(Style.RESET_ALL)

        # np.set_printoptions(edgeitems=30, linewidth=100000)
        # print(self._obs)


class SingleChannelObs(PacmanObservation):

    GHOST = 0
    WALL = 51
    EMPTY = 102
    ENERGY = 153
    BOOST = 204
    GHOST_ZOMBIE = 255

    def __init__(self, game_map, max_lives):
        super().__init__(game_map, max_lives)

        # First dimension is for the image channels required by tf.nn.conv2d
        self._shape = (1, self.height, self.width)

        self._obs = np.full(self._shape, self.EMPTY, dtype=np.uint8)

    def get_obs(self, game_state, mapa):

        self.center_x, self.center_y = int(self.width / 2), int(self.height / 2)

        self._obs[0][...] = self.EMPTY

        self.pac_x, self.pac_y = game_state['pacman']

        for x, y in self.walls[mapa.filename]:
            for x, y in self._new_points(x, y, mapa):
                self._obs[0][y][x] = self.WALL

        for x, y in game_state['energy']:
            for x, y in self._new_points(x, y, mapa):
                self._obs[0][y][x] = self.ENERGY

        for x, y in game_state['boost']:
            for x, y in self._new_points(x, y, mapa):
                self._obs[0][y][x] = self.BOOST

        for ghost in game_state['ghosts']:
            x, y = ghost[0]
            for x, y in self._new_points(x, y, mapa):

                if ghost[1]:
                    self._obs[0][y][x] = self.GHOST_ZOMBIE
                else:
                    self._obs[0][y][x] = self.GHOST

        return self._obs

    def render(self):
        for y in range(self.height):
            for x in range(self.width):
                color = None
                value = self._obs[0][y][x]

                if value == self.GHOST:
                    color = Back.MAGENTA
                elif value == self.GHOST_ZOMBIE:
                    color = Back.BLUE
                elif value == self.EMPTY:
                    color = Back.BLACK

                    if x == self.center_x and y == self.center_y:
                        color = Back.YELLOW

                elif value == self.WALL:
                    color = Back.WHITE
                elif value == self.ENERGY:
                    color = Back.RED
                elif value == self.BOOST:
                    color = Back.CYAN

                print(color, ' ', end='')
            print(Style.RESET_ALL)

        # np.set_printoptions(edgeitems=30, linewidth=100000)
        # print(self._obs)