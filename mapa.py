import os
import pygame
import logging
from enum import Enum
import platform

WALL = 0xff000000
ENERGY = 0xffffd7d6
BOOST = 0xffff2600
PACMAN = 0xffd4fdd5
GHOST = 0xff00f900

if platform.system() == "Windows" or os.uname().machine.startswith("arm"):
    MASK = 0xFFFFFFFF
    WALL = ~(WALL ^ MASK)
    ENERGY = ~(ENERGY ^ MASK)
    BOOST = ~(BOOST ^ MASK)
    PACMAN = ~(PACMAN ^ MASK)
    GHOST = ~(GHOST ^ MASK)

class Tiles(Enum):
    ENERGY = 1
    BOOST = 3

class Map:
    def __init__(self, filename):
        self._filename = filename
        image = pygame.image.load(filename)
        self.pxarray = pygame.PixelArray(image)
        self.hor_tiles=len(self.pxarray)
        self.ver_tiles=len(self.pxarray[0])

        self._energy = []
        self._boost = []
        
        # Initialize with default values
        self._pacman_spawn = (15, 14)
        self._ghost_spawn = (1, 2)
        
        # Debug: collect all unique colors
        unique_colors = set()
        color_positions = {}

        for x in range(self.hor_tiles):
            for y in range(self.ver_tiles):
                p = self.pxarray[x][y]
                unique_colors.add(p)
                
                if p not in color_positions:
                    color_positions[p] = []
                color_positions[p].append((x, y))
                
                # Try to match colors more flexibly
                if self.is_black_ish(p):  # Wall
                    continue
                elif self.is_blue_ish(p):  # Likely boost
                    self._boost.append((x,y))
                elif self.is_light_green_ish(p):  # Likely pacman spawn
                    self._pacman_spawn = (15, 14)
                elif self.is_bright_green_ish(p):  # Likely ghost spawn
                    self._ghost_spawn = (x, y)
        
        # ADDED: Create energy dots in walkable areas
        self._create_energy_dots()
        
        # Debug output
        print(f"Map loaded: {filename}")
        print(f"  - Size: {self.hor_tiles}x{self.ver_tiles}")
        print(f"  - Unique colors found: {len(unique_colors)}")
        
        for color in sorted(unique_colors):
            count = len(color_positions[color])
            hex_color = f"0x{color:08x}"
            print(f"    {hex_color}: {count} pixels")
            
        print(f"  - Pacman spawn: {self._pacman_spawn}")
        print(f"  - Ghost spawn: {self._ghost_spawn}")
        print(f"  - Energy dots: {len(self._energy)}")
        print(f"  - Boost items: {len(self._boost)}")

    def _create_energy_dots(self):
        """Create energy dots in walkable areas"""
        for x in range(1, self.hor_tiles - 1):
            for y in range(1, self.ver_tiles - 1):
                # Skip if it's a wall
                if self.is_wall((x, y)):
                    continue
                
                # Skip spawn points
                if (x, y) == self._pacman_spawn or (x, y) == self._ghost_spawn:
                    continue
                    
                # Skip boost positions
                if (x, y) in self._boost:
                    continue
                
                # Add energy dots in a pattern (every 2nd position)
                if x % 2 == 1 and y % 2 == 1:
                    self._energy.append((x, y))

    def is_black_ish(self, color):
        """Check if color is black or very dark"""
        # Extract RGB values
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF
        return r < 50 and g < 50 and b < 50

    def is_blue_ish(self, color):
        """Check if color is red/orange (boost items)"""
        r = color & 0xFF
        g = (color >> 8) & 0xFF
        b = (color >> 16) & 0xFF
        return r > 150 and g < 100 and b < 100

    def is_light_green_ish(self, color):
        """Check if color is light green (pacman spawn)"""
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF
        return g > 200 and r > 150 and b > 150

    def is_bright_green_ish(self, color):
        """Check if color is bright green (ghost spawn)"""
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF
        return g > 200 and r < 100 and b < 100

    def is_pink_ish(self, color):
        """Check if color is pink/light (empty space or energy)"""
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF
        return r > 200 and g > 150 and b > 150

    @property
    def filename(self):
        return self._filename

    @property
    def size(self):
        return self.hor_tiles, self.ver_tiles 

    @property
    def energy(self):
        return self._energy

    @property
    def boost(self):
        return self._boost

    @property
    def pacman_spawn(self):
        return self._pacman_spawn

    @property
    def ghost_spawn(self):
        return self._ghost_spawn

    def is_wall(self, pos):
        x, y = pos
        if x not in range(self.hor_tiles) or y not in range(self.ver_tiles):
            return True
        p = self.pxarray[x][y]
        return self.is_black_ish(p)

    def calc_pos(self, cur, direction):
        assert direction in "wasd"

        cx, cy = cur
        npos = cur
        if direction == 'w':
            npos = cx, cy-1
        if direction == 'a':
            npos = cx-1, cy
        if direction == 's':
            npos = cx, cy+1
        if direction == 'd':
            npos = cx+1, cy

        #wrap map
        nx, ny = npos
        if nx < 0:
            nx = self.hor_tiles-1
        if nx == self.hor_tiles:
            nx = 0
        if ny < 0:
            ny = self.ver_tiles-1
        if ny == self.ver_tiles:
            ny = 0
        npos = nx, ny 

        #test wall
        if self.is_wall(npos):
            return cur
   
        return npos