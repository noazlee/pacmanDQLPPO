#!/usr/bin/env python3
"""
Create a properly structured classic Pac-Man map with clean edges
"""

import pygame
import os

# Colors from mapa.py
WALL = 0x000000      # Black walls
EMPTY = 0xFFFFFF     # White walkable space  
ENERGY = 0xD6D7FF    # Light pink for energy dots
BOOST = 0x0026FF     # Orange for boost items
PACMAN = 0xD5FDD4    # Light green for Pac-Man spawn
GHOST = 0x00F900     # Bright green for ghost spawn

def create_fixed_classic_map(filename="data/fixed_classic.bmp"):
    """Create a clean classic Pac-Man style map with proper borders"""
    width, height = 25, 29  # Adjusted dimensions
    
    # Create surface - start with all walls
    surface = pygame.Surface((width, height))
    surface.fill(WALL)
    
    # Define a cleaner maze pattern
    maze = [
        # Each row is width=25 characters
        "1111111111111111111111111",  # 0  - Top border
        "1.......................1",  # 1
        "1.111.111111.1111.1111.",    # 2
        "1O111.111111.1111.1111O",    # 3  - O = boost items
        "1.111.111111.1111.1111.",    # 4
        "1.......................1",  # 5
        "1.111.11.1111111.11.111.1",  # 6
        "1.111.11.1111111.11.111.1",  # 7
        "1.....11....1....11.....1",  # 8
        "11111.11111.1.11111.11111",  # 9
        "11111.11111.1.11111.11111",  # 10
        "11111.11....1....11.11111",  # 11
        "11111.11.1111111.11.11111",  # 12
        "11111.11.1.......11.111111", # 13
        ".........1..G....1........",  # 14 - Tunnel row, G = ghost
        "11111.11.1.......11.111111", # 15
        "11111.11.1111111.11.11111",  # 16
        "11111.11....1....11.11111",  # 17
        "11111.11111.1.11111.11111",  # 18
        "11111.11111.1.11111.11111",  # 19
        "1.....11....1....11.....1",  # 20
        "1.111.11.1111111.11.111.1",  # 21
        "1.111.11.1111111.11.111.1",  # 22
        "1.......................1",  # 23
        "1.111.111111.1111.1111.",    # 24
        "1O111.111111.1111.1111O",    # 25 - O = boost items
        "1.111.111111.1111.1111.",    # 26
        "1...........P...........1",    # 27 - P = Pacman spawn
        "1111111111111111111111111",   # 28 - Bottom border
    ]
    
    # Convert maze to pixels
    for y, row in enumerate(maze):
        for x, char in enumerate(row):
            if x >= width or y >= height:
                continue
                
            if char == '1':      # Wall
                surface.set_at((x, y), WALL)
            elif char == '.':    # Walkable with energy dot
                surface.set_at((x, y), ENERGY)
            elif char == 'O':    # Boost item
                surface.set_at((x, y), BOOST)
            elif char == 'G':    # Ghost spawn
                surface.set_at((x, y), GHOST)
            elif char == 'P':    # Pacman spawn
                surface.set_at((x, y), PACMAN)
            elif char == ' ':    # Empty walkable space
                surface.set_at((x, y), EMPTY)
    
    # Clean up any remaining issues - ensure solid borders
    # Top and bottom borders
    for x in range(width):
        surface.set_at((x, 0), WALL)           # Top
        surface.set_at((x, height-1), WALL)    # Bottom
    
    # Left and right borders (except tunnel row)
    for y in range(height):
        if y != 14:  # Skip tunnel row
            surface.set_at((0, y), WALL)           # Left
            surface.set_at((width-1, y), WALL)    # Right
    
    # Make sure tunnel row (14) has proper openings
    for x in range(9):  # Left tunnel
        surface.set_at((x, 14), EMPTY)
    for x in range(16, width):  # Right tunnel  
        surface.set_at((x, 14), EMPTY)
    
    pygame.image.save(surface, filename)
    print(f"✓ Created fixed classic map: {filename}")
    print(f"  - Size: {width}x{height}")
    print(f"  - Clean borders with side tunnels")
    print(f"  - Ghost spawn in center")
    print(f"  - Pacman spawn at bottom")
    print(f"  - 4 power pellets in corners")
    
    return filename

if __name__ == "__main__":
    pygame.init()
    os.makedirs("data", exist_ok=True)
    
    print("Creating fixed Pac-Man maps...\n")
    
    # Create fixed classic map
    create_fixed_classic_map()
    