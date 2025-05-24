#!/usr/bin/env python3
"""
Debug why ghosts aren't moving
"""

from game import Game
from ghost1 import Ghost
from mapa import Map
import json

def test_ghost_movement():
    print("Testing ghost movement logic...")
    
    # Create game
    game = Game("data/map1.bmp", 2, 0, 3, 1000)  # Level 0, 2 ghosts
    game.start("DebugTest")
    
    print(f"Map size: {game.map.size}")
    print(f"Ghost spawn: {game.map.ghost_spawn}")
    print(f"Pacman spawn: {game.map.pacman_spawn}")
    
    # Test first few game steps
    for step in range(10):
        print(f"\n--- Step {step} ---")
        
        # Get current state
        state = json.loads(game.state)
        
        print(f"Pacman: {state.get('pacman', 'N/A')}")
        print(f"Ghosts: {[(g.x, g.y) for g in game._ghosts]}")
        print(f"Ghost waits: {[g.wait for g in game._ghosts]}")
        
        # Check what each ghost is doing
        for i, ghost in enumerate(game._ghosts):
            print(f"Ghost {i}:")
            print(f"  Position: ({ghost.x}, {ghost.y})")
            print(f"  Wait: {ghost.wait}")
            print(f"  Direction: {ghost.direction}")
            print(f"  Level: {ghost.level}")
            print(f"  Visibility: {ghost.visibility}")
            
            # Test if ghost can move
            test_directions = ['w', 'a', 's', 'd']
            valid_moves = []
            for d in test_directions:
                new_pos = game.map.calc_pos((ghost.x, ghost.y), d)
                if new_pos != (ghost.x, ghost.y):  # Can move
                    valid_moves.append((d, new_pos))
            
            print(f"  Valid moves: {valid_moves}")
        
        # Advance game
        game.keypress('d')  # Pacman moves right
        game.compute_next_frame()

if __name__ == "__main__":
    test_ghost_movement()