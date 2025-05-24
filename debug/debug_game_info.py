#!/usr/bin/env python3
"""
Debug the game.info() method specifically
"""

from game import Game
import json

def test_game_info():
    print("Testing game.info() method...")
    
    try:
        print("1. Creating game...")
        game = Game("data/map1.bmp", 2, 1, 3, 1000)
        print("✓ Game created")
        
        print("2. Calling game.info()...")
        try:
            info = game.info()
            print("✓ game.info() returned successfully")
            print(f"   Type: {type(info)}")
            print(f"   Length: {len(info)} characters")
            print(f"   First 200 chars: {info[:200]}")
            
            print("3. Testing JSON parsing...")
            try:
                parsed = json.loads(info)
                print("✓ JSON is valid")
                print(f"   Keys: {list(parsed.keys())}")
                for key, value in parsed.items():
                    print(f"     {key}: {type(value)} = {value}")
            except json.JSONDecodeError as e:
                print(f"✗ JSON parsing failed: {e}")
                print("Raw info:")
                print(repr(info))
            
        except Exception as e:
            print(f"✗ game.info() failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"✗ Game creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_game_info()