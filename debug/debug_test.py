#!/usr/bin/env python3
"""
Debug the server connection step by step - Updated for single endpoint
"""

import asyncio
import websockets
import json
import sys

async def test_connection():
    print("Testing server connection...")
    
    try:
        print("1. Connecting to ws://localhost:8000/...")  # Updated URL
        async with websockets.connect('ws://localhost:8000/') as ws:
            print("Connected successfully!")
            
            print("2. Sending join message...")
            join_msg = json.dumps({'cmd': 'join', 'name': 'DebugBot'})
            await ws.send(join_msg)
            print("Join message sent")
            
            print("3. Waiting for game info...")
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                print("Received response:")
                print(f" Length: {len(response)} characters")
                print(f" First 200 chars: {response[:200]}")
                
                # Try to parse as JSON
                try:
                    game_info = json.loads(response)
                    print("Response is valid JSON")
                    print(f" Keys: {list(game_info.keys())}")
                except json.JSONDecodeError as e:
                    print(f"Response is not valid JSON: {e}")
                    return
                
                print("4. Testing a single key press...")
                key_msg = json.dumps({'cmd': 'key', 'key': 'd'})
                await ws.send(key_msg)
                print("Key message sent")
                
                print("5. Waiting for game state...")
                try:
                    state = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    print("Received game state")
                    print(f"   Length: {len(state)} characters")
                    
                    # Try to parse game state
                    try:
                        state_data = json.loads(state)
                        print("✓ Game state is valid JSON")
                        print(f"   Keys: {list(state_data.keys())}")
                        if 'score' in state_data:
                            print(f"   Score: {state_data['score']}")
                        if 'lives' in state_data:
                            print(f"   Lives: {state_data['lives']}")
                        if 'pacman' in state_data:
                            print(f"   Pacman position: {state_data['pacman']}")
                    except json.JSONDecodeError as e:
                        print(f"✗ Game state is not valid JSON: {e}")
                        print(f"   Raw state: {state[:100]}...")
                        
                except asyncio.TimeoutError:
                    print("No game state received (timeout)")
                    
                print("\nBasic connection test completed successfully!")
                
            except asyncio.TimeoutError:
                print("✗ Timeout waiting for game info")
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}")
    except Exception as e:
        print(f"Connection error: {e}")
        import traceback
        traceback.print_exc()

async def test_game_creation():
    """Test if we can create the game components locally"""
    print("\nTesting local game creation...")
    
    try:
        from game import Game
        print("1. Creating game instance...")
        game = Game("data/map1.bmp", 2, 1, 3, 1000)
        print("✓ Game created")
        
        print("2. Starting game...")
        game.start("TestPlayer")
        print("✓ Game started")
        
        print("3. Testing a few steps...")
        for i in range(3):
            game.keypress('d')
            game.compute_next_frame()
            state = json.loads(game.state)
            print(f"   Step {i+1}: Score={state.get('score', 0)}, Lives={state.get('lives', 0)}")
        
        print("Local game test completed!")
        return True
        
    except Exception as e:
        print(f"Local game test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Debug Test for Pac-Man Server")
    print("=" * 60)
    
    print("Make sure server is running:")
    print("  python server.py --map data/map1.bmp --ghosts 2")
    print()
    
    # Test 1: Local game creation
    if not asyncio.run(test_game_creation()):
        print("\nLocal game creation failed - fix this first!")
        return
    
    print("\n" + "-" * 40)
    
    # Test 2: Server connection
    asyncio.run(test_connection())

if __name__ == "__main__":
    main()