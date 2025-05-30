import os
import asyncio
import pygame
import random
from functools import partial
from mapa import Map
import json
import websockets
import logging
import argparse
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('websockets')
logger.setLevel(logging.WARN)

PACMAN = {'up': (24, 72), 'left': (0, 72), 'down': (120, 72), 'right': (96, 72)}

RED_GHOST = {'up': (168, 144), 'left': (96, 144), 'down': (48, 144), 'right': (0, 144)}
PINK_GHOST = {'up': (168, 192), 'left': (96, 192), 'down': (48, 192), 'right': (0, 192)}
ORANGE_GHOST = {'up': (168, 216), 'left': (96, 216), 'down': (48, 216), 'right': (0, 216)}
BLUE_GHOST = {'up': (8*24+168, 192), 'left': (8*24+96, 192), 'down': (8*24+48, 192), 'right': (8*24+0, 192)}
GHOSTS = [RED_GHOST, PINK_GHOST, ORANGE_GHOST, BLUE_GHOST]

CHAR_LENGTH = 26
CHAR_SIZE= CHAR_LENGTH, CHAR_LENGTH #22 + 2px border
ENERGY_RADIUS = 4
BOOST_RADIUS = 8
SCALE = None 

COLORS = {'white':(255,255,255), 'red':(255,0,0), 'pink':(255,105,180), 'blue':(135,206,235), 'orange':(255,165,0), 'yellow':(255,255,0)}
BACKGROUND = (0, 0, 0)
RANKS = {1:"1ST", 2:"2ND", 3:"3RD", 4:"4TH", 5:"5TH", 6:"6TH", 7:"7TH", 8:"8TH", 9:"9TH", 10:"10TH"}

async def messages_handler(ws_path, queue):
    print(f"Connecting to {ws_path}...")
    try:
        async with websockets.connect(ws_path) as websocket:
            print("✓ Connected to server")
            
            # Join as viewer - use special name to indicate viewer
            await websocket.send(json.dumps({"cmd": "join", "name": "pygame_viewer"}))
            initial_msg = await websocket.recv()
            print("✓ Received initial game info")
            queue.put_nowait(initial_msg)

            message_count = 0
            while True:
                try:
                    r = await websocket.recv()
                    queue.put_nowait(r)
                    message_count += 1
                    if message_count % 50 == 0:
                        print(f"Received {message_count} game state messages")
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                    break
                except Exception as e:
                    print(f"Error receiving message: {e}")
                    break
    except Exception as e:
        print(f"Connection error: {e}")

class GameOver(BaseException):
    pass

class PacMan(pygame.sprite.Sprite):
    def __init__(self, *args, **kw):
        self.x, self.y = (kw.pop("pos", ((kw.pop("x", 0), kw.pop("y", 0)))))
        self.images = kw.get("images")
        self.rect = pygame.Rect((self.x, self.y) + CHAR_SIZE)
        self.image = pygame.Surface(CHAR_SIZE)
        self.direction = "left"
        if self.images:
            self.image.blit(*self.sprite_pos())
        self.image = pygame.transform.scale(self.image, scale((1,1)))
        super().__init__()
   
    def sprite_pos(self, new_pos=(0,0)):
        if not self.images:
            return (pygame.Surface((22, 22)), (2,2), (0, 0, 22, 22))
            
        CROP = 22 
        x, y = new_pos 
        
        if x > self.x:
            self.direction = "right"
        if x < self.x:
            self.direction = "left"
        if y > self.y:
            self.direction = "down"
        if y < self.y:
            self.direction = "up"

        x, y = PACMAN[self.direction]
        return (self.images, (2,2), (x, y, x+CROP, y+CROP))

    def update(self, state):
        if 'pacman' in state:
            x, y = state['pacman']
            sx, sy = scale((x, y))
            self.rect = pygame.Rect((sx, sy) + CHAR_SIZE)
            self.image = pygame.Surface(CHAR_SIZE)
            self.image.fill((255, 255, 0))  # Yellow for Pacman
            if self.images:
                self.image.fill((0,0,0))
                self.image.blit(*self.sprite_pos((sx, sy)))
            self.image = pygame.transform.scale(self.image, scale((1, 1)))
            self.x, self.y = sx, sy

class Ghost(pygame.sprite.Sprite):
    def __init__(self, *args, **kw):
        self.x, self.y = (kw.pop("pos", ((kw.pop("x", 0), kw.pop("y", 0)))))
        self.index = kw.pop("index", 0)
        self.images = kw.get("images")
        self.direction = "left"
        self.rect = pygame.Rect((self.x, self.y) + CHAR_SIZE)
        self.image = pygame.Surface(CHAR_SIZE)
        if self.images:
            self.image.blit(*self.sprite_pos((self.x, self.y)))
        else:
            # Fallback colors for ghosts
            colors = [(255, 0, 0), (255, 105, 180), (255, 165, 0), (0, 0, 255)]
            self.image.fill(colors[self.index % len(colors)])
        self.image = pygame.transform.scale(self.image, scale((1,1)))
        super().__init__()
   
    def sprite_pos(self, new_pos, boost=False):
        if not self.images:
            return (pygame.Surface((22, 22)), (2,2), (0, 0, 22, 22))
            
        CROP = 22 
        x, y = new_pos 

        if x > self.x:
            self.direction = "right"
        if x < self.x:
            self.direction = "left"
        if y > self.y:
            self.direction = "down"
        if y < self.y:
            self.direction = "up"

        x, y = GHOSTS[self.index][self.direction] 

        if boost:
            x, y = 168, 96
        return (self.images, (2,2), (x, y, x+CROP, y+CROP))

    def update(self, state):
        if 'ghosts' in state and self.index < len(state['ghosts']):
            (x, y), zombie, z_timeout = state['ghosts'][self.index]
            sx, sy = scale((x, y))
            self.rect = pygame.Rect((sx, sy) + CHAR_SIZE)
            self.image = pygame.Surface(CHAR_SIZE)
            
            if zombie:
                self.image.fill((0, 0, 255))  # Blue for zombie ghosts
            else:
                colors = [(255, 0, 0), (255, 105, 180), (255, 165, 0), (0, 255, 0)]
                self.image.fill(colors[self.index % len(colors)])
                
            if self.images:
                self.image.fill((0,0,0))
                self.image.blit(*self.sprite_pos((sx, sy), zombie))
            self.image = pygame.transform.scale(self.image, scale((1,1)))
            self.x, self.y = sx, sy

def clear_callback(surf, rect):
    color = 0, 0, 0
    surf.fill(color, rect)

def scale(pos):
    x, y = pos
    return int(x * CHAR_LENGTH / SCALE), int(y * CHAR_LENGTH / SCALE)

def draw_background(mapa, SCREEN):
    for x in range(int(mapa.size[0])):
        for y in range(int(mapa.size[1])):
            if mapa.is_wall((x,y)):
                draw_wall(SCREEN, x, y)
        
def draw_wall(SCREEN, x, y):
    wx, wy = scale((x, y))
    wall_color = (100,100,100)
    pygame.draw.rect(SCREEN, wall_color,
                       (wx,wy,*scale((1,1))), 0)

def draw_energy(SCREEN, x, y, boost=False):
    ex, ey = scale((x, y))
    color = (255, 255, 255) if not boost else (255, 255, 0)
    radius = int(BOOST_RADIUS/SCALE) if boost else int(ENERGY_RADIUS/SCALE)
    pygame.draw.circle(SCREEN, color,
                       (ex+int(CHAR_LENGTH/SCALE/2),ey+int(CHAR_LENGTH/SCALE/2)),
                       radius, 0)

def draw_info(SCREEN, text, pos, color=(255,255,255), background=None):
    try:
        myfont = pygame.font.Font(None, int(30/SCALE))
        textsurface = myfont.render(text, True, color, background)

        erase = pygame.Surface(textsurface.get_size())
        erase.fill((0,0,0))

        if pos[0] > SCREEN.get_size()[0]:
            pos = SCREEN.get_size()[0] - textsurface.get_size()[0], pos[1]
        if pos[1] > SCREEN.get_size()[1]:
            pos = pos[0], SCREEN.get_size()[1] - textsurface.get_size()[1]

        SCREEN.blit(erase,pos)
        SCREEN.blit(textsurface,pos)
    except:
        pass  # Skip if font rendering fails

async def main_loop(q):
    main_group = pygame.sprite.OrderedUpdates()
    
    # Try to load sprites - fallback to colored rectangles if not available
    images = None
    try:
        images = pygame.image.load("data/sprites/spritemap.png")
        print(" Loaded sprite images")
    except:
        print("Could not load sprites - using colored rectangles")
   
    logging.info("Waiting for map information from server") 
    
    # Wait for initial game info
    state = await q.get()
    print(f"Received initial state: {len(state)} characters")

    try:
        newgame_json = json.loads(state)
        print(f"Parsed game info: {list(newgame_json.keys())}")
        
        mapa = Map(newgame_json["map"])
        print(f"Loaded map: {mapa.size}")
        
        GAME_SPEED = newgame_json.get("fps", 10)
        print(f"Game speed: {GAME_SPEED} FPS")
        
    except Exception as e:
        print(f"Error parsing initial game info: {e}")
        return
    
    global SCALE
    SCALE = 1  # You can adjust this to make the game bigger/smaller
    
    SCREEN = pygame.display.set_mode(scale(mapa.size))
    pygame.display.set_caption("Pac-Man Live Game")
    print(f"Created display: {scale(mapa.size)}")
   
    draw_background(mapa, SCREEN)
    
    # Create sprites
    main_group.add(PacMan(pos=scale(mapa.pacman_spawn), images=images))
    
    n_ghosts = newgame_json.get("ghosts", 2)
    for i in range(n_ghosts):
        main_group.add(Ghost(pos=scale(mapa.ghost_spawn), images=images, index=i))
    
    print(f"Created {n_ghosts+1} sprites")
    
    state = dict()
    game_active = False
    last_update = time.time()
    
    print("Starting main game loop...")
    
    while True:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return
 
        # Get new game state
        try:
            new_state_str = q.get_nowait()
            try:
                new_state = json.loads(new_state_str)
                if 'pacman' in new_state:  # Valid game state
                    state = new_state
                    game_active = True
                    last_update = time.time()
            except json.JSONDecodeError:
                pass  # Skip invalid JSON
        except:
            pass  # No new state available
        
        # Clear screen
        SCREEN.fill((0, 0, 0))
        main_group.clear(SCREEN, clear_callback)
        
        if game_active and state:
            # Draw game elements
            if "energy" in state:
                for x, y in state["energy"]:
                    draw_energy(SCREEN, x, y)
            if "boost" in state:
                for x, y in state["boost"]:
                    draw_energy(SCREEN, x, y, True)
            
            # Update and draw sprites
            main_group.update(state)
            main_group.draw(SCREEN)
            
            # Draw game info
            if "score" in state:
                draw_info(SCREEN, f"Score: {state['score']}", (10, 10))
            if "lives" in state:
                draw_info(SCREEN, f"Lives: {state['lives']}", (10, 40))
            if "step" in state:
                draw_info(SCREEN, f"Step: {state['step']}", (10, 70))
                
            # Check if game is still active
            if time.time() - last_update > 5:  # No update for 5 seconds
                game_active = False
                
        else:
            # Show waiting message
            draw_info(SCREEN, "Waiting for game...", scale((5, 5)))
            draw_info(SCREEN, "Connect a client to start playing!", scale((5, 7)))
        
        # Draw background walls
        draw_background(mapa, SCREEN)
        
        pygame.display.flip()
        await asyncio.sleep(1./max(GAME_SPEED, 10))  # Limit FPS

async def main():
    pygame.init()
    pygame.font.init()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", help="IP address of the server", default='localhost')
    parser.add_argument("--scale", help="reduce size of window by x times", type=int, default=1)
    parser.add_argument("--port", help="TCP port", type=int, default=8000)
    args = parser.parse_args()
    
    global SCALE
    SCALE = args.scale

    q = asyncio.Queue()
    
    ws_path = f'ws://{args.server}:{args.port}/'
    
    print("Starting Pac-Man Viewer")
    print(f"Connecting to: {ws_path}")
    
    try:
        await asyncio.gather(
            messages_handler(ws_path, q), 
            main_loop(q)
        )
    except KeyboardInterrupt:
        print("\nViewer stopped by user")
    except Exception as e:
        print(f"Viewer error: {e}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    asyncio.run(main())