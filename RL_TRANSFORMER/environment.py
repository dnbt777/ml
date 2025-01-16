import pygame
import random
from levels import *

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
grid_rows = 11
grid_cols = 15
GRID_SIZE = min(WIDTH // grid_cols, HEIGHT // grid_rows)
WIDTH, HEIGHT = grid_cols*GRID_SIZE, grid_rows*GRID_SIZE

# Colors
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
GREEN = (100, 200, 100)
BROWN = (139, 69, 19)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Panda Bomb Game")

# Clock for controlling frame rate
clock = pygame.time.Clock()

# Font for rendering text
font = pygame.font.SysFont(None, 24)

# Load panda image
panda_img = pygame.image.load("panda.png")
panda_img = pygame.transform.scale(panda_img, (int(GRID_SIZE * 0.8), int(GRID_SIZE * 0.8)))
panda_dead_img = pygame.transform.flip(panda_img, False, True)

# Player properties
panda_x, panda_y = 0, 0
panda_speed = 3
panda_dir = None  # Direction the panda is moving: 'up', 'down', 'left', 'right'
moving_keys = set()  # Tracks currently pressed movement keys
max_bombs = 1  # Max bombs the panda can place
active_bombs = 0  # Current number of active bombs
panda_alive = True

# Grid properties
level = make_default_level(grid_rows, grid_cols)
grid = level.terrain
spawn_x, spawn_y = level.spawn1
panda_x, panda_y = spawn_x * GRID_SIZE + GRID_SIZE // 2, spawn_y * GRID_SIZE + GRID_SIZE // 2


# Bomb properties
exploded_cells = []  # Stores cells temporarily turned orange
bombs = []  # List of bombs: each bomb is a dictionary {x, y, timer}
EXPLOSION_RADIUS = 1  # Radius in grid cells

# Powerup properties
powerups = []  # List of powerups: each powerup is a dictionary {x, y, type}
POWERUP_TYPES = ["B", "R", "E"]
panda_powerups = {"B": 0, "R": 0, "E": 0}  # Collected powerups

# Helper functions
def is_walkable(x0, y0, x1, y1):
    """
    Checks if the panda can walk from (x0, y0) to (x1, y1).
    Allows walking out of walls (including bombs) but prevents walking into them.
    """
    panda_rect = pygame.Rect(
        x1 - panda_img.get_width() // 2, y1 - panda_img.get_height() // 2,
        panda_img.get_width(), panda_img.get_height()
    )

    grid_x0, grid_y0 = int(x0 // GRID_SIZE), int(y0 // GRID_SIZE)
    grid_x1, grid_y1 = int(x1 // GRID_SIZE), int(y1 // GRID_SIZE)

    if 0 <= grid_x1 < grid_cols and 0 <= grid_y1 < grid_rows:
        # Check for obstacles in the target cell
        if grid[grid_y1][grid_x1] in ["wood", "stone"]:
            return False

        # Check for bombs in the target cell
        for bomb in bombs:
            if bomb["x"] == grid_x1 and bomb["y"] == grid_y1:
                # Allow exiting the current bomb cell but block entry into new bomb cells
                if grid_x0 == grid_x1 and grid_y0 == grid_y1:
                    return True
                return False

        # If no obstacles, movement is allowed
        return True

    # Out of bounds
    return False


def explode_bomb(bomb, visited):
    """Handles bomb explosion and chain reactions."""
    global active_bombs
    bomb_coords = (bomb["x"], bomb["y"])
    if bomb_coords in visited:
        return
    visited.add(bomb_coords)

    # Add the bomb's cell to exploded cells
    exploded_cells.append((bomb["x"], bomb["y"], pygame.time.get_ticks()))

    # Adjust explosion radius based on powerups
    explosion_radius = EXPLOSION_RADIUS + panda_powerups["E"]

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Cardinal directions only
        for step in range(1, explosion_radius + 1):
            nx, ny = bomb["x"] + dx * step, bomb["y"] + dy * step
            if 0 <= nx < grid_cols and 0 <= ny < grid_rows:
                if grid[ny][nx] == "stone":
                    break  # Stop at stone
                exploded_cells.append((nx, ny, pygame.time.get_ticks()))  # Mark cell as exploded
                if grid[ny][nx] == "wood":
                    grid[ny][nx] = "empty"
                    if random.random() < 0.8:  # 80% chance of dropping a powerup
                        powerup_type = random.choice(POWERUP_TYPES)
                        powerups.append({"x": nx, "y": ny, "type": powerup_type})
                    break  # Stop explosion at wood
                for other_bomb in bombs:
                    if other_bomb["x"] == nx and other_bomb["y"] == ny:
                        explode_bomb(other_bomb, visited)

    bombs.remove(bomb)
    active_bombs -= 1

# Main game loop
running = True
while running:
    screen.fill(GREEN)

    # Remove orange effect from exploded cells after a short time
    current_time = pygame.time.get_ticks()
    exploded_cells = [(x, y, t) for x, y, t in exploded_cells if current_time - t < 500]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Drop bomb
        if event.type == pygame.KEYDOWN and panda_alive:
            if event.key == pygame.K_SPACE:
                if active_bombs < max_bombs + panda_powerups["B"]:
                    grid_x, grid_y = int(panda_x // GRID_SIZE), int(panda_y // GRID_SIZE)
                    if grid[grid_y][grid_x] == "empty":
                        bombs.append({"x": grid_x, "y": grid_y, "timer": 5000})  # Timer in ms
                        active_bombs += 1

            # Start moving the panda
            if event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                moving_keys.add(event.key)

        if event.type == pygame.KEYUP:
            if event.key in moving_keys:
                moving_keys.remove(event.key)

    # Movement
    if panda_alive and moving_keys:
        speed = panda_speed + panda_powerups["R"]
        new_x, new_y = panda_x, panda_y
        if pygame.K_UP in moving_keys:
            new_x, new_y = panda_x, panda_y - speed
        elif pygame.K_DOWN in moving_keys:
            new_x, new_y = panda_x, panda_y + speed
        elif pygame.K_LEFT in moving_keys:
            new_x, new_y = panda_x - speed, panda_y
        elif pygame.K_RIGHT in moving_keys:
            new_x, new_y = panda_x + speed, panda_y

        if is_walkable(panda_x, panda_y, new_x, new_y):
            panda_x, panda_y = new_x, new_y


    # Check if panda touches an exploded cell
    if panda_alive:
        panda_grid_x, panda_grid_y = int(panda_x // GRID_SIZE), int(panda_y // GRID_SIZE)
        for x, y, _ in exploded_cells:
            if panda_grid_x == x and panda_grid_y == y:
                panda_alive = False
                break

    # Collect powerups
    for powerup in powerups[:]:
        if powerup["x"] == panda_grid_x and powerup["y"] == panda_grid_y:
            panda_powerups[powerup["type"]] += 1
            powerups.remove(powerup)

    # Draw grid
    for row in range(grid_rows):
        for col in range(grid_cols):
            x, y = col * GRID_SIZE, row * GRID_SIZE
            if grid[row][col] == "wood":
                pygame.draw.rect(screen, BROWN, (x, y, GRID_SIZE, GRID_SIZE))
            elif grid[row][col] == "stone":
                pygame.draw.rect(screen, GRAY, (x, y, GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(screen, BLACK, (x, y, GRID_SIZE, GRID_SIZE), 1)

    # Draw exploded cells
    for x, y, _ in exploded_cells:
        pygame.draw.rect(screen, ORANGE, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    # Draw bombs
    for bomb in bombs[:]:
        bomb_x, bomb_y = bomb["x"] * GRID_SIZE, bomb["y"] * GRID_SIZE
        pygame.draw.circle(screen, BLACK, (bomb_x + GRID_SIZE // 2, bomb_y + GRID_SIZE // 2), GRID_SIZE // 4)
        bomb["timer"] -= clock.get_time()

        # Display timer on bomb
        time_left = max(0, bomb["timer"] / 1000)
        timer_text = font.render(f"{time_left:.2f}", True, WHITE)
        screen.blit(timer_text, (bomb_x + GRID_SIZE // 4, bomb_y + GRID_SIZE // 4))

        # Bomb explosion
        if bomb["timer"] <= 0:
            explode_bomb(bomb, set())

    # Draw powerups
    for powerup in powerups:
        powerup_x, powerup_y = powerup["x"] * GRID_SIZE, powerup["y"] * GRID_SIZE
        powerup_text = font.render(powerup["type"], True, WHITE)
        screen.blit(powerup_text, (powerup_x + GRID_SIZE // 4, powerup_y + GRID_SIZE // 4))

    # Draw player
    if panda_alive:
        screen.blit(panda_img, (panda_x - panda_img.get_width() // 2, panda_y - panda_img.get_height() // 2))
    else:
        screen.blit(panda_dead_img, (panda_x - panda_img.get_width() // 2, panda_y - panda_img.get_height() // 2))

    # Update display
    pygame.display.flip()

    # Cap frame rate
    clock.tick(60)

pygame.quit()
