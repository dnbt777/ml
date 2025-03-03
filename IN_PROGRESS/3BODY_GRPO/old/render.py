import pygame
import numpy as np
from environment import init_solarsystems, step_simulation, true_simulation_size, simulation_size, SolarSystem
import jax.random as jrand
import time

from render_utils import *

pygame.init()

WIDTH, HEIGHT = 800, 600
SCALE = simulation_size
PLANET_COLOR = (200, 150, 0)
SUN_COLOR = (255, 255, 0)
GRID_COLOR = (200, 200, 200)  # Light grey grid color
FPS = 60
MOUSE_SENSITIVITY = 0.1
MOVEMENT_SPEED = 0.1
GRID_LINES = 1  # Number of grid lines on each axis
MIN_SUN_RADIUS = 7
MIN_PLANET_RADIUS = 2


# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Solar System Simulation")
pygame.event.set_grab(True)
pygame.mouse.set_visible(False)

key = jrand.PRNGKey(int(10000 * time.time()))
simulations = 1  # Run n simulations simultaneously
solar_system = init_solarsystems(key, simulations)

# Camera settings
camera_position = np.array([8.0, 2.0, -6.0])  # Camera position
yaw, pitch = 49.4, 8.4  # Camera angles (degrees)
focal_length = 5.0  # Perspective strength


def get_right_vector_xz(yaw):
    yaw_rad = np.radians(yaw)
    x = np.cos(yaw_rad)
    z = np.sin(yaw_rad)
    return np.array([x, 0, z]) 


def get_forward_vector_xz(yaw):
    yaw_rad = np.radians(yaw)
    x = -np.cos(yaw_rad - np.pi / 2)
    z = -np.sin(yaw_rad - np.pi / 2)
    return np.array([x, 0, z])


def to_screen_coords(position):
    global yaw, pitch, camera_position

    relative_position = position - camera_position

    yaw_rad = np.radians(-yaw)
    pitch_rad = np.radians(-pitch)
    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
    cos_pitch, sin_pitch = np.cos(pitch_rad), np.sin(pitch_rad)

    x1 = cos_yaw * relative_position[0] - sin_yaw * relative_position[2]
    z1 = sin_yaw * relative_position[0] + cos_yaw * relative_position[2]

    y2 = cos_pitch * relative_position[1] - sin_pitch * z1
    z2 = sin_pitch * relative_position[1] + cos_pitch * z1

    if z2 <= 0:
        return None

    x_proj = (focal_length / z2) * x1
    y_proj = (focal_length / z2) * y2

    screen_x = int((x_proj + 0.5) * WIDTH)
    screen_y = int((1 - (y_proj + 0.5)) * HEIGHT)

    return screen_x, screen_y




def draw_grid_skybox():
    grid_spacing = simulation_size / GRID_LINES
    grid_size = simulation_size
    grid_base_corner = (0, 0, 0)
    offset = np.array(grid_base_corner)


    for x in np.arange(0, grid_size, grid_spacing):
        for y in np.arange(0, grid_size, grid_spacing):

            start = to_screen_coords([x, y, grid_size] + offset)
            end_x = to_screen_coords([x + grid_spacing, y, grid_size] + offset)
            end_y = to_screen_coords([x, y + grid_spacing, grid_size] + offset)
            if start and end_x:
                pygame.draw.aaline(screen, GRID_COLOR, start, end_x)
            if start and end_y:
                pygame.draw.aaline(screen, GRID_COLOR, start, end_y)

            start = to_screen_coords([x, y, 0] + offset)
            end_x = to_screen_coords([x + grid_spacing, y, 0] + offset)
            end_y = to_screen_coords([x, y + grid_spacing, 0] + offset)
            if start and end_x:
                pygame.draw.aaline(screen, GRID_COLOR, start, end_x)
            if start and end_y:
                pygame.draw.aaline(screen, GRID_COLOR, start, end_y)

    for x in np.arange(0, grid_size, grid_spacing):
        for z in np.arange(0, grid_size, grid_spacing):
            start = to_screen_coords([x, 0, z] + offset)
            end_x = to_screen_coords([x + grid_spacing, 0, z] + offset)
            end_z = to_screen_coords([x, 0, z + grid_spacing] + offset)
            if start and end_x:
                pygame.draw.aaline(screen, GRID_COLOR, start, end_x)
            if start and end_z:
                pygame.draw.aaline(screen, GRID_COLOR, start, end_z)


            start = to_screen_coords([x, grid_size, z] + offset)
            end_x = to_screen_coords([x + grid_spacing, grid_size, z] + offset)
            end_z = to_screen_coords([x, grid_size, z + grid_spacing] + offset)
            if start and end_x:
                pygame.draw.aaline(screen, GRID_COLOR, start, end_x)
            if start and end_z:
                pygame.draw.aaline(screen, GRID_COLOR, start, end_z)

    for y in np.arange(0, grid_size, grid_spacing):
        for z in np.arange(0, grid_size, grid_spacing):
            start = to_screen_coords([0, y, z] + offset)
            end_y = to_screen_coords([0, y + grid_spacing, z] + offset)
            end_z = to_screen_coords([0, y, z + grid_spacing] + offset)
            if start and end_y:
                pygame.draw.aaline(screen, GRID_COLOR, start, end_y)
            if start and end_z:
                pygame.draw.aaline(screen, GRID_COLOR, start, end_z)

            start = to_screen_coords([grid_size, y, z] + offset)
            end_y = to_screen_coords([grid_size, y + grid_spacing, z] + offset)
            end_z = to_screen_coords([grid_size, y, z + grid_spacing] + offset)
            if start and end_y:
                pygame.draw.aaline(screen, GRID_COLOR, start, end_y)
            if start and end_z:
                pygame.draw.aaline(screen, GRID_COLOR, start, end_z)



def aim_at_body(camera_position, target_position):
    relative_position = target_position - camera_position
    yaw = np.degrees(np.arctan2(relative_position[0], relative_position[2]))
    pitch = np.degrees(np.arctan2(relative_position[1], np.sqrt(relative_position[0]**2 + relative_position[2]**2)))
    return yaw, pitch


running = True
clock = pygame.time.Clock()

sim_steps = 0
cycles = 0
last_time_check = time.time()
recheck_every_n_cycles = 1000
render = True

while running:
    solar_system = step_simulation(solar_system)

    if render:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    pygame.event.set_grab(False)
                    pygame.mouse.set_visible(True)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                random_index = np.random.randint(0, solar_system.bodies.position.shape[1])
                target_position = solar_system.bodies.position[0, random_index]
                yaw, pitch = aim_at_body(camera_position, target_position)


        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        yaw -= mouse_dx * MOUSE_SENSITIVITY
        pitch += mouse_dy * MOUSE_SENSITIVITY
        pitch = max(-89.0, min(89.0, pitch))

        keys = pygame.key.get_pressed()


        movement = np.array([0.0, 0.0, 0.0]) 
        forward = get_forward_vector_xz(yaw)
        right = get_right_vector_xz(yaw)
        up = np.array([0, 1, 0])

        if keys[pygame.K_w]:
            movement += forward * MOVEMENT_SPEED
        if keys[pygame.K_s]:
            movement -= forward * MOVEMENT_SPEED
        if keys[pygame.K_a]:
            movement -= right * MOVEMENT_SPEED
        if keys[pygame.K_d]:
            movement += right * MOVEMENT_SPEED
        if keys[pygame.K_SPACE]:
            movement += up * MOVEMENT_SPEED
        if keys[pygame.K_LSHIFT]:
            movement -= up * MOVEMENT_SPEED

        camera_position += movement

        screen.fill((0, 0, 0))

        for i in range(solar_system.bodies.position.shape[1]):
            position = solar_system.bodies.position[0, i]
            radius = float(solar_system.bodies.radius[0, i]) * HEIGHT
            screen_coords = to_screen_coords(position)

            if screen_coords is not None:
                if i == 0:
                    color = PLANET_COLOR
                    min_radius = MIN_PLANET_RADIUS
                else:
                    color = SUN_COLOR
                    min_radius = MIN_SUN_RADIUS
                pygame.draw.circle(screen, color, screen_coords, max(min_radius, int(radius)))

        draw_grid_skybox()

        pygame.display.flip()
        clock.tick(FPS)

    sim_steps += 1 * simulations
    cycles += 1
    if cycles == recheck_every_n_cycles:
        duration = time.time() - last_time_check
        sim_steps_per_sec = sim_steps / duration
        last_time_check = time.time()
        print(f"{sim_steps_per_sec=}")
        sim_steps = 0
        cycles = 0

pygame.quit()
