import numpy as np
import time
import jax.random as jrand

# Your existing imports for environment & rendering utilities
from environment import init_solarsystems, step_simulation, simulation_size, SolarSystem

# ---------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------
WIDTH, HEIGHT = 800, 600
SCALE = simulation_size
PLANET_COLOR = (200, 150, 0)
SUN_COLOR = (255, 255, 0)
GRID_COLOR = (200, 200, 200)  # Light grey grid color
FPS = 60
MOUSE_SENSITIVITY = 0.1
MOVEMENT_SPEED = 0.1
GRID_LINES = 1  # Number of grid lines on each axis
MIN_SUN_RADIUS = 14
MIN_PLANET_RADIUS = 6
SKYBOX = "galaxy2.png"  # This is the relative path to your skybox image
suns = 1
planets = 2 # code only supports one planet rn
focal_length = 3.0  # Perspective strength
render = True
simulations = 1_000_000  # Run n simulations simultaneously
steps_per_simulation = 2_000
if render:
    steps_per_simulation = 1_000_000
    simulations = 1
    trail_iterations = 100

# ---------------------------------------------------------------------
# Initialize Pygame Display with an OpenGL context
# ---------------------------------------------------------------------
if render:
    import pygame

    from OpenGL.GL import *
    from OpenGL.GLU import *
    # Import our refactored utility functions
    from render_utils import *
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Solar System Simulation")
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)

    # Set up an orthographic projection (2D) that spans [0..WIDTH] x [0..HEIGHT].
    glViewport(0, 0, WIDTH, HEIGHT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, WIDTH, 0, HEIGHT)  # Our manual projection (via to_screen_coords) maps to these coordinates
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glScalef(-1, -1, 1)
    glRotatef(180, 0, 1, 0)

    # We disable depth test for 2D draws
    #glDisable(GL_DEPTH_TEST)
    # Clear once
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)

    # ---------------------------------------------------------------------
    # Camera settings
    # ---------------------------------------------------------------------
    camera_position = np.array([8.0, 2.0, -6.0])  # Camera position
    yaw, pitch = 49.3, 27  # Camera angles (degrees)

    # ---------------------------------------------------------------------
    # Skybox Setup
    # ---------------------------------------------------------------------
    skybox_texture_id, quadric_sphere = init_skybox(f"3BODY/res/{SKYBOX}")

    # Main loop
    clock = pygame.time.Clock()


# ---------------------------------------------------------------------
# Initialize the environment
# ---------------------------------------------------------------------
key = jrand.PRNGKey(int(10000 * time.time()))
solar_system = init_solarsystems(key, simulations, suns, planets)


sim_steps = 0
cycles = 0
last_time_check = time.time()
recheck_every_n_cycles = 1000
running = True

re_init = False
while running:
    # Update simulation
    solar_system = step_simulation(solar_system)

    if render:
        # ----------------------------------
        # Handle events
        # ----------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    pygame.event.set_grab(False)
                    pygame.mouse.set_visible(True)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Aim camera at a random body
                random_index = np.random.randint(0, solar_system.bodies.position.shape[1])
                target_position = solar_system.bodies.position[0, random_index]
                yaw, pitch = aim_at_body(camera_position, target_position)

        # Mouse look
        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        yaw -= mouse_dx * MOUSE_SENSITIVITY
        pitch += mouse_dy * MOUSE_SENSITIVITY
        pitch = max(-89.0, min(89.0, pitch))

        # Keyboard movement
        keys = pygame.key.get_pressed()
        movement = np.array([0.0, 0.0, 0.0]) 
        forward = get_forward_vector_xz(yaw)
        right   = get_right_vector_xz(yaw)
        up      = np.array([0.0, 1.0, 0.0])

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
        if keys[pygame.K_r]:
            re_init = True

        camera_position += movement

        # --------------------------------
        # Clear screen
        # --------------------------------
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # --------------------------------
        # 1) Draw the Skybox Sphere
        # --------------------------------
        draw_skybox_sphere(skybox_texture_id, quadric_sphere, yaw, pitch, WIDTH, HEIGHT)

        # --------------------------------
        # 2) Draw the bodies in 2D
        # --------------------------------
        for i in range(solar_system.bodies.position.shape[1]):
            position = solar_system.bodies.position[0, i]
            radius = float(solar_system.bodies.radius[0, i]) * HEIGHT
            screen_coords = to_screen_coords(position, camera_position,
                                             yaw, pitch, focal_length,
                                             WIDTH, HEIGHT)

            if i < planets:
                color = PLANET_COLOR
                min_radius = MIN_PLANET_RADIUS
            else:
                color = SUN_COLOR
                min_radius = MIN_SUN_RADIUS

            if screen_coords is not None:
                draw_circle(color, screen_coords, max(min_radius, int(radius)))

        # --------------------------------
        # 3) Draw the grid lines
        # --------------------------------
        draw_grid_skybox(
            GRID_COLOR, GRID_LINES, simulation_size,
            camera_position, yaw, pitch, focal_length,
            WIDTH, HEIGHT
        )

        # --------------------------------
        # Swap buffers, tick FPS
        # --------------------------------
        pygame.display.flip()
        clock.tick(FPS)

    # Count simulation steps
    sim_steps += 1 * simulations
    cycles += 1
    if sim_steps % steps_per_simulation == 0:
        re_init = True
    if cycles == recheck_every_n_cycles:
        duration = time.time() - last_time_check
        sim_steps_per_sec = sim_steps / duration
        last_time_check = time.time()
        print(f"steps/s={int(sim_steps_per_sec):,}")
        sim_steps = 0
        cycles = 0
        #re_init = True
    if re_init:
        key = jrand.PRNGKey(int(10000 * time.time()))
        solar_system = init_solarsystems(key, simulations, suns, planets) # restart
        re_init = False

if render:
    pygame.quit()
