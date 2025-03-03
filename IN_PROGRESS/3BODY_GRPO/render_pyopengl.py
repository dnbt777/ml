import numpy as np
import time
import jax.random as jrand
import jax
import os

from environment import init_solarsystems, step_simulation, downscaled_simulation_size, get_reward, get_state_summary
from GRPO import *
from file_utils import load_model_params

# most important params
render = True
debug = False
load_params_from_file = True # else, generate a random model. useful for seeing growth
train_run = -1

# other
WIDTH, HEIGHT = 800, 800
SCALE = downscaled_simulation_size
PLANET_COLOR = (200, 150, 0)
SUN_COLOR = (255, 255, 0)
GRID_COLOR = (200, 200, 200) 
FPS = 60
MOUSE_SENSITIVITY = 0.1
MOVEMENT_SPEED = 0.1
GRID_LINES = 1  # Number of grid lines on each axis
MIN_SUN_RADIUS = 0.05
MIN_PLANET_RADIUS = 0.02
SKYBOX = "galaxy2.png"  # This is the relative path to your skybox image
suns = 3
planets = 1
focal_length = 3.0
simulations = 2 #1_000_000  # Run n simulations simultaneously
steps_per_simulation = 2_000
#model
hidden_size = 16
hidden_layers = 10
input_datapoints = 3*4 + 3*4 + 1*4
output_actions = 7 # lr/ud/bf/nothing
project_dir = "IN_PROGRESS/3BODY_GRPO"
train_run_folder = f"{project_dir}/train_runs"

# setup params that require conditional logic
if render:
    steps_per_simulation = 1_000_000
    simulations = 1 # can be more, for testing, if need be
    TRAIL_LENGTH = 100 
    TRAIL_FADE = 0.01
    lock_mouse = True
if debug:
    jax.config.update("jax_disable_jit", True)
    lock_mouse = False
if train_run == -1:
    # load the last train run
    train_run = len(os.listdir(train_run_folder))
params_filename = f"{train_run_folder}/{train_run}/params.pkl"

## Initialize model and solar systems
key = jrand.PRNGKey(int(10000 * time.time()))
if load_params_from_file:
    policy_model_params = load_model_params(params_filename)
else:
    policy_model_params = init_policy_model(key, hidden_layers, hidden_size, input_datapoints, output_actions)
key, _ = jrand.split(key, 2)
solar_system = init_solarsystems(key, simulations, planets, suns)


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
    if lock_mouse:
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

    font = pygame.font.SysFont("Arial", 18)  # Choose a font and size
    text_color = (255, 255, 255)

    # Set up an orthographic projection (2D) that spans [0..WIDTH] x [0..HEIGHT].
    glViewport(0, 0, WIDTH, HEIGHT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(90.0, float(WIDTH) / float(HEIGHT), 0.1, 10000.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glScalef(-1, -1, 1)
    glRotatef(180, 0, 1, 0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_DEPTH_TEST)  # Enable depth testing
    glDepthFunc(GL_LESS)     # Accept fragment if it is closer
    glDepthMask(GL_TRUE) 


    # We disable depth test for 2D draws
    #glDisable(GL_DEPTH_TEST)
    # Clear once
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # ---------------------------------------------------------------------
    # Camera settings
    # ---------------------------------------------------------------------
    camera_position = np.array([1.5, 1.5, -1.5])  # Camera position
    yaw, pitch = 49.3, 27  # Camera angles (degrees)

    # ---------------------------------------------------------------------
    # Skybox Setup
    # ---------------------------------------------------------------------
    skybox_texture_id, quadric_sphere = init_skybox(f"{project_dir}/res/{SKYBOX}")

    # Initialize position history for fading trails
    position_history = np.zeros((TRAIL_LENGTH, solar_system.bodies.position.shape[1], 3))

    camera_target_index = -1
    MIN_DISTANCE = 0.01  # Minimum zoom level
    MAX_DISTANCE = 100.0  # Maximum zoom level
    camera_distance = 1.5

    show_debug_menu = True

    # Main loop
    clock = pygame.time.Clock()


sim_steps = 0
cycles = 0
last_time_check = time.time()
recheck_every_n_cycles = 1000
running = True

re_init = False
while running:
    # Update simulation
    key = jrand.PRNGKey(int(time.time()*10))
    action = take_action(key, policy_model_params, solar_system)
    key, _ = jrand.split(key, 2)
    solar_system = step_simulation(solar_system, action)
    reward = jax.vmap(get_reward, in_axes=0)(solar_system)
    debug_data = get_state_summary(solar_system)
    model_vision = safe_concat_current_state(solar_system) # see what the model sees

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
                if event.key == pygame.K_r:
                    re_init = True
                if event.key == pygame.K_SPACE:
                    camera_target_index = -1
                if event.key == pygame.K_F3:
                    show_debug_menu = not show_debug_menu # toggle debug menu MC style
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    # Cycle through bodies and the origin
                    camera_target_index = (camera_target_index + 1) % (solar_system.bodies.position.shape[1] + 1)
                if event.button == 3:  # Right mouse button
                    # Cycle through bodies and the origin
                    camera_target_index = (camera_target_index - 1) % (solar_system.bodies.position.shape[1] + 1)
                if event.button == 4:  # Scroll up
                    camera_distance = max(MIN_DISTANCE, camera_distance - 0.05)  # Zoom in
                elif event.button == 5:  # Scroll down
                    camera_distance = min(MAX_DISTANCE, camera_distance + 0.05)  # Zoom out

        if camera_target_index == -1:
            # Camera follows the origin
            target_position = np.array([0.0, 0.0, 0.0])
        else:
            # Camera follows a specific body
            target_position = solar_system.bodies.position[0, camera_target_index]

        # Compute the direction vector from the camera to the target
        direction_vector = camera_position - target_position

        # Normalize the direction vector
        direction_magnitude = np.linalg.norm(direction_vector)
        if direction_magnitude > 0:
            direction_vector /= direction_magnitude

        # Position the camera at the desired distance from the target
        camera_position = target_position + direction_vector * camera_distance
        # Compute the desired camera position
        desired_camera_position = target_position + direction_vector * camera_distance

        # Smoothly interpolate to the desired position
        camera_position += (desired_camera_position - camera_position) * 0.1  # Adjust smoothing factor as needed

        # Mouse look
        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        yaw -= mouse_dx * MOUSE_SENSITIVITY
        pitch += mouse_dy * MOUSE_SENSITIVITY
        pitch = max(-89.0, min(89.0, pitch))

        # --------------------------------
        # Clear screen
        # --------------------------------
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)

        camera_position = np.array([
            target_position[0] + camera_distance * np.cos(pitch_rad) * np.sin(yaw_rad),  # X
            target_position[1] + camera_distance * np.sin(pitch_rad),                   # Y
            target_position[2] + camera_distance * np.cos(pitch_rad) * np.cos(yaw_rad)  # Z
        ])


        glLoadIdentity()
        gluLookAt(
            camera_position[0], camera_position[1], camera_position[2],  # Camera position
            target_position[0], target_position[1], target_position[2],  # Target to look at
            0, 1, 0  # Up vector
        )

        # --------------------------------
        # 1) Draw the Skybox Sphere
        # --------------------------------
        draw_skybox_sphere(skybox_texture_id, quadric_sphere, yaw, pitch, WIDTH, HEIGHT)

        # --------------------------------
        # 2) Draw the bodies in 2D
        # --------------------------------
        for i in range(solar_system.bodies.position.shape[1]):
            position = solar_system.bodies.position[0, i]
            if i < planets:
                radius = max(MIN_PLANET_RADIUS,float(solar_system.bodies.radius[0, i]))
                color = PLANET_COLOR
                wireframe_color = PLANET_COLOR
            else:
                radius = max(MIN_SUN_RADIUS,float(solar_system.bodies.radius[0, i]))
                color = SUN_COLOR
                wireframe_color = (255, 220, 0)
            if i == camera_target_index:
                color = (255, 255, 255)
                wireframe_color = (150, 150, 255)

            draw_sphere_3d_with_wireframe(position, radius, color, wireframe_color)

        # draw trails
        # Update position history
        position_history = np.roll(position_history, shift=1, axis=0)
        position_history[0] = solar_system.bodies.position[0]

        # Render fading trail lines
        for i in range(solar_system.bodies.position.shape[1]):
            if i < planets:
                trail_color = PLANET_COLOR
            else:
                trail_color = SUN_COLOR
            if i == camera_target_index:
                trail_color = (50, 50, 255)
            draw_trail_lines(position_history, i, trail_color, TRAIL_LENGTH, TRAIL_FADE)

        # --------------------------------
        # 3) Draw the grid lines
        # --------------------------------
        draw_cube_edges(GRID_COLOR, SCALE)

        # Create the text surface
        body_count = planets + suns
        if show_debug_menu:
            jnp.set_printoptions(precision=3)
            text_lines = [
                "",
                f"Reward: {reward}",
                f"Steps: {sim_steps}",
                f"FPS: {int(clock.get_fps())}",
                f"Action: {action}",
                f"Debug Data:{debug_data}",
                f"Model vision:",
                f"Planet:",
                f"Pos : {model_vision[0:3]}",
                f"Vel : {model_vision[3*body_count:3*body_count + 3]}",
                f"Mass: {model_vision[6*body_count:6*body_count + 1]}"
                f"",
                f"Suns:",
                f"Pos : {model_vision[3:3*body_count]}",
                f"Vel : {model_vision[3*body_count + 3:6*body_count]}",
                f"Mass: {model_vision[6*body_count + 1:]}"
            ]
            text_surfaces = [font.render(line, True, text_color) for line in text_lines]

            # Render each line at the top-left corner
            for i, text_surface in enumerate(text_surfaces):
                text_data = pygame.image.tostring(text_surface, "RGBA", True)
                glWindowPos2d(10, HEIGHT - 10 - i * 20)  # Position text (10px margin, 20px line height)
                glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)


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
        solar_system = init_solarsystems(key, simulations, planets, suns) # restart
        sim_steps = 0
        if render:
            position_history = np.zeros((TRAIL_LENGTH, solar_system.bodies.position.shape[1], 3))
        re_init = False

if render:
    pygame.quit()
