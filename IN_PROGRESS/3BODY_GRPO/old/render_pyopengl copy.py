import pygame
import numpy as np
import math
import time
from OpenGL.GL import *
from OpenGL.GLU import *
import jax.random as jrand
from PIL import Image
from environment import init_solarsystems, step_simulation, true_simulation_size, simulation_size, SolarSystem
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
SKYBOX = "galaxy2.png"

# ---------------------------------------------------------------------
# Initialize Pygame Display with an OpenGL context
# ---------------------------------------------------------------------
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


# We disable depth test for your 2D draws
glDisable(GL_DEPTH_TEST)
# Clear once just to be safe
glClearColor(0.0, 0.0, 0.0, 1.0)
glClear(GL_COLOR_BUFFER_BIT)

key = jrand.PRNGKey(int(10000 * time.time()))
simulations = 1  # Run n simulations simultaneously
solar_system = init_solarsystems(key, simulations)

# Camera settings
camera_position = np.array([8.0, 2.0, -6.0])  # Camera position
yaw, pitch = 49.3, 27  # Camera angles (degrees)
focal_length = 3.0  # Perspective strength

# ---------------------------------------------------------
# Skybox Setup
# ---------------------------------------------------------
skybox_texture_id = None
quadric_sphere = None

def init_skybox():
    global skybox_texture_id, quadric_sphere

    # Load the image using PIL
    image = Image.open(f"3BODY/res/{SKYBOX}")
    image = image.transpose(Image.FLIP_TOP_BOTTOM)  # OpenGL typically expects bottom-to-top
    img_data = image.convert("RGB").tobytes()

    # Generate a new texture ID
    skybox_texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, skybox_texture_id)

    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    # Build the texture
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB,
        image.width, image.height,
        0, GL_RGB, GL_UNSIGNED_BYTE,
        img_data
    )

    glBindTexture(GL_TEXTURE_2D, 0)

    # Create a quadric
    quadric_sphere = gluNewQuadric()
    gluQuadricTexture(quadric_sphere, GL_TRUE)
    # This makes the texture inside-facing
    gluQuadricOrientation(quadric_sphere, GLU_INSIDE)

def draw_skybox_sphere(yaw, pitch):
    """
    Temporarily switch to a real 3D perspective and draw a sphere around the camera,
    but only applying rotation (no translation). That way the skybox appears stationary
    as the camera rotates.
    """

    # Save current projection & modelview
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()

    # Enable depth test just for the skybox so it draws behind everything
    glEnable(GL_DEPTH_TEST)

    # -- Switch to a 3D perspective
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # A 90-degree FOV, typical near/far planes
    gluPerspective(90.0, float(WIDTH) / float(HEIGHT), 0.1, 5000.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # We apply the opposite of the camera's rotation:
    # (camera has yaw/pitch, so we rotate -pitch, -yaw to keep the skybox aligned)
    glRotatef(pitch, 1.0, 0.0, 0.0)
    glRotatef(-yaw,   0.0, 1.0, 0.0)

    # -- Draw the sphere with the skybox texture
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, skybox_texture_id)

    radius = 3000.0
    gluSphere(quadric_sphere, radius, 64, 64)

    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)

    # Disable depth test again so your 2D draws work as before
    glDisable(GL_DEPTH_TEST)

    # -- Restore your original orthographic 2D setup
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

# Initialize the skybox
init_skybox()

# ---------------------------------------------------------
# Helper functions to replace Pygame drawing with OpenGL
# ---------------------------------------------------------
def draw_line(color, start, end):
    if start is None or end is None:
        return
    r, g, b = [c / 255.0 for c in color]
    glColor3f(r, g, b)
    glBegin(GL_LINES)
    glVertex2f(float(start[0]), float(start[1]))
    glVertex2f(float(end[0]),   float(end[1]))
    glEnd()

def draw_circle(color, center, radius):
    if center is None:
        return
    r, g, b = [c / 255.0 for c in color]
    cx, cy = center
    glColor3f(r, g, b)
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(cx, cy)   # Center of circle
    steps = 36  # Smoothness
    for i in range(steps+1):
        theta = 2.0 * math.pi * (float(i) / steps)
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        glVertex2f(x, y)
    glEnd()

def draw_circle_with_shading(color, center, radius, light_dir=(0, 0, 1)):
    """
    Draw a circle with shading based on a light direction.
    
    Args:
        color: Base RGB color of the circle.
        center: (cx, cy) Screen-space coordinates for the center.
        radius: Radius of the circle.
        light_dir: Direction of the light source as a 3D vector.
    """
    if center is None or radius <= 0:
        return
    
    r, g, b = [c / 255.0 for c in color]
    cx, cy = center
    steps = 100  # Number of segments for the circle
    
    # Normalize the light direction
    light_dir = np.array(light_dir)
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    glBegin(GL_TRIANGLE_FAN)
    glColor3f(r, g, b)  # Base color at the center
    glVertex2f(cx, cy)  # Center of the circle
    
    for i in range(steps + 1):
        # Angle for this segment
        theta = 2.0 * math.pi * (float(i) / steps)
        x = math.cos(theta)
        y = math.sin(theta)
        
        # Normal at this vertex
        normal = np.array([x, y, 0])
        normal = normal / np.linalg.norm(normal)
        
        # Shading intensity based on the dot product
        intensity = max(0.0, np.dot(normal, light_dir))
        shaded_color = (r * intensity, g * intensity, b * intensity)
        
        # Vertex position
        vx = cx + radius * x
        vy = cy + radius * y
        glColor3f(*shaded_color)
        glVertex2f(vx, vy)
    
    glEnd()


# ---------------------------------------------------------
# Remainder of your original helper functions
# ---------------------------------------------------------
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

    return (screen_x, screen_y)

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
                draw_line(GRID_COLOR, start, end_x)
            if start and end_y:
                draw_line(GRID_COLOR, start, end_y)

            start = to_screen_coords([x, y, 0] + offset)
            end_x = to_screen_coords([x + grid_spacing, y, 0] + offset)
            end_y = to_screen_coords([x, y + grid_spacing, 0] + offset)
            if start and end_x:
                draw_line(GRID_COLOR, start, end_x)
            if start and end_y:
                draw_line(GRID_COLOR, start, end_y)

    for x in np.arange(0, grid_size, grid_spacing):
        for z in np.arange(0, grid_size, grid_spacing):
            start = to_screen_coords([x, 0, z] + offset)
            end_x = to_screen_coords([x + grid_spacing, 0, z] + offset)
            end_z = to_screen_coords([x, 0, z + grid_spacing] + offset)
            if start and end_x:
                draw_line(GRID_COLOR, start, end_x)
            if start and end_z:
                draw_line(GRID_COLOR, start, end_z)

            start = to_screen_coords([x, grid_size, z] + offset)
            end_x = to_screen_coords([x + grid_spacing, grid_size, z] + offset)
            end_z = to_screen_coords([x, grid_size, z + grid_spacing] + offset)
            if start and end_x:
                draw_line(GRID_COLOR, start, end_x)
            if start and end_z:
                draw_line(GRID_COLOR, start, end_z)

    for y in np.arange(0, grid_size, grid_spacing):
        for z in np.arange(0, grid_size, grid_spacing):
            start = to_screen_coords([0, y, z] + offset)
            end_y = to_screen_coords([0, y + grid_spacing, z] + offset)
            end_z = to_screen_coords([0, y, z + grid_spacing] + offset)
            if start and end_y:
                draw_line(GRID_COLOR, start, end_y)
            if start and end_z:
                draw_line(GRID_COLOR, start, end_z)

            start = to_screen_coords([grid_size, y, z] + offset)
            end_y = to_screen_coords([grid_size, y + grid_spacing, z] + offset)
            end_z = to_screen_coords([grid_size, y, z + grid_spacing] + offset)
            if start and end_y:
                draw_line(GRID_COLOR, start, end_y)
            if start and end_z:
                draw_line(GRID_COLOR, start, end_z)

def aim_at_body(camera_position, target_position):
    relative_position = target_position - camera_position
    yaw = np.degrees(np.arctan2(relative_position[0], relative_position[2]))
    pitch = np.degrees(
        np.arctan2(
            relative_position[1],
            np.sqrt(relative_position[0]**2 + relative_position[2]**2)
        )
    )
    return (yaw, pitch)

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
                random_index = np.random.randint(0, solar_system.bodies.position.shape[1])
                target_position = solar_system.bodies.position[0, random_index]
                yaw, pitch = aim_at_body(camera_position, target_position)

        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        yaw -= mouse_dx * MOUSE_SENSITIVITY
        pitch += mouse_dy * MOUSE_SENSITIVITY
        pitch = max(-89.0, min(89.0, pitch))

        print(yaw, pitch)

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

        camera_position += movement

        # --------------------------------
        # Clear screen
        # --------------------------------
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # --------------------------------
        # 1) Draw the Skybox Sphere (with camera rotation)
        # --------------------------------
        # Pass in current yaw and pitch
        draw_skybox_sphere(yaw, pitch)

        # --------------------------------
        # 2) Draw the bodies in 2D
        # --------------------------------
        for i in range(solar_system.bodies.position.shape[1]):
            position = solar_system.bodies.position[0, i]
            radius   = float(solar_system.bodies.radius[0, i]) * HEIGHT
            screen_coords = to_screen_coords(position)

            if i == 0:
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
        draw_grid_skybox()

        # --------------------------------
        # Swap buffers, tick FPS
        # --------------------------------
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
