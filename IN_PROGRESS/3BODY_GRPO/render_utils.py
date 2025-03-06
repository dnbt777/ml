import math
import numpy as np

# PyOpenGL
from OpenGL.GL import *
from OpenGL.GLU import *

# PIL for loading textures
from PIL import Image

## NOTE: Most of this is chatgpt generated. I usually use chatgpt for renderers when practicing ML stuff

def init_skybox(skybox_image_path):
    """
    Creates and returns the OpenGL texture ID and quadric object
    needed to draw the skybox sphere.
    """
    # Load the image using PIL
    image = Image.open(skybox_image_path)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip to match OpenGL's expected orientation
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
    # Inside-facing sphere so the texture is on the inside
    gluQuadricOrientation(quadric_sphere, GLU_INSIDE)

    return skybox_texture_id, quadric_sphere


def draw_skybox_sphere(skybox_texture_id, quadric_sphere, yaw, pitch, screen_width, screen_height):
    """
    Temporarily switch to a 3D perspective, draw a large sphere around the camera,
    then restore the original orthographic 2D projection.
    """
    # Save current projection & modelview
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()

    # Enable depth for the skybox so it draws behind everything else
    glEnable(GL_DEPTH_TEST)

    # Switch to 3D perspective
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # A 90-degree FOV, typical near/far planes
    gluPerspective(90.0, float(screen_width) / float(screen_height), 0.1, 5000.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Apply the opposite of the camera's rotation
    glRotatef(pitch, 1.0, 0.0, 0.0)
    glRotatef(-yaw,   0.0, 1.0, 0.0)

    # Draw the sphere with the skybox texture
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, skybox_texture_id)

    radius = 3000.0
    gluSphere(quadric_sphere, radius, 64, 64)

    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)

    # Disable depth again so our 2D draws (lines, circles) work as before
    glDisable(GL_DEPTH_TEST)

    # Restore the original orthographic 2D setup
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()


def draw_line(color, start, end):
    """
    Draw a 2D line in OpenGL from start -> end in screen coordinates.
    """
    if start is None or end is None:
        return
    r, g, b = [c / 255.0 for c in color]
    glColor3f(r, g, b)
    glBegin(GL_LINES)
    glVertex2f(float(start[0]), float(start[1]))
    glVertex2f(float(end[0]),   float(end[1]))
    glEnd()


def draw_circle(color, center, radius):
    """
    Draw a solid 2D circle in screen coordinates.
    """
    if center is None or radius <= 0:
        return
    r, g, b = [c / 255.0 for c in color]
    cx, cy = center
    glColor3f(r, g, b)
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(cx, cy)   # Center of the circle
    steps = 36  # Smoothness
    for i in range(steps + 1):
        theta = 2.0 * math.pi * (float(i) / steps)
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        glVertex2f(x, y)
    glEnd()


def draw_shaded_sphere(position, camera_position, yaw, pitch, focal_length, WIDTH, HEIGHT, color, radius):
    """
    Draw a fake 'sphere' with consistent shading depending on the angle of view.

    Args:
        position: (x, y, z) tuple of the sphere's position in 3D space.
        camera_position: (x, y, z) tuple of the camera's position in 3D space.
        yaw: Yaw angle of the camera in radians.
        pitch: Pitch angle of the camera in radians.
        focal_length: Focal length of the camera (e.g., in pixels).
        WIDTH: Width of the screen in pixels.
        HEIGHT: Height of the screen in pixels.
        color: (r, g, b) tuple representing the base color of the sphere (0-255).
        radius: Radius of the sphere in world units.
    """
    # Calculate the sphere's position relative to the camera
    dx, dy, dz = [position[i] - camera_position[i] for i in range(3)]

    # Apply yaw and pitch rotations to transform into camera space
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)

    # Yaw rotation (around the Y-axis)
    x_cam = cos_yaw * dx + sin_yaw * dz
    y_cam = dy
    z_cam = -sin_yaw * dx + cos_yaw * dz

    # Pitch rotation (around the X-axis)
    x_cam = x_cam
    y_cam = cos_pitch * y_cam - sin_pitch * z_cam
    z_cam = sin_pitch * y_cam + cos_pitch * z_cam

    # If the sphere is behind the camera, skip rendering
    if z_cam <= 0:
        return

    # Project the sphere's center into 2D screen space
    screen_x = int(WIDTH / 2 + (focal_length * x_cam / z_cam))
    screen_y = int(HEIGHT / 2 - (focal_length * y_cam / z_cam))

    # Convert the sphere's radius in world units to screen space
    sphere_radius_screen = int(focal_length * radius / z_cam)

    # Determine the light direction (e.g., light coming from the camera direction)
    light_direction = [0, 0, -1]

    # Normalize the light direction
    light_mag = math.sqrt(sum([l ** 2 for l in light_direction]))
    light_direction = [l / light_mag for l in light_direction]

    # Compute shading based on the angle between the light direction and the sphere's surface normals
    def compute_shading(normal):
        dot = sum([light_direction[i] * normal[i] for i in range(3)])
        return max(dot, 0.1)  # Ensure a minimum brightness for visibility

    # Draw the shaded sphere using concentric circles
    steps = 36
    for i in range(sphere_radius_screen, 0, -1):  # Draw from outermost to innermost
        # Calculate the normalized distance for shading
        normalized_distance = i / sphere_radius_screen

        # Calculate the shading factor
        normal = [0, 0, 1]  # Approximation of normal vector (assuming flat view)
        brightness = compute_shading(normal)

        # Adjust the color by brightness
        shaded_color = [int(c * brightness * normalized_distance) for c in color]

        # Draw the circle
        draw_circle(shaded_color, (screen_x, screen_y), i)

def draw_circle(color, center, radius, steps=72, gradient=False, gradient_color=None):
    """
    Draw a solid 2D circle in screen coordinates with optional gradient.

    Parameters:
    - color (tuple): The base RGB color of the circle as a tuple of three integers (0-255).
    - center (tuple): The (x, y) coordinates of the circle's center.
    - radius (float): The radius of the circle.
    - steps (int, optional): The smoothness of the circle (default is 72).
    - gradient (bool, optional): Enable gradient fill (default is False).
    - gradient_color (tuple, optional): Secondary RGB color for the gradient (required if gradient is True).

    Returns:
    None
    """
    if center is None or radius <= 0:
        return

    # Normalize base color to OpenGL's [0, 1] range
    r, g, b = [c / 255.0 for c in color]

    # Handle gradient
    if gradient and gradient_color:
        gr, gg, gb = [c / 255.0 for c in gradient_color]
    else:
        gradient = False  # Disable gradient if no secondary color provided

    cx, cy = center
    glBegin(GL_TRIANGLE_FAN)

    # Center point of the circle
    if gradient:
        glColor3f(gr, gg, gb)
    else:
        glColor3f(r, g, b)
    glVertex2f(cx, cy)

    # Outer points of the circle
    for i in range(steps + 1):
        theta = 2.0 * math.pi * (float(i) / steps)
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)

        # Blend colors if gradient is enabled
        if gradient:
            t = i / steps  # Interpolation factor
            blended_r = (1 - t) * r + t * gr
            blended_g = (1 - t) * g + t * gg
            blended_b = (1 - t) * b + t * gb
            glColor3f(blended_r, blended_g, blended_b)
        else:
            glColor3f(r, g, b)

        glVertex2f(x, y)

    glEnd()



def draw_circle_with_shading(color, center, radius, light_dir=(0, 0, 1)):
    """
    Draw a circle with simple Lambertian shading based on a light direction.
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
    glVertex2f(cx, cy)  # Center
    
    for i in range(steps + 1):
        theta = 2.0 * math.pi * (float(i) / steps)
        x = math.cos(theta)
        y = math.sin(theta)
        
        # Normal at this vertex
        normal = np.array([x, y, 0])
        normal = normal / np.linalg.norm(normal)
        
        # Shading intensity
        intensity = max(0.0, np.dot(normal, light_dir))
        shaded_color = (r * intensity, g * intensity, b * intensity)
        
        # Vertex position
        vx = cx + radius * x
        vy = cy + radius * y
        glColor3f(*shaded_color)
        glVertex2f(vx, vy)
    
    glEnd()


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


def to_screen_coords(position, camera_position, yaw, pitch, focal_length, screen_width, screen_height):
    """
    Convert a 3D world-space position to 2D screen coordinates,
    based on the camera position, orientation, and focal length.
    Returns None if the point is behind the camera (z2 <= 0).
    """
    relative_position = position - camera_position

    yaw_rad = np.radians(-yaw)
    pitch_rad = np.radians(-pitch)
    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
    cos_pitch, sin_pitch = np.cos(pitch_rad), np.sin(pitch_rad)

    x1 = cos_yaw * relative_position[0] - sin_yaw * relative_position[2]
    z1 = sin_yaw * relative_position[0] + cos_yaw * relative_position[2]

    y2 = cos_pitch * relative_position[1] - sin_pitch * z1
    z2 = sin_pitch * relative_position[1] + cos_pitch * z1

    # If behind camera, don't draw
    if z2 <= 0:
        return None

    x_proj = (focal_length / z2) * x1
    y_proj = (focal_length / z2) * y2

    screen_x = int((x_proj + 0.5) * screen_width)
    screen_y = int((1 - (y_proj + 0.5)) * screen_height)

    return (screen_x, screen_y)


def draw_grid_skybox(grid_color, grid_lines, simulation_size, camera_position,
                     yaw, pitch, focal_length, screen_width, screen_height):
    """
    Draw a simple set of "cube" edges by sampling a 3D grid from [0..simulation_size].
    """
    grid_spacing = simulation_size / grid_lines
    grid_size = simulation_size
    offset = np.array([0, 0, 0])

    # Helper to reduce repetition
    def sc(pos):
        return to_screen_coords(
            np.array(pos) + offset,
            camera_position,
            yaw, pitch,
            focal_length,
            screen_width, screen_height
        )

    # X-Y planes at z=0 and z=grid_size
    for x in np.arange(0, grid_size, grid_spacing):
        for y in np.arange(0, grid_size, grid_spacing):
            start = sc([x, y, grid_size])
            end_x = sc([x + grid_spacing, y, grid_size])
            end_y = sc([x, y + grid_spacing, grid_size])
            if start and end_x:
                draw_line(grid_color, start, end_x)
            if start and end_y:
                draw_line(grid_color, start, end_y)

            start = sc([x, y, 0])
            end_x = sc([x + grid_spacing, y, 0])
            end_y = sc([x, y + grid_spacing, 0])
            if start and end_x:
                draw_line(grid_color, start, end_x)
            if start and end_y:
                draw_line(grid_color, start, end_y)

    # X-Z planes at y=0 and y=grid_size
    for x in np.arange(0, grid_size, grid_spacing):
        for z in np.arange(0, grid_size, grid_spacing):
            start = sc([x, 0, z])
            end_x = sc([x + grid_spacing, 0, z])
            end_z = sc([x, 0, z + grid_spacing])
            if start and end_x:
                draw_line(grid_color, start, end_x)
            if start and end_z:
                draw_line(grid_color, start, end_z)

            start = sc([x, grid_size, z])
            end_x = sc([x + grid_spacing, grid_size, z])
            end_z = sc([x, grid_size, z + grid_spacing])
            if start and end_x:
                draw_line(grid_color, start, end_x)
            if start and end_z:
                draw_line(grid_color, start, end_z)

    # Y-Z planes at x=0 and x=grid_size
    for y in np.arange(0, grid_size, grid_spacing):
        for z in np.arange(0, grid_size, grid_spacing):
            start = sc([0, y, z])
            end_y = sc([0, y + grid_spacing, z])
            end_z = sc([0, y, z + grid_spacing])
            if start and end_y:
                draw_line(grid_color, start, end_y)
            if start and end_z:
                draw_line(grid_color, start, end_z)

            start = sc([grid_size, y, z])
            end_y = sc([grid_size, y + grid_spacing, z])
            end_z = sc([grid_size, y, z + grid_spacing])
            if start and end_y:
                draw_line(grid_color, start, end_y)
            if start and end_z:
                draw_line(grid_color, start, end_z)


def aim_at_body(camera_position, target_position):
    """
    Helper to compute yaw/pitch angles that make camera_position
    'look at' target_position.
    """
    relative_position = target_position - camera_position
    yaw = np.degrees(np.arctan2(relative_position[0], relative_position[2]))
    pitch = np.degrees(
        np.arctan2(
            relative_position[1],
            np.sqrt(relative_position[0]**2 + relative_position[2]**2)
        )
    )
    return (yaw, pitch)



def draw_trail_lines(position_history, body_index, base_color, trail_length, trail_fade):
    """
    Draw fading trail lines for a given body.

    Args:
        position_history: The history of positions for all bodies.
        body_index: Index of the body for which to draw the trail.
        base_color: RGB tuple for the trail's color.
        trail_length: Number of historical positions to render.
        trail_fade: Fade factor for each step in the trail.
    """
    r, g, b = [c / 255.0 for c in base_color]

    glBegin(GL_LINES)
    for t in range(trail_length - 1):
        # Current and next positions in the trail
        current_position = position_history[t, body_index]
        next_position = position_history[t + 1, body_index]

        # Skip if the positions are invalid
        if np.any(np.isnan(current_position)) or np.any(np.isnan(next_position)):
            continue

        # Calculate fading opacity
        opacity = max(0, 1.0 - t * trail_fade)

        # Set the color for the current segment
        glColor4f(r, g, b, opacity)
        glVertex3f(*current_position)
        glVertex3f(*next_position)
    glEnd()




def draw_cube_edges(edge_color, size, shift=(0.0, 0.0, 0.0)):
    """
    Draw a wireframe cube centered at the origin with given size and edge color,
    shifted by a specified translation vector.

    Args:
        edge_color (tuple): RGB color of the cube edges (0-255).
        size (float): Length of the cube's edges.
        shift (tuple): (x, y, z) shift applied to the cube's position.
    """
    half_size = size / 2.0
    shift_x, shift_y, shift_z = (half_size, half_size, half_size)

    # Define cube vertices with the shift applied
    vertices = [
        (-half_size + shift_x, -half_size + shift_y, -half_size + shift_z),
        ( half_size + shift_x, -half_size + shift_y, -half_size + shift_z),
        ( half_size + shift_x,  half_size + shift_y, -half_size + shift_z),
        (-half_size + shift_x,  half_size + shift_y, -half_size + shift_z),
        (-half_size + shift_x, -half_size + shift_y,  half_size + shift_z),
        ( half_size + shift_x, -half_size + shift_y,  half_size + shift_z),
        ( half_size + shift_x,  half_size + shift_y,  half_size + shift_z),
        (-half_size + shift_x,  half_size + shift_y,  half_size + shift_z)
    ]

    # Define cube edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Set the edge color
    r, g, b = [c / 255.0 for c in edge_color]
    glColor3f(r, g, b)

    # Draw the edges
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3f(*vertices[vertex])
    glEnd()



def draw_sphere_3d(position, radius, color):
    """
    Draws a 3D sphere at the specified position with optional transparency.
    """
    r, g, b = [c / 255.0 for c in color[:3]]
    alpha = color[3] / 255.0 if len(color) == 4 else 1.0  # Optional alpha
    glPushMatrix()
    glColor4f(r, g, b, alpha)
    glTranslatef(*position)
    quadric = gluNewQuadric()
    gluSphere(quadric, radius, 32, 32)
    gluDeleteQuadric(quadric)
    glPopMatrix()



def draw_sphere_3d_with_normals(position, radius, color, light_direction=(1, 1, 1)):
    """
    Draws a 3D sphere at the specified position with normal-based Lambertian shading.

    Args:
        position: (x, y, z) tuple of the sphere's center.
        radius: Radius of the sphere.
        color: Base RGB color of the sphere (0-255).
        light_direction: (x, y, z) direction of the light source.
    """
    r, g, b = [c / 255.0 for c in color[:3]]  # Normalize color to [0, 1]
    light_direction = np.array(light_direction)
    light_direction = light_direction / np.linalg.norm(light_direction)  # Normalize light direction

    glPushMatrix()
    glTranslatef(*position)

    # Subdivide the sphere into stacks and slices
    stacks = 8
    slices = 8

    for i in range(stacks):
        theta1 = math.pi * i / stacks
        theta2 = math.pi * (i + 1) / stacks

        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            phi = 2 * math.pi * j / slices

            # Compute two vertices on the sphere
            for theta in [theta1, theta2]:
                x = radius * math.sin(theta) * math.cos(phi)
                y = radius * math.cos(theta)
                z = radius * math.sin(theta) * math.sin(phi)

                # Calculate normal
                normal = np.array([x, y, z]) / radius  # Normalize the normal
                brightness = max(np.dot(normal, light_direction), 0.0)  # Lambertian reflection

                # Adjust color based on brightness
                shaded_color = (r * brightness, g * brightness, b * brightness)
                glColor3f(*shaded_color)

                # Specify the vertex
                glNormal3f(*normal)  # Set the normal for lighting calculations
                glVertex3f(x, y, z)
        glEnd()

    glPopMatrix()



def draw_sphere_3d_with_wireframe(position, radius, color, wireframe_color):
    """
    Draws a 3D sphere with a low-poly wireframe overlay.
    The wireframe overlay only appears on the front-facing surfaces.

    Args:
        position: (x, y, z) tuple of the sphere's center.
        radius: Radius of the sphere.
        color: Base RGB color of the sphere (0-255).
        wireframe_color: RGB color for the wireframe overlay (0-255).
    """
    # Render the solid sphere
    r, g, b = [c / 255.0 for c in color[:3]]
    glPushMatrix()
    glColor3f(r, g, b)
    glTranslatef(*position)
    quadric = gluNewQuadric()
    gluSphere(quadric, radius, 32, 32)  # Smooth sphere
    gluDeleteQuadric(quadric)
    glPopMatrix()

    # Render the wireframe overlay
    glLineWidth(2.0)
    wr, wg, wb = [c / 255.0 for c in wireframe_color[:3]]
    glPushMatrix()
    glColor3f(wr, wg, wb)
    glTranslatef(*position)

    # Enable face culling for the wireframe
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)  # Cull back-facing polygons

    quadric = gluNewQuadric()
    gluQuadricDrawStyle(quadric, GLU_LINE)  # Wireframe mode
    gluSphere(quadric, radius, 12, 12)  # Low-poly sphere
    gluDeleteQuadric(quadric)

    # Disable face culling after rendering
    glDisable(GL_CULL_FACE)

    glPopMatrix()

