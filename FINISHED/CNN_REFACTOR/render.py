import pygame
import numpy as np
import jax.numpy as jnp
from train_utils import cnn_forward

## This loads the model params and makes a canvas you can draw digits on
## I used chatGPT to make this
## Controls:
#   - rclick: draw
#   - p: predict the digit
#   - c: clear
#   - l: load a digit from the test set


def apply_brush(grid, x, y, intensity=40, radius=2):
    """
    Applies a Gaussian-like brush stroke to the grid.
    Args:
        grid: The drawing grid (28x28 numpy array).
        x, y: The coordinates of the center of the brush.
        intensity: The max intensity to add.
        radius: The size of the brush.
    """
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                distance = (dx ** 2 + dy ** 2) ** 0.5
                weight = max(0, 1 - distance / radius)  # Linear fall-off
                grid[ny, nx] = min(255, int(grid[ny, nx]) + int(intensity * weight))


def render(cnn_params, x_test):
  # Initialize Pygame
  pygame.init()
  # Grid settings
  grid_size = 28
  square_size = 20  # Each square is 20x20 pixels, making a 560x560 window
  screen_size = grid_size * square_size
  # Create the drawing grid (28x28)
  drawing_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
  # Initialize Pygame screen
  screen = pygame.display.set_mode((screen_size, screen_size))
  pygame.display.set_caption("Drawing")  # Default status
  # Colors
  black = (0, 0, 0)
  # Function to draw the grid on the screen
  def draw_grid():
      for x in range(grid_size):
          for y in range(grid_size):
              intensity = drawing_grid[y][x]
              color = (intensity, intensity, intensity)  # Grayscale color based on intensity
              pygame.draw.rect(
                  screen,
                  color,
                  (x * square_size, y * square_size, square_size, square_size)
              )
  # Main loop
  running = True
  drawing = False
  while running:
      for event in pygame.event.get():
          if event.type == pygame.QUIT:
              running = False
          elif event.type == pygame.MOUSEBUTTONDOWN:
              drawing = True
              pygame.display.set_caption("Drawing")  # Set title to indicate drawing
          elif event.type == pygame.MOUSEBUTTONUP:
              drawing = False
          elif event.type == pygame.KEYDOWN:
              if event.key == pygame.K_c:  # Clear the grid
                  drawing_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
                  screen.fill(black)
                  pygame.display.set_caption("Cleared")  # Update title
                  pygame.display.flip()
              elif event.key == pygame.K_p:  # Predict
                  # Pass the 28x28 numpy array to the CNN
                  img = drawing_grid
                  img_jax = jnp.expand_dims(jnp.expand_dims(jnp.array(img, dtype=jnp.float32), axis=0), axis=0)  # Shape: (1, 1, 28, 28)
                  prediction = cnn_forward(cnn_params, img_jax[0])  # Forward pass with your JAX CNN
                  predicted_digit = int(jnp.argmax(prediction))
                  pygame.display.set_caption(f"Predicted Digit: {predicted_digit}")  # Update title with prediction
              elif event.key == pygame.K_l:  # Load a jrand image from x_train
                  random_index = np.random.randint(0, len(x_test))
                  drawing_grid = (np.array(x_test[random_index][0]) * 255).astype(np.uint8)  # Ensure NumPy and scale to [0, 255]
                  screen.fill(black)
                  pygame.display.set_caption("Loaded Example")  # Update title
                  pygame.display.flip()
      # Handle drawing
      if drawing:
          mouse_x, mouse_y = pygame.mouse.get_pos()
          grid_x = mouse_x // square_size
          grid_y = mouse_y // square_size
          if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
              apply_brush(drawing_grid, grid_x, grid_y, intensity=80, radius=2)
      # Update the screen
      draw_grid()
      pygame.display.flip()
  pygame.quit()
