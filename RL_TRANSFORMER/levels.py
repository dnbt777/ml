from typing import List, NamedTuple
import random


class Level(NamedTuple):
  terrain : List[List[str]]
  spawn1 : List[int]



def make_default_level(grid_rows, grid_cols):
  # fill in with wood
  terrain = [["wood" for _ in range(grid_cols)] for _ in range(grid_rows)]

  # add stone to all the walls
  terrain[0] = ["stone" for _ in range(grid_cols)]
  terrain[-1] = ["stone" for _ in range(grid_cols)]
  for grid_row in range(grid_rows):
     terrain[grid_row][0] = "stone"
     terrain[grid_row][-1] = "stone"


  # add stone in the middle
  # rule: stone if row % 2 == 0 and col % 2 == 0
  for col in range(grid_cols):
     for row in range(grid_rows):
        if row % 2 == 0 and col % 2 == 0:
           terrain[row][col] = "stone"


  # empty the spawn points (corners)
  spawn1 = (1, 1)
  spawn2 = (-2, -2)
  spawn3 = (1, -2)
  spawn4 = (-2, 1)

  for x, y in [spawn1, spawn2, spawn3, spawn4]:
     terrain[y][x] = "empty"
     terrain[y + int(y/abs(y))][x] = "empty"
     terrain[y][x + int(x/abs(x))] = "empty"


  return Level(
     terrain=terrain,
     spawn1=spawn1
  )
  



def make_random_level(grid_rows, grid_cols):
  terrain = [["empty" for _ in range(grid_cols)] for _ in range(grid_rows)]

    # Add some wood and stone blocks
  for row in range(grid_rows):
      for col in range(grid_cols):
          if random.random() < 0.2:
              terrain[row][col] = "wood"
          elif random.random() < 0.1:
              terrain[row][col] = "stone"
  
  # Spawn panda at an empty cell
  while True:
    spawn_x, spawn_y = random.randint(0, grid_cols - 1), random.randint(0, grid_rows - 1)
    if terrain[spawn_y][spawn_x] == "empty":
      break
  
  return Level(
     terrain=terrain,
     spawn1=[spawn_x, spawn_y]
  )
