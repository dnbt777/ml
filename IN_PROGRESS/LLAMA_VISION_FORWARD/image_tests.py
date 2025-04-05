from vision_forward import image_to_tiles 
from PIL import Image
import numpy as np
from einops import rearrange

imgpath = "./image-1.png"
img = Image.open(imgpath)
tiles, aspect_ratio_id = image_to_tiles(img, (224, 224), (16, 16))
tiles = rearrange(tiles, "B T PH PW H W C -> (B T) (PH H) (PW W) C")


import time
for T in range(tiles.shape[0]):
  temparray = np.array(tiles[T]).astype(np.uint8)
  print(T, temparray.size, aspect_ratio_id)
  tempimg = Image.fromarray(temparray)
  tempimg.show()
  time.sleep(3)