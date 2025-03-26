from vision_forward import image_to_patches
from PIL import Image
import jax
import jax.numpy as jnp
import numpy as np

imgpath = "./image.png"
img = Image.open(imgpath)
patches = image_to_patches(img, (224, 224), (32, 32))

print(patches.shape)

img.resize((224, 224)).show()
import time
for i in range(patches.shape[0]):
  for j in range(patches.shape[1]):
    tempimg = Image.fromarray(np.array(patches[j, i]))
    print(i, j, tempimg.size)
    tempimg.show()
    time.sleep(3)