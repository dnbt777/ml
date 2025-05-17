from tokenizer import load_tokenizer, encode, decode
import time

path = 'Llama/tokenizer.json'
tokenizer = load_tokenizer(path)
padding_token = 128004
bot_token = 128000
eot_token = 128001

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #



            ## # ## ## ## ##
            ## Inference ##
            ## ## ## ## ##
# ======================================== #
import jax.random as jrand
import jax.numpy as jnp
import jax
from inference import inference
from PIL import Image

rolling_key = jrand.PRNGKey(  (7_7)-7  )
prompt = "<|image|><|begin_of_text|> Describe the image."
prompt = "uhhhhhhhhhhhhhhhhhhhhhhhhhhh hello??? 🦙"
image_path = "./images/bed.jpg"
temp = 0.001 
print(f"Prompt: \"{prompt}\"")

DEBUG = False 
if DEBUG:
  jax.config.update("jax_disable_jit", True)

answer = ""
context_window_size = 32 # if this is not done, it will recompile repeatedly forever at each new context window size
# inference loop
start = time.time()
image = Image.open(image_path)
print("IMG SHAPE: ", jnp.array(image).shape)
for i in range(context_window_size - len(encode(tokenizer, prompt)) - 1):
  context = prompt + answer
  context_tokens = jax.lax.concatenate([jnp.array([bot_token]), jnp.array(encode(tokenizer, context))], 0)
  context_tokens = jnp.pad(context_tokens, (0, context_window_size - len(context_tokens)), constant_values=padding_token)
  print(context_tokens)
  print([decode(tokenizer, [token]) for token in context_tokens])
  print(decode(tokenizer, context_tokens))
  from vision_forward import image_to_tiles 
  # CLIP norm - norm across R, G, and B separately
  tile_resolution = (448, 448)
  pixel_values, aspect_ratio = image_to_tiles(image, tile_resolution)
  print("pixel_values: ",aspect_ratio, pixel_values)
  
  break
