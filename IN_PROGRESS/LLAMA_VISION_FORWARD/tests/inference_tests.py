from setup_utils import load_model_params, load_text_model_params
import time

import jax.random as jrand
import jax.numpy as jnp
import jax
from inference import inference
from llama_forward import *


DEBUG =False 
if DEBUG:
  jax.config.update("jax_disable_jit", True)


path = "./Llama"
paths = [
    f"./{path}/model-0000{n}-of-00005.safetensors"
    for n in range(1, 5+1)
]
# llama_params = load_as_jnp_dict(paths)
print("Initializing...")
start = time.time()
llama_params = load_text_model_params(paths)
print(f"Initialized in {time.time() - start:0.2f}s.")



from tokenizer import load_tokenizer, encode, decode

print("DECODER/ENCODER TEST")
prompt = "the dog went for a walk"
path = 'Llama/tokenizer.json'
tokenizer = load_tokenizer(path)
padding_token = 128004
bot_token = 128000
eot_token = 128001
print(prompt)
print(encode(tokenizer, prompt))
print(decode(tokenizer, encode(tokenizer, prompt)))









rolling_key = jrand.PRNGKey(int(7/(7_7)/7))
prompt = r"Hello from Earth"
temp = 0.001 
print(f"Prompt: \"{prompt}\"")

print("EMBEDDING/UNEMBEDDING TEST")

context_window_size = 16
context = prompt
context_tokens = jax.lax.concatenate([jnp.array([bot_token]), jnp.array(encode(tokenizer, context))], 0)
context_tokens = jnp.pad(context_tokens, (0, context_window_size - len(context_tokens)), constant_values=padding_token)
embedded = embed_tokens(llama_params.language_model, context_tokens)
unembedded_logits = embedded @ jnp.transpose(llama_params.language_model.lm_head_weight)
unembedded = jnp.argmax(unembedded_logits, axis=-1)
print(context_tokens)
print(embedded)
print(unembedded)

print("RMSNORM TEST")
x = jnp.arange(5)
print(x)
weight = jnp.ones_like(x)
print(RMSnorm(x, weight))


print("INFERENCE TEST")

from PIL import Image

image_path = "./image.png"
image = Image.open(image_path)
prompt = "<|image|><|begin_of_text|>What is 2 + 2?\nAnswer:"

answer = ""
context_window_size = 32 # if this is not done, it will recompile repeatedly forever at each new context window size
# inference loop
start = time.time()
for i in range(context_window_size - len(encode(tokenizer, prompt)) - 1):
  context = prompt + answer
  context_tokens = jax.lax.concatenate([jnp.array([bot_token]), jnp.array(encode(tokenizer, context))], 0)
  context_tokens = jnp.pad(context_tokens, (0, context_window_size - len(context_tokens)), constant_values=padding_token)
  next_token, predicted_tokens = inference(llama_params, image, context_tokens, temp, rolling_key)
  next_chunk = decode(tokenizer, jnp.array([next_token]))
  print("tokens:", predicted_tokens)
  print(f"predicted: {decode(tokenizer, predicted_tokens)}")
  answer += next_chunk 
  #print(str(next_chunk), end="", flush=True)
  rolling_key, _ = jrand.split(rolling_key, 2)
  #if next_token == "<end token>":
  #  break # TODO add actual end token to inference loop

print("\nFinal output:")
print("--------------")
print(context)
# ======================================== #


# future optimizations:
# OPTIMIZATION: Store jaxpr, somehow (compilation takes forever)
# TODO add a padding mask for each batch. currently a single padding mask is broadcast over all batches