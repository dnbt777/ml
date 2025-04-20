import jax
import jax.numpy as jnp
import jax.random as jrand


## load params
from setup_utils import load_params, load_dummy_params

filepath = './pixtral/consolidated.safetensors'
pixtral_params = load_dummy_params(filepath)


## set up inference prompt
prompt = "Introduce yourself"
image = "image.png"
temp = 0
max_tokens = 64

## do inference
from inference import inference

context = prompt
tokens = tokenize(prompt)
for i in range(max_tokens - tokens):
    out = inference(pixtral_params, tokens, image, temp)
    context += decode(out)
    tokens = tokenize(context)

## print result
print("completion: ", context)


