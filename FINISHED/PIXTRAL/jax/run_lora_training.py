import jax
import jax.numpy as jnp
import jax.random as jrand

## param loading
from load_model import load_params, fast_load_params

# preprocessing
from forward_common import encode, decode, tokenize_messages_dict_with_masks

# fine-tuning
from forward_training import *

# logging
import time


## load data (for now just overfit on one prompt)
prompt = "Describe the image."
response = "nah lmao i aint doin that" # fine tune it to say this!
url_1 = "../images/chess.png"
x = [
  {
      "role":
      "user",
      "content": [
          {
              "type": "image_url",
              "image_url": {
                  "url": url_1
              }
          },
          {
              "type": "text",
              "text": prompt
          },
      ],
  },
]

y = x + [ {
  "role":
  "assistant",
  "content": [
      {
          "type": "text",
          "text": response
      },
  ],
},]
messages = y

## TODO: prompts = parse_dataset(file)
## not that hard, just store [messages]


## do training
seed = (0_0)-7
key = jrand.PRNGKey(seed)


# hyperparameters and other fine tuning advice
# https://mistral.ai/news/unlocking-potential-vision-language-models-satellite-imagery-fine-tuning


## encode dataset (right now, just one prompt to overfit on)
prompt_tokens, images, image_start_indices, context_mask, image_mask = tokenize_messages_dict_with_masks(messages)
#print(prompt_tokens) # am i training on an intermediate EOS?
prompt_tokens = jnp.array(prompt_tokens, dtype=int)
context_mask = jnp.array(context_mask, dtype=bool)[None, ..., None]
image_mask = jnp.array(image_mask, dtype=bool)[None, ..., None]
print(f"input tokens: {len(prompt_tokens)}")


## initialize lora
# dense
channel_dim = 5120 # from pixtral
vocab_size = 131072 # from pixtral
# attn lora
rank = 32 # intermediate lora size
layers = 40
k_proj_shape = (1024, 5120)
o_proj_shape = (5120, 4096)
q_proj_shape = (4096, 5120)
v_proj_shape = (1024, 5120)
lora_params = init_attn_lora(
    key,
    q_proj_shape[0],
    q_proj_shape[1],
    rank,
    k_proj_shape[0],
    k_proj_shape[1],
    rank,
    v_proj_shape[0],
    v_proj_shape[1],
    rank,
    o_proj_shape[0],
    o_proj_shape[1],
    rank,
    layers,
)


## load params
load_start = time.time()
print("loading params")
paths = ['../pixtral/consolidated.safetensors']
pixtral_params = fast_load_params(paths)
print(f"Loaded params in {time.time() - load_start:.2f}s")


## begin fine-tuning
for i in range(1000):
    # get loss and grads
    loss, grads = jax.value_and_grad(attn_lora_loss_fn)(lora_params, pixtral_params, prompt_tokens, images, image_start_indices, context_mask, key)
    
    print(jax.tree_util.tree_map(lambda g: jax.numpy.linalg.norm(g), grads))
    # update
    lr = 1e-3 * (0.5 + jnp.abs(jnp.cos(i/20)))
    print(f"it: {i} || loss: {loss:.2f} || lr: {lr:.7f}")
    lora_params = jax.tree_util.tree_map(lambda p, g: p - lr*g, lora_params, grads) # TODO implement adam, adamw, muon


## save lora for future use in inference/chat or in continued fine tuning
filepath = "loras/test.safetensors"
save_attn_lora(lora_params, filepath)
print(f"Saved lora to {filepath}")



