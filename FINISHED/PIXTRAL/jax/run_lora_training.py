import jax
import jax.numpy as jnp
import jax.random as jrand


## load params
from load_model import load_params
from forward_common import encode, decode, tokenize_messages_dict

# logging
import time

## load params
load_start = time.time()

paths = ['../pixtral/consolidated.safetensors']
pixtral_params = load_params(paths, dummy=False)

print(f"Loaded params in {time.time() - load_start:.2f}s")

## set up training prompt (for now just overfit on one prompt)
prompt = "Analyze the chess position in the image, from the perspective of a grandmaster, listing each impactful attribute of the position and its corresponding effect on net centipawn score. Then determine who is winning, and the net centipawn score."
response = "nah i aint doin that nerd shit" # fine tune it to say this!
url_1 = "../images/chess.png"
messages = [
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
  {
  "role":
  "assistant",
  "content": [
      {
          "type": "text",
          "text": response
      },
  ],
},
]

## TODO: prompts = parse_dataset(file)
## not that hard, just store [messages]

## do training
seed = (0_0)-7
key = jrand.PRNGKey(seed)

## encode dataset (right now, just one prompt to overfit on)
prompt_tokens, images, image_start_indices = tokenize_messages_dict(messages)
print(f"input tokens: {len(prompt_tokens)}")

## initialize lora
channel_dim = 1024
vocab_size = 128000 # placeholders
lora_size = 256
lora = init_lora(channel_dim, vocab_size, lora_size)

from forward_training import loss
for i in range(1000):
    # get loss and grads
    loss, grads = jax.value_and_grad(loss)(lora, pixtral_params, tokens, images, image_start_indices, y, key)
    
    # update
    lr = 1e-4
    lora = jax.tree_util.tree_map(lambda p, g: p - lr*g, lora, grads)

    print(loss)
    # repeat

# future: include the lora in inference



