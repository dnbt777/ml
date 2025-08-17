import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.profiler

## param loading
from load_model import load_params, fast_load_params

# fine-tuning
from forward_training import *

# logging
import time



completions = []

## load data (for now just overfit on one prompt)
"""
          {
              "type": "image_url",
              "image_url": {
                  "url": url_1
              }
          },
"""
prompt = "Say hi!"
response = "nah lmao i aint doin that" # fine tune it to say this!
url_1 = "../images/chess.png"
context = [
  {
      "role":
      "user",
      "content": [
          {
              "type": "text",
              "text": prompt
          },
      ],
  },
]

completion = context + [ {
  "role":
  "assistant",
  "content": [
      {
          "type": "text",
          "text": response
      },
  ],
},]

batch_size = 1
for _ in range(batch_size):
    completions.append(completion)

## TODO: prompts = parse_dataset(file)
## not that hard, just store [messages]


## do training
seed = (0_0)-7
key = jrand.PRNGKey(seed)


# hyperparameters and other fine tuning advice:
# https://mistral.ai/news/unlocking-potential-vision-language-models-satellite-imagery-fine-tuning


batch_completions = batch_parse_completions(completions)
print(batch_completions["tokens"])
print(batch_completions["context_mask"])
print(batch_completions["padding_mask"])

## initialize lora
# dense
dense_rank = 0 # disable
channel_dim = 5120 # from pixtral
vocab_size = 131072 # from pixtral
dense_in_dim = channel_dim
dense_out_dim = vocab_size
# attn lora
attn_rank = 1
layers = 40
k_proj_shape = (5120, 1024)
o_proj_shape = (4096, 5120)
q_proj_shape = (5120, 4096)
v_proj_shape = (5120, 1024)

lora_params = init_lora(
    key,
    dense_in_dim, dense_out_dim, dense_rank,
    q_proj_shape[0], q_proj_shape[1], attn_rank,
    k_proj_shape[0], k_proj_shape[1], attn_rank,
    v_proj_shape[0], v_proj_shape[1], attn_rank,
    o_proj_shape[0], o_proj_shape[1], attn_rank,
    layers,
)


## load params
load_start = time.time()
print("loading params")
paths = ['./pixtral/consolidated.safetensors']
pixtral_params = fast_load_params(paths)
print(f"Loaded params in {time.time() - load_start:.2f}s")


## begin fine-tuning
#batch_completions["context_mask"] = jnp.zeros_like(batch_completions["context_mask"], dtype=bool)

for i in range(1000):
    # get loss and grads
    loss, grads = jax.value_and_grad(text_lora_loss_fn, argnums=1)(
        pixtral_params,
        lora_params, # arg 1
        batch_completions["tokens"],
        batch_completions["context_mask"], batch_completions["padding_mask"],
        key
    )

    # print grads (debug, shows how params are learning)
    #print(jax.tree_util.tree_map(lambda g: jax.numpy.linalg.norm(g), grads))
    # update
    lr = 1e-3 * (0.5 + jnp.abs(jnp.cos(i/20))) * (0.995**i)
    print(f"it: {i} || loss: {loss:.5f} || lr: {lr:.7f}")
    lora_params = jax.tree_util.tree_map(lambda p, g: p - lr*g, lora_params, grads) # TODO implement adam, adamw, muon


## save lora for future use in inference/chat or in continued fine tuning
filepath = "loras/test.safetensors"
save_lora(lora_params, filepath)
print(f"Saved lora to {filepath}")

## test completion
completions = _get_completions(pixtral_params, [context], max_tokens=64, temp=0.0, lora_path="loras/test.safetensors")
print(completions)
