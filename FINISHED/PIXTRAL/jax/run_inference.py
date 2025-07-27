import jax
import jax.numpy as jnp
import jax.random as jrand


## load params
from load_model import load_params
from forward import encode, decode


paths = ['../pixtral/consolidated.safetensors']
pixtral_params = load_params(paths, dummy=False)

debug = False
jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_default_dtype_bits", "32")
if debug:
    jax.config.update("jax_disable_jit", True)


## set up inference prompt
seed = (0 - 0)-7
key = jrand.PRNGKey(seed)
prompt = "What is in the image?" 
url_1 = "../images/bed.jpg"
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
]
temp = 0.0
generate_tokens = 32

## do inference
from forward import inference, tokenize_messages_dict

prompt_tokens, images, image_start_indices = tokenize_messages_dict(messages)
print(len(prompt_tokens), "input prompt text token len")

completion = ""
tokens = prompt_tokens

# pad tokens to make shape constant
max_tokens = len(tokens) + generate_tokens

import time

for i in range(generate_tokens):
    if i == 0:
        start = time.time()
    # pad tokens to constant size
    context_tokens = len(tokens)
    padding_token_count = max_tokens - context_tokens
    padding_token_id = 11 # https://github.com/mistralai/mistral-common/issues/105#issuecomment-2997200779
    padding_tokens = jnp.array([padding_token_id for _ in range(padding_token_count)]) # single batch for now
    padded_tokens = jnp.concatenate([padding_tokens, jnp.array(tokens)])

    # do inference
    shifted_sequence = inference(key, pixtral_params, padded_tokens, images, image_start_indices)
    next_token = shifted_sequence[0, -1]
    if i == 0:
        print("Human:\n",prompt,"\n\nAssistant:")
    next_token_chars = decode([next_token])
    print(next_token_chars, end="", flush=True)
    completion += next_token_chars
    tokens.append(int(next_token))

## print result
print("\n\n\ncompletion: ", completion)
print("tok/sec", (generate_tokens - 1) / (time.time() - start))


