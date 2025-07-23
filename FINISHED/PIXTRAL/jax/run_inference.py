import jax
import jax.numpy as jnp
import jax.random as jrand


## load params
from load_model import load_params
from forward import encode, decode


paths = ['../pixtral/consolidated.safetensors']
pixtral_params = load_params(paths, dummy=False)

debug = True
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
max_tokens = 32

## do inference
from forward import inference, tokenize_messages_dict

prompt_tokens, images, image_start_indices = tokenize_messages_dict(messages)
print(len(prompt_tokens), "input prompt text token len")

completion = ""
tokens = prompt_tokens

for i in range(max_tokens):
    next_token = inference(key, pixtral_params, tokens, images, image_start_indices)
    if i == 0:
        print("Human:\n",prompt,"\n\nAssistant:")
    next_token_chars = decode(next_token)
    completion += next_token_chars
    tokens = prompt_tokens + encode(completion, add_special=False)
    print(next_token_chars, end="", flush=True)

## print result
print("\n\n\ncompletion: ", completion)


