import jax
import jax.numpy as jnp
import jax.random as jrand


## load params
from load_model import load_params
from forward import encode, decode



paths = ['./pixtral/consolidated.safetensors']
pixtral_params = load_params(paths, dummy=False)


## set up inference prompt
seed = (0 - 0)-7
key = jrand.PRNGKey(seed)
prompt = "What is the capitol of France?" 
url_1 = "./images/bed.jpg"
url_2 = "./images/bed.jpg"
url_3 = "./images/bed.jpg"
messages = [
  {
      "role":
      "user",
      "content": [
          {
              "type": "text",
              "text": prompt
          },
          {
              "type": "image_url",
              "image_url": {
                  "url": url_1
              }
          },
          {
              "type": "image_url",
              "image_url": {
                  "url": url_2
              }
          },
      ],
  },
  {
      "role": "assistant",
      "content": "The images show nature.",
  },
  {
      "role": "user",
      "content": "More details please and answer only in French!.",
  },
  {
      "role": "user",
      "content": [
          {
              "type": "image_url",
              "image_url": {
                  "url": url_3
              }
          },
      ],
  },
]
temp = 0
max_tokens = 64

## do inference
from forward import inference
context = prompt
tokens = encode(prompt)
for i in range(max_tokens - len(tokens)):
    print(i, "start")
    out = inference(key, pixtral_params, messages)
    context += decode(out)
    tokens = encode(context)
    print(i)

## print result
print("completion: ", context)


