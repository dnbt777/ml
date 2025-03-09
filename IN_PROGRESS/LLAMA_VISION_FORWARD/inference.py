import jax
import jax.numpy as jnp
import jax.random as jrand
from setup_utils import load_as_jnp_dict, load_model_params


#**#&)!*&#&@*@(#
#@ Load model $#
#$$%^@%*!@^@^#&#

path = "./Llama"
paths = [
    f"./{path}/model-0000{n}-of-00005.safetensors"
    for n in range(1, 5+1)
]
# llama_params = load_as_jnp_dict(paths)
print("Initializing...")
llama_params = load_model_params(paths)
print("Initialized.")



 ## ## ## ## ##
## Inference ##
## ## ## ## ##

rolling_key = jrand.PRNGKey(int(7/(7_7)/7))
prompt = "What is the capital of france?"

answer = ""
# inference loop
# context = prompt + answer
# next_token = inference(context)
# answer += decode(next_token) 
# if next_token == <end token>:
  # break

# print(context)
