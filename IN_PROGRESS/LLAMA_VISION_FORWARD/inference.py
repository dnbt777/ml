            #**#&)!*&#&@*@(#
            #@ Load model $#
            #$$%^@%*!@^@^#&#
# --------------------------------------- #
from setup_utils import load_model_params
import time

path = "./Llama"
paths = [
    f"./{path}/model-0000{n}-of-00005.safetensors"
    for n in range(1, 5+1)
]
# llama_params = load_as_jnp_dict(paths)
print("Initializing...")
start = time.time()
llama_params = load_model_params(paths)
print(f"Initialized in {time.time() - start:0.2f}s.")
# --------------------------------------- #



            ##-#-##-#-##-#-##-##
            ## Load tokenizer -#
            #-##-#-##-##-####-##
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
from tokenizer import load_tokenizer, encode, decode

path = 'Llama/tokenizer.json'
tokenizer = load_tokenizer(path)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #



            ## # ## ## ## ##
            ## Inference ##
            ## ## ## ## ##
# ======================================== #
import jax.random as jrand
import jax.numpy as jnp
from inference_utils import inference

rolling_key = jrand.PRNGKey(int(7/(7_7)/7))
prompt = "What is the capital of france?"
temp = 0
print(f"Prompt: \"{prompt}\"")

answer = ""
# inference loop
start = time.time()
for i in range(100):
  context = prompt + answer
  context_tokens = jnp.array(encode(tokenizer, context))
  next_token = inference(llama_params, context_tokens, temp, rolling_key)
  if i == 0:
    print(f"jitted in {time.time() - start:0.2f}s")
  next_chunk = decode(tokenizer, jnp.array([next_token]))
  answer += next_chunk 
  print(str(next_chunk), end="")
  rolling_key, _ = jrand.split(rolling_key, 2)
  #if next_token == "<end token>":
  #  break # TODO add actual end token to inference loop

print("\nFinal output:")
print("--------------")
print(context)
# ======================================== #


# future optimizations:
# OPTIMIZATION: Store jaxpr, somehow (compilation takes forever)