import warnings
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import jax.random as jrand

## load params
from load_model import load_params, fast_load_params
from forward_common import encode, decode, tokenize_messages_dict
from forward_training import (
    load_dense_lora, apply_dense_lora,
    load_attn_lora, apply_attn_lora
)

# typing
from typing import Tuple, List, Optional
from model_types import PixtralModel

# logging
import time

# chat formatting
from PIL import Image
from display_common import *
from pprint import pprint # pretty debug outputs
import codecs # used for streaming unicode (an emoji is >1 tokens - breaks if printed token by token)

# model inferencing (prefill + kvcache)
from forward_prefill import inference_prefill
from forward_cached import inference_cached



left_bracket_token = "<<"
right_bracket_token = ">>"

HELP_MESSAGE = f"""
Example prompt: "This is a prompt. Here {left_bracket_token}../images/image.png{right_bracket_token} is a picture. Please describe it."

## Commands ##
/help                       displays this message
/delete                     deletes the last message
/set temp <value>           sets the model temperature to any float value
/set max_tokens <value>     change the max tokens generated per response
/info                       shows the current params
/reset                      resets the chat
/exit
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace") # emoji streaming stuff (currently broken)
decoder = codecs.getincrementaldecoder("latin-1")()



# splits a string, keeps the delimiter
# keep_split("how cool") => ["how", " ", "cool"]
def keep_split(
    string: str,
    delimiter: str
) -> List[str]:
    xs = string.split(delimiter)
    xs = [(substr, delimiter) for substr in xs]
    xs = [item for sublist in xs for item in sublist]
    xs = xs[:-1]
    xs = [x for x in xs if x]
    return xs



# parse each user input to the chat
# /xyz -> command
# xyz -> chat message
# xyz <<images/abc.png>> bla -> chat message with image
def parse_user_message(user_message: str) -> Tuple[str, str]:
    message = {
        "role": "user",
        "content": [],
    }
    tokenized = keep_split(user_message, left_bracket_token)
    tokenized = [keep_split(chars, right_bracket_token) for chars in tokenized]
    tokenized = [item for sublist in tokenized for item in sublist] # horrible syntax
    i = 0
    try:
        while i < len(tokenized):
            token = tokenized[i]
            if token == "<<":
                if i + 1 > len(tokenized):
                    raise ValueError(f"unmatched '{left_bracket_token}'")
                elif tokenized[i+2] != right_bracket_token:
                    raise ValueError(f"no closing tag found, or a second open tag was found early (e.g. '{left_bracket_token} {left_bracket_token}')")
                elif tokenized[i+1] == right_bracket_token:
                    raise ValueError("Tag must contain an image path. Type '/help' for an example")
                elif i + 2 > len(tokenized):
                    raise ValueError(f"unmatched '{left_bracket_token}'")
                else:
                    content = {
                      "type": "image_url",
                      "image_url": {
                          "url": tokenized[i+1]
                      }
                    }
                    message["content"].append(content)
                    i += 3
                    continue
            elif token == right_bracket_token:
                raise ValueError(f"unmatched '{right_bracket_token}'")
            else:
                content = {
                  "type": "text",
                  "text": token
                }
                message["content"].append(content)
                i += 1
                continue
    except Exception as e:
        error_msg = str(e)
        return None, error_msg

    return message, None



def print_user_message(user_message):
    for content in user_message["content"]:
        if content["type"] == "image_url":
            image_url = content["image_url"]["url"]
            image = Image.open(image_url)
            show_image(image)
        else:
            pass
            #print(f"{content['text']}")
    
    

def chat(
    verbose=True, # pretty print statuses
    debug=False, # output internal info
    safetensors_paths=['./pixtral/consolidated.safetensors'], # pixtral param safetensor paths
    lora_path=None # paths of additional LoRAs
    ):
    params_loaded = False

    load_and_show_image("../logo.png", height=16, resample=Image.NEAREST) # if alpha ..
    if int(time.time()) % 2 == 7:
        _j = highlight_string("j", color_jax_blue_light)
        _a = highlight_string("a", color_jax_green_light)
        _x = highlight_string("x", color_jax_purple_light)
        print(f"/{_j}{_a}{_x}tral chat/") # make this grey or something
        print()
    else:
        _j = color_string("j", color_jax_blue_light)
        _a = color_string("a", color_jax_green_light)
        _x = color_string("x", color_jax_purple_light)
        _start = color_string("/", color_grey)
        _end = color_string("tral chat/", color_grey)
        print(f"{_start}{_j}{_a}{_x}{_end}") # make this grey or something
        print()

    # starting params
    temp = 0.0
    max_tokens = 1024

    messages = [] # initialize chat
    while 1:
        # get user input
        user_input = input(color_string("+> ", color_jax_purple_light))

        # run command
        if user_input.startswith("/"):
            command, *args = user_input[1:].split(' ')
            if command == "help":
                print_color(HELP_MESSAGE, color=color_grey)
            elif command == "set":
                if args[0] == "temp":
                    temp = float(args[1])
                elif args[0] == "max_tokens":
                    max_tokens = int(args[1])
            elif command == "reset":
                messages = []
            elif command == "info":
                print_color(f"max_tokens: {max_tokens}, temp: {temp}", color=color_grey)
            elif command == "exit":
                break
            continue

        # parse user message
        user_message, err = parse_user_message(user_input)
        if err:
            print_color("Error: ", err, color=color_red)
            continue
        messages.append(user_message)

        # show user's message
        # remove input text they typed
        print_user_message(user_message)

        # get ai response
        if debug:
            pprint(messages)
        if not params_loaded:
            if lora_path:
                lora_params = load_attn_lora(lora_path)
            # load params
            load_start = time.time()
            print_color("loading params...", color=color_grey)
            pixtral_params = fast_load_params(safetensors_paths)
            if verbose: print_color(f"Loaded params in {time.time() - load_start:.2f}s", color=color_grey)
            params_loaded = True
            # apply loras
            if lora_path:
                #lora_params = load_lora("loras/test.safetensors")
                pixtral_params = apply_attn_lora(pixtral_params, lora_params)
        completion = _get_completion(pixtral_params, messages, max_tokens, temp, verbose=False, color=color_jax_blue_light)
        print() # add a newline

        # add ai response to chat
        response = {"role": "assistant", "content": [{"type": "text", "text": completion}]}
        messages.append(response)

        if debug:
            # show current messages json
            pprint(messages)



# user interface
def get_completion(
    messages,
    max_tokens: int,
    temp: float,
    seed: float = (0_0)-7,
    verbose: bool = True,
    lora_path: str = None
) -> str:
    load_start = time.time()
    safetensors_paths = ['./pixtral/consolidated.safetensors']
    pixtral_params = load_params(safetensors_paths)
    if verbose: print(f"Loaded params in {time.time() - load_start:.2f}s")
    if lora_path:
                lora_params = load_attn_lora(lora_path)
                pixtral_params = apply_attn_lora(pixtral_params, lora_params)
    return _get_completion(pixtral_params, messages, max_tokens, temp, seed=seed, verbose=verbose)



# loads completion using existing params
def _get_completion(
    pixtral_params: PixtralModel,
    messages,
    max_tokens: int,
    temp: float,
    seed: float = (0_0)-7,
    verbose: bool = True,
    color: Optional[Tuple[int, int, int]] = None
) -> str:
    log = lambda *args: None
    if verbose:
        log = print

    # tokenize prompt
    prompt_tokens, images, image_start_indices = tokenize_messages_dict(messages)
    log(f"input tokens: {len(prompt_tokens)}", prompt_tokens)
    
    # init completion
    completion = ""
    completion_token_count = 0
    completion_tokens = []
    tokens = prompt_tokens
    max_tokens = len(tokens) + max_tokens # pad tokens => make shape constant => only one jit compilation
    
    ## run inference loop
    key = jrand.PRNGKey(seed)
    if color:
        set_text_color(color)
    for i in range(max_tokens):
        finished_jit = False
        if i == 0:
            ### PREFILL
            prefill_start = time.time()
            next_token_batch, kvcache = inference_prefill(key, pixtral_params, tokens, images, image_start_indices)
            next_token = next_token_batch[0]
            # pad kvcache to max token size (jax jit will require constant shapes)
            blocks, B, Hk, r, T, d = kvcache.K.shape
            initial_token_count = T-1
            max_tokens = initial_token_count + max_tokens
            padding_tokens = (max_tokens - 1)
            padding_dims = [(0, 0), (0, 0),(0, 0),(0, 0),(0, padding_tokens),(0, 0)] # pad on T dim: KV is (xfmr_blocks, B, Hk, r=1, T, d)
            kvcache = kvcache._replace(
                K = jnp.pad(kvcache.K, padding_dims, mode="constant", constant_values=0),
                V = jnp.pad(kvcache.V, padding_dims, mode="constant", constant_values=0),
            )
            next_token_index = T
            log(f"prefill duration: {time.time() - prefill_start:.2f}")
        else:
            ### INFERENCE WITH KVCACHE
            if i == 1:
                jit_start = time.time()
            elif i == 2:
                token_generation_start = time.time()
            next_token_batch, kvcache = inference_cached(key, pixtral_params, next_token_batch, kvcache, next_token_index)
            next_token_batch = next_token_batch[0] # double batched for some reason (bug)
            next_token = next_token_batch[0]
            next_token_index += 1
            if i == 1:
                jit_end = time.time()
                finished_jit = True
        ## Print in-progress completion
        if int(next_token) == 2:
            completion_token_count += 1
            break # EOS - stop inference loop
        else:
            next_token_chars = decode([next_token])
        if color:
            next_bytes = next_token_chars.encode("utf-8")
            next_chars = decoder.decode(next_bytes, final=False)
            if next_chars:
                print(next_chars, end="", flush=True)   
            #print_color(next_token_chars, color=color, end="")
        else:
            print(next_token_chars, end="", flush=True)
        completion += next_token_chars
        completion_token_count += 1
        tokens.append(int(next_token))
        completion_tokens.append(int(next_token))
        #print(tokens) # debug
    
    ## print result
    print("done")
    if finished_jit:
        log(f"\n\njit duration:{jit_end - jit_start:.2f}")
    log("tokens generated: ", completion_token_count)
    duration = time.time() - token_generation_start
    log("generation duration", duration)
    log("tok/sec", (completion_token_count - 2)/duration) # dont count tokens generated during prefill or jit
    return decode(completion_tokens)

