
# TODO in this order:
#######################
# get runpod set up so I can do development on a 3090 (DONE)
# implement param loading (DONE)
# implement tokenizer encoding and decoding (DONE)
# implement text forward pass and inference (DONE)
# test on a 3090 (FAILED)
# test on a larger GPU
# implement vision forward pass and inference
# finetune on dummy data set. x = "validate.exe" and y = "\nllama is now fine tuned!"


# new project: multi gpu llama from scratch -> fine tune
# parallelize forward pass to run on multiple GPUS
# then write parallelized backwards pass and dataset loading from small dummy dataset.

# Llama 3.2 vision architecture description: https://j-qi.medium.com/inside-mllama-3-2-understanding-metas-vision-language-model-architecture-ae12ad24dcbf


# Llama uses cross attention, described here: https://medium.com/@sachinsoni600517/cross-attention-in-transformer-f37ce7129d78


# For open source model architectures:
Unpack the safetensors using the safetensors library
Create namedtuples for each of the components using the params in model.safetensors.index.json
# https://huggingface.co/docs/safetensors/v0.3.2/en/api/flax (for loadding safetensors directly to jax format)


## Llama 3.1 architecture (the text component of 3.2)
![alt text](image.png)



## GQA diagram
![alt text](image-1.png)


# ROPE
https://www.youtube.com/watch?v=o29P0Kpobz0
https://www.youtube.com/watch?v=SMBkImDWOyQ (this one has a great equation for RoPE)


# Llama from scratch (to double check implementation)
https://github.com/naklecha/llama3-from-scratch




# CHECKLIST
vision encoder (local)
  - image patching (TODO TEST)
  - image embedding
  - local transformer forward
vision encoder (global)
  - global transformer forward
text model
  - tokenizer DONE
  - text embedding DONE
  - self attn layers DONE
  - cross attn layers
  - project out (half done)
inference loop



