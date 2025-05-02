# Llama 3.2 Vision 11B

## Progress
### TODO in this order:
- get runpod set up so I can do development on a 3090 (DONE)
- implement param loading (DONE)
- implement tokenizer encoding and decoding (DONE)
- implement text forward pass and inference (DONE)
- test on a 3090 (FAILED)
- test on a larger GPU (DONE)
- implement vision forward pass and inference (DONE)
- test vision forward (IN PROGRESS)
- finetune on dummy data set. x = "validate.exe" and y = "\nllama is now fine tuned!"


### ARHITECTURE IMPLEMENTATION CHECKLIST
vision encoder
  - image patching (DONE)
  - image embedding (DONE)
  - local transformer forward (DONE) 
  - global transformer forward (DONE) 
  
text model
  - tokenizer (DONE)
  - text embedding (DONE)
  - self attn layers (DONE)
  - cross attn layers (DONE)
  - project to output (DONE)

inference loop (DONE)


## Resources


### Llama 3.2 vision architecture description
https://j-qi.medium.com/inside-mllama-3-2-understanding-metas-vision-language-model-architecture-ae12ad24dcbf


### Llama uses cross attention, described here
https://medium.com/@sachinsoni600517/cross-attention-in-transformer-f37ce7129d78


### For open source model architectures
- Unpack the safetensors using the safetensors library
- Create namedtuples for each of the components using the params in model.safetensors.index.json
https://huggingface.co/docs/safetensors/v0.3.2/en/api/flax (for loading safetensors directly to jax format)


### Llama 3.1 architecture (the text component of 3.2)
https://magazine.sebastianraschka.com/p/ai-research-papers-2024-part-2

![alt text](image.png)


### GQA diagram
![alt text](image-1.png)


### ROPE
- https://www.youtube.com/watch?v=o29P0Kpobz0
- https://www.youtube.com/watch?v=SMBkImDWOyQ (this one has a great equation for RoPE)


### Llama from scratch (to double check implementation)
https://github.com/naklecha/llama3-from-scratch







### Small details:
- Each TILE gets its own CLS token



### Proof checklist
Proving that each section is implemented correctly
Tokenizer
Image patcher
- Text forward
Vision-
  - local
  - global