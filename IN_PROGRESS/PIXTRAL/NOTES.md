

Pixtral overview:
https://mistral.ai/news/pixtral-12b



https://x.com/i/grok/share/JJ37wVyKy5SJKb9SXr0ctnb9b



NeMo paper:
https://arxiv.org/pdf/2310.06825





CHECKLIST
~~implement parameter loading/classes~~
~~implement tokenizer~~
~~implement mm_preprocessor~~
implement transformerblock
  - ~~feed forward~~
  - attention
  - ~~the rest~~
implement vision_encoder (img -> embeddings)
  - ~~conv2D~~
  - ~~RMSnorm~~
  - rope2D
implement embedding(message_tokens, processed_images)
implement out + decode
implement inference loop


testing: do shapes work?




testing : do outputs match reference?
test tokenizer
test preprocessor
test message

final testing: test across multiple prompts/images

clean up code

optimizations
  - rope freqs

implement tokenizer from scratch (instead of importing)
replace einops w jax


then post
then record baseline tok/sec

then improve this as much as possible
then post new tok/sec



do moondream






organization:
last time I had too many files all over the place.

mm_preprocessor :: [Image] -> [image_token] 

tokenizer :: [string] -> [[token]]
  - preprocess
  - tokenize
  

vision_encoder :: [[image_token]] -> [???]
  - vision_encoder

text_forward :: [[token]] -> [logprobs]
  - sliding window attention
  - kv cache ring buffer
  - GQA

mm_forward :: [Union(Image, string)] -> [logprobs]


inference :: [Union(Image, string)] -> [string]



each file will be named after its primary function


