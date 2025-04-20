# pixtral forward:
# vision encode
# text encode
# transformer


# vision encoder
# get patches of each image
# get embeddings of each patch
# rope2D (rope1d x and rope1d y)
# alternating attention mask
# put patch embeds through Pixtral ViT
# put embeds through vision-language projector MLP => output embeddings
# add embeddings of break tokens to each row end (except last)
# add embeddings of end tokens to each img end


# text encoder:
# get text embeddings


# transformer
# 40 attention blocks
#   pre attn layernorm
#   attention
  #   compute Q, K, V
  #   split via GQA
  #   softmax attention table
  #   recombine shapes
  #   outproject
#   recombine residual
#   post attn layernorm
#   feed forward (swiglu)
#   recombine residual
# layer norm
# head projection
# softmax
# sample categorical per token