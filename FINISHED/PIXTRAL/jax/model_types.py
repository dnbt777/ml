import jax
import jax.numpy as jnp
from typing import NamedTuple, List

#|==============================================================>
#|_______ model_types.py _______________________________________
#|
#|
#|  Contains dataclasses for pixtral and its subcomponents
#| 
#|  PixtralModel
#|    - VisionEncoder
#|    - VisionLanguageAdapter
#|    - Transformer
#|
#|
#|===============================================================>>>



#######
## vision encoder

class VisionEncoderLayer(NamedTuple):
  attention_wk_weight: jax.Array # (1024, 1024)
  attention_wo_weight: jax.Array # (1024, 1024)
  attention_wq_weight: jax.Array # (1024, 1024)
  attention_wv_weight: jax.Array # (1024, 1024)
  attention_norm_weight: jax.Array # (1024,)
  feed_forward_w1_weight: jax.Array # (4096, 1024)
  feed_forward_w2_weight: jax.Array # (1024, 4096)
  feed_forward_w3_weight: jax.Array # (4096, 1024)
  ffn_norm_weight: jax.Array # (1024,)

class VisionEncoder(NamedTuple):
  ln_pre_weight: jax.Array # (1024,)
  patch_conv_weight: jax.Array # (1024, 3, 16, 16)
  vision_encoder_layers: List[VisionEncoderLayer] # 23



#######
## vision-language adapter

class VisionLanguageAdapter(NamedTuple):
  w_in_bias: jax.Array     # (5120,)
  w_in_weight: jax.Array   # (5120, 1024)
  w_out_bias: jax.Array    # (5120,)
  w_out_weight: jax.Array  # (5120, 5120)
 


#######
## transformer

class TransformerLayer(NamedTuple):
  attention_wk_weight: jax.Array # ()
  attention_wo_weight: jax.Array # ()
  attention_wq_weight: jax.Array # ()
  attention_wv_weight: jax.Array # ()
  attention_norm_weight: jax.Array # ()
  feed_forward_w1_weight: jax.Array # ()
  feed_forward_w2_weight: jax.Array # ()
  feed_forward_w3_weight: jax.Array # ()
  ffn_norm_weight: jax.Array # ()

class Transformer(NamedTuple):
  transformer_layers: TransformerLayer # 39



#######
## pixtral

class PixtralModel(NamedTuple):
  norm_weight: jax.Array # (5120,)
  output_weight: jax.Array # (131072, 5120)
  tok_embeddings_weight: jax.Array # (131072, 5120) 
  vision_encoder: VisionEncoder
  vision_language_adapter: VisionLanguageAdapter
  transformer: Transformer


#######
## additional types

from typing import TypeAlias, Union
TransformerBlock: TypeAlias = Union[TransformerLayer, VisionEncoderLayer]