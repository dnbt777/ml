import jax
import jax.numpy as jnp
import jax.random as jrand
import PIL

from setup_utils import LlamaParams
from typing import TypeAlias
from llama_types import LlamaParams

# Type aliases for better static typing
TensorBTC: TypeAlias = jax.Array
TensorBC: TypeAlias = jax.Array
TensorBTHd: TypeAlias = jax.Array
ImageFloat32: TypeAlias = jax.Array # PIL uses float32 for F mode.



# def vision_encoder
# ImageFloat32 -> TensorBTC


# def text_encoder
# text -> embeddings


# trainable if needed
# model -> TensorBTC -> TensorBTC -> float -> TensorBTC
def forward(model_params: LlamaParams, xBTC: TensorBTC, xBTC_image: TensorBTC, temperature: float) -> TensorBTC:
  ### VISION ENCODING
  ## local


  ## global


  ### -> xBTC_image

  ### TRANSFORMER
  # for each attention layer:
    # if self attention:
      # Reshape input into attention heads
      # xBTC => xBTHd 
      
      # do self attention

      
      # reshape back into xBTC_residual
      
      # xBTC_residual = layer_norm(xBTC_residual)
      # forward layer => xBTC_residual = ffn(xBTC_residual)

      # xBTC = xBTC + xBTC_residual
    # if cross attention:
      # reshape input xBTC and input xBTC_image into attention heads
      # xBTC => xBTHd, xBTC_image => xBTHd_image
      # do cross attention: Q = xBTHd, K = xBTHd_image

      # reshape output into xBTC_residual

      # xBTC_residual = layer_norm(xBTC_residual)
      # forward mlp layer. xBTC_residual = ffn(xBTC_residual)

      # xBTC = xBTC + xBTC_residual
  
  # xBTC = layernorm(xBTC)

  # yBTC = ffn(xBTC)

  # yBTC_probs = softmax(xBTC/(temperature + eta), axis=-1)
  
  # return yBTC_probs
  pass


# model -> str -> ImageFloat32 -> float -> str
def inference(model_params: LlamaParams, prompt: str, image: ImageFloat32, temperature: float) -> str:
  # yBTC_probs = forward(model_params, encode(prompt), temperature) 

  # yBT = jnp.argmax(yBTC_probs, axis=-1)
  
  # return yBT
  pass