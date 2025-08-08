import jax
import jax.numpy as jnp
import numpy as np
from typing import List
from model_types import *
from einops import rearrange
import jax.random as jrand
from functools import partial

from PIL import Image
import cv2

from forward_common import vision_encoder, vision_language_adapter, text_embedding, multimodal_embedding
from forward_common import *

from typing import NamedTuple, List
from model_types import PixtralModel

from safetensors.flax import save_file, load_file





@jax.jit
def cross_entropy_loss(
    next_token_logprobs: jax.Array,
    target_probs: jax.Array,
    mask: jax.Array
) -> float:
    return -jnp.sum(target_probs * next_token_logprobs * mask.astype(jnp.bfloat16)) / jnp.sum(mask.astype(jnp.bfloat16))



################################
## DenseLora
## applies a lora to the lm head

# Lora: trains the model to have different facts/instructions
# small matrix that goes over lm head
# https://ash-01xor.github.io/blog/posts/LoRA/
class DenseLoRA(NamedTuple):
    in_matrix: jax.Array # (channel, lora_dim)
    out_matrix: jax.Array # (lora_dim, vocab)
    alpha: jnp.bfloat16



#@partial(jax.jit, donate_args=["model_params", "lora_params"])
def apply_dense_lora(
    model_params: PixtralModel,
    lora_params: DenseLoRA
) -> PixtralModel:
    # add lora to dense and return the params
    # this would be called in the model's training code. just slip in 'model_params = apply_lora(model_params, lora) before calling forward(model_params, ...)
    return model_params._replace(
        output_weight = model_params.output_weight + lora_params.alpha*(lora_params.in_matrix @ lora_params.out_matrix).T
    )


def init_dense_lora(
    key: jax.Array,
    in_dim: int,
    out_dim: int,
    lora_dim: int
) -> DenseLoRA:
    # initialize lora params using xavier normal
    # https://medium.com/@himalayaashish/unveiling-the-weighty-world-of-neural-networks-a-deep-dive-into-weights-605f3fcabe0b
    in_key, out_key = jrand.split(key)
    return LoRA(
        in_matrix=jrand.normal(in_key, (in_dim, lora_dim), dtype=jnp.bfloat16)*jnp.sqrt(2/in_dim),
        out_matrix=0.0*jrand.normal(out_key, (lora_dim, out_dim), dtype=jnp.bfloat16)*jnp.sqrt(2/lora_dim),
        alpha=jnp.bfloat16(1.0),
    )


def save_dense_lora(lora_params: DenseLoRA, filepath: str):
    tensors = {
        "in_matrix": lora_params.in_matrix,
        "out_matrix": lora_params.out_matrix,
        "alpha": lora_params.alpha,
    }
    save_file(tensors, filepath)



def load_dense_lora(filepath: str, device: jax.Device = None) -> DenseLoRA:
    tensors = load_file(filepath)
    if device:
        tensors = {k: jax.device_put(v, device) for k, v in tensors.items()}
    return DenseLoRA(**tensors)


@jax.jit
def dense_lora_loss_fn(
     lora_params: DenseLoRA,
     pixtral_params: PixtralModel,
     message_tokens: jax.Array,
     processed_images,
     intext_image_start_indices: List[int],
     context_mask: jax.Array,
     key: jax.Array
) -> float:
    # apply lora to params
    pixtral_params = apply_dense_lora(pixtral_params, lora_params)
    # forward
    next_token_logits = mm_forward(pixtral_params, message_tokens, processed_images, intext_image_start_indices)
    next_token_logprobs = jax.nn.log_softmax(next_token_logits[:, :-1, :], axis=-1)
    # mask out context tokens (i.e. only do assistant response)
    target_probs = jax.nn.one_hot(message_tokens[1:], 131072, axis=-1, dtype=jnp.bfloat16)
    next_token_context_mask = context_mask[:, 1:]
    return cross_entropy_loss(next_token_logprobs, target_probs, next_token_context_mask)




#########################
## AttentionLora
## list of QLoRA, KLoRA, and VLoRA (QKV - NOT 'quantized')

class AttentionLoRA(NamedTuple):
    in_q:  jax.Array
    out_q: jax.Array
    alpha_q: jax.Array 
    in_k:  jax.Array
    out_k: jax.Array
    alpha_k: jax.Array
    in_v:  jax.Array
    out_v: jax.Array
    alpha_v: jax.Array
    in_o:  jax.Array
    out_o: jax.Array
    alpha_o: jax.Array


"""
class AttentionLoRA(NamedTuple):
    lora_layers: AttentionLayerLoRA
"""


#@partial(jax.jit, donate_args=["model_params", "lora_params"])
# a scan would compile faster btw and take up less memory. would not be surprised if OOM
def apply_attn_lora(
    model_params: PixtralModel,
    lora_params: AttentionLoRA
) -> PixtralModel:
    # add lora to attn layers and return the params
    # this would be called in the model's training code. just slip in 'model_params = apply_lora(model_params, lora) before calling forward(model_params, ...)
    for i in range(len(model_params.transformer.transformer_layers)):
        model_params.transformer.transformer_layers[i]._replace(
            attention_wq_weight=model_params.transformer.transformer_layers[i].attention_wq_weight + lora_params.alpha_q[i]*(lora_params.in_q[i] @ lora_params.out_q[i]),
            attention_wk_weight=model_params.transformer.transformer_layers[i].attention_wk_weight + lora_params.alpha_k[i]*(lora_params.in_k[i] @ lora_params.out_k[i]),
            attention_wv_weight=model_params.transformer.transformer_layers[i].attention_wv_weight + lora_params.alpha_v[i]*(lora_params.in_v[i] @ lora_params.out_v[i]),
            attention_wo_weight=model_params.transformer.transformer_layers[i].attention_wo_weight + lora_params.alpha_o[i]*(lora_params.in_o[i] @ lora_params.out_o[i]),
        )
    return model_params



def init_attn_lora(
    key: jax.Array,
    in_q_dim: int,
    out_q_dim: int,
    rank_q: int,
    in_k_dim: int,
    out_k_dim: int,
    rank_k: int,
    in_v_dim: int,
    out_v_dim: int,
    rank_v: int,
    in_o_dim: int,
    out_o_dim: int,
    rank_o: int,
    layers: int,
) -> AttentionLoRA:
    # initialize lora params using xavier normal
    # https://medium.com/@himalayaashish/unveiling-the-weighty-world-of-neural-networks-a-deep-dive-into-weights-605f3fcabe0b
    in_key, out_key = jrand.split(key) # TODO update
    return AttentionLoRA(
        in_q=jrand.normal(in_key, (layers, in_q_dim, rank_q), dtype=jnp.bfloat16)*jnp.sqrt(2/in_q_dim),
        out_q=0.0*jrand.normal(out_key, (rank_q, out_q_dim), dtype=jnp.bfloat16)*jnp.sqrt(2/rank_q),
        alpha_q=jnp.bfloat16(1.0),
        in_k=jrand.normal(in_key, (layers, in_k_dim, rank_k), dtype=jnp.bfloat16)*jnp.sqrt(2/in_k_dim),
        out_k=0.0*jrand.normal(out_key, (rank_k, out_k_dim), dtype=jnp.bfloat16)*jnp.sqrt(2/rank_k),
        alpha_k=jnp.bfloat16(1.0),
        in_v=jrand.normal(in_key, (layers, in_v_dim, rank_v), dtype=jnp.bfloat16)*jnp.sqrt(2/in_v_dim),
        out_v=0.0*jrand.normal(out_key, (rank_v, out_v_dim), dtype=jnp.bfloat16)*jnp.sqrt(2/rank_v),
        alpha_v=jnp.bfloat16(1.0),
        in_o=jrand.normal(in_key, (layers, in_o_dim, rank_o), dtype=jnp.bfloat16)*jnp.sqrt(2/in_o_dim),
        out_o=0.0*jrand.normal(out_key, (rank_o, out_o_dim), dtype=jnp.bfloat16)*jnp.sqrt(2/rank_o),
        alpha_o=jnp.bfloat16(1.0),
    )



def save_attn_lora(lora_params: AttentionLoRA, filepath: str):
    tensors = {
        "in_q": lora_params.in_q,
        "out_q": lora_params.out_q,
        "alpha_q": lora_params.alpha_q,
        "in_k": lora_params.in_k,
        "out_k": lora_params.out_k,
        "alpha_k": lora_params.alpha_k,
        "in_v": lora_params.in_v,
        "out_v": lora_params.out_v,
        "alpha_v": lora_params.alpha_v,
        "in_o": lora_params.in_o,
        "out_o": lora_params.out_o,
        "alpha_o": lora_params.alpha_o,
    }
    save_file(tensors, filepath)



def load_attn_lora(filepath: str, device: jax.Device = None) -> AttentionLoRA:
    tensors = load_file(filepath)
    if device:
        tensors = {k: jax.device_put(v, device) for k, v in tensors.items()}
    return AttentionLoRA(**tensors)



@jax.jit
def attn_lora_loss_fn(
     lora_params: AttentionLoRA,
     pixtral_params: PixtralModel,
     message_tokens: jax.Array,
     processed_images,
     intext_image_start_indices: List[int],
     context_mask: jax.Array,
     key: jax.Array
) -> float:
    # apply lora to params
    pixtral_params = apply_attn_lora(pixtral_params, lora_params)
    # forward
    next_token_logits = mm_forward(pixtral_params, message_tokens, processed_images, intext_image_start_indices)
    next_token_logprobs = jax.nn.log_softmax(next_token_logits[:, :-1, :], axis=-1)
    # mask out context tokens (i.e. only do assistant response)
    target_probs = jax.nn.one_hot(message_tokens[1:], 131072, axis=-1, dtype=jnp.bfloat16)
    next_token_context_mask = context_mask[:, 1:]
    return cross_entropy_loss(next_token_logprobs, target_probs, next_token_context_mask)







# experiments
# train a simple lora that writes in all caps
# train a simple lora that 
# do in context learning


# experiments to consider
# sparse autoencoder for chesstral (constantly steers the conversation towards chess)


# write a blog post on how to implement this
# write little blog posts about experiments (make it fun)


# for chesstral: this NEEDS a lora on everything related to vision processing
