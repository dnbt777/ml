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

from forward_common import (
    vision_encoder, vision_language_adapter, text_embedding, multimodal_embedding,
    transformer_block
)
from forward_common import *

from typing import NamedTuple, List
from model_types import PixtralModel

from safetensors.flax import save_file, load_file




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


def init_dense_lora(
    key: jax.Array,
    in_dim: int,
    out_dim: int,
    rank: int
) -> DenseLoRA:
    # initialize lora params using xavier normal
    # https://medium.com/@himalayaashish/unveiling-the-weighty-world-of-neural-networks-a-deep-dive-into-weights-605f3fcabe0b
    return DenseLoRA(
        in_matrix=jrand.normal(key, (in_dim, rank), dtype=jnp.bfloat16)*jnp.sqrt(2/in_dim),
        out_matrix=jnp.zeros((rank, out_dim), dtype=jnp.bfloat16),
        alpha=jnp.bfloat16(1.0),
    )



#########################
## AttentionLora
## list of QLoRA, KLoRA, and VLoRA (QKV - NOT 'quantized')
## layer-based. this will be scanned over

class AttentionLoRALayer(NamedTuple):
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


class AttentionLoRA(NamedTuple):
    layers: AttentionLoRALayer


def init_attention_lora(
    key: jax.Array,
    in_q_dim: int, out_q_dim: int, rank_q: int,
    in_k_dim: int, out_k_dim: int, rank_k: int,
    in_v_dim: int, out_v_dim: int, rank_v: int,
    in_o_dim: int, out_o_dim: int, rank_o: int,
    layers: int,
) -> AttentionLoRA:
    # initialize lora params using xavier normal
    # https://medium.com/@himalayaashish/unveiling-the-weighty-world-of-neural-networks-a-deep-dive-into-weights-605f3fcabe0b
    key_q, key_k, key_v, key_o = jrand.split(key, 4)
    dtype = jnp.bfloat16
    initial_alpha = dtype(1.0)
    return AttentionLoRA(
        layers=AttentionLoRALayer(
            in_q=jrand.normal(key_q, (layers, in_q_dim, rank_q), dtype=dtype)*jnp.sqrt(2.0/in_q_dim),
            out_q=jnp.zeros((layers, rank_q, out_q_dim), dtype=dtype),
            alpha_q=jnp.ones((layers,), dtype=dtype)*initial_alpha,
            in_k=jrand.normal(key_k, (layers, in_k_dim, rank_k), dtype=dtype)*jnp.sqrt(2.0/in_k_dim),
            out_k=jnp.zeros((layers, rank_k, out_k_dim), dtype=dtype),
            alpha_k=jnp.ones((layers,), dtype=dtype)*initial_alpha,
            in_v=jrand.normal(key_v, (layers, in_v_dim, rank_v), dtype=dtype)*jnp.sqrt(2.0/in_v_dim),
            out_v=jnp.zeros((layers, rank_v, out_v_dim), dtype=dtype),
            alpha_v=jnp.ones((layers,), dtype=dtype)*initial_alpha,
            in_o=jrand.normal(key_o, (layers, in_o_dim, rank_o), dtype=dtype)*jnp.sqrt(2.0/in_o_dim),
            out_o=jnp.zeros((layers, rank_o, out_o_dim), dtype=dtype),
            alpha_o=jnp.ones((layers,), dtype=dtype)*initial_alpha,
        )
    )



###########################
### LoRA
## all loras combined into one
## general-purpose lora type. used for function logic/signatures

class LoRA(NamedTuple):
    attention_lora: AttentionLoRA
    dense_lora: DenseLoRA


def init_lora(
    key,
    dense_in_dim: int, dense_out_dim: int, dense_rank: int,
    attn_in_q_dim: int, attn_out_q_dim: int, attn_rank_q: int,
    attn_in_k_dim: int, attn_out_k_dim: int, attn_rank_k: int,
    attn_in_v_dim: int, attn_out_v_dim: int, attn_rank_v: int,
    attn_in_o_dim: int, attn_out_o_dim: int, attn_rank_o: int,
    attn_layers: int,
) -> LoRA:
    attn_key, dense_key = jrand.split(key)
    return LoRA(
        attention_lora=init_attention_lora(
            attn_key,
            attn_in_q_dim, attn_out_q_dim, attn_rank_q,
            attn_in_k_dim, attn_out_k_dim, attn_rank_k,
            attn_in_v_dim, attn_out_v_dim, attn_rank_v,
            attn_in_o_dim, attn_out_o_dim, attn_rank_o,
            attn_layers,
        ),
        dense_lora=init_dense_lora(
            dense_key, dense_in_dim, dense_out_dim, dense_rank
        )
    )


#@partial(jax.jit, donate_argnames=["model_params", "lora_params"])
def merge_lora(
    model_params,
    lora_params
):
    """
    Adds the lora params onto the model's params
    """
    # dense
    #model_params = model_params._replace(transformer=)

    # attention
    model_params = model_params._replace(
        transformer=Transformer(
            TransformerLayer(
                  attention_wk_weight=model_params.transformer.transformer_layers.attention_wk_weight + jnp.swapaxes(
                      lora_params.attention_lora.layers.alpha_k[:, None, None]*(lora_params.attention_lora.layers.in_k @ lora_params.attention_lora.layers.out_k), -1, -2),
                  attention_wo_weight=model_params.transformer.transformer_layers.attention_wo_weight + jnp.swapaxes(
                      lora_params.attention_lora.layers.alpha_o[:, None, None]*(lora_params.attention_lora.layers.in_o @ lora_params.attention_lora.layers.out_o), -1, -2),
                  attention_wq_weight=model_params.transformer.transformer_layers.attention_wq_weight + jnp.swapaxes(
                      lora_params.attention_lora.layers.alpha_q[:, None, None]*(lora_params.attention_lora.layers.in_q @ lora_params.attention_lora.layers.out_q), -1, -2),
                  attention_wv_weight=model_params.transformer.transformer_layers.attention_wv_weight + jnp.swapaxes(
                      lora_params.attention_lora.layers.alpha_v[:, None, None]*(lora_params.attention_lora.layers.in_v @ lora_params.attention_lora.layers.out_v), -1, -2),
                  attention_norm_weight=model_params.transformer.transformer_layers.attention_norm_weight,
                  feed_forward_w1_weight=model_params.transformer.transformer_layers.feed_forward_w1_weight,
                  feed_forward_w2_weight=model_params.transformer.transformer_layers.feed_forward_w2_weight,
                  feed_forward_w3_weight=model_params.transformer.transformer_layers.feed_forward_w3_weight,
                  ffn_norm_weight=model_params.transformer.transformer_layers.ffn_norm_weight,
            )
        )
    )
    return model_params
                                        



def load_lora(filepath: str, device: jax.Device = None) -> AttentionLoRA:
    tensors = load_file(filepath)
    attn_prefix = "attn"
    dense_prefix = "dense"
    mlp_proj_prefix = "mlp"
    if True or device: # just do this anyways, for now
        attention_lora_tensors = {
            k.split('.')[1]: jax.device_put(v, device)
            for k, v in tensors.items()
            if k.split('.')[0] == attn_prefix
        }
        dense_lora_tensors = {
            k.split('.')[1]: jax.device_put(v, device)
            for k, v in tensors.items()
            if k.split('.')[0] == dense_prefix
        }
    return LoRA(
        attention_lora=AttentionLoRA(AttentionLoRALayer(**attention_lora_tensors)),
        dense_lora=DenseLoRA(**dense_lora_tensors),
    )


def save_lora(lora_params: AttentionLoRA, filepath: str):
    attn_prefix = "attn"
    dense_prefix = "dense"
    mlp_proj_prefix = "mlp"
    tensors = {
        # attention lora
        f"{attn_prefix}.in_q": lora_params.attention_lora.layers.in_q,
        f"{attn_prefix}.out_q": lora_params.attention_lora.layers.out_q,
        f"{attn_prefix}.alpha_q": lora_params.attention_lora.layers.alpha_q,
        f"{attn_prefix}.in_k": lora_params.attention_lora.layers.in_k,
        f"{attn_prefix}.out_k": lora_params.attention_lora.layers.out_k,
        f"{attn_prefix}.alpha_k": lora_params.attention_lora.layers.alpha_k,
        f"{attn_prefix}.in_v": lora_params.attention_lora.layers.in_v,
        f"{attn_prefix}.out_v": lora_params.attention_lora.layers.out_v,
        f"{attn_prefix}.alpha_v": lora_params.attention_lora.layers.alpha_v,
        f"{attn_prefix}.in_o": lora_params.attention_lora.layers.in_o,
        f"{attn_prefix}.out_o": lora_params.attention_lora.layers.out_o,
        f"{attn_prefix}.alpha_o": lora_params.attention_lora.layers.alpha_o,
        # todo add mlp/proj lora
        # dense lora
        f"{dense_prefix}.in_matrix": lora_params.dense_lora.in_matrix,
        f"{dense_prefix}.out_matrix": lora_params.dense_lora.out_matrix,
        f"{dense_prefix}.alpha": lora_params.dense_lora.alpha,
    }
    save_file(tensors, filepath)



#@jax.jit
def mm_lora_loss_fn(
     lora_params: LoRA,
     pixtral_params: PixtralModel,
     batch_message_tokens: jax.Array,
     batch_processed_images,
     batch_intext_image_start_indices: List[int],
     batch_context_mask: jax.Array,
     batch_padding_mask: jax.Array,
     key: jax.Array
) -> float:
    # forward
    next_token_logits = mm_forward(pixtral_params, batch_message_tokens, batch_processed_images, batch_intext_image_start_indices, lora_params=lora_params)
    next_token_logprobs = jax.nn.log_softmax(next_token_logits[:, :-1, :], axis=-1)
    # mask out context tokens (i.e. only do assistant response)
    target_probs = jax.nn.one_hot(batch_message_tokens[1:], 131072, axis=-1, dtype=jnp.bfloat16)
    batch_loss_mask = jnp.logical_or(batch_context_mask, batch_padding_mask) # mask out context and padding, only grade on response
    batch_loss_mask = batch_loss_mask[:, :-1, None]
    # get loss
    batch_loss = cross_entropy_loss(batch_next_token_logprobs, batch_target_probs, batch_loss_mask)


def batch_parse_completions(completions):
    """
    Batches chat completions for training

    completions: list of json chat completions, each ending with an assistant message
    """
    # tokenize prompt
    # SoA (for logic related (not performance) reasons)
    batch_completions = {
        "processed_images"      : [],
        "image_start_indices"   : [],
        "tokens"                : [],
        "padding_mask"          : [],
        "context_mask"          : [],
        "image_mask"            : [],
    }

    # store processed prompts and initialize values
    largest_completion_tokens = 0
    for completion in completions:
        completion_tokens, processed_images, image_start_indices, context_mask, image_mask = tokenize_messages_dict_with_masks(completion)
        print(f"input tokens: {len(completion_tokens)}", completion_tokens)
        largest_completion_tokens = max(largest_completion_tokens, len(completion_tokens))
        batch_completions["processed_images"].append(processed_images)
        batch_completions["image_start_indices"].append(image_start_indices)
        batch_completions["tokens"].append(completion_tokens)
        batch_completions["padding_mask"].append(None)
        batch_completions["context_mask"].append(jnp.array(context_mask))
        batch_completions["image_mask"].append(jnp.array(image_mask))
        
    # pad all completions to be the size of the largest completion
    completion_count = len(completions)
    for i in range(completion_count):
        initial_completion_length = len(batch_completions["tokens"][i])
        new_completion_length = largest_completion_tokens
        padding_length = new_completion_length - initial_completion_length
        batch_completions["tokens"][i] = jnp.append(jnp.array(batch_completions["tokens"][i]), jnp.zeros((padding_length,), dtype=int))
        padding_mask = (jnp.arange(new_completion_length) >= initial_completion_length).astype(bool) # mask padding with True
        assert jnp.sum(padding_mask).astype(int) == padding_length
        batch_completions["padding_mask"][i] = padding_mask

    # stack jax arrays into a batch
    batch_completions["tokens"] = jnp.stack(batch_completions["tokens"]) # list[arrayT] -> arrayBT
    batch_completions["padding_mask"] = jnp.stack(batch_completions["padding_mask"]) # list[arrayT] -> arrayBT
    batch_completions["context_mask"] = jnp.stack(batch_completions["context_mask"]) # list[arrayT] -> arrayBT
    
    return batch_completions



    
def text_forward_train(model_params: PixtralModel, batch_tokens, batch_attn_mask, lora_params=None):
  hidden_state_BTC = text_embedding(model_params, batch_tokens)
  return forward_train(model_params, hidden_state_BTC, batch_attn_mask, lora_params=lora_params)


def mm_forward_train(model_params: PixtralModel, batch_tokens, batch_image_sets, batch_intext_image_start_indices, batch_attn_mask, lora_params=None):
  hidden_state_BTC = multimodal_embedding(model_params, batch_tokens, batch_image_sets, batch_intext_image_start_indices)
  return forward_train(model_params, hidden_state_BTC, batch_attn_mask, lora_params=lora_params)


def forward_train(model_params, hidden_state_BTC, batch_attn_mask, lora_params=None):
  B, T, C = hidden_state_BTC.shape
  head_dim = 128 # params.json
  max_pos, d = T, head_dim
  freqs = precompute_rope_freqs_1d(max_pos, d) # mistral does rope after splitting k and q into gqa heads. q and k are split into the same channel size per head

  # attention layers
  Hq = 32 # params.json
  Hk = 8 # params.json
  attn_mask = get_causal_mask(T)[None, None, :, :]
  attn_mask = jnp.logical_or(batch_attn_mask[:, None, None, None, :], attn_mask)  # if True in either mask, mask out token
  # head dim defined above - it's used to calculate rope1d frequencies
  # scan compiles faster than a for loop
  if lora_params:
    def _block_fn(hidden_state, freqs, attn_mask, carry):
      xfmr_block_params, block_lora_params = carry
      return transformer_block(xfmr_block_params, hidden_state, freqs, Hq, Hk, head_dim, attn_mask, block_lora_params=block_lora_params)
    block_fn = jax.checkpoint(_block_fn)
    def scanf(hidden_state, carry):
      hidden_state = block_fn(hidden_state, freqs, attn_mask, carry)
      return hidden_state, None
    hidden_state_BTC, _ = jax.lax.scan(scanf, hidden_state_BTC, (model_params.transformer.transformer_layers, lora_params.attention_lora.layers))
  else:
    def scanf(hidden_state, block_params):
      hidden_state = transformer_block(block_params, hidden_state, freqs, Hq, Hk, head_dim, attn_mask, block_lora_params=None)
      return hidden_state, None
    hidden_state_BTC, _ = jax.lax.scan(scanf, hidden_state_BTC, model_params.transformer.transformer_layers)
    
  # training - we care about all tokens here
  # layernorm
  hidden_state_BTC = layernorm(hidden_state_BTC, model_params.norm_weight, jnp.zeros((1, hidden_state_BTC.shape[-1])))
  # lm_head: channel -> vocab logits
  if lora_params:
      lora_out = lora_params.dense_lora.alpha * (hidden_state_BTC @ lora_params.dense_lora.in_matrix) @ lora_params.dense_lora.out_matrix
      hidden_state_BTC = lora_out + (hidden_state_BTC @ model_params.output_weight.T) # (B, C) @ (C, vocab) => (B, vocab)
  else:
      hidden_state_BTC = hidden_state_BTC @ model_params.output_weight.T # (B, C) @ (C, vocab) => (B, vocab)
      
  return hidden_state_BTC




@jax.jit
def cross_entropy_loss(
    batch_next_token_logits: jax.Array,
    batch_target_tokens: jax.Array,
    batch_loss_mask: jax.Array
) -> float:
    batch_next_token_logprobs = jax.nn.log_softmax(batch_next_token_logits, axis=-1)
    batch_sub_crossentropies = jnp.take_along_axis(batch_next_token_logprobs, batch_target_tokens[..., None], axis=-1) # B, ->T<-, C
    return -jnp.sum(jnp.where(batch_loss_mask, jnp.bfloat16(0), batch_sub_crossentropies))



@jax.jit
def text_lora_loss_fn(
     pixtral_params: PixtralModel,
     lora_params: LoRA,
     batch_message_tokens: jax.Array,
     batch_context_mask: jax.Array,
     batch_padding_mask: jax.Array,
     key: jax.Array
) -> float:
    #pixtral_params = merge_lora(pixtral_params, lora_params) # not mem efficient in backwards
    # forward
    batch_input_tokens = batch_message_tokens[:, :-1]
    batch_target_tokens = batch_message_tokens[:, 1:]
    batch_attn_mask = batch_padding_mask[:, :-1] # align with inputs
    batch_next_token_logits = text_forward_train(pixtral_params, batch_input_tokens, batch_attn_mask, lora_params=lora_params)
    # mask out context tokens (i.e. only train on assistant's response)
    # mask out padding tokens (padding_mask) and user prompt tokens (context_mask)
    batch_loss_mask = jnp.logical_or(batch_context_mask, batch_padding_mask) # mask out context and padding, only grade on response
    batch_loss_mask = batch_loss_mask[:, 1:, None] # align with targets
    # get loss
    batch_loss = cross_entropy_loss(batch_next_token_logits, batch_target_tokens, batch_loss_mask)
    return batch_loss



# experiments
# train a simple lora that writes in all caps
# train a simple lora that 
# do in context learning


# experiments to consider
# sparse autoencoder for chesstral (constantly steers the conversation towards chess)


# write a blog post on how to implement this
# write little blog posts about experiments (make it fun)


# for chesstral: this NEEDS a lora on everything related to vision processing
