import jax
from typing import NamedTuple, List, TypeAlias, Union 


## Tokenizer types
class Tokenizer(NamedTuple):
  vocab: dict
  merges: dict

Text: TypeAlias = str
Tokens: TypeAlias = jax.Array
Token: TypeAlias = int

## Linalg types


## Llama Model Types
class MultiModalProjector(NamedTuple):
    weight: jax.Array # (4096, 7680)
    bias: jax.Array # (4096,)


class LangModelCrossAttentionLayer(NamedTuple):
    cross_attn_k_norm_weight: jax.Array
    cross_attn_k_proj_weight: jax.Array
    cross_attn_o_proj_weight: jax.Array
    cross_attn_q_norm_weight: jax.Array
    cross_attn_q_proj_weight: jax.Array
    cross_attn_v_proj_weight: jax.Array
    cross_attn_attn_gate: jax.Array
    cross_attn_mlp_gate: jax.Array
    input_layernorm_weight: jax.Array
    mlp_down_proj_weight: jax.Array
    mlp_gate_proj_weight: jax.Array
    mlp_up_proj_weight: jax.Array
    post_attention_layernorm_weight: jax.Array


class LangModelSelfAttentionLayer(NamedTuple):
    input_layernorm_weight: jax.Array
    mlp_down_proj_weight: jax.Array
    mlp_gate_proj_weight: jax.Array
    mlp_up_proj_weight: jax.Array
    post_attention_layernorm_weight: jax.Array
    self_attn_k_proj_weight: jax.Array
    self_attn_o_proj_weight: jax.Array
    self_attn_q_proj_weight: jax.Array
    self_attn_v_proj_weight: jax.Array


class LangModelModel(NamedTuple):
    norm_weight: jax.Array 
    self_attention_layers: List[LangModelSelfAttentionLayer]
    cross_attention_layers: List[LangModelCrossAttentionLayer]

class LangModel(NamedTuple):
    lm_head_weight: jax.Array 
    model: LangModelModel 

    
class VisionModelGlobalLayer(NamedTuple):
    gate_attn: jax.Array
    gate_ffn: jax.Array
    input_layernorm_bias: jax.Array
    input_layernorm_weight: jax.Array
    mlp_fc1_bias: jax.Array
    mlp_fc1_weight: jax.Array
    mlp_fc2_bias: jax.Array
    mlp_fc2_weight: jax.Array
    post_attention_layernorm_bias: jax.Array
    post_attention_layernorm_weight: jax.Array
    self_attn_k_proj_weight: jax.Array
    self_attn_o_proj_weight: jax.Array
    self_attn_q_proj_weight: jax.Array
    self_attn_v_proj_weight: jax.Array



class VisionModelLocalLayer(NamedTuple):
    input_layernorm_bias: jax.Array
    input_layernorm_weight: jax.Array
    mlp_fc1_bias: jax.Array
    mlp_fc1_weight: jax.Array
    mlp_fc2_bias: jax.Array
    mlp_fc2_weight: jax.Array
    post_attention_layernorm_bias: jax.Array
    post_attention_layernorm_weight: jax.Array
    self_attn_k_proj_weight: jax.Array
    self_attn_o_proj_weight: jax.Array
    self_attn_q_proj_weight: jax.Array
    self_attn_v_proj_weight: jax.Array


class VisionModelTransformer(NamedTuple):
    layers: List[VisionModelLocalLayer]


class VisionModelGlobalTransformer(NamedTuple):
    layers: List[VisionModelGlobalLayer]


class VisionModel(NamedTuple):
    transformer: VisionModelTransformer
    global_transformer: VisionModelGlobalTransformer    
    class_embedding: jax.Array
    gated_positional_embedding_embedding: jax.Array
    gated_positional_embedding_gate: jax.Array
    gated_positional_embedding_tile_embedding_weight: jax.Array
    layernorm_post_bias: jax.Array
    layernorm_post_weight: jax.Array
    layernorm_pre_bias: jax.Array
    layernorm_pre_weight: jax.Array
    patch_embedding_weight: jax.Array
    post_tile_positional_embedding_embedding_weight: jax.Array
    post_tile_positional_embedding_gate: jax.Array
    pre_tile_positional_embedding_embedding_weight: jax.Array
    pre_tile_positional_embedding_gate: jax.Array



class LlamaParams(NamedTuple):
    language_model: LangModel
    vision_model: VisionModel
    multi_modal_projector: MultiModalProjector

