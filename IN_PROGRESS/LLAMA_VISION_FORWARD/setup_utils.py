import jax
import jax.numpy as jnp

from safetensors.numpy import load_file
from safetensors import safe_open
from llama_types import * # TODO replace glob with declared imports 
from jax import ShapeDtypeStruct
from jax.core import ShapedArray 


# Load params from safetensors paths. returns a dict of jnp arrays
def load_as_jnp_dict(paths: str) -> dict:
    # for now just set all of the namedtuples to super small jax arrays
    model = {}
    for path in paths:
        tensors = load_file(path)
        print("loaded", path)
        for key, tensor in tensors.items():
            model.update({key: tensor})
            print(f"updated {key}: {tensor.shape}, {tensor.dtype}")
        del tensors 
    


def display_shapes(paths: str):
    for path in paths:
        tensors = load_file(path)
        print("loaded", path)
        for key, tensor in tensors.items():
            print(f"{key}: {tensor.shape}, {tensor.dtype}")
        del tensors 



def load_dummy_params(paths: str) -> LlamaParams:
    """
    Outputs a fake LlamaParams loaded from safetensors
    Takes up very little actual memory
    """
    # get config
    # TODO optimize getting config for model
    lang_model_cross_attn_layers = [3, 8, 13, 18, 23, 28, 33, 38]
    lang_model_self_attn_layers = [i for i in range(39+1) if i not in lang_model_cross_attn_layers]
    vision_model_local_layers = list(range(32))
    vision_model_global_layers = list(range(8))

    def load_tensor(key, count=0):
        for path in paths:
            with safe_open(path, framework="numpy") as f:
                if key in f.keys():
                    tensor = f.get_tensor(key).astype("bfloat16")
                    if count > 0:
                        return ShapedArray((count, *tensor.shape), dtype="bfloat16")
                    else:
                        return ShapedArray(tensor.shape, dtype="bfloat16")
        raise KeyError(f"Tensor with key '{key}' not found")

    # OPTIMIZATION: fix ugly param initialization
    llama_params = LlamaParams(
        language_model=LangModel(
            lm_head_weight=load_tensor("language_model.lm_head.weight"),
            model=LangModelModel(
                embed_tokens=load_tensor("language_model.model.embed_tokens.weight"),
                norm_weight=load_tensor("language_model.model.norm.weight"),
                self_attention_layers=LangModelSelfAttentionLayer(
                    input_layernorm_weight=load_tensor(f"language_model.model.layers.0.input_layernorm.weight", len(lang_model_self_attn_layers)),
                    mlp_down_proj_weight=load_tensor(f"language_model.model.layers.0.mlp.down_proj.weight", len(lang_model_self_attn_layers)),
                    mlp_gate_proj_weight=load_tensor(f"language_model.model.layers.0.mlp.gate_proj.weight", len(lang_model_self_attn_layers)),
                    mlp_up_proj_weight=load_tensor(f"language_model.model.layers.0.mlp.up_proj.weight", len(lang_model_self_attn_layers)),
                    post_attention_layernorm_weight=load_tensor(f"language_model.model.layers.0.post_attention_layernorm.weight", len(lang_model_self_attn_layers)),
                    self_attn_k_proj_weight=load_tensor(f"language_model.model.layers.0.self_attn.k_proj.weight", len(lang_model_self_attn_layers)),
                    self_attn_o_proj_weight=load_tensor(f"language_model.model.layers.0.self_attn.o_proj.weight", len(lang_model_self_attn_layers)),
                    self_attn_q_proj_weight=load_tensor(f"language_model.model.layers.0.self_attn.q_proj.weight", len(lang_model_self_attn_layers)),
                    self_attn_v_proj_weight=load_tensor(f"language_model.model.layers.0.self_attn.v_proj.weight", len(lang_model_self_attn_layers)),
                ),
                cross_attention_layers=LangModelCrossAttentionLayer(
                    cross_attn_k_norm_weight=load_tensor(f"language_model.model.layers.3.cross_attn.k_norm.weight", len(lang_model_cross_attn_layers)),
                    cross_attn_k_proj_weight=load_tensor(f"language_model.model.layers.3.cross_attn.k_proj.weight", len(lang_model_cross_attn_layers)),
                    cross_attn_o_proj_weight=load_tensor(f"language_model.model.layers.3.cross_attn.o_proj.weight", len(lang_model_cross_attn_layers)),
                    cross_attn_q_norm_weight=load_tensor(f"language_model.model.layers.3.cross_attn.q_norm.weight", len(lang_model_cross_attn_layers)),
                    cross_attn_q_proj_weight=load_tensor(f"language_model.model.layers.3.cross_attn.q_proj.weight", len(lang_model_cross_attn_layers)),
                    cross_attn_v_proj_weight=load_tensor(f"language_model.model.layers.3.cross_attn.v_proj.weight", len(lang_model_cross_attn_layers)),
                    cross_attn_attn_gate=load_tensor(f"language_model.model.layers.3.cross_attn_attn_gate", len(lang_model_cross_attn_layers)),
                    cross_attn_mlp_gate=load_tensor(f"language_model.model.layers.3.cross_attn_mlp_gate", len(lang_model_cross_attn_layers)),
                    input_layernorm_weight=load_tensor(f"language_model.model.layers.3.input_layernorm.weight", len(lang_model_cross_attn_layers)),
                    mlp_down_proj_weight=load_tensor(f"language_model.model.layers.3.mlp.down_proj.weight", len(lang_model_cross_attn_layers)),
                    mlp_gate_proj_weight=load_tensor(f"language_model.model.layers.3.mlp.gate_proj.weight", len(lang_model_cross_attn_layers)),
                    mlp_up_proj_weight=load_tensor(f"language_model.model.layers.3.mlp.up_proj.weight", len(lang_model_cross_attn_layers)),
                    post_attention_layernorm_weight=load_tensor(f"language_model.model.layers.3.post_attention_layernorm.weight", len(lang_model_cross_attn_layers)),
                ),
            ),
        ),
        vision_model=VisionModel(
            transformer=VisionModelTransformer(
                layers=VisionModelLocalLayer(
                    input_layernorm_bias=load_tensor(f"vision_model.transformer.layers.0.input_layernorm.bias", len(vision_model_local_layers)),
                    input_layernorm_weight=load_tensor(f"vision_model.transformer.layers.0.input_layernorm.weight", len(vision_model_local_layers)),
                    mlp_fc1_bias=load_tensor(f"vision_model.transformer.layers.0.mlp.fc1.bias", len(vision_model_local_layers)),
                    mlp_fc1_weight=load_tensor(f"vision_model.transformer.layers.0.mlp.fc1.weight", len(vision_model_local_layers)),
                    mlp_fc2_bias=load_tensor(f"vision_model.transformer.layers.0.mlp.fc2.bias", len(vision_model_local_layers)),
                    mlp_fc2_weight=load_tensor(f"vision_model.transformer.layers.0.mlp.fc2.weight", len(vision_model_local_layers)),
                    post_attention_layernorm_bias=load_tensor(f"vision_model.transformer.layers.0.post_attention_layernorm.bias", len(vision_model_local_layers)),
                    post_attention_layernorm_weight=load_tensor(f"vision_model.transformer.layers.0.post_attention_layernorm.weight", len(vision_model_local_layers)),
                    self_attn_k_proj_weight=load_tensor(f"vision_model.transformer.layers.0.self_attn.k_proj.weight", len(vision_model_local_layers)),
                    self_attn_o_proj_weight=load_tensor(f"vision_model.transformer.layers.0.self_attn.o_proj.weight", len(vision_model_local_layers)),
                    self_attn_q_proj_weight=load_tensor(f"vision_model.transformer.layers.0.self_attn.q_proj.weight", len(vision_model_local_layers)),
                    self_attn_v_proj_weight=load_tensor(f"vision_model.transformer.layers.0.self_attn.v_proj.weight", len(vision_model_local_layers)),
                ),
            ),
            global_transformer=VisionModelGlobalTransformer(
                layers=VisionModelGlobalLayer(
                    gate_attn=load_tensor(f"vision_model.global_transformer.layers.0.gate_attn", len(vision_model_global_layers)),
                    gate_ffn=load_tensor(f"vision_model.global_transformer.layers.0.gate_ffn", len(vision_model_global_layers)),
                    input_layernorm_bias=load_tensor(f"vision_model.global_transformer.layers.0.input_layernorm.bias", len(vision_model_global_layers)),
                    input_layernorm_weight=load_tensor(f"vision_model.global_transformer.layers.0.input_layernorm.weight", len(vision_model_global_layers)),
                    mlp_fc1_bias=load_tensor(f"vision_model.global_transformer.layers.0.mlp.fc1.bias", len(vision_model_global_layers)),
                    mlp_fc1_weight=load_tensor(f"vision_model.global_transformer.layers.0.mlp.fc1.weight", len(vision_model_global_layers)),
                    mlp_fc2_bias=load_tensor(f"vision_model.global_transformer.layers.0.mlp.fc2.bias", len(vision_model_global_layers)),
                    mlp_fc2_weight=load_tensor(f"vision_model.global_transformer.layers.0.mlp.fc2.weight", len(vision_model_global_layers)),
                    post_attention_layernorm_bias=load_tensor(f"vision_model.global_transformer.layers.0.post_attention_layernorm.bias", len(vision_model_global_layers)),
                    post_attention_layernorm_weight=load_tensor(f"vision_model.global_transformer.layers.0.post_attention_layernorm.weight", len(vision_model_global_layers)),
                    self_attn_k_proj_weight=load_tensor(f"vision_model.global_transformer.layers.0.self_attn.k_proj.weight", len(vision_model_global_layers)),
                    self_attn_o_proj_weight=load_tensor(f"vision_model.global_transformer.layers.0.self_attn.o_proj.weight", len(vision_model_global_layers)),
                    self_attn_q_proj_weight=load_tensor(f"vision_model.global_transformer.layers.0.self_attn.q_proj.weight", len(vision_model_global_layers)),
                    self_attn_v_proj_weight=load_tensor(f"vision_model.global_transformer.layers.0.self_attn.v_proj.weight", len(vision_model_global_layers)),
                ),
            ),
            class_embedding=load_tensor("vision_model.class_embedding"),
            gated_positional_embedding_embedding=load_tensor("vision_model.gated_positional_embedding.embedding"),
            gated_positional_embedding_gate=load_tensor("vision_model.gated_positional_embedding.gate"),
            gated_positional_embedding_tile_embedding_weight=load_tensor("vision_model.gated_positional_embedding.tile_embedding.weight"),
            layernorm_post_bias=load_tensor("vision_model.layernorm_post.bias"),
            layernorm_post_weight=load_tensor("vision_model.layernorm_post.weight"),
            layernorm_pre_bias=load_tensor("vision_model.layernorm_pre.bias"),
            layernorm_pre_weight=load_tensor("vision_model.layernorm_pre.weight"),
            patch_embedding_weight=load_tensor("vision_model.patch_embedding.weight"),
            post_tile_positional_embedding_embedding_weight=load_tensor("vision_model.post_tile_positional_embedding.embedding.weight"),
            post_tile_positional_embedding_gate=load_tensor("vision_model.post_tile_positional_embedding.gate"),
            pre_tile_positional_embedding_embedding_weight=load_tensor("vision_model.pre_tile_positional_embedding.embedding.weight"),
            pre_tile_positional_embedding_gate=load_tensor("vision_model.pre_tile_positional_embedding.gate"),
        ),
        multi_modal_projector=MultiModalProjector(
            weight=load_tensor("multi_modal_projector.weight"),
            bias=load_tensor("multi_modal_projector.bias"),
        ),
    )

    return llama_params



# import json
# config = json.load(f"{llama_path}/config.json")
def load_model_params(paths: str) -> LlamaParams:
    """
    Outputs a LlamaParams loaded from safetensors
    """
    # get config
    # TODO optimize getting config for model
    lang_model_cross_attn_layers = [3, 8, 13, 18, 23, 28, 33, 38]
    lang_model_self_attn_layers = [i for i in range(39+1) if i not in lang_model_cross_attn_layers]
    vision_model_local_layers = list(range(32))
    vision_model_global_layers = list(range(8))

    def load_tensor(key):
        for path in paths:
            with safe_open(path, framework="numpy") as f:
                if key in f.keys():
                    tensor = f.get_tensor(key).astype("bfloat16")
                    return jax.device_put(tensor)
        raise KeyError(f"Tensor with key '{key}' not found")

    # OPTIMIZATION: fix ugly param initialization
    llama_params = LlamaParams(
        language_model=LangModel(
            lm_head_weight=load_tensor("language_model.lm_head.weight"),
            model=LangModelModel(
                embed_tokens=load_tensor("language_model.model.embed_tokens.weight"),
                norm_weight=load_tensor("language_model.model.norm.weight"),
                self_attention_layers=LangModelSelfAttentionLayer(
                    input_layernorm_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.input_layernorm.weight") for i in lang_model_self_attn_layers]),
                    mlp_down_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.mlp.down_proj.weight") for i in lang_model_self_attn_layers]),
                    mlp_gate_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.mlp.gate_proj.weight") for i in lang_model_self_attn_layers]),
                    mlp_up_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.mlp.up_proj.weight") for i in lang_model_self_attn_layers]),
                    post_attention_layernorm_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.post_attention_layernorm.weight") for i in lang_model_self_attn_layers]),
                    self_attn_k_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.self_attn.k_proj.weight") for i in lang_model_self_attn_layers]),
                    self_attn_o_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.self_attn.o_proj.weight") for i in lang_model_self_attn_layers]),
                    self_attn_q_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.self_attn.q_proj.weight") for i in lang_model_self_attn_layers]),
                    self_attn_v_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.self_attn.v_proj.weight") for i in lang_model_self_attn_layers]),
                ),
                cross_attention_layers=LangModelCrossAttentionLayer(
                    cross_attn_k_norm_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.cross_attn.k_norm.weight") for i in lang_model_cross_attn_layers]),
                    cross_attn_k_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.cross_attn.k_proj.weight") for i in lang_model_cross_attn_layers]),
                    cross_attn_o_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.cross_attn.o_proj.weight") for i in lang_model_cross_attn_layers]),
                    cross_attn_q_norm_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.cross_attn.q_norm.weight") for i in lang_model_cross_attn_layers]),
                    cross_attn_q_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.cross_attn.q_proj.weight") for i in lang_model_cross_attn_layers]),
                    cross_attn_v_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.cross_attn.v_proj.weight") for i in lang_model_cross_attn_layers]),
                    cross_attn_attn_gate=jnp.array([load_tensor(f"language_model.model.layers.{i}.cross_attn_attn_gate") for i in lang_model_cross_attn_layers]),
                    cross_attn_mlp_gate=jnp.array([load_tensor(f"language_model.model.layers.{i}.cross_attn_mlp_gate") for i in lang_model_cross_attn_layers]),
                    input_layernorm_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.input_layernorm.weight") for i in lang_model_cross_attn_layers]),
                    mlp_down_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.mlp.down_proj.weight") for i in lang_model_cross_attn_layers]),
                    mlp_gate_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.mlp.gate_proj.weight") for i in lang_model_cross_attn_layers]),
                    mlp_up_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.mlp.up_proj.weight") for i in lang_model_cross_attn_layers]),
                    post_attention_layernorm_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.post_attention_layernorm.weight") for i in lang_model_cross_attn_layers]),
                ),
            ),
        ),
        vision_model=VisionModel(
            transformer=VisionModelTransformer(
                layers=VisionModelLocalLayer(
                    input_layernorm_bias=jnp.array([load_tensor(f"vision_model.transformer.layers.{i}.input_layernorm.bias") for i in vision_model_local_layers]),
                    input_layernorm_weight=jnp.array([load_tensor(f"vision_model.transformer.layers.{i}.input_layernorm.weight") for i in vision_model_local_layers]),
                    mlp_fc1_bias=jnp.array([load_tensor(f"vision_model.transformer.layers.{i}.mlp.fc1.bias") for i in vision_model_local_layers]),
                    mlp_fc1_weight=jnp.array([load_tensor(f"vision_model.transformer.layers.{i}.mlp.fc1.weight") for i in vision_model_local_layers]),
                    mlp_fc2_bias=jnp.array([load_tensor(f"vision_model.transformer.layers.{i}.mlp.fc2.bias") for i in vision_model_local_layers]),
                    mlp_fc2_weight=jnp.array([load_tensor(f"vision_model.transformer.layers.{i}.mlp.fc2.weight") for i in vision_model_local_layers]),
                    post_attention_layernorm_bias=jnp.array([load_tensor(f"vision_model.transformer.layers.{i}.post_attention_layernorm.bias") for i in vision_model_local_layers]),
                    post_attention_layernorm_weight=jnp.array([load_tensor(f"vision_model.transformer.layers.{i}.post_attention_layernorm.weight") for i in vision_model_local_layers]),
                    self_attn_k_proj_weight=jnp.array([load_tensor(f"vision_model.transformer.layers.{i}.self_attn.k_proj.weight") for i in vision_model_local_layers]),
                    self_attn_o_proj_weight=jnp.array([load_tensor(f"vision_model.transformer.layers.{i}.self_attn.o_proj.weight") for i in vision_model_local_layers]),
                    self_attn_q_proj_weight=jnp.array([load_tensor(f"vision_model.transformer.layers.{i}.self_attn.q_proj.weight") for i in vision_model_local_layers]),
                    self_attn_v_proj_weight=jnp.array([load_tensor(f"vision_model.transformer.layers.{i}.self_attn.v_proj.weight") for i in vision_model_local_layers]),
                ),
            ),
            global_transformer=VisionModelGlobalTransformer(
                layers=VisionModelGlobalLayer(
                    gate_attn=jnp.array([load_tensor(f"vision_model.global_transformer.layers.{i}.gate_attn") for i in vision_model_global_layers]),
                    gate_ffn=jnp.array([load_tensor(f"vision_model.global_transformer.layers.{i}.gate_ffn") for i in vision_model_global_layers]),
                    input_layernorm_bias=jnp.array([load_tensor(f"vision_model.global_transformer.layers.{i}.input_layernorm.bias") for i in vision_model_global_layers]),
                    input_layernorm_weight=jnp.array([load_tensor(f"vision_model.global_transformer.layers.{i}.input_layernorm.weight") for i in vision_model_global_layers]),
                    mlp_fc1_bias=jnp.array([load_tensor(f"vision_model.global_transformer.layers.{i}.mlp.fc1.bias") for i in vision_model_global_layers]),
                    mlp_fc1_weight=jnp.array([load_tensor(f"vision_model.global_transformer.layers.{i}.mlp.fc1.weight") for i in vision_model_global_layers]),
                    mlp_fc2_bias=jnp.array([load_tensor(f"vision_model.global_transformer.layers.{i}.mlp.fc2.bias") for i in vision_model_global_layers]),
                    mlp_fc2_weight=jnp.array([load_tensor(f"vision_model.global_transformer.layers.{i}.mlp.fc2.weight") for i in vision_model_global_layers]),
                    post_attention_layernorm_bias=jnp.array([load_tensor(f"vision_model.global_transformer.layers.{i}.post_attention_layernorm.bias") for i in vision_model_global_layers]),
                    post_attention_layernorm_weight=jnp.array([load_tensor(f"vision_model.global_transformer.layers.{i}.post_attention_layernorm.weight") for i in vision_model_global_layers]),
                    self_attn_k_proj_weight=jnp.array([load_tensor(f"vision_model.global_transformer.layers.{i}.self_attn.k_proj.weight") for i in vision_model_global_layers]),
                    self_attn_o_proj_weight=jnp.array([load_tensor(f"vision_model.global_transformer.layers.{i}.self_attn.o_proj.weight") for i in vision_model_global_layers]),
                    self_attn_q_proj_weight=jnp.array([load_tensor(f"vision_model.global_transformer.layers.{i}.self_attn.q_proj.weight") for i in vision_model_global_layers]),
                    self_attn_v_proj_weight=jnp.array([load_tensor(f"vision_model.global_transformer.layers.{i}.self_attn.v_proj.weight") for i in vision_model_global_layers]),
                ),
            ),
            class_embedding=load_tensor("vision_model.class_embedding"),
            gated_positional_embedding_embedding=load_tensor("vision_model.gated_positional_embedding.embedding"),
            gated_positional_embedding_gate=load_tensor("vision_model.gated_positional_embedding.gate"),
            gated_positional_embedding_tile_embedding_weight=load_tensor("vision_model.gated_positional_embedding.tile_embedding.weight"),
            layernorm_post_bias=load_tensor("vision_model.layernorm_post.bias"),
            layernorm_post_weight=load_tensor("vision_model.layernorm_post.weight"),
            layernorm_pre_bias=load_tensor("vision_model.layernorm_pre.bias"),
            layernorm_pre_weight=load_tensor("vision_model.layernorm_pre.weight"),
            patch_embedding_weight=load_tensor("vision_model.patch_embedding.weight"),
            post_tile_positional_embedding_embedding_weight=load_tensor("vision_model.post_tile_positional_embedding.embedding.weight"),
            post_tile_positional_embedding_gate=load_tensor("vision_model.post_tile_positional_embedding.gate"),
            pre_tile_positional_embedding_embedding_weight=load_tensor("vision_model.pre_tile_positional_embedding.embedding.weight"),
            pre_tile_positional_embedding_gate=load_tensor("vision_model.pre_tile_positional_embedding.gate"),
        ),
        multi_modal_projector=MultiModalProjector(
            weight=load_tensor("multi_modal_projector.weight"),
            bias=load_tensor("multi_modal_projector.bias"),
        ),
    )

    return llama_params



# import json
# config = json.load(f"{llama_path}/config.json")
def load_text_model_params(paths: str) -> LlamaParams:
    """
    Outputs a LlamaParams loaded from safetensors
    """
    # get config
    # TODO optimize getting config for model
    lang_model_cross_attn_layers = [3, 8, 13, 18, 23, 28, 33, 38]
    lang_model_self_attn_layers = [i for i in range(39+1) if i not in lang_model_cross_attn_layers]

    def load_tensor(key):
        for path in paths:
            with safe_open(path, framework="numpy") as f:
                if key in f.keys():
                    tensor = f.get_tensor(key).astype("bfloat16")
                    return jax.device_put(tensor)
        raise KeyError(f"Tensor with key '{key}' not found")

    llama_params = LlamaParams(
        language_model=LangModel(
            lm_head_weight=load_tensor("language_model.lm_head.weight"),
            model=LangModelModel(
                embed_tokens=load_tensor("language_model.model.embed_tokens.weight"),
                norm_weight=load_tensor("language_model.model.norm.weight"),
                self_attention_layers=LangModelSelfAttentionLayer(
                    input_layernorm_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.input_layernorm.weight") for i in lang_model_self_attn_layers]),
                    mlp_down_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.mlp.down_proj.weight") for i in lang_model_self_attn_layers]),
                    mlp_gate_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.mlp.gate_proj.weight") for i in lang_model_self_attn_layers]),
                    mlp_up_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.mlp.up_proj.weight") for i in lang_model_self_attn_layers]),
                    post_attention_layernorm_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.post_attention_layernorm.weight") for i in lang_model_self_attn_layers]),
                    self_attn_k_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.self_attn.k_proj.weight") for i in lang_model_self_attn_layers]),
                    self_attn_o_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.self_attn.o_proj.weight") for i in lang_model_self_attn_layers]),
                    self_attn_q_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.self_attn.q_proj.weight") for i in lang_model_self_attn_layers]),
                    self_attn_v_proj_weight=jnp.array([load_tensor(f"language_model.model.layers.{i}.self_attn.v_proj.weight") for i in lang_model_self_attn_layers]),
                ),
                cross_attention_layers=None,
            )
        ),
        vision_model=None,
        multi_modal_projector=None,
    )

    return llama_params


