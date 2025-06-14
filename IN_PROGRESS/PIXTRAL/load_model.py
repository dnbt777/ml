import jax
import jax.numpy as jnp
from safetensors.numpy import load_file
from safetensors import safe_open
from model_types import * 
from jax.core import ShapedArray # for loading dummy tensors with no data 

#|=====================================>
#|_______ load_model.py _______________
#|
#|
#|  loads pixtral from safetensors
#| 
#| 
#|======================>==============>>



def display_shapes(paths: str):
    for path in paths:
        tensors = load_file(path)
        print("loaded", path)
        for key, tensor in tensors.items():
            print(f"{key}: {tensor.shape}, {tensor.dtype}")
        del tensors 



def load_params(paths: str, dummy: bool = False) -> PixtralModel:
    """
    Loads pixtral's params

    dummy:
        - Outputs a fake PixtralModel loaded from safetensors
        - Takes up very little actual memory. Used for development on my laptop ($0) versus a cloud GPU ($0.40/hr)
    """
    vision_encoder_layers = list(range(23+1)) # [0, 23]
    transformer_layers = list(range(39+1)) # [0, 39]
    model_dtype = "bfloat16"

    if dummy:
        def load_tensor(key, count=0):
            for path in paths:
                with safe_open(path, framework="numpy") as f:
                    if key in f.keys():
                        tensor = f.get_tensor(key).astype(model_dtype)
                        if count > 0:
                            return ShapedArray((count, *tensor.shape), dtype=model_dtype)
                        else:
                            return ShapedArray(tensor.shape, dtype=model_dtype)
            raise KeyError(f"Tensor with key '{key}' not found")
    else:
        def load_tensor(key):
            for path in paths:
                with safe_open(path, framework="numpy") as f:
                    if key in f.keys():
                        tensor = f.get_tensor(key).astype("bfloat16")
                        return jax.device_put(tensor)
                        # would be better in c/rust probably
            raise KeyError(f"Tensor with key '{key}' not found")

    model_params = PixtralModel(
      norm_weight=load_tensor(f"norm.weight"),
      output_weight=load_tensor(f"output.weight"),
      tok_embeddings_weight=load_tensor(f"tok_embeddings.weight"),
      vision_encoder=VisionEncoder(
          ln_pre_weight=load_tensor(f"vision_encoder.ln_pre.weight"),
          patch_conv_weight=load_tensor(f"vision_encoder.patch_conv.weight"),
          vision_encoder_layers=VisionEncoderLayer(
              attention_wk_weight=jnp.array([load_tensor(f"vision_encoder.transformer.layers.{i}.attention_wk_weight") for i in vision_encoder_layers]),
              attention_wo_weight=jnp.array([load_tensor(f"vision_encoder.transformer.layers.{i}.attention_wo_weight") for i in vision_encoder_layers]),
              attention_wq_weight=jnp.array([load_tensor(f"vision_encoder.transformer.layers.{i}.attention_wq_weight") for i in vision_encoder_layers]),
              attention_wv_weight=jnp.array([load_tensor(f"vision_encoder.transformer.layers.{i}.attention_wv_weight") for i in vision_encoder_layers]),
              attention_norm_weight=jnp.array([load_tensor(f"vision_encoder.transformer.layers.{i}.attention_norm_weight") for i in vision_encoder_layers]),
              feed_forward_w1_weight=jnp.array([load_tensor(f"vision_encoder.transformer.layers.{i}.feed_forward_w1_weight") for i in vision_encoder_layers]),
              feed_forward_w2_weight=jnp.array([load_tensor(f"vision_encoder.transformer.layers.{i}.feed_forward_w2_weight") for i in vision_encoder_layers]),
              feed_forward_w3_weight=jnp.array([load_tensor(f"vision_encoder.transformer.layers.{i}.feed_forward_w3_weight") for i in vision_encoder_layers]),
              ffn_norm_weight=jnp.array([load_tensor(f"vision_encoder.transformer.layers.{i}.ffn_norm_weight") for i in vision_encoder_layers]),
          )
      ),
      vision_language_adapter=VisionLanguageAdapter(
          w_in_bias=load_tensor(f"vision_language_adapter.w_in.bias"),
          w_in_weight=load_tensor(f"vision_language_adapter.w_in.weight"),
          w_out_bias=load_tensor(f"vvision_language_adapter.w_out.bias"),
          w_out_weight=load_tensor(f"vision_language_adapter.w_out.weight"),
      ),
      transformer=Transformer(
          transformer_layers=TransformerLayer(
              attention_wk_weight=jnp.array([load_tensor(f"layers.{i}.attention_wk_weight") for i in transformer_layers]),
              attention_wo_weight=jnp.array([load_tensor(f"layers.{i}.attention_wo_weight") for i in transformer_layers]),
              attention_wq_weight=jnp.array([load_tensor(f"layers.{i}.attention_wq_weight") for i in transformer_layers]),
              attention_wv_weight=jnp.array([load_tensor(f"layers.{i}.attention_wv_weight") for i in transformer_layers]),
              attention_norm_weight=jnp.array([load_tensor(f"layers.{i}.attention_norm_weight") for i in transformer_layers]),
              feed_forward_w1_weight=jnp.array([load_tensor(f"layers.{i}.feed_forward_w1_weight") for i in transformer_layers]),
              feed_forward_w2_weight=jnp.array([load_tensor(f"layers.{i}.feed_forward_w2_weight") for i in transformer_layers]),
              feed_forward_w3_weight=jnp.array([load_tensor(f"layers.{i}.feed_forward_w3_weight") for i in transformer_layers]),
              ffn_norm_weight=jnp.array([load_tensor(f"layers.{i}.ffn_norm_weight") for i in transformer_layers]),
          )
      ),
    )

    return model_params
