import jax
import jax.numpy as jnp
from safetensors.numpy import load_file
from safetensors import safe_open
from model_types import * 
from jax.core import ShapedArray # for loading dummy tensors with no data 

import numpy as np

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
                        tensor = f.get_tensor(key).astype(model_dtype)
                        return jax.device_put(tensor)
                        # would be better in c/rust probably
            raise KeyError(f"Tensor with key '{key}' not found")
        def load_tensors(fmt_key, count):
            mats = []
            for path in paths:
                with safe_open(path, framework="numpy") as f:
                    for i in range(count):
                        key = fmt_key.format(i)
                        mats.append(f.get_tensor(key).astype(model_dtype))
            host_stack = np.stack(mats).astype(model_dtype)
            
            return jax.device_put(jnp.array(mats, dtype=model_dtype))

    if dummy:
        model_params = PixtralModel(
          norm_weight=load_tensor(f"norm.weight"),
          output_weight=load_tensor(f"output.weight"),
          tok_embeddings_weight=load_tensor(f"tok_embeddings.weight"),
          vision_encoder=VisionEncoder(
              ln_pre_weight=load_tensor(f"vision_encoder.ln_pre.weight"),
              patch_conv_weight=load_tensor(f"vision_encoder.patch_conv.weight"),
              vision_encoder_layers=VisionEncoderLayer(
                  attention_wk_weight=load_tensor(f"vision_encoder.transformer.layers.0.attention.wk.weight", count=len(vision_encoder_layers)),
                  attention_wo_weight=load_tensor(f"vision_encoder.transformer.layers.0.attention.wo.weight", count=len(vision_encoder_layers)),
                  attention_wq_weight=load_tensor(f"vision_encoder.transformer.layers.0.attention.wq.weight", count=len(vision_encoder_layers)),
                  attention_wv_weight=load_tensor(f"vision_encoder.transformer.layers.0.attention.wv.weight", count=len(vision_encoder_layers)),
                  attention_norm_weight=load_tensor(f"vision_encoder.transformer.layers.0.attention_norm.weight", count=len(vision_encoder_layers)),
                  feed_forward_w1_weight=load_tensor(f"vision_encoder.transformer.layers.0.feed_forward.w1.weight", count=len(vision_encoder_layers)),
                  feed_forward_w2_weight=load_tensor(f"vision_encoder.transformer.layers.0.feed_forward.w2.weight", count=len(vision_encoder_layers)),
                  feed_forward_w3_weight=load_tensor(f"vision_encoder.transformer.layers.0.feed_forward.w3.weight", count=len(vision_encoder_layers)),
                  ffn_norm_weight=load_tensor(f"vision_encoder.transformer.layers.0.ffn_norm.weight", count=len(vision_encoder_layers)),
              )
          ),
          vision_language_adapter=VisionLanguageAdapter(
              w_in_bias=load_tensor(f"vision_language_adapter.w_in.bias"),
              w_in_weight=load_tensor(f"vision_language_adapter.w_in.weight"),
              w_out_bias=load_tensor(f"vision_language_adapter.w_out.bias"),
              w_out_weight=load_tensor(f"vision_language_adapter.w_out.weight"),
          ),
          transformer=Transformer(
              transformer_layers=TransformerLayer(
                  attention_wk_weight=load_tensor(f"layers.0.attention.wk.weight", count=len(transformer_layers)),
                  attention_wo_weight=load_tensor(f"layers.0.attention.wo.weight", count=len(transformer_layers)),
                  attention_wq_weight=load_tensor(f"layers.0.attention.wq.weight", count=len(transformer_layers)),
                  attention_wv_weight=load_tensor(f"layers.0.attention.wv.weight", count=len(transformer_layers)),
                  attention_norm_weight=load_tensor(f"layers.0.attention_norm.weight", count=len(transformer_layers)),
                  feed_forward_w1_weight=load_tensor(f"layers.0.feed_forward.w1.weight", count=len(transformer_layers)),
                  feed_forward_w2_weight=load_tensor(f"layers.0.feed_forward.w2.weight", count=len(transformer_layers)),
                  feed_forward_w3_weight=load_tensor(f"layers.0.feed_forward.w3.weight", count=len(transformer_layers)),
                  ffn_norm_weight=load_tensor(f"layers.0.ffn_norm.weight", count=len(transformer_layers)),
              )
          ),
        )
    else:
        model_params = PixtralModel(
          norm_weight=load_tensor(f"norm.weight"),
          output_weight=load_tensor(f"output.weight"),
          tok_embeddings_weight=load_tensor(f"tok_embeddings.weight"),
          vision_encoder=VisionEncoder(
              ln_pre_weight=load_tensor(f"vision_encoder.ln_pre.weight"),
              patch_conv_weight=load_tensor(f"vision_encoder.patch_conv.weight"),
              vision_encoder_layers=VisionEncoderLayer(
                  attention_wk_weight=load_tensors("vision_encoder.transformer.layers.{}.attention.wk.weight", count=len(vision_encoder_layers)),
                  attention_wo_weight=load_tensors("vision_encoder.transformer.layers.{}.attention.wo.weight", count=len(vision_encoder_layers)),
                  attention_wq_weight=load_tensors("vision_encoder.transformer.layers.{}.attention.wq.weight", count=len(vision_encoder_layers)),
                  attention_wv_weight=load_tensors("vision_encoder.transformer.layers.{}.attention.wv.weight", count=len(vision_encoder_layers)),
                  attention_norm_weight=load_tensors("vision_encoder.transformer.layers.{}.attention_norm.weight", count=len(vision_encoder_layers)),
                  feed_forward_w1_weight=load_tensors("vision_encoder.transformer.layers.{}.feed_forward.w1.weight", count=len(vision_encoder_layers)),
                  feed_forward_w2_weight=load_tensors("vision_encoder.transformer.layers.{}.feed_forward.w2.weight", count=len(vision_encoder_layers)),
                  feed_forward_w3_weight=load_tensors("vision_encoder.transformer.layers.{}.feed_forward.w3.weight", count=len(vision_encoder_layers)),
                  ffn_norm_weight=load_tensors("vision_encoder.transformer.layers.{}.ffn_norm.weight", count=len(vision_encoder_layers)),
              )
          ),
          vision_language_adapter=VisionLanguageAdapter(
              w_in_bias=load_tensor(f"vision_language_adapter.w_in.bias"),
              w_in_weight=load_tensor(f"vision_language_adapter.w_in.weight"),
              w_out_bias=load_tensor(f"vision_language_adapter.w_out.bias"),
              w_out_weight=load_tensor(f"vision_language_adapter.w_out.weight"),
          ),
          transformer=Transformer(
              transformer_layers=TransformerLayer(
                  attention_wk_weight=load_tensors("layers.{}.attention.wk.weight", count=len(transformer_layers)),
                  attention_wo_weight=load_tensors("layers.{}.attention.wo.weight", count=len(transformer_layers)),
                  attention_wq_weight=load_tensors("layers.{}.attention.wq.weight", count=len(transformer_layers)),
                  attention_wv_weight=load_tensors("layers.{}.attention.wv.weight", count=len(transformer_layers)),
                  attention_norm_weight=load_tensors("layers.{}.attention_norm.weight", count=len(transformer_layers)),
                  feed_forward_w1_weight=load_tensors("layers.{}.feed_forward.w1.weight", count=len(transformer_layers)),
                  feed_forward_w2_weight=load_tensors("layers.{}.feed_forward.w2.weight", count=len(transformer_layers)),
                  feed_forward_w3_weight=load_tensors("layers.{}.feed_forward.w3.weight", count=len(transformer_layers)),
                  ffn_norm_weight=load_tensors("layers.{}.ffn_norm.weight", count=len(transformer_layers)),
              )
          ),
        )


    return model_params
