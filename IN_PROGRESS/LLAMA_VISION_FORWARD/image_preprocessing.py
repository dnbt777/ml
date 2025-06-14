import jax
import jax.numpy as jnp
from einops import rearrange
from PIL import Image
from forward_utils import (
    RMSnorm, RMSnorm_bias,
    layer_norm,
    feed_forward, rope, rope_channel,
    vision_model_local_feed_forward, vision_model_global_feed_forward
)
from llama_types import (
  Text, Tokens, TensorBTC
)
from llama_types import *
from typing import Tuple
import PIL
# many redundant imports here

import tdiff


# https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py


# https://github.com/huggingface/transformers/blob/593e29c5e2a9b17baec010e8dc7c1431fed6e841/src/transformers/models/mllama/image_processing_mllama.py#L496
def convert_img_to_rgb(image:PIL.Image.Image) -> PIL.Image.Image:
    if image.mode == "RGB":
        return image

    alpha_img = image.convert("RGBA")
    background = Image.new("RGBA", alpha_img.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, alpha_img)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


def preprocess_image(image: jax.Array) -> jax.Array:
  rescale_factor = 0.00392156862745098 # 1 / 255.0
  #rescale_factor = jnp.array([1.0], dtype=image.dtype) / jnp.array([255.0], dtype=image.dtype)
  image = image*rescale_factor # doesnt match vllm unless op takes place in float64
  image = image.astype(jnp.float32) # without this line, doesnt match vllm...
  # the multiplication results in a float64 array that we downcast to float32. this gives an exact match
  # with vllm.
  tdiff.capture(image, name="preprocessing_post_rescale.img", project="jax")
  image_mean = jnp.array([
    0.48145466,
    0.4578275,
    0.40821073
  ]).astype(image.dtype)
  image_std = jnp.array([
    0.26862954,
    0.26130258,
    0.27577711
  ]).astype(image.dtype) # maybe try converting to float64? doing the op is float64? idk
  tdiff.capture(image, name="preprocessing_pre_norm.img", project="jax")
  tdiff.capture(image_mean, name="preprocessing_pre_norm.mean", project="jax")
  tdiff.capture(image_std, name="preprocessing_pre_norm.std", project="jax")
  #image = (
  #    (image.T - image_mean)/image_std
  #).T
  image = jnp.transpose(image)
  image = image - image_mean
  image = image.astype(jnp.float64) / image_std # in order to match VLLM, mul ops must be done in float64. but add/sub must NOT be done in float64
  image = jnp.transpose(image)
  image = image.astype(jnp.float32)
  tdiff.capture(image, name="preprocessing_post_norm.img", project="jax")
  return image


# img -> (h, w) -> (p, p) -> 2D array of patches(patch = 2D array of pixels)
def image_to_tiles(image: Image, tile_resolution) -> jax.Array:
    tile_width, tile_height = tile_resolution
    max_tiles = 4 # hardcoded for now
    ## get aspect ratio
    width, height = image.size
    aspect_ratio = int(jnp.ceil(height/min(width, height))), int(jnp.ceil(width/min(width, height))) # convention here is (h, w)
    if aspect_ratio == (1, 1) and (width > tile_width or height > tile_height):
        aspect_ratio = (2, 2)
    
    ## get aspect ratio, scale, and aspect ratio id
    # https://github.com/huggingface/transformers/blob/716819b8309324302e00a3488a3c3d6faa427f79/src/transformers/models/mllama/image_processing_mllama.py#L71
    aspect_ratios = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (3, 1), (4, 1)] # 9 is 'none'
    # https://github.com/huggingface/transformers/blob/716819b8309324302e00a3488a3c3d6faa427f79/src/transformers/models/mllama/image_processing_mllama.py#L134
    scaling_options = [
        min(
            tile_height*canvas_height/height, # (pixels/tile x tiles) / pixels 
            tile_width*canvas_width/width     # (pixels/tile x tiles) / pixels 
        )
        for canvas_height, canvas_width in aspect_ratios   # convention here is (h, w)
    ] # min scaling needed for each aspect ratio to fill its tile canvas
    # prefer upscaling. pick the smallest option, if there is one.
    upscaling_options = [scale for scale in scaling_options if scale >= 1]
    if len(upscaling_options) > 0:
        scale = min(upscaling_options)
    # otherwise pick the largest downscaling (minimizing changes)
    else:
        downscaling_options = [scale for scale in scaling_options if scale < 1]
        scale = max(scaling_options)
    # pick the canvas size with the smallest area
    smallest_area, best_idx = 100000000, None
    for idx, scaling_option in enumerate(scaling_options):
        if scaling_option != scale:
            continue
        else:
            tiles_h, tiles_w = aspect_ratios[idx]
            tile_width, tile_height = tile_resolution
            area = tiles_h*tile_height + tiles_w*tile_width
            if area < smallest_area:
                best_idx, smallest_area = idx, area
    aspect_ratio = aspect_ratios[best_idx]
    # https://github.com/huggingface/transformers/blob/716819b8309324302e00a3488a3c3d6faa427f79/src/transformers/models/mllama/image_processing_mllama.py#L451
    aspect_ratio_id = aspect_ratios.index(aspect_ratio) + 1
    # https://github.com/huggingface/transformers/blob/e3b70b0d1c15c87ba2010b00830fbd92b2c50252/src/transformers/models/mllama/image_processing_mllama.py#L314
    aspect_ratio_mask = jnp.zeros((4,))
    for idx in range(smallest_area):
        aspect_ratio_mask = aspect_ratio_mask.at[idx].set(1) # leave unused for now. in future optimize 1) for fast loading and 2) so batches have same tile count

    # scale to tiles
    tile_width, tile_height = tile_resolution
    canvas_width, canvas_height = tile_width*aspect_ratio[1], tile_height*aspect_ratio[0]

    new_width, new_height = int(scale*width), int(scale*height)
    # https://github.com/huggingface/transformers/blob/716819b8309324302e00a3488a3c3d6faa427f79/src/transformers/models/mllama/image_processing_mllama.py#L840
    image = image.resize((new_width, new_height), resample=Image.BILINEAR) # PIL.Image.BILINEAR. default is BICUBIC, but thats not whats used here
    
    # create canvas of 0s in the shape of the tiles
    image = jnp.array(image) # go from (w, h) convention (images) to (h, w) convention (linalg arrays)
    image = rearrange(image, "H W C -> C H W")
    tdiff.capture(image, name="preprocessing_post_resize.img", project="jax")
    canvas = jnp.zeros((3, canvas_height, canvas_width), dtype=image.dtype) # doesnt work with batches rn

    # add img to the top left corner
    canvas = canvas.at[:, 0:new_height, 0:new_width].set(image)
    #canvas = canvas.astype(jnp.float32)
    tdiff.capture(canvas, name="preprocessing_post_pad.img", project="jax")

    # return tiles
    pixel_values = preprocess_image(canvas)
    tiles = rearrange(pixel_values, "C (Th h) (Tw w)-> (Th Tw) C h w", Th=int(aspect_ratio[0]), Tw=int(aspect_ratio[1]))
    # pad tiles to 4
    tile_count = aspect_ratio[0]*aspect_ratio[1]
    tile_canvas = jnp.zeros((max_tiles, 3, tile_height, tile_width), dtype=tiles.dtype)
    tile_canvas = tile_canvas.at[0:tile_count].set(tiles)
    print("ASPECT RATIO", aspect_ratio)
    #print("TILE 0", tile_canvas[0])
    #print("TILE 1", tile_canvas[1])
    #print("TILE 2", tile_canvas[2])
    #print("TILE 3", tile_canvas[3])

    tdiff.capture(image, name="preprocessing_post_tilesplit.img", project="vllm")
    
    return tile_canvas, aspect_ratio_id
