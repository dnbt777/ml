import jax
import jax.random as jrand
import jax.numpy as jnp

from typing import NamedTuple

class FullyConnectedLayer(NamedTuple):
  w: jax.Array
  b: jax.Array

class CNNParams(NamedTuple):
  conv1: jax.Array
  conv2: jax.Array
  conv3: jax.Array
  conv4: jax.Array
  fc1: FullyConnectedLayer
  fc2: FullyConnectedLayer
   

def init_cnn_params(rolling_key, output_shape, dtype=jnp.float32):
  # shape: (1, 28, 28)
  # this is (layers, height, width)
  # for example, an RGB image would be (3, height, width)
  
  # (out_channels, in_channels, kernel_row, kernel_col)
  # 4x4 kernel. 1 input (mnist), 3 outputs
  # here we use mode="same", so the height/width dont change
  # we have 3 kernels, so we will have 3 channels in the next layer. 
  conv1 = jax.nn.initializers.xavier_uniform()(rolling_key, (3, 1, 4, 4)).astype(dtype)
  rolling_key, _ = jrand.split(rolling_key, 2)
  # current_shape = (3, 28, 28) 

  conv2 = jax.nn.initializers.xavier_uniform()(rolling_key, (3, 3, 4, 4)).astype(dtype)
  rolling_key, _ = jrand.split(rolling_key, 2)
  # current_shape = (3, 28, 28) # this conv will have a skip connection and mode will be 'same'

  # Here we maxpool
  # We choose a maxpool window of 2x2, so our height and width shrink by 1 each
  # current_shape = (3, 27, 27) # (3, 28-1, 28-1)

  # here we have 7 kernels. each one processes all three channels. so the next output has 7 channels
  # the number of channels in the next layer is the number of kernels in the current layer
  conv3 = jax.nn.initializers.xavier_uniform()(rolling_key, (7, 3, 10, 10)).astype(dtype)
  rolling_key, _ = jrand.split(rolling_key, 2)
  # current_shape = (7, 27, 27)

  conv4 = jax.nn.initializers.xavier_uniform()(rolling_key, (7, 7, 10, 10)).astype(dtype)
  rolling_key, _ = jrand.split(rolling_key, 2)
  # current_shape = (7, 27, 27)
  
  # shape: (7, 27, 27) 
  # => ravel => (7*27*27,) is the input shape to the weight
  # we choose 128 activations for the next layer size
  layer_input_size = 7*27*27
  layer_output_size = 128
  # since its xW for batching, W shape=(input, output)
  fc1 = FullyConnectedLayer(
    w = (jrand.normal(rolling_key, shape=(layer_input_size, layer_output_size))*jnp.sqrt(2 / (layer_input_size + layer_output_size))).astype(dtype),
    b = jnp.zeros(shape=(layer_output_size,)).astype(dtype)
  )
  rolling_key, _ = jrand.split(rolling_key, 2)
  
  layer_input_size = layer_output_size
  fc2 = FullyConnectedLayer( 
      w = (jrand.normal(rolling_key, shape=(layer_input_size, output_shape[0])) * jnp.sqrt(2 / (layer_input_size + output_shape[0]))).astype(dtype),
      b = jnp.zeros(shape=output_shape).astype(dtype)
  )
  rolling_key, _ = jrand.split(rolling_key, 2)
  
  cnn_params = CNNParams(
    conv1=conv1,
    conv2=conv2,
    conv3=conv3,
    conv4=conv4,
    fc1=fc1,
    fc2=fc2
  )

  return cnn_params


