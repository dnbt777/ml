###
# deep q agent
# used to test train and 'inference' setup.
# input: SolarSystem
# output: momentum shift (soft capped with tanh)

import jax
import jax.numpy as jnp
import jax.random as jrand
from typing import NamedTuple, List


# SoA
class DQNLayer(NamedTuple):
  weight : jax.Array
  bias : jax.Array

class DQNParams(NamedTuple):
  wi : jax.Array
  bi :jax.Array
  wo : jax.Array
  bo : jax.Array
  hidden_layers : List[DQNLayer] # do it this way for scanning



def init_deep_q_net(hidden_layers, hidden_size, input_size, output_size):
  hidden_shape_w = (hidden_layers, hidden_size, hidden_size)
  hidden_shape_b = (hidden_layers, hidden_size)
  input_shape_w = (input_size, hidden_size)
  input_shape_b = (hidden_size,)
  output_shape_w = (hidden_size, input_size)
  output_shape_b = (output_size)
  return DQNParams(
    layers=DQNLayer(
      weight=jax.nn.initializers.glorot_uniform(hidden_shape_w),
      bias=jax.nn.initializers.glorot_uniform(hidden_shape_b)
    ),
    wi = jax.nn.initializers.glorot_uniform(input_shape_w),
    bi = jax.nn.initializers.glorot_uniform(input_shape_b),
    wo = jax.nn.initializers.glorot_uniform(output_shape_w),
    bo = jax.nn.initializers.glorot_uniform(output_shape_b)
  )



# move to environment or environment_utils idk
def concat_current_state(solar_system_batch):
  return jax.lax.concatenate(solar_system_batch.position, solar_system_batch.momentum, solar_system_batch.mass)


# init model params(input=total, hidden layers, output=6 (udlrbf for now))


# pretrained. no epsilon
def agent_forward(model_params : DQNParams, current_state):
  # input:
  # 4 bodies:
  # 4 x (x, y, z) = 12
  # 4 x (mx, my, mz) = 12
  # 4 x (mass) = 4
  # ignore radius stuff
  # total = 4*12 + 4*12 + 4 = 100

  concatted = concat_current_state(current_state)

  x = jax.nn.relu(concatted @ model_params.wi + model_params.bi)

  # scanf : (carry, input_i) -> (next_carry, output_i)
  scanf = lambda x, hidden_layer : jax.nn.relu(x @ hidden_layer.weight + hidden_layer.bias, None)
  x = jax.lax.scan(scanf, x, (model_params.hidden_layers))

  x = jax.nn.relu(x @ model_params.wo + model_params.bo)

  return x