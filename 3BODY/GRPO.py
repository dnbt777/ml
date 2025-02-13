###
# deep q agent
# used to test train and 'inference' setup.
# input: SolarSystem
# output: momentum shift (soft capped with tanh)

import jax
import jax.numpy as jnp
import jax.random as jrand
from typing import NamedTuple, List
from functools import partial
import time


# SoA
# PolicyModel
class PMLayer(NamedTuple):
  weight : jax.Array
  bias : jax.Array

class PMParams(NamedTuple):
  wi : jax.Array
  bi :jax.Array
  wo : jax.Array
  bo : jax.Array
  hidden_layers : List[PMLayer] # do it this way for scanning

# @jax.jit # breaks :(
def init_policy_model(hidden_layers, hidden_size, input_size, output_size):
  initializer = jax.nn.initializers.glorot_uniform()
  key = jrand.PRNGKey(int(time.time()*10000))
  hidden_shape_w = (hidden_layers, hidden_size, hidden_size)
  hidden_shape_b = (hidden_layers, hidden_size)
  input_shape_w = (input_size, hidden_size)
  input_shape_b = (hidden_size,)
  output_shape_w = (hidden_size, output_size)
  output_shape_b = (output_size)
  return PMParams(
    hidden_layers=PMLayer(
      weight=initializer(key, hidden_shape_w),
      bias=jnp.zeros(hidden_shape_b)
    ),
    wi = initializer(key, input_shape_w),
    bi = jnp.zeros(input_shape_b),
    wo = initializer(key, output_shape_w),
    bo = jnp.zeros(output_shape_b)
  )


# move to environment or environment_utils idk
@jax.jit
def concat_current_state(solar_system_batch):
  velocity = jnp.ravel(solar_system_batch.bodies.velocity)
  #norm_velocity = (velocity - jnp.mean(velocity, axis=-1, keepdims=True)) / jnp.std(velocity, axis=-1, keepdims=True) 
  # reason for disabling norm velocity: velocity needs to be precise, like position
  # should be false velocity? divide by sim size maybe? hmm...

  mass = jnp.ravel(solar_system_batch.bodies.mass)
  #norm_mass = (mass - jnp.mean(mass, axis=-1, keepdims=True)) / jnp.std(mass, axis=-1, keepdims=True) # constant transform, does not need to be recalculated
  log_mass = jnp.log(mass)

  position = jnp.ravel(solar_system_batch.bodies.position) # not true position, but sim position. should be fine

  return jax.lax.concatenate([
    position, # should be relative/small
    velocity,
    log_mass
    ],
    dimension=0
  )

@jax.jit
def safe_concat_current_state(solar_system_batch):
  # sacrifices accuracy for num safety
  velocity = jnp.ravel(solar_system_batch.bodies.velocity)
  safe_velocity = velocity / jnp.max(jnp.abs(velocity), axis=-1, keepdims=True)
  #norm_velocity = (velocity - jnp.mean(velocity, axis=-1, keepdims=True)) / jnp.std(velocity, axis=-1, keepdims=True) 
  # reason for disabling norm velocity: velocity needs to be precise, like position
  # should be false velocity? divide by sim size maybe? hmm...

  mass = jnp.ravel(solar_system_batch.bodies.mass)
  safe_mass = mass / jnp.max(jnp.abs(mass), axis=-1, keepdims=True)
  #norm_mass = (mass - jnp.mean(mass, axis=-1, keepdims=True)) / jnp.std(mass, axis=-1, keepdims=True) # constant transform, does not need to be recalculated
  #log_mass = jnp.log(mass)

  position = jnp.ravel(solar_system_batch.bodies.position) # not true position, but sim position. should be fine
  safe_position = position / jnp.max(jnp.abs(position), axis=-1, keepdims=True)

  return jax.lax.concatenate([
    position, # should be relative/small
    velocity,
    safe_mass
    ],
    dimension=0
  )

@jax.jit
def get_decision_logits(policy_model_params : PMParams, current_state):
  # input:
  # 4 bodies:
  # 4 x (x, y, z) = 12
  # 4 x (v, v, v) = 12
  # 4 x (mass) = 4
  # ignore radius stuff
  # total = 12 + 12 + 4 = (batchsize, 28)
  concatted = jax.vmap(safe_concat_current_state, in_axes=0)(current_state) # vmap across batch axis - this is the way to go, as opposed to writing batched functions
  x = jax.nn.tanh(concatted @ policy_model_params.wi + policy_model_params.bi)
  # scanf : (carry, input_i) -> (next_carry, output_i)
  scanf = lambda x, hidden_layer : (jax.nn.tanh(x @ hidden_layer.weight + hidden_layer.bias), None)
  x = jax.lax.scan(scanf, x, (policy_model_params.hidden_layers))[0] # scan => (x, None)
  x = jax.nn.tanh(x @ policy_model_params.wo + policy_model_params.bo)
  return x


@jax.jit
def get_decision_probs(policy_model_params : PMParams, current_state):
  x = get_decision_logits(policy_model_params, current_state)
  #x_safer = x - jnp.max(x, axis=-1, keepdims=True) # subtract by max along relevant axis. reduces nan https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
  # nvm jax.nn.softmax already does safe softmax
  #jax.config.update("jax_debug_infs", False)
  p = jax.nn.softmax(x, axis=-1)
  #jax.config.update("jax_debug_infs", True)
  return p


# pretrained. no epsilon
@jax.jit
def take_action(key, policy_model_params : PMParams, current_state):
  logits = get_decision_logits(policy_model_params, current_state)
  # randomly choose action from policy
  action = jrand.categorical(key, logits, axis=-1)
  return action


