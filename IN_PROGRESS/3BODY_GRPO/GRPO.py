###
# deep q agent
# used to test train and 'inference' setup.
# input: SolarSystem
# output: momentum shift (soft capped with tanh)

import jax
import jax.numpy as jnp
import jax.random as jrand
from custom_types import *
from functools import partial
import time


# @jax.jit # breaks :(
def init_policy_model(init_key, hidden_layers, hidden_size, input_size, output_size):
  xavier = jax.nn.initializers.xavier_uniform()
  initializer = lambda key, hidden_shape_w: xavier(key, hidden_shape_w)
  # get keys for random initialization
  io_init_keys = jrand.split(init_key, 2)
  hidden_init_keys = jrand.split(init_key, hidden_layers)
  # get shapes for initializing arrays
  hidden_shape_w = (hidden_size, hidden_size)
  hidden_shape_b = (hidden_layers, hidden_size)
  input_shape_w = (input_size, hidden_size)
  input_shape_b = (hidden_size,)
  output_shape_w = (hidden_size, output_size)
  output_shape_b = (output_size)
  # return initialized params
  return PMParams(
    hidden_layers=PMLayer(
      # initialize each layer separately. otherwise most become too small
      lnw=jnp.ones(hidden_shape_b), # layer norm
      lnb=jnp.zeros(hidden_shape_b),
      weight=jax.vmap(initializer, in_axes=(0, None))(hidden_init_keys, hidden_shape_w),
      bias=jnp.zeros(hidden_shape_b)
    ),
    wi = xavier(io_init_keys[0], input_shape_w),
    bi = jnp.zeros(input_shape_b),
    lnwi = jnp.ones(input_shape_b),
    lnbi=jnp.zeros(input_shape_b),
    wo = xavier(io_init_keys[1], output_shape_w),
    bo = jnp.zeros(output_shape_b)
  )


@jax.jit
def safe_concat_current_state(solar_system):
  # sacrifices accuracy for num safety
  velocity = jnp.ravel(solar_system.bodies.velocity)
  safe_velocity = velocity / jnp.max(jnp.abs(velocity), axis=-1, keepdims=True)
  #norm_velocity = (velocity - jnp.mean(velocity, axis=-1, keepdims=True)) / jnp.std(velocity, axis=-1, keepdims=True) 
  # reason for disabling norm velocity: velocity needs to be precise, like position
  # should be false velocity? divide by sim size maybe? hmm...

  mass = jnp.ravel(solar_system.bodies.mass)
  safe_mass = mass / jnp.max(jnp.abs(mass), axis=-1, keepdims=True)
  # OPTIMIZATION: relative masses may confuse the model. replace with safe_mass = mass / avg_sun_mass (a known constant)
  #log_mass = jnp.log(mass)

  position = jnp.ravel(solar_system.bodies.position) # not true position, but sim position

  return jax.lax.concatenate([
    position, # should be relative/small
    velocity,
    safe_mass
    ],
    dimension=0
  )

@jax.jit
def model_forward(policy_model_params: PMParams, current_state_batch):
  # current_state shape:
  # 4 bodies:
  # 4 x (x, y, z) = 12
  # 4 x (v, v, v) = 12
  # 4 x (mass) = 4
  # ignore radius
  # total = 12 + 12 + 4 = (batchsize, 28)
  concatted = jax.vmap(safe_concat_current_state, in_axes=0)(current_state_batch) # vmap across batch axis - this is the way to go, as opposed to writing batched functions
  # initial projection
  x = jax.nn.relu(concatted @ policy_model_params.wi + policy_model_params.bi)
  # scan through each hidden layer
  # scanf : (carry, input_i) -> (next_carry, output_i)
  def scanf(x, hidden_layer_i):
    next_x = x + jax.nn.relu(x @ hidden_layer_i.weight + hidden_layer_i.bias) # TODO relu causes grad explosion? debug why relu -> grad explosion (versus tanh)
    return next_x, None 
  x, _ = jax.lax.scan(scanf, x, policy_model_params.hidden_layers) # scan => (x, None)
  # final projection to output size
  x = jax.nn.relu(x @ policy_model_params.wo + policy_model_params.bo)
  return x


@jax.jit
def get_decision_probs(policy_model_params: PMParams, current_state):
  x = model_forward(policy_model_params, current_state)
  #x_safer = x - jnp.max(x, axis=-1, keepdims=True) # subtract by max along relevant axis. reduces nan https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
  # nvm jax.nn.softmax already does safe softmax
  jax.config.update("jax_debug_infs", False)
  p = jax.nn.softmax(x, axis=-1)
  jax.config.update("jax_debug_infs", True)
  return p


# pretrained. no epsilon
@jax.jit
def take_action(key, policy_model_params: PMParams, current_state):
  logits = model_forward(policy_model_params, current_state)
  # randomly choose action from policy
  action = jrand.categorical(key, logits, axis=-1)
  return action


