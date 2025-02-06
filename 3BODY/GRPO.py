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
# TODO: store velocities rather than momentum. requires changing the simulation but not by much. (massless momentum)
# reason: prevents infs in momentum. then, use velocities as an input to the neural net. should be MUCH smaller.
# then: use log(mass) as an input to the NN as well.
@jax.jit
def concat_current_state(solar_system_batch):
  velocity = jnp.ravel(solar_system_batch.bodies.velocity) # should just be divided by planet mass tbh, this is a hack
  velocity = (velocity - jnp.mean(velocity, axis=-1, keepdims=True)) / jnp.std(velocity, axis=-1, keepdims=True) 

  mass = jnp.ravel(solar_system_batch.bodies.mass)
  mass = (mass - jnp.mean(mass, axis=-1, keepdims=True)) / jnp.std(mass, axis=-1, keepdims=True) # constant transform, does not need to be recalculated

  return jax.lax.concatenate([
    jnp.ravel(solar_system_batch.bodies.position), # should be relative/small
    velocity, # log is temp to test perf improvement
    mass # this will break when batch size > 1
    ],
    dimension=0
  )

# init model params(input=total, hidden layers, output=6 (udlrbf for now))


@jax.jit
def get_decision_logits(policy_model_params : PMParams, current_state):
  # input:
  # 4 bodies:
  # 4 x (x, y, z) = 12
  # 4 x (mx, my, mz) = 12
  # 4 x (mass) = 4
  # ignore radius stuff
  # total = 4*12 + 4*12 + 4 = 100
  concatted = concat_current_state(current_state)
  x = jax.nn.relu(concatted @ policy_model_params.wi + policy_model_params.bi)
  # scanf : (carry, input_i) -> (next_carry, output_i)
  scanf = lambda x, hidden_layer : (jax.nn.relu(x @ hidden_layer.weight + hidden_layer.bias), None)
  x = jax.lax.scan(scanf, x, (policy_model_params.hidden_layers))[0] # scan => (x, None)
  x = jax.nn.relu(x @ policy_model_params.wo + policy_model_params.bo)
  return x


@jax.jit
def get_decision_probs(policy_model_params : PMParams, current_state):
  x = get_decision_logits(policy_model_params, current_state)
  p = jax.nn.softmax(x)
  return p


# pretrained. no epsilon
@jax.jit
def take_action(key, policy_model_params : PMParams, current_state):
  logits = get_decision_logits(policy_model_params, current_state)
  # randomly choose action from policy
  action = jrand.categorical(key, logits, axis=-1)
  return action


