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


hidden_size = 16
hidden_layers = 10


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

# returns 'agent_forward' function. basically it returns inference() func
def init_agent():
  dqnparams = init_policy_model(hidden_layers, hidden_size, 3*4 + 3*4 + 1*4, 6)
  agent_forward = lambda key, current_state: take_action(key, dqnparams, current_state)
  return agent_forward


# move to environment or environment_utils idk
def concat_current_state(solar_system_batch):
  return jax.lax.concatenate([
    jnp.ravel(solar_system_batch.bodies.position),
    jnp.ravel(solar_system_batch.bodies.momentum),
    jnp.ravel(solar_system_batch.bodies.mass) # this will break when batch size > 1
    ],
    dimension=0
  )

# init model params(input=total, hidden layers, output=6 (udlrbf for now))

# pretrained. no epsilon
@jax.jit
def take_action(key, policy_model_params : PMParams, current_state):
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

  # choose highest value action
  # choices are comprised of possible momentum shifts
  action_index = jnp.argmax(x)
  action = jnp.array([
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1)
  ])[action_index]
  return jnp.array(action)





# GRPO
planets = 1
suns = 3

# policy = init policy (model)
hidden_size = 16
hidden_layers = 10
input_datapoints = 3*4 + 3*4 + 1*4
output_actions = 7 # lr/ud/bf/nothing
policy_model = init_policy_model(hidden_layers, hidden_size, input_datapoints, output_actions)
# frozen model = a copy of this ( will not be updated)
# learning model = this (will be updated)

old_policy_model = policy_model
new_policy_model = policy_model
# policy model: the ML model
# policy: the actual logprobs of the actions

G = 100
episode_steps = 2000 # run simulation for n steps
# logprob_ratios = []
# end_rewards = []
for train_iter in range(train_iters):
  # for g in G:
  for g in range(G):
      # init new state
      batches = 1 # turn into G batches
      solar_systems = init_solarsystems(key, batches, planets, suns)
      # old_logprobs_g = []
      # new_logprobs_g = []
      # for step in range 2000:
      for step in range(episode_steps):
          # old_policy_logprobs = old_policy(state)
          # new_policy_logprobs = new_policy(state)
          old_policy_step = take_action(key, old_policy_model, solar_systems) # (batches, actions)
          new_policy_step = take_action(key, new_policy_model, solar_systems)
          # old_logprobs_g.append(old_policy_logprobs)
          # new_logprobs_g.append(new_policy_logprobs)
          old_policy[:, step] = old_policy_step # stores logprobs of actions for each steo for each batch. (batches, step, actions)
          new_policy[:, step] = new_policy_step # in the future: (batches, g, step, actions)
          # action = rand.choice(p=old_policy_logprobs)
          action = np.rand.weighted_choice(range(output_actions), p=old_policy_step) # (batches,)
          # state = take_step(state, action)
          solar_systems = step_simulation(key, lambda *args: action, solar_systems)
      # end_reward = reward(state)
      end_reward[:, g], _ = get_reward(solar_systems) # (batch, g)
  # advantages = (end_rewards - avg(end_rewards)) / standard_deviation(end_rewards)
  advantages = (end_reward - jnp.mean(end_reward, axis=-1, keepdims=True)) / jnp.std(end_reward, axis=-1, keepdims=True) # (batch, g)
  # loss = - (1/G) * (1/2000) * sum_across_G(sum_across_steps((min(logprob_ratios * advantages.extend(), min(logprob_ratios, 1 + epsilon, 1 - epsilon)) - kl_divergence)))
  # policy += learning_rate * jax.grad(loss_func)(policy)