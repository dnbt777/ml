import jax.numpy as jnp
import jax.random as jrand
import jax

import functools

from environment import init_solarsystems, step_simulation, get_reward
from GRPO import get_decision_probs, init_policy_model
from file_utils import load_model_params, save_model_params

# todo
  # jax init everything. make sure its all jitted
    # does moving the funcs outside of get_loss matter?

@functools.partial(jax.jit, static_argnames=["G", "epsilon", "trajectory_steps", "planets", "suns", "batch_size"])
def get_loss_outcome_supervised(new_policy_model_params, old_policy_model_params, key,
             G, epsilon, trajectory_steps,
             planets, suns, batch_size):
  def run_single_trajectory_g(trajectory_key):
    # init new state
    trajectory_key, _ = jrand.split(trajectory_key, 2) # roll key
    solar_systems = init_solarsystems(trajectory_key, batch_size, planets, suns)

    # carry, [a] => (carry, [b])
    # state, None => (state, (old_policy_probs, new_policy_probs))
    # state: trajectory_key, solar_systems
    # [a]: None
    def step_scanf(state, i):
      solar_systems, scan_key = state

      old_policy_step = get_decision_probs(old_policy_model_params, solar_systems) # (batch_size, actions)
      new_policy_step = get_decision_probs(new_policy_model_params, solar_systems)

      scan_key, _ = jrand.split(scan_key, 2) # roll scan_key
      action = jrand.categorical(scan_key, old_policy_step, axis=-1) # (batch_size,)

      scan_key, _ = jrand.split(scan_key, 2) # roll scan_key
      solar_systems = step_simulation(solar_systems, action)
      return (solar_systems, scan_key), (old_policy_step, new_policy_step) # state, b

    ## SCAN OVER SIM STEPS
    # pack
    init_state = (solar_systems, trajectory_key)
    # scan
    state, policies = jax.lax.scan(step_scanf, init_state, None, length=trajectory_steps)
    # unpack
    solar_systems, trajectory_key = state
    old_policy_trajectory_g, new_policy_trajectory_g = policies
    # format axes (b, step, logits)
    old_policy_trajectory_g = old_policy_trajectory_g.swapaxes(0, 1) # (step, batch, logits) => (batch, step, logits)
    new_policy_trajectory_g = new_policy_trajectory_g.swapaxes(0, 1) # (step, batch, logits) => (batch, step, logits)

    # end_reward = reward(state)
    outcome_reward_g = jax.vmap(get_reward, in_axes=0)(solar_systems) # vmap over (batch,)
    return old_policy_trajectory_g, new_policy_trajectory_g, outcome_reward_g

  random_keys = jrand.split(key, G).reshape((G, 2)) # batching happens INSIDE run_single_trajectory_g
  #key_func = functools.partial(run_single_trajectory_g, new_policy_model_params)
  #vmap_func = jax.vmap(jax.vmap(run_single_trajectory_g))
  old_policy, new_policy, outcome_rewards = jax.vmap(run_single_trajectory_g)(random_keys)
  # in_axes=-1 because each key is a 1D array of two values. (batch, g, key_data) where key_data is always of size 2.

  # advantages = (end_rewards - avg(end_rewards)) / standard_deviation(end_rewards)
  advantages = (outcome_rewards - jnp.mean(outcome_rewards, axis=-1, keepdims=True)) / (jnp.std(outcome_rewards, axis=-1, keepdims=True) + 1e-7) # (batch, g)
  # loss = - (1/G) * (1/2000) * sum_across_G(sum_across_steps((min(prob_ratios * advantages.extend(), min(prob_ratios, 1 + epsilon, 1 - epsilon)) - kl_divergence)))
  advantages = advantages[:, :, None, None] # (batch, g, trajectory_steps, probs) => (batch, g, trajectory_steps, probs)
  prob_ratios = new_policy / (old_policy + 1e-7)
  kl_divergence = old_policy / (new_policy + 1e-7) - jnp.log(old_policy + 1e-7) - jnp.log(new_policy + 1e-7) - 1

  # get loss for each g
  xa = prob_ratios * advantages
  xb = jnp.clip(prob_ratios, 1 + epsilon, 1 - epsilon) * advantages
  xc = jnp.minimum(xa,xb)
  xd = xc - kl_divergence
  xe = jnp.sum(xd, axis=-1) * (1/7) # across logit axis (batch, g, step, logit) => (batch, g, step) 
  xf = jnp.sum(xe, axis=-1) * (1/trajectory_steps) # across step axis (batch, g, step) => (batch, g) -- get loss for each g
  xg = jnp.sum(xf, axis=-1) * (1/batch_size) # combine all batch losses.
  xh = jnp.sum(xg, axis=-1) * (1/G) # across g axis (batch, g) => batch. get loss for each batch.
  loss = -xh
  return loss


@functools.partial(jax.jit, static_argnames=["G", "trajectory_steps", "planets", "suns", "batch_size"])
def get_episode_rewards(new_policy_model_params, key,
             G, trajectory_steps,
             planets, suns, batch_size):
  def run_single_trajectory_g(trajectory_key):
    # init new state
    trajectory_key, _ = jrand.split(trajectory_key, 2) # roll key
    solar_systems = init_solarsystems(trajectory_key, batch_size, planets, suns)

    # carry, [a] => (carry, [b])
    # state, None => (state, (old_policy_probs, new_policy_probs))
    # state: trajectory_key, solar_systems
    # [a]: None
    def step_scanf(state, i):
      solar_systems, scan_key = state

      new_policy_step = get_decision_probs(new_policy_model_params, solar_systems)

      scan_key, _ = jrand.split(scan_key, 2) # roll scan_key
      action = jrand.categorical(scan_key, new_policy_step, axis=-1) # (batch_size,)

      scan_key, _ = jrand.split(scan_key, 2) # roll scan_key
      solar_systems = step_simulation(solar_systems, action)
      next_step_rewards = jax.vmap(get_reward, in_axes=0)(solar_systems)
      return (solar_systems, scan_key), (next_step_rewards) # state, b

    ## SCAN OVER SIM STEPS
    # pack
    init_state = (solar_systems, trajectory_key)
    # scan
    state, intermittent_rewards_g = jax.lax.scan(step_scanf, init_state, None, length=trajectory_steps)
    solar_systems, trajectory_key = state
    # format axes (b, step, logits)
    # end_reward = reward(state)
    outcome_reward_g = jax.vmap(get_reward, in_axes=0)(solar_systems) # vmap over (batch,)
    return intermittent_rewards_g, outcome_reward_g

  random_keys = jrand.split(key, G).reshape((G, 2)) # batching happens INSIDE run_single_trajectory_g
  intermittent_rewards, outcome_rewards = jax.vmap(run_single_trajectory_g)(random_keys)
  return intermittent_rewards, outcome_rewards


# policy = init policy (model)
hidden_size = 16
hidden_layers = 10
input_datapoints = 3*4 + 3*4 + 1*4
output_actions = 7 # lr/ud/bf/nothing
policy_model_params = init_policy_model(hidden_layers, hidden_size, input_datapoints, output_actions)
# frozen model = a copy of this ( will not be updated)
# learning model = this (will be updated)

old_policy_model_params = policy_model_params
new_policy_model_params = policy_model_params
# policy model: the ML model
# policy: the actual probs of the actions


# sim stuff
planets = 1
suns = 3
trajectory_horizon = 200 # run simulation for n steps

# GRPO https://arxiv.org/pdf/2402.03300
G = 64 # 512 # paper: 64 outputs
batch_size = 128 # 16 # paper uses 1024
update_old_policy = 10 # update every 10 iters

epsilon = 0.1 # for clipping - https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
train_iters = 4000 # arbitrary. how long we want to train for
dkl_beta = 0.0001 # https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
key = jrand.PRNGKey(0) # init rolling key

# train
debug = False
if debug:
  jax.config.update("jax_disable_jit", True)
  jax.config.update("jax_debug_infs", False)
  jax.config.update("jax_debug_nans", True)
else:
  import warnings
  warnings.filterwarnings("ignore")


learning_rate = 3e-2
learning_rate_decay = 0.001

import time
start = time.time()
print("training")
for train_iter in range(train_iters):
  key, _ = jrand.split(key, 2) # roll key
  loss, grads = jax.value_and_grad(get_loss_outcome_supervised)(new_policy_model_params, old_policy_model_params, key,
                                            G=G, epsilon=epsilon, trajectory_steps=trajectory_horizon,
                                            planets=planets, suns=suns, batch_size=batch_size)
  step_rewards, outcome_rewards = get_episode_rewards(new_policy_model_params, key,
                                                          G, trajectory_horizon,
                                                          planets, suns, batch_size)
  if train_iter == 0:
      jit_end = time.time()
      print((jit_end - start)/60)
  elif train_iter % update_old_policy == 0:
      old_policy_model_params = new_policy_model_params # update old policy model
      learning_rate = learning_rate * (1 - learning_rate_decay)
      step_rewards, outcome_rewards = get_episode_rewards(new_policy_model_params, key,
                                                          G, trajectory_horizon,
                                                          planets, suns, batch_size)
      #print(f"{train_iter} Mean outcome reward: {jnp.mean(outcome_rewards)}")
      #print(f"{train_iter} Mean episode return: {jnp.mean(step_rewards)}")
  grads = jax.tree_util.tree_map(lambda g: g*learning_rate, grads)
  new_policy_model_params = jax.tree_util.tree_map(lambda g, p: p + g, grads, new_policy_model_params) # maximize objective (its different in RL)
  print(train_iter, loss, learning_rate, jnp.mean(step_rewards), jnp.mean(outcome_rewards))
end = time.time()
#print((end - start)/60)

#print(end - jit_end)
print(f"steps/sec: {train_iters*trajectory_horizon*G*batch_size / (end - jit_end):,.0f}")
save_model_params(new_policy_model_params)


# differences
# - they train a reward model to come up with rewards
    # mine just assigns each step the reward of the outcome
    # actually nvm this was also in the paper as 'outcome supervision'
# - they use a reference policy for KL divergence instead of old_policy (idk what this means)