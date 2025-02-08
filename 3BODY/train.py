import jax.numpy as jnp
import jax.random as jrand
import jax

import functools

from environment import init_solarsystems, step_simulation, get_reward
from GRPO import get_decision_probs, init_policy_model
from utils import load_model_params, save_model_params

# todo
  # loop -> scan for:
    # sim step func
  # jax init everything. make sure its all jitted
    # does moving the funcs outside of get_loss matter?
    # jit init_solarsystem
  # change the reward from outcome-supervised to the avg of the states. i.e., advantage_i = reward_i.

@functools.partial(jax.jit, static_argnames=["G", "epsilon", "trajectory_steps", "planets", "suns", "batch_size"])
def get_loss(new_policy_model_params, old_policy_model_params, key,
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
    old_policy_trajectory_g, new_policy_trajectory_g = policies # may break. [b] is [(old_p_step, new_p_step)] not ([old_p_step], [new_p_step])
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


# GRPO https://arxiv.org/pdf/2402.03300
update_old_policy = 50 # update every 10 iters
planets = 1
suns = 3

G = 512 # paper: 64 outputs
batch_size = 16 # paper uses 1024
trajectory_steps = 50 # run simulation for n steps

epsilon = 1e-3 # for clipping
train_iters = 20000 # arbitrary. how long we want to train for
dkl_beta = 0.04 # from paper
key = jrand.PRNGKey(0) # init rolling key

# train
debug = False
if debug:
    jax.config.update("jax_disable_jit", True)
    
jax.config.update("jax_debug_infs", True)
jax.config.update("jax_debug_nans", True)

learning_rate = 0.1
learning_rate_decay = 0.1

import time
start = time.time()
print("training")
for train_iter in range(train_iters):
  
  key, _ = jrand.split(key, 2) # roll key
  #try:
  #  loss = get_loss(new_policy_model_params, old_policy_model_params, key,
  #                  G=G, epsilon=epsilon, trajectory_steps=trajectory_steps,
  #                  planets=planets, suns=suns, batch_size=batch_size)
  #except:
  #   jax.config.update("jax_disable_jit", True)
  #   loss = get_loss(new_policy_model_params, old_policy_model_params, key,
  #                  G=G, epsilon=epsilon, trajectory_steps=trajectory_steps,
  #                  planets=planets, suns=suns, batch_size=batch_size)
  try:
    loss, grads = jax.value_and_grad(get_loss)(new_policy_model_params, old_policy_model_params, key,
                                              G=G, epsilon=epsilon, trajectory_steps=trajectory_steps,
                                              planets=planets, suns=suns, batch_size=batch_size)
  except:
    jax.config.update("jax_disable_jit", True)
    loss = get_loss(new_policy_model_params, old_policy_model_params, key,
                  G=G, epsilon=epsilon, trajectory_steps=trajectory_steps,
                  planets=planets, suns=suns, batch_size=batch_size)
  if train_iter == 0:
      jit_end = time.time()
      print((jit_end - start)/60)
  elif train_iter % update_old_policy == 0:
      old_policy_model_params = new_policy_model_params # update old policy model
      learning_rate = learning_rate * (1 - learning_rate_decay)
  grads = jax.tree_util.tree_map(lambda g: g*learning_rate, grads)
  new_policy_model_params = jax.tree_util.tree_map(lambda g, p: p - g, grads, new_policy_model_params)
  print(train_iter, loss, learning_rate)
end = time.time()
print((end - start)/60)

print(end - jit_end)
print(f"steps/sec:{train_iters*trajectory_steps*G*batch_size / (end - jit_end)}")
save_model_params(new_policy_model_params)


# differences
# - they train a reward model to come up with rewards
    # mine just assigns each step the reward of the outcome
    # actually nvm this was also in the paper as 'outcome supervision'
# - they use a reference policy for KL divergence instead of old_policy (idk what this means)