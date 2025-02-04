import jax.numpy as jnp
import jax.random as jrand

from environment import *
from GRPO import *


# GRPO https://arxiv.org/pdf/2402.03300
planets = 1
suns = 3

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
# policy: the actual logprobs of the actions

G = 100
trajectory_steps = 2000 # run simulation for n steps

epsilon = 1e-4 # for clipping
train_iters = 100 # arbitrary. how long we want to train for
key = jrand.PRNGKey(0) # init rolling key

for train_iter in range(train_iters):
  # for g in G:
  def get_loss(new_policy, key):
    for g in range(G):
        # init new state
        batches = 1 # turn into G batches
        key, _ = jrand.split(key, 2) # roll key
        solar_systems = init_solarsystems(key, batches, planets, suns)
        # old_logprobs_g = [] # end result shape will be (batch, g, trajectory_step, action_logprobs)
        # new_logprobs_g = [] # in the future i may combine batch and g...
        policy_shape = (batches, G, trajectory_steps, output_actions)
        old_policy = jnp.zeros(policy_shape) # store the decisions made at each step, for each g, for each batch
        new_policy = jnp.zeros(policy_shape)
        end_reward = jnp.zeros((batches, G))
        # for step in range 2000:
        for step in range(trajectory_steps):
            # old_policy_logprobs = old_policy(state)
            # new_policy_logprobs = new_policy(state)
            key, _ = jrand.split(key, 2) # roll key
            old_policy_step = take_action(key, old_policy_model_params, solar_systems) # (batches, actions)
            key, _ = jrand.split(key, 2) # roll key
            new_policy_step = take_action(key, new_policy_model_params, solar_systems)
            # old_logprobs_g.append(old_policy_logprobs)
            # new_logprobs_g.append(new_policy_logprobs)
            old_policy[:, step] = old_policy_step # stores logprobs of actions for each steo for each batch. (batches, step, actions)
            new_policy[:, step] = new_policy_step # in the future: (batches, g, step, actions)
              # action = rand.choice(p=old_policy_logprobs)
            import numpy as np # temp, replace with jax and a key
            key, _ = jrand.split(key, 2) # roll key
            action = jrand.choice(key, range(output_actions), p=old_policy_step) # (batches,)
            # state = take_step(state, action)
            key, _ = jrand.split(key, 2) # roll key
            solar_systems = step_simulation(key, action, solar_systems)
        # end_reward = reward(state)
        end_reward[:, g] = end_reward.at[:, g].set(get_reward(solar_systems)) # (batch, g)
    # advantages = (end_rewards - avg(end_rewards)) / standard_deviation(end_rewards)
    advantages = (end_reward - jnp.mean(end_reward, axis=-1, keepdims=True)) / jnp.std(end_reward, axis=-1, keepdims=True) # (batch, g)
    # loss = - (1/G) * (1/2000) * sum_across_G(sum_across_steps((min(logprob_ratios * advantages.extend(), min(logprob_ratios, 1 + epsilon, 1 - epsilon)) - kl_divergence)))
    advantages = advantages[:, :, None, None] # (batch, g, trajectory_steps, logprobs)
    logprob_ratios = new_policy / (old_policy + 1e-7)
    kl_divergence = old_policy / (new_policy + 1e-7) - jnp.log(old_policy / (new_policy + 1e-7)) - 1
    loss = - (1/G) * (1/trajectory_steps) * jnp.sum(
      jnp.min(
        logprob_ratios * advantages,
        jnp.clip(logprob_ratios, 1 + epsilon, 1 - epsilon) * advantages
        ) - kl_divergence,
      axis=(2, 3)
    ) #(batch, g, step, logprob) => (batch, g). sum across (2, 3)
    return loss
  
  learning_rate = 0.001
  key, _ = jrand.split(key, 2) # roll key
  loss, grads = jax.value_and_grad(get_loss)(new_policy_model_params, key)
  new_policy_model_params = grads * learning_rate + new_policy_model_params
  print(train_iter, loss)




# differences
# - they train a reward model to come up with rewards
    # mine just assigns each step the reward of the outcome
# - they use a reference policy for KL divergence instead of old_policy (idk what this means)