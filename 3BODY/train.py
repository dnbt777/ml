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
# policy: the actual probs of the actions

G = 16
trajectory_steps = 10 # run simulation for n steps

epsilon = 1e-4 # for clipping
train_iters = 100 # arbitrary. how long we want to train for
key = jrand.PRNGKey(0) # init rolling key


# outcome supervision
@jax.jit
def get_loss_old(new_policy, key,
             G=G, epsilon=epsilon, trajectory_steps=10,
             planets=planets, suns=suns):
    batches = 1 # turn into G batches
    policy_shape = (batches, G, trajectory_steps, output_actions)
    old_policy = jnp.zeros(policy_shape) # store the decisions made at each step, for each g, for each batch
    new_policy = jnp.zeros(policy_shape)
    end_reward = jnp.zeros((batches, G))
    for g in range(G):
        # init new state
        key, _ = jrand.split(key, 2) # roll key
        solar_systems = init_solarsystems(key, batches, planets, suns)
        # old_probs_g = [] # end result shape will be (batch, g, trajectory_step, action_probs)
        # new_probs_g = [] # in the future i may combine batch and g...

        # for step in range 2000:
        for step in range(trajectory_steps):
            # old_policy_probs = old_policy(state)
            # new_policy_probs = new_policy(state)
            old_policy_step = get_decision_probs(old_policy_model_params, solar_systems) # (batches, actions)
            new_policy_step = get_decision_probs(new_policy_model_params, solar_systems)
            # old_probs_g.append(old_policy_probs)
            # new_probs_g.append(new_policy_probs)
            old_policy = old_policy.at[:, step].set(old_policy_step) # stores probs of actions for each steo for each batch. (batches, step, actions)
            new_policy = new_policy.at[:, step].set(new_policy_step) # in the future: (batches, g, step, actions)
              # action = rand.choice(p=old_policy_probs)
            import numpy as np # temp, replace with jax and a key
            key, _ = jrand.split(key, 2) # roll key
            action = jrand.categorical(key, old_policy_step, axis=-1) # (batches,)
            # state = take_step(state, action)
            key, _ = jrand.split(key, 2) # roll key
            solar_systems = step_simulation(solar_systems, action)
        # end_reward = reward(state)
        end_reward = end_reward.at[:, g].set(get_reward(solar_systems)) # (batch, g)
    # advantages = (end_rewards - avg(end_rewards)) / standard_deviation(end_rewards)
    advantages = (end_reward - jnp.mean(end_reward, axis=-1, keepdims=True)) / jnp.std(end_reward, axis=-1, keepdims=True) # (batch, g)
    # loss = - (1/G) * (1/2000) * sum_across_G(sum_across_steps((min(prob_ratios * advantages.extend(), min(prob_ratios, 1 + epsilon, 1 - epsilon)) - kl_divergence)))
    advantages = advantages[:, :, None, None] # (batch, g, trajectory_steps, probs)
    prob_ratios = new_policy / (old_policy + 1e-7)
    kl_divergence = old_policy / (new_policy + 1e-7) - jnp.log(old_policy + 1e-7) - jnp.log(new_policy + 1e-7) - 1

    # get loss for each g
    xa = prob_ratios * advantages
    xb = jnp.clip(prob_ratios, 1 + epsilon, 1 - epsilon) * advantages
    xc = jnp.minimum(xa,xb)
    xd = xc - kl_divergence
    xe = jnp.sum(xd, axis=-1) # across logit axis (batch, g, step, logit) => (batch, g, step) 
    xf = jnp.sum(xe, axis=-1) * (1/trajectory_steps) # across step axis (batch, g, step) => (batch, g) -- get loss for each g
    xg = jnp.sum(xf, axis=-1) * (1/G) # across g axis (batch, g) => batch. get loss for each batch.
    xh = jnp.sum(xg, axis=-1) # combine all batch losses.
    loss = -xh
    return loss

# todo
  # loop -> scan for:
    # sim step func
    # two loops in get_loss
  # test with faster compile time. get it to train on batchsize=1
  # increase batch size... or just have large G and batchsize=1 forever


@jax.jit
def get_loss(new_policy, key,
             G=G, epsilon=epsilon, trajectory_steps=10,
             planets=planets, suns=suns):
    batches = 1 # turn into G batches
    policy_shape = (batches, G, trajectory_steps, output_actions)
    old_policy = jnp.zeros(policy_shape) # store the decisions made at each step, for each g, for each batch
    new_policy = jnp.zeros(policy_shape)
    end_reward = jnp.zeros((batches, G))
    for g in range(G):
        # init new state
        key, _ = jrand.split(key, 2) # roll key
        solar_systems = init_solarsystems(key, batches, planets, suns)

        # carry, [a] => (carry, [b])
        # state, None => (state, (old_policy_probs, new_policy_probs))
        # state: key, solar_systems
        # [a]: None
        def scanf(state, i):
            solar_systems, key = state

            old_policy_step = get_decision_probs(old_policy_model_params, solar_systems) # (batches, actions)
            new_policy_step = get_decision_probs(new_policy_model_params, solar_systems)

            #old_policy = old_policy.at[:, step].set(old_policy_step) # stores probs of actions for each steo for each batch. (batches, step, actions)
            #new_policy = new_policy.at[:, step].set(new_policy_step) # in the future: (batches, g, step, actions)

            key, _ = jrand.split(key, 2) # roll key
            action = jrand.categorical(key, old_policy_step, axis=-1) # (batches,)

            key, _ = jrand.split(key, 2) # roll key
            solar_systems = step_simulation(solar_systems, action)
            return (solar_systems, key), (old_policy_step, new_policy_step) # state, b
      
        ## SCAN OVER SIM STEPS
        # pack
        init_state = (solar_systems, key)
        # scan
        state, policies = jax.lax.scan(scanf, init_state, None, length=trajectory_steps)
        # unpack
        solar_systems, key = state
        old_policy, new_policy = policies # may break. [b] is [(old_p_step, new_p_step)] not ([old_p_step], [new_p_step])

        # end_reward = reward(state)
        end_reward = end_reward.at[:, g].set(get_reward(solar_systems)) # (batch, g)
    # advantages = (end_rewards - avg(end_rewards)) / standard_deviation(end_rewards)
    advantages = (end_reward - jnp.mean(end_reward, axis=-1, keepdims=True)) / jnp.std(end_reward, axis=-1, keepdims=True) # (batch, g)
    # loss = - (1/G) * (1/2000) * sum_across_G(sum_across_steps((min(prob_ratios * advantages.extend(), min(prob_ratios, 1 + epsilon, 1 - epsilon)) - kl_divergence)))
    advantages = advantages[:, :, None, None] # (batch, g, trajectory_steps, probs)
    prob_ratios = new_policy / (old_policy + 1e-7)
    kl_divergence = old_policy / (new_policy + 1e-7) - jnp.log(old_policy + 1e-7) - jnp.log(new_policy + 1e-7) - 1

    # get loss for each g
    xa = prob_ratios * advantages
    xb = jnp.clip(prob_ratios, 1 + epsilon, 1 - epsilon) * advantages
    xc = jnp.minimum(xa,xb)
    xd = xc - kl_divergence
    xe = jnp.sum(xd, axis=-1) # across logit axis (batch, g, step, logit) => (batch, g, step) 
    xf = jnp.sum(xe, axis=-1) * (1/trajectory_steps) # across step axis (batch, g, step) => (batch, g) -- get loss for each g
    xg = jnp.sum(xf, axis=-1) * (1/G) # across g axis (batch, g) => batch. get loss for each batch.
    xh = jnp.sum(xg, axis=-1) # combine all batch losses.
    loss = -xh
    return loss



print("training")
for train_iter in range(train_iters):
  learning_rate = 0.001
  key, _ = jrand.split(key, 2) # roll key
  loss, grads = jax.value_and_grad(get_loss)(new_policy_model_params, key)
  grads = jax.tree_util.tree_map(lambda g: g*learning_rate, grads)
  new_policy_model_params = jax.tree_util.tree_map(lambda g, p: g + p, grads, new_policy_model_params)
  print(train_iter, loss)



# differences
# - they train a reward model to come up with rewards
    # mine just assigns each step the reward of the outcome
    # actually nvm this was also in the paper as 'outcome supervision'
# - they use a reference policy for KL divergence instead of old_policy (idk what this means)