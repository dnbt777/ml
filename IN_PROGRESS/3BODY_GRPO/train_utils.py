import jax.numpy as jnp
import jax.random as jrand
import jax
import functools
from typing import Tuple
from environment import init_solarsystems, step_simulation, get_reward
from GRPO import get_decision_probs, init_policy_model
from file_utils import load_model_params, save_model_params
from custom_types import *
import time


# todo:
# figure out why grads are 0. where precisely is this happening?
# possibly debug w vjp and pullback
# https://www.kaggle.com/code/goktugguvercin/automatic-differentiation-in-jax#Pushforward-and-Pullback:
# create train_utils and then jit the entire training process there


## ------------- ######
#### - FUNCTIONS - ####
###### ------------- ##

## ALL ##


@functools.partial(jax.jit, static_argnames=["epsilon", "dkl_beta"])
def get_objective(policy_model_params, reference_policy_outputs,
                                    old_policy_outputs, trajectory_states, A,
                                    epsilon, dkl_beta):
    # set up keys, get shape sizes
    G, batch_size, trajectory_horizon, _ = old_policy_outputs.shape # (G, batch_size, trajectory_horizon, logits)
    # replay all trajectories to get new_policy probs
    replay_group_trajectories = jax.vmap(replay_batch_trajectories, in_axes=(None, 0))            
    new_policy_outputs = replay_group_trajectories(policy_model_params, trajectory_states) # (G, batch_size, trajectory_horizon, logits)
    # outcome supervision GRPO equation
    # loss = - (1/G) * (1/2000) * sum_across_G(sum_across_steps((min(prob_ratios * advantages.extend(), min(prob_ratios, 1 + epsilon, 1 - epsilon)) - kl_divergence)))
    prob_ratios = new_policy_outputs / (old_policy_outputs + 1e-7)
    # get kl divergence
    ref_policy_ratio = reference_policy_outputs / (new_policy_outputs + 1e-7)
    kl_divergence = ref_policy_ratio - jnp.log(ref_policy_ratio) - 1
    # calculate the equations in parts
    # this gets compiled anyways, so its likely not memory inefficient
    xa = prob_ratios * A
    xb = jnp.clip(prob_ratios, min=1-epsilon, max=1+epsilon) * A
    xc = jnp.minimum(xa,xb)
    xd = xc - (dkl_beta * kl_divergence)
    # xe = jnp.sum(xd, axis=-1) #* (1/7) # across logit axis (batch, g, step, logit) => (batch, g, step) 
    # xf = jnp.sum(xe, axis=-1) / trajectory_horizon # across step axis (batch, g, step) => (batch, g) -- get loss for each g
    # xg = jnp.sum(xf, axis=-1) #* (1/batch_size) # combine all batch losses.
    # xh = jnp.sum(xg, axis=-1) / G # across g axis (batch, g) => batch. get loss for each batch.
    objective = jnp.mean(xd)
    return objective


@jax.jit
def replay_batch_trajectories(policy_model_params, trajectory_states):
    def scan_through_trajectories(state, solar_systems_oi):
        decision_probs_oi = get_decision_probs(policy_model_params, solar_systems_oi)
        return None, decision_probs_oi
    # scan
    _, policy_outputs = jax.lax.scan(scan_through_trajectories, None, trajectory_states)
    # scan has timewise axes for each batch. so swap
    # policy_outputs = jnp.swapaxes(policy_outputs, 0, 1) # (G, traj_horizon, batch, decision) => (G, batch, traj_horizon, decision)
    # DEBUG: check if axes make sense from the value returned from the scan
    return policy_outputs # (batch_size, trajectory_horizon, logits)


### OUTCOME SUPERVISION ###

# given a starting state solar_system, runs a trajectory for $trajectory_horizon steps
@functools.partial(jax.jit, static_argnames=["trajectory_horizon"])
def run_batch_trajectories_outcome_supervised(policy_model_params, solar_system, trajectory_horizon, trajectory_key):
    # carry, [a] => (carry, [b])
    # state, None => (state, (old_policy_probs, new_policy_probs))
    # state: trajectory_key, solar_systems
    # [a]: None
    def scan_through_trajectories(state, i):
        solar_system, scan_key = state
        decision_probs_oi = get_decision_probs(policy_model_params, solar_system)
        # roll scan_key and sample action from probs
        scan_key, _ = jrand.split(scan_key, 2)
        action = jrand.categorical(scan_key, decision_probs_oi, axis=-1) # Scalar
        solar_system = step_simulation(solar_system, action)
        return (solar_system, scan_key), (decision_probs_oi, solar_system) # state, b
    ## SCAN OVER SIM STEPS
    # pack
    init_state = (solar_system, trajectory_key)
    # scan
    state, trajectory = jax.lax.scan(scan_through_trajectories, init_state, None, length=trajectory_horizon)
    # unpack
    solar_system_end_state, final_trajectory_key = state # scalar scalar
    policy_outputs, trajectory_states = trajectory # (step, logits) (step, solar_system)
    policy_outputs = jnp.swapaxes(policy_outputs, 0, 1) # (G, traj_horizon, batch, decision) => (G, batch, traj_horizon, decision)
    # end_reward = reward(state)
    outcome_reward = jax.vmap(get_reward, in_axes=0)(solar_system_end_state) # scalar
    return policy_outputs, trajectory_states, outcome_reward # (trajectory_horizon, logits) (trajectory_horizon, solar_system) scalar


## PROCESS SUPERVISION ##

# todo
@functools.partial(jax.jit, static_argnames=["trajectory_horizon"])
def run_batch_trajectories_process_supervised(policy_model_params, solar_system, trajectory_horizon, trajectory_key):
    # carry, [a] => (carry, [b])
    # state, None => (state, (old_policy_probs, new_policy_probs))
    # state: trajectory_key, solar_systems
    # [a]: None
    def scan_through_trajectories(state, i):
        solar_system, scan_key = state
        decision_probs_oi = get_decision_probs(policy_model_params, solar_system)
        # roll scan_key and sample action from probs
        scan_key, _ = jrand.split(scan_key, 2)
        action = jrand.categorical(scan_key, decision_probs_oi, axis=-1) # Scalar
        solar_system = step_simulation(solar_system, action)
        reward = jax.vmap(get_reward, in_axes=0)(solar_system)
        return (solar_system, scan_key), (decision_probs_oi, reward, solar_system) # state, b
    ## SCAN OVER SIM STEPS
    # pack
    init_state = (solar_system, trajectory_key)
    # scan
    state, trajectory = jax.lax.scan(scan_through_trajectories, init_state, None, length=trajectory_horizon)
    # unpack
    solar_system_end_state, final_trajectory_key = state # scalar scalar
    policy_outputs, process_rewards, trajectory_states = trajectory # (step, logits) (step,) (step, solar_system)
    policy_outputs = jnp.swapaxes(policy_outputs, 0, 1) # (traj_horizon, batch, decision) => (batch, traj_horizon, decision)
    trajectory_states = jax.tree_util.tree_map(lambda arr: jnp.swapaxes(arr, 0, 1), trajectory_states) # => solar_system(batch, step, ...)
    process_rewards = jnp.swapaxes(process_rewards, 0, 1) # (step, batch) => (batch, step)
    return policy_outputs, trajectory_states, process_rewards # (trajectory_horizon, logits) (trajectory_horizon, solar_system) scalar



## TRAIN FUNCTIONS
# training is comprised of:
# train_steps(I)
    # GRPO_steps(M)
        # grad_update_iterasions(mu)
@functools.partial(jax.jit, static_argnames=["mu", "epsilon", "dkl_beta"])
def run_grad_update_iterations(
        # arrays and giant values
        policy_model_params: PMParams,
        reference_policy_outputs: jax.Array,
        old_policy_outputs: jax.Array,
        trajectory_states: jax.Array,
        A: jax.Array,
        # dynamic args
        learning_rate: float,
        # static args
        mu: int,
        epsilon: float,
        dkl_beta: float
        ) -> Tuple[PMParams, float, PMParams]:
    def scan_fn(params, _unused):
        # GRPO objective function and grads
        objective, grads = jax.value_and_grad(get_objective)(
            params, reference_policy_outputs,
            old_policy_outputs, trajectory_states, A,
            epsilon, dkl_beta
        )
        # maximize objective function
        updated_params = jax.tree_util.tree_map(lambda p, g: p + g*learning_rate, params, grads) # element-wise sum of two structures
        return updated_params, (objective, jax.tree_util.tree_map(jnp.linalg.norm, grads))
    updated_policy_model_params, output = jax.lax.scan(scan_fn, policy_model_params, None, length=mu)
    objectives, grad_norms = output
    mean_objective = jnp.mean(objectives)
    grad_norms = jax.tree_util.tree_map(jnp.mean, grad_norms)
    return updated_policy_model_params, mean_objective, grad_norms




# run M iterations of generating groups, scoring the advantages, and updating the policy model gradients mu times
@functools.partial(jax.jit, static_argnames=[
    "M",
    "batch_size",
    "planets",
    "suns",
    "G",
    "outcome_supervised",
    "trajectory_horizon",
    "mu",
    "epsilon",
    "dkl_beta",
])
def run_GRPO_iterations(
        # giant arrays
        policy_model_params: PMParams,
        # dynamic args
        key: jax.Array,
        learning_rate: float,
        # static args
        M: int,
        batch_size: int,
        planets: int,
        suns: int,
        G: int, # the size of each group in GRPO
        outcome_supervised: bool, # determines whether the function is outcome or x supervised. static. unused branch is removed after compilation
        trajectory_horizon: int,
        mu: int,
        epsilon: float,
        dkl_beta: float,
        ) -> Tuple[PMParams, float, float]:
    reference_policy_model_params = policy_model_params # update reference policy
    def scan_fn(params, scan_key):
        # sample a batch Db
        rolling_key, _ = jrand.split(scan_key, 2) # reroll key before every use
        solar_system_batch = init_solarsystems(rolling_key, batch_size, planets, suns) # (batch_size, solar_system)
        # old_policy <- policy
        # old_policy_model_params = policy_model_params
        # generate G outputs for each q in Db
        # convert (batch_size, *) arrays in SoA to (G, batch_size, *) arrays
        duplicate_batch_G_times = lambda batch: jnp.repeat(jnp.expand_dims(batch, 0), G, axis=0)
        solar_system_G = jax.tree_util.tree_map(duplicate_batch_G_times, solar_system_batch) # ugly but i am not rewriting the entire codebase to get around it
        # setup a function to run a trajectory for each initial state (solar_system) in (G, batch_size)
        # first get (G, batch_size) keys
        rolling_key, _ = jrand.split(rolling_key, 2)
        trajectory_keys = jnp.reshape(jrand.split(rolling_key, G), (G, 2)) # each batch group only requires a single key, since rand ops are batched.
        #run_batch_trajectory = jax.vmap(run_single_trajectory, in_axes=(None, 0, None, 0))
        if outcome_supervised:
            run_group_trajectory = jax.vmap(run_batch_trajectories_outcome_supervised, in_axes=(None, 0, None, 0))
            old_policy_outputs, trajectory_states, outcome_rewards = run_group_trajectory(
                params, solar_system_G, trajectory_horizon, trajectory_keys
            )    # (G, batch_size, trajectory_horizon, logits) (G, batch_size, trajectory_horizon, solar_system) (G, batch_size)
            A = (outcome_rewards - jnp.mean(outcome_rewards, axis=0, keepdims=True)) / (1e-7 + jnp.std(outcome_rewards, axis=0, keepdims=True)) # (G, batch_size)
            A = A[:, :, jnp.newaxis, jnp.newaxis] # (G, batch_size) => (G, batch_size, step_o, logprobs)
        else:
            run_group_trajectory = jax.vmap(run_batch_trajectories_process_supervised, in_axes=(None, 0, None, 0))
            old_policy_outputs, trajectory_states, process_rewards = run_group_trajectory(
                params, solar_system_G, trajectory_horizon, trajectory_keys
            )
            # avr/std of R are across G and oi
            # A = (process_rewards - jnp.mean(process_rewards, axis=(0, 2), keepdims=True)) / (1e-7 + jnp.std(process_rewards, axis=(0, 2), keepdims=True))
            A = (process_rewards - jnp.mean(process_rewards, axis=0, keepdims=True)) / (1e-7 + jnp.std(process_rewards, axis=0, keepdims=True))
            A = A[:, :, :, jnp.newaxis] # (G, batch_size, step_o) => (G, batch_size, step_o, logprobs)
            A = jnp.cumsum(A[:, :, ::-1, :], axis=2)[:, :, ::-1, :] / A.shape[2] # sum along step axis
        
        # get the reference policy
        replay_group_trajectories = jax.vmap(replay_batch_trajectories, in_axes=(None, 0)) # get reference policy of current group
        reference_policy_outputs = replay_group_trajectories(reference_policy_model_params, trajectory_states)
        # run iterations of gradient descent
        updated_policy_model_params, grad_update_mean_objective, grad_norms = run_grad_update_iterations(
            params,
            reference_policy_outputs,
            old_policy_outputs,
            trajectory_states,
            A,
            learning_rate,
            mu,
            epsilon,
            dkl_beta
        )
        # get mean reward
        if outcome_supervised:
            mean_reward = jnp.mean(outcome_rewards)
        else:
            mean_reward = jnp.mean(process_rewards)
        return updated_policy_model_params, (mean_reward, grad_update_mean_objective, grad_norms)

    # setup and do scan
    scan_keys = jrand.split(key, M)    
    updated_policy_model_params, stats = jax.lax.scan(scan_fn, policy_model_params, scan_keys)
    # unpack and clean stats
    rewards, objectives, grad_norms = stats
    mean_rewards = jnp.mean(rewards)
    mean_objective = jnp.mean(objectives)
    grad_norms = jax.tree_util.tree_map(jnp.mean, grad_norms)
    # return everything 
    return updated_policy_model_params, mean_rewards, mean_objective, grad_norms


