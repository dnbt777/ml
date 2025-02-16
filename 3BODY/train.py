import jax.numpy as jnp
import jax.random as jrand
import jax
import functools
from environment import init_solarsystems, step_simulation, get_reward
from GRPO import get_decision_probs, init_policy_model
from file_utils import load_model_params, save_model_params
import time



## ------------- ######
#### - FUNCTIONS - ####
###### ------------- ##

@functools.partial(jax.jit, static_argnames=["epsilon", "dkl_beta"])
def get_objective_outcome_supervised(new_policy_model_params,
                                old_policy_outputs, trajectory_states,
                                epsilon, dkl_beta):
  # set up keys, get shape sizes
  G, batch_size, trajectory_horizon, _ = old_policy_outputs.shape # (G, batch_size, trajectory_horizon, logits)
  random_keys = jrand.split(key, G*batch_size).reshape((G, batch_size, 2))
  # replay all trajectories to get new_policy probs
  replay_batch_trajectories = jax.vmap(replay_single_trajectory, in_axes=(None, 0, 0))
  replay_group_trajectories = jax.vmap(replay_batch_trajectories, in_axes=(None, 0, 0))                           
  new_policy_outputs = replay_group_trajectories(new_policy_model_params, trajectory_states, random_keys) # (G, batch_size, trajectory_horizon, logits)
  # outcome supervision GRPO equation
  # loss = - (1/G) * (1/2000) * sum_across_G(sum_across_steps((min(prob_ratios * advantages.extend(), min(prob_ratios, 1 + epsilon, 1 - epsilon)) - kl_divergence)))
  prob_ratios = new_policy_outputs / (old_policy_outputs + 1e-7)
  kl_divergence = old_policy_outputs / (new_policy_outputs + 1e-7) - (jnp.log(old_policy_outputs + 1e-7) - jnp.log(new_policy_outputs + 1e-7)) - 1
  # calculate the equations in parts for debugging
  # this gets compiled anyways, so its likely not memory inefficient
  xa = prob_ratios * A
  xb = jnp.clip(prob_ratios, 1 + epsilon, 1 - epsilon) * A
  xc = jnp.minimum(xa,xb)
  xd = xc - (dkl_beta * kl_divergence)
  xe = jnp.sum(xd, axis=-1) #* (1/7) # across logit axis (batch, g, step, logit) => (batch, g, step) 
  xf = jnp.sum(xe, axis=-1) * (1/trajectory_horizon) # across step axis (batch, g, step) => (batch, g) -- get loss for each g
  xg = jnp.sum(xf, axis=-1) #* (1/batch_size) # combine all batch losses.
  xh = jnp.sum(xg, axis=-1) * (1/G) # across g axis (batch, g) => batch. get loss for each batch.
  objective = xh
  return objective

# given a starting state solar_system, runs a trajectory for $trajectory_horizon steps
@functools.partial(jax.jit, static_argnames=["trajectory_horizon"])
def run_single_trajectory(policy_model_params, solar_system, trajectory_horizon, trajectory_key):
  # carry, [a] => (carry, [b])
  # state, None => (state, (old_policy_probs, new_policy_probs))
  # state: trajectory_key, solar_systems
  # [a]: None
  def scan_through_trajectory(state, i):
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
  state, trajectory = jax.lax.scan(scan_through_trajectory, init_state, None, length=trajectory_horizon)
  # unpack
  solar_system_end_state, final_trajectory_key = state # scalar scalar
  policy_outputs, trajectory_states = trajectory # (step, logits) (step, solar_system)
  # end_reward = reward(state)
  outcome_reward = get_reward(solar_system_end_state) # scalar
  return policy_outputs, trajectory_states, outcome_reward # (trajectory_horizon, logits) (trajectory_horizon, solar_system) scalar

# replay a single trajectory
@functools.partial(jax.jit, static_argnames=["trajectory_horizon"])
def replay_single_trajectory(policy_model_params, solar_system, trajectory_horizon, trajectory_key):
  def scan_through_trajectory(state):
    solar_systems, scan_key = state
    decision_probs_oi = get_decision_probs(policy_model_params, solar_systems)
    scan_key, _ = jrand.split(scan_key, 2) # roll scan_key
    action = jrand.categorical(scan_key, decision_probs_oi, axis=-1) # Scalar
    solar_system = step_simulation(solar_system, action)
    return (solar_system, scan_key), (decision_probs_oi, solar_system)
  # pack
  init_state = (solar_system, trajectory_key)
  # scan
  state, trajectory = jax.lax.scan(scan_through_trajectory, init_state, None, length=trajectory_horizon)
  # unpack
  solar_system_end_state, final_trajectory_key = state # scalar scalar
  policy_outputs, trajectory_states = trajectory # (step, logits) (step, solar_system)
  # end_reward = reward(state)
  outcome_reward = get_reward(solar_system_end_state) # scalar
  return policy_outputs, trajectory_states, outcome_reward # (trajectory_horizon, logits) (trajectory_horizon, solar_system) scalar



###### --------------- ##
#### - PARAM SETUP - ####
## --------------- ######

# model hyperparams
hidden_size = 64
hidden_layers = 16
input_datapoints = 3*4 + 3*4 + 1*4
output_actions = 7 # lr/ud/bf/nothing

# sim hyperparams
planets = 1
suns = 3
trajectory_horizon = 20 # run simulation for n steps

# GRPO hyperparams https://arxiv.org/pdf/2402.03300
G = 64 # 512 # paper: 64 outputs
batch_size = 128 # 16 # paper uses 1024
update_old_policy = 20 # update every 10 iters

epsilon = 0.1 # for clipping - https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
train_iters = 4000 # arbitrary. how long we want to train for
dkl_beta = 0.0001 # https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
rolling_key = jrand.PRNGKey(0) # init rolling key

learning_rate = 3e-3
learning_rate_decay = 0.001

debug = False
if debug:
  jax.config.update("jax_disable_jit", True)
  jax.config.update("jax_debug_infs", False)
  jax.config.update("jax_debug_nans", True)
else:
  import warnings
  warnings.filterwarnings("ignore")



###### --------- ######
  #### - TRAIN - ####
###### --------- ######

start = time.time()
###################


#> page 14 - https://arxiv.org/pdf/2402.03300 <#
initial_policy_model_params = init_policy_model(hidden_layers, hidden_size, input_datapoints, output_actions)
policy_model_params = initial_policy_model_params
I = 20
for iteration in range(I):
  reference_policy_model_params = policy_model_params
  M = 100
  for step in range(M):
    # sample a batch Db
    rolling_key, _ = jrand.split(rolling_key, 2) # reroll key before every use
    solar_system_batch = init_solarsystems(rolling_key, batch_size, planets, suns) # (batch_size, solar_system)
    # old_policy <- policy
    old_policy_model_params = policy_model_params
    # generate G outputs for each q in Db
    # convert (batch_size, *) arrays in SoA to (G, batch_size, *) arrays
    duplicate_batch_G_times = lambda batch: jnp.repeat(jnp.expand_dims(batch, 0), G, axis=0)
    solar_system_G = jax.tree_util.tree_map(duplicate_batch_G_times, solar_system_batch) # ugly but i am not rewriting the entire codebase to get around it
    # setup a function to run a trajectory for each initial state (solar_system) in (G, batch_size)
    # first get (G, batch_size) keys
    rolling_key, _ = jrand.split(rolling_key, 2)
    trajectory_keys = jnp.reshape(jrand.split(rolling_key, G*batch_size), (G, batch_size, 2))
    run_batch_trajectory = jax.vmap(run_single_trajectory, in_axes=(None, 0, None, 0))
    run_group_trajectory = jax.vmap(run_batch_trajectory, in_axes=(None, 0, None, 0))
    old_policy_outputs, trajectory_states, outcome_rewards = run_group_trajectory(
      old_policy_model_params, solar_system_G, trajectory_horizon, trajectory_keys
      )  # (G, batch_size, trajectory_horizon, logits) (G, batch_size, trajectory_horizon, solar_system) (G, batch_size)
    # get the advantage across the G dim (first axis)
    A = (outcome_rewards - jnp.mean(outcome_rewards, axis=0, keepdims=True)) / jnp.std(outcome_rewards, axis=0, keepdims=True) # (G, batch_size)
    A = A[:, :, jnp.newaxis, jnp.newaxis] # (G, batch_size) => (G, batch_size, step_o, logprobs) ## this lets us do (A * old_policy / new_policy) later
    # new_policy <- policy
    new_policy_model_params = policy_model_params
    mu = 100
    for GRPO_iteration in range(mu):
      # GRPO objective function and grads
      objective, grads = jax.value_and_grad(get_objective_outcome_supervised)(
        new_policy_model_params,
        old_policy_outputs, trajectory_states, A,
        epsilon, dkl_beta
      )
      # maximize objective function
      new_policy_model_params = jax.tree_util.tree_map(lambda p, g: p + g, new_policy_model_params, grads) # element-wise sum of two structures
      print(iteration, step, GRPO_iteration, objective)
    # update policy
    # the 4th policy model var 'policy_model_params' is not needed
    policy_model_params = new_policy_model_params


#################
end = time.time()

save_model_params(new_policy_model_params, "params.pkl")
