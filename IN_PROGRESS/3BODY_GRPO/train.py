import jax.numpy as jnp
import jax.random as jrand
import jax
from GRPO import init_policy_model
from file_utils import save_model_params
from train_utils import run_GRPO_iterations
import time


# TODO CLEANUP: replace all glob imports with precise ones
# TODO CLEANUP: add proper type signatures to all functions




###### --------------- ##
#### - PARAM SETUP - ####
## --------------- ######

# sim hyperparams
planets = 1 # 1
suns = 1 # 3 # TODO test 1 sun instead of 3
trajectory_horizon = 50 # run simulation for n steps

# model hyperparams
hidden_size = 64 
hidden_layers = 32 # TODO try shorter layers 
input_datapoints = 3*4 + 3*4 + 1*4 # TODO update to calculate based on the suns and planets vars. will break in current form
output_actions = 7 # lr/ud/bf/nothing


# GRPO hyperparams https://arxiv.org/pdf/2402.03300
outcome_supervised = False 
G = 128 # 512 # paper: 64 outputs
batch_size = 8 # paper uses 1024
epsilon = 0.5 # for clipping - https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
dkl_beta = 0.004 # https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
learning_rate = 3e-4
learning_rate_decay = 0.99

# OPTIMIZATION THE PROBLEM IS THAT THE MODEL CAN'T SEE RIGHT!! ITS AN ARCHITECTURE AND ENV VISION PROBLEM!!!!

# Train loop hyperparams
I = 200 # iterations of M (and reference policy updates)
M = 4 # iterations of GRPO generation. (Iterations before updating the old model)
mu = 4 # iterations of grad updates on grpo results


debug = False 
if debug:
  jax.config.update("jax_disable_jit", True)
  # jax.config.update("jax_debug_infs", True)
  # jax.config.update("jax_debug_nans", True)
else:
  import warnings
  warnings.filterwarnings("ignore")


# TODO hyperparameter optimization - convert hparams to structure and then sweep across SoA (pmap + cloud GPUs? create hparam_sweep.py)


###### --------- ######
  #### - TRAIN - ####
###### --------- ######

start = time.time()
###################


#> page 14 - https://arxiv.org/pdf/2402.03300 <#
rolling_key = jrand.PRNGKey(0) # init rolling key
policy_model_params = init_policy_model(rolling_key, hidden_layers, hidden_size, input_datapoints, output_actions)
rolling_key, _ = jrand.split(rolling_key, 2)
data = [] # graphing progress

# cast model to new dtype
# MODEL_DTYPE = jnp.float32 # experimental, seems to train worse if not f32
# policy_model_params = jax.tree_util.tree_map(lambda arr: arr.astype(MODEL_DTYPE), policy_model_params)

# train
data = []
for iteration in range(I):
    learning_rate = learning_rate*learning_rate_decay
    rolling_key, _ = jrand.split(rolling_key, 2)
    policy_model_params, mean_rewards, mean_objective, grad_norms = run_GRPO_iterations(
        # giant arrays
        policy_model_params,
        # dynamic args
        rolling_key,
        learning_rate,
        # static args
        M,
        batch_size,
        planets,
        suns,
        G, # the size of each group in GRPO
        outcome_supervised, # determines whether the function is outcome or x supervised. static. unused branch is removed after compilation
        trajectory_horizon,
        mu,
        epsilon,
        dkl_beta,
    )
    print(f"I={iteration}|rewards={mean_rewards:.06f}|objective={mean_objective:.06f}|lr={learning_rate:.4e}")
    data.append((iteration, mean_objective, mean_rewards))
    print(grad_norms) # (extremely useful if model is not learning)
    # print(jax.tree_util.tree_map(jnp.mean, policy_model_params)) # print model average weight sizes



#################
end = time.time()
duration = end - start
grad_updates = I*M*mu
samples_trained = grad_updates*batch_size*G
print(f"{duration:0.2f}s, {samples_trained/duration:0.2f} samples/s")


##                ###
##  -  logging  -  ##
###                ##

import os
import pandas as pd
from matplotlib import pyplot as plt
train_run_folder = "./train_runs"
existing_train_runs = len(os.listdir(train_run_folder))
current_run = existing_train_runs + 1
current_run_folder = f"{train_run_folder}/{current_run}"
os.mkdir(current_run_folder)
# save model
save_model_params(policy_model_params, f"{current_run_folder}/params.pkl")
# save data
col_names = ["iteration", "objective", "outcome_rewards"]
df = pd.DataFrame(data, columns=col_names)
df.to_csv(f"{current_run_folder}/data.csv")
# output plots
xstep, objectives, avg_reward = zip(*[(I*iteration, objective, avg_reward) for iteration, objective, avg_reward in data])
plt.figure(0)
plt.plot(xstep, objectives)
plt.title('step vs objective value')
plt.savefig(f"{current_run_folder}/objective")
plt.figure(1)
plt.plot(xstep, avg_reward)
plt.title('step vs mean group reward')
plt.savefig(f"{current_run_folder}/reward")