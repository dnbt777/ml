import jax.numpy as jnp
import jax.random as jrand
import jax
import functools
from environment import init_solarsystems, step_simulation, get_reward
from GRPO import get_decision_probs, init_policy_model
from file_utils import load_model_params, save_model_params
from train_utils import *
import time


# todo:
# figure out why grads are 0. where precisely is this happening?
# possibly debug w vjp and pullback
# https://www.kaggle.com/code/goktugguvercin/automatic-differentiation-in-jax#Pushforward-and-Pullback:
# create train_utils and then jit the entire training process there


###### --------------- ##
#### - PARAM SETUP - ####
## --------------- ######

# model hyperparams
hidden_size = 16 
hidden_layers = 16 
input_datapoints = 3*4 + 3*4 + 1*4
output_actions = 7 # lr/ud/bf/nothing

# sim hyperparams
planets = 1
suns = 3
trajectory_horizon = 200 # run simulation for n steps

# GRPO hyperparams https://arxiv.org/pdf/2402.03300
outcome_supervised = False
G = 12 # 512 # paper: 64 outputs
batch_size = 8 # paper uses 1024
epsilon = 0.3 # for clipping - https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
dkl_beta = 0.03 # https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
learning_rate = 3e-4
learning_rate_decay = 0.9

# Train loop hyperparams
I = 4 # iterations of M (and reference policy updates)
M = 8 # iterations of GRPO generation. (Iterations before updating the old model)
mu = 8 # iterations of grad updates on grpo results


debug = False 
if debug:
  jax.config.update("jax_disable_jit", False)
  jax.config.update("jax_debug_infs", True)
  jax.config.update("jax_debug_nans", True)
else:
  import warnings
  warnings.filterwarnings("ignore")

# TODO
# hyperparameter optimization?


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
MODEL_DTYPE = jnp.float32 # experimental, seems to train worse if not f32
policy_model_params = jax.tree_util.tree_map(lambda arr: arr.astype(MODEL_DTYPE), policy_model_params)

# train
for iteration in range(I):
    learning_rate = learning_rate*learning_rate_decay
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
    print(grad_norms)



#################
end = time.time()
print(end - start, "seconds")


##                ###
##  -  logging  -  ##
###                ##

import os
import pandas as pd
from matplotlib import pyplot as plt
train_run_folder = "IN_PROGRESS/3BODY_GRPO/train_runs"
existing_train_runs = len(os.listdir(train_run_folder))
current_run = existing_train_runs + 1
current_run_folder = f"{train_run_folder}/{current_run}"
os.mkdir(current_run_folder)
# save model
save_model_params(policy_model_params, f"{current_run_folder}/params.pkl")
# save data
col_names = ["iteration", "step", "GRPO_iteration", "objective", "outcome_rewards"]
df = pd.DataFrame(data, columns=col_names)
df.to_csv(f"{current_run_folder}/data.csv")
# output plots
xstep, objectives, avg_reward = zip(*[(I*iteration + mu*step + GRPO_iteration, objective, avg_reward) for iteration, step, GRPO_iteration, objective, avg_reward in data])
plt.figure(0)
plt.plot(xstep, objectives)
plt.title('step vs objective value')
plt.savefig(f"{current_run_folder}/objective")
plt.figure(1)
plt.plot(xstep, avg_reward)
plt.title('step vs mean group reward')
plt.savefig(f"{current_run_folder}/reward")