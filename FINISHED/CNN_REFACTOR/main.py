import jax.numpy as jnp
import jax.random as jrand 
import optax
import time

from train_utils import get_accuracy, grad_update_step
from cnn_utils import init_cnn_params
from data_loading_utils import load_MNIST



####################
### DATALOADING: ###
### MNIST ~~~~~ ####
####################

print("Loading MNIST...")

data_load_start = time.time()
classes, x_train, y_train, x_test, y_test = load_MNIST()
data_load_end = time.time()
data_load_time = data_load_end - data_load_start

print(f"MNIST loaded in {data_load_time:0.2f}s")



##               ##
### TRAIN SETUP ###
##               ##

# initialize parameters
input_shape = (1, 28, 28)
output_shape = (10,)
model_dtype = jnp.float32 # breaks if not float32 :(
rolling_key = jrand.PRNGKey(29383)
cnn_params = init_cnn_params(rolling_key, output_shape, dtype=model_dtype)
rolling_key, _ = jrand.split(rolling_key, 2)

# initialize adam optimizer
learning_rate = 3e-3
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(cnn_params)

# train loop params
epochs = 50
train_batch_size = 8192



## ############## ##
### ~~ TRAIN ~~  ###
## ############## ##

start = time.time()
#########################


# Train!
for epoch in range(epochs):
  batches = x_train.shape[0] // train_batch_size
  total_samples = batches*train_batch_size
  # prep train data - randomize and reshape into batches
  train_indices = jrand.permutation(rolling_key, x_train.shape[0])
  rolling_key, _ = jrand.split(rolling_key, 2)
  x_train_randomized = x_train[train_indices][:total_samples] # make sure sample size is a multiple of batch size
  y_train_randomized = y_train[train_indices][:total_samples] # same indices, same x->y pairs
  reshape_to_batch = lambda arr, batch_size: jnp.reshape(arr, (batch_size, arr.shape[0]//batch_size, *arr.shape[1:])) 
  x_train_batches = reshape_to_batch(x_train_randomized, train_batch_size)
  y_train_batches = reshape_to_batch(y_train_randomized, train_batch_size)
  # train over each batch
  # i attempted to jit this part, but the compute graph took up too much memory.
  for batch_idx in range(batches):
    # get grads  from train data
    cnn_params, opt_state, loss = grad_update_step(cnn_params, x_train_batches[batch_idx], y_train_batches[batch_idx], opt_state, optimizer)
  # randomize test set, then get val loss and accuracy
  test_indices = jrand.permutation(rolling_key, x_test.shape[0])
  rolling_key, _ = jrand.split(rolling_key, 2)
  test_batch_size = train_batch_size//4
  val_loss, val_accuracy = get_accuracy(cnn_params, x_test[test_indices][:test_batch_size], y_test[test_indices][:test_batch_size])
  # print epoch results
  print(f"epoch{epoch}/{epochs}|batch{batch_idx}/{batches}|loss={loss:0.4f}|val_loss={val_loss:0.4f}|val_accuracy={val_accuracy:0.5f}")


#########################
end = time.time()


## Print stats ##
train_time = end - start
samples = epochs*batches*train_batch_size
samples_per_second = samples / train_time
print(f"samples/sec={samples_per_second:0.4f}") # includes jit compilation time, which makes the rate seem slower



##########################
######### RENDER
######

from render import render
render(cnn_params, x_test)

## Controls:
#   - rclick: draw
#   - p: predict the digit
#   - c: clear
#   - l: load a digit from the test set
