import jax
import jax.numpy as jnp
import functools
import optax

from typing import Tuple

from math_utils import crossentropyloss, maxpool, convolve2D
from cnn_utils import CNNParams


@functools.partial(jax.jit, static_argnames=["optimizer"])
def grad_update_step(
    cnn_params: CNNParams,
    x_batch: jax.Array,
    y_batch: jax.Array,
    opt_state: optax.OptState,
    optimizer
    ) -> Tuple[CNNParams, optax.OptState, float]:
  loss, grads = jax.value_and_grad(get_loss)(cnn_params, x_batch, y_batch)
  updates, updated_opt_state = optimizer.update(grads, opt_state)
  updated_cnn_params = optax.apply_updates(cnn_params, updates)
  return updated_cnn_params, updated_opt_state, loss


# Predict yhat from x, and get the loss 
@jax.jit
def get_loss(
    cnn_params: CNNParams,
    x_batch: jax.Array,
    y_batch: jax.Array
    ) -> float:
  logits = jax.vmap(cnn_forward, in_axes=(None, 0))(cnn_params, x_batch)
  loss = jax.vmap(crossentropyloss, in_axes=(0, 0))(logits, y_batch)
  return jnp.mean(loss)


# Get the accuracy of the model on a validation batch
@jax.jit
def get_accuracy(
    cnn_params: CNNParams,
    x_test_batch: jax.Array,
    y_test_batch: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
  # get the outputs of the validation batch
  logits = jax.vmap(cnn_forward, in_axes=(None,0))(cnn_params, x_test_batch)
  losses = jax.vmap(crossentropyloss, in_axes=(0, 0))(logits, y_test_batch)
  yhat_test_batch = jnp.argmax(logits, axis=-1)
  y_test_batch = jnp.argmax(y_test_batch, axis=-1)
  # calculate how many accurately predicted the correct digit
  number_correct = jnp.sum(yhat_test_batch == y_test_batch)
  samples = x_test_batch.shape[0]
  fraction_correct = number_correct / samples 
  return jnp.mean(losses), fraction_correct 


@jax.jit
def cnn_forward(
    cnn_params: CNNParams,
    x: jax.Array
    ) -> jax.Array:
  # do conv, then relu, then maxpool, then fc.
  x = convolve2D(x, cnn_params.conv1)
  x = jax.nn.relu(x)

  # residual/skip connnection. mode=same to output the same shape.
  x_residual = convolve2D(x, cnn_params.conv2)
  x_residual = jax.nn.relu(x_residual)
  x = x + x_residual
  x = jax.nn.relu(x)

  # maxpool 2x2 
  x = maxpool(x)
  x = jnp.squeeze(x, axis=-1) # (24, 24, 1) => (24, 24)

  # convolve again
  x = convolve2D(x, cnn_params.conv3)
  x = jax.nn.relu(x)

  # residual convolve again
  x_residual = convolve2D(x, cnn_params.conv4)
  x_residual = jax.nn.relu(x_residual)
  x = x + x_residual
  x = jax.nn.relu(x)

  # collapse all dims to make a 1D array 
  x = jnp.ravel(x)
  
  # fully connected layer - do one projection, then another to output space
  x = jax.nn.relu(x @ cnn_params.fc1.w + cnn_params.fc1.b)
  x = x @ cnn_params.fc2.w + cnn_params.fc2.b
  
  return x

