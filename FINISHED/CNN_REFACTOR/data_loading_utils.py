
import torch
import torchvision
import torchvision.transforms as transforms
import jax
import jax.numpy as jnp

def load_MNIST():
  # first load the dataset
  train_data = torchvision.datasets.MNIST(root = './', train=True, download=True, transform=transforms.ToTensor())
  test_data = torchvision.datasets.MNIST(root = './', train=False, download=True, transform=transforms.ToTensor())

  # convert to jnp/np
  x_train, y_train = zip(*train_data)
  x_train, y_train = jnp.array(x_train), jnp.array(y_train)

  x_test, y_test = zip(*test_data)
  x_test, y_test = jnp.array(x_test), jnp.array(y_test)

  # convert ys to one-hot
  classes = len(set(y_train.tolist()))
  y_train = jax.nn.one_hot(y_train, classes) # from n -> one-hot of n
  y_test = jax.nn.one_hot(y_test, classes)

  return classes, x_train, y_train, x_test, y_test