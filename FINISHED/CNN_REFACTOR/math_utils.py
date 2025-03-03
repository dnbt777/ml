import jax
import jax.numpy as jnp
import flax

# jit is probably not necessary here.
# since this func is only called inside other jitted functions, it automatically gets jitted
@jax.jit
def crossentropyloss(
    logits: jax.Array,
    y: jax.Array
    ) -> float:
  yhat = jax.nn.log_softmax(logits)
  return jnp.sum(-yhat * y)


@jax.jit
def convolve2D(
    layers: jax.Array,
    kernels: jax.Array
    ) -> jax.Array:
  def convolve_layer_scan_fn(carry_state, kernel):
      layer = jax.scipy.signal.convolve(layers, kernel, mode="same")[0]
      return None, layer
  _, layers = jax.lax.scan(convolve_layer_scan_fn, None, kernels)
  return layers


@jax.jit
def maxpool(
    layers: jax.Array
    ) -> jax.Array:
  maxpool_channel = lambda channel: flax.linen.max_pool(channel[:, :, jnp.newaxis], window_shape=(2, 2))
  return jax.vmap(maxpool_channel, in_axes=0)(layers)