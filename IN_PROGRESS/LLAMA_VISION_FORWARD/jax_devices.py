import jax
import jax.numpy as jnp


print(jax.devices())

from jax.sharding import PartitionSpec as P


mesh = jax.make_mesh((2, 4), ('x', 'y'))
sharding = jax.sharding.NamedSharding(mesh, P('x', 'y'))

arr = jnp.arange(32).reshape(4,8)
arr = jax.device_put(arr, sharding)


print(arr.devices())
print(arr.sharding)