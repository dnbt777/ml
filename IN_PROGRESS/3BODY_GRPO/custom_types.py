import jax
from typing import NamedTuple, List


# SoA
# PolicyModel
class PMLayer(NamedTuple):
  weight : jax.Array
  bias : jax.Array
  lnw: jax.Array
  lnb: jax.Array

class PMParams(NamedTuple):
  wi : jax.Array
  bi :jax.Array
  lnwi: jax.Array
  lnbi: jax.Array
  wo : jax.Array
  bo : jax.Array
  hidden_layers : List[PMLayer] # do it this way for scanning