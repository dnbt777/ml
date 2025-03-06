import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import optax
import functools # for partial
import time
from typing import List, NamedTuple


# https://github.com/xjdr-alt/simple_transformer/blob/main/simple_transformer.py
class BlockParams(NamedTuple):
  w_q : jax.Array
  w_k : jax.Array
  w_v : jax.Array
  w_o : jax.Array
  w1 : jax.Array
  w2 : jax.Array
  w3 : jax.Array
  attn_norm_w : jax.Array
  attn_norm_b : jax.Array
  ffnorm_w : jax.Array
  ffnorm_b : jax.Array


class ModelParams(NamedTuple):
  blocks : List[BlockParams]
  embedding_projection : jax.Array
  to_logits_w : jax.Array # after entire network
  to_logits_b : jax.Array
  positional_embeddings : jax.Array
  output_norm_w : jax.Array
  output_norm_b : jax.Array


def init_model_params(blocks, model_dim, d_k, qkv_dim, ff_hidden_size, vocab_size, block_size):
  D = d_k
  K = qkv_dim
  attention_heads = model_dim // d_k
  H = attention_heads
  DH = attention_heads * d_k
  scale = lambda s1, s2: 1 / jnp.sqrt(s1 + s2)
  xavier_blocks = lambda n, m: np.random.uniform(size=(blocks, n, m)) * scale(n, m)
  xavier_multihead_blocks = lambda n, m: np.random.uniform(size=(blocks, H, n, m))
  xavier = lambda n, m: np.random.uniform(size=(n, m)) * scale(n, m)
  
  block_params = BlockParams(
    # multi head attention
    # C => D, H
    w_q=xavier_multihead_blocks(D, K),
    w_k=xavier_multihead_blocks(D, K),
    w_v=xavier_multihead_blocks(D, K),
    w_o=xavier_multihead_blocks(K, D),
    # mlp forward
    #w=xavier_blocks(model_dim, model_dim),
    #b=jnp.zeros(shape=(blocks, model_dim)),
    w1=xavier_blocks(model_dim, ff_hidden_size),
    w2=xavier_blocks(model_dim, ff_hidden_size),
    w3=xavier_blocks(ff_hidden_size, model_dim),
    # norm stuff
    attn_norm_w=xavier_blocks(1, model_dim), # B, T, C (x) B, 1, C
    attn_norm_b=jnp.zeros(shape=(blocks, model_dim,)),
    ffnorm_w=xavier_blocks(1, model_dim),
    ffnorm_b=jnp.zeros(shape=(blocks, model_dim,)),
  )

  model_params = ModelParams(
    blocks=block_params,
    embedding_projection=xavier(vocab_size, model_dim),
    to_logits_w=xavier(model_dim, vocab_size),
    to_logits_b=jnp.zeros(shape=(vocab_size,)),
    positional_embeddings=xavier(block_size, model_dim), # bias: T, C
    output_norm_w=xavier(1, model_dim),
    output_norm_b=jnp.zeros(shape=(model_dim,)),
  )

  return model_params


@functools.partial(jax.jit, static_argnames=["dropout_rate", "d_k"])
def attention(block_params : BlockParams, xBTC, dropout_key, dropout_rate=0.0, d_k=128):
  # xBTC -> xBTHD via reshape | H = head, D = dim (like channel, but split into H segments)
  B, T, C = xBTC.shape
  H = block_params.w_q.shape[0] # attention head
  D = block_params.w_q.shape[1] # model dim after split into heads
  K = block_params.w_q.shape[2] # query dim
  BTHD = (B, T, H, D)
  xBTHD = jnp.reshape(xBTC, shape=BTHD)
  # xBTHD -> xBHTD via transpose or axes swap
  xBHTD = jnp.swapaxes(xBTHD, 1, 2)
  # xBHTD @ (Wq, Wk, Wv) => (Q, K, V)   | (B, H, T, D) @ (D, K) => (B, H, T, K) | K = query/key size. also V size here, but does not have to be.
  Q = jnp.einsum("BHTD,HDK->BHTK", xBHTD, block_params.w_q) # (B, H, T, D) @ (H, D, K) => (B, H, T, K)
  K = jnp.einsum("BHTD,HDK->BHTK", xBHTD, block_params.w_k)
  V = jnp.einsum("BHTD,HDK->BHTK", xBHTD, block_params.w_v)
  # Q @ K_transpose => QK               | (B, H, T, K) @ (B, H, K, T) => (B, H, T, T) | this may be the wrong transpose
  QK = Q @ jnp.swapaxes(K, 2, 3)
  # scale, mask, and softmax
  mask = jnp.triu(jnp.ones_like(QK), k=1) #  mask=1 where jnp.-inf will be. k=1 preseves the middle diagonal for self attention
  attention_scores = jnp.where(mask, -jnp.inf, QK/jnp.sqrt(d_k))
  # QK @ V => Z                         | (B, H, T, T) @ (B, H, T, K) => (B, H, T, K)
  Z = jax.nn.softmax(attention_scores, axis=-1) @ V # softmax in the T and K dims, for each B and H
  # Z swap axes                         | (B, H, T, K) => (B, T, H, K)
  Z = dropout(dropout_key, Z, dropout_rate)
  Z = jnp.swapaxes(Z, 1, 2)
  # Z @ w_out       | project to BTHD   | (B, T, H, K) @ (H, K, D) => (B, T, H, D)
  xBTHD = jnp.einsum("BTHK,HKD->BTHD", Z, block_params.w_o)
  # reshape to BTC                      | (B, T, H, D) => (B, T, C)
  xBTC = jnp.reshape(xBTHD, (B, T, C))
  # return xBTC_out
  xBTC = dropout(dropout_key, xBTC, dropout_rate)
  return xBTC



@functools.partial(jax.jit, static_argnames=["dropout_rate", "vocab_size"])
def forward(rolling_key, model_params: ModelParams, xBT, vocab_size, dropout_rate=0.0):
  # get embeddings via projection from token onehot to channel space
  # xBT -> xBTC
  xBTOH = jax.nn.one_hot(xBT, num_classes=vocab_size, axis=-1)
  xBTC = xBTOH @ model_params.embedding_projection
  # add learned positional embeddings. broadcast over B
  B, T, C = xBTC.shape
  xBTC = xBTC + model_params.positional_embeddings[None, :T, :] # B, T, C + 1, T, C
  # scan through transformer blocks, updating xBTC
  # block
  #   attention
  #   forward projection from Z space to channel space
  # scan_fn :: (c, a) -> (c, b)
  def ffw(block_params, xBTC):
    #return jax.nn.silu(xBTC @ block_params.w1) * ((xBTC @ block_params.w2 + block_params.b2) @ block_params.w3 + block_params.b3)
    return (jax.nn.silu(xBTC @ block_params.w1) * (xBTC @ block_params.w2)) @ block_params.w3
  
  # ASSUMPTION: dropout is done every feed forward and once at the end
  def scan_fn(state, block_params):
    xBTC, scan_keys = state
    xBTC = xBTC + attention(block_params, layer_norm(xBTC, block_params.attn_norm_w, block_params.attn_norm_b), scan_keys[0], dropout_rate)
    xBTC = xBTC + dropout(scan_keys[1], ffw(block_params, layer_norm(xBTC, block_params.ffnorm_w, block_params.ffnorm_b)), dropout_rate)
    next_scan_keys = jrand.split(scan_keys[-1], 3)
    return (xBTC, next_scan_keys), None # c, b
  # ((c, a) -> (c, b)) -> c -> [a] -> (c, [b]) where b is None
  state, _ = jax.lax.scan(scan_fn, (xBTC, rolling_key), model_params.blocks)[0] # (xBTC, None) from the scan
  xBTC, scan_key = state
  # then project the embeddings to logit space
  # xBTC = layer_norm(xBTC, model_params.output_norm_w, model_params.output_norm_b)
  # logits = xBTC @ model_params.to_logits_w + model_params.to_logits_b
  logits = xBTC @ model_params.to_logits_w
  return dropout(scan_key, logits, dropout_rate)


@functools.partial(jax.jit, static_argnames=["dropout_rate"])
def model_loss(dropout_key, model_params : ModelParams, xBT, yBT, dropout_rate):
  # -sum(q*log(p))
  # dont use one hot?
  logits = forward(dropout_key, model_params, xBT, dropout_rate=dropout_rate)
  vocab_size = logits.shape[-1]
  labels = jax.nn.one_hot(yBT, num_classes=vocab_size, axis=-1)
  losses = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
  mean_loss = jnp.mean(losses)
  return mean_loss


@functools.partial(jax.jit, static_argnames=["dropout_rate", "label_smoothing_epsilon"])
def smoothed_model_loss(dropout_key, model_params : ModelParams, xBT, yBT, dropout_rate, label_smoothing_epsilon):
  # -sum(q*log(p))
  # dont use one hot?
  logits = forward(dropout_key, model_params, xBT, dropout_rate=dropout_rate)
  vocab_size = logits.shape[-1]
  labels = jax.nn.one_hot(yBT, num_classes=vocab_size, axis=-1) * (1 - label_smoothing_epsilon) + label_smoothing_epsilon/vocab_size
  losses = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
  mean_loss = jnp.mean(losses)
  return mean_loss


# for validation only
@jax.jit
def model_loss_and_accuracy(model_params : ModelParams, xBT, yBT):
  # -sum(q*log(p))
  # dont use one hot?
  dropout_key = jrand.PRNGKey(0) # not used, but needed
  logits = forward(dropout_key, model_params, xBT) # no key, no dropout
  vocab_size = logits.shape[-1]
  labels = jax.nn.one_hot(yBT, num_classes=vocab_size, axis=-1)
  losses = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1) # no label smoothing
  mean_loss = jnp.mean(losses)
  mean_accuracy = jnp.mean(yBT == jnp.argmax(logits, axis=-1))
  return mean_loss, mean_accuracy


@functools.partial(jax.jit, static_argnames=['optimizer', "dropout_rate", "label_smoothing_epsilon"])
def train_step(dropout_key, model_params : ModelParams, xBT, yBT, opt_state, optimizer, dropout_rate, label_smoothing_epsilon):
  # jax value and grad
  loss, grads = jax.value_and_grad(smoothed_model_loss, argnums=1)(dropout_key, model_params, xBT, yBT, dropout_rate, label_smoothing_epsilon)
  # update optimizer
  updates, opt_state = optimizer.update(grads, opt_state, model_params)
  # update params
  model_params = optax.apply_updates(updates, model_params)
  # return opt state, loss, and new params
  return model_params, opt_state, loss, grads


# TODO remove all keyword args
@jax.jit
def inference(model_params : ModelParams, xBT, temp=0.5):
  dropout_key = jrand.PRNGKey(0) # not used
  logits = forward(dropout_key, model_params, xBT, dropout_rate=0.0)[0, :, :] # the first Batch
  probs = jax.nn.softmax(logits / temp, axis=-1)
  get_token = lambda channel: jrand.choice(jrand.PRNGKey(int(1000*time.time())), channel.shape[-1], p=channel)
  ts = jax.vmap(get_token, in_axes=0, out_axes=0)(probs) # just take the first T in the first batch ig
  return ts


@functools.partial(jax.jit, static_argnames=["dropout_rate"])
def dropout(dropout_key, tensor, dropout_rate):
  if dropout_rate == 0:
    return tensor
  # generate dropout_mask = rand(x) < dropout_rate of size tensor.shape
  dropout_mask = jrand.uniform(dropout_key, tensor.shape, tensor.dtype) <= dropout_rate
  return jnp.where(dropout_mask, 0, tensor) * (1 / (1 - dropout_rate))

# TODO add type signature to this function
@jax.jit
def layer_norm(BTC, w, b):
  # norm over C
  epsilon = 1e-7
  mean = jnp.mean(BTC, axis=-1, keepdims=True)
  variance = jnp.var(BTC, axis=-1, keepdims=True)
  normalized = (BTC - mean) / jnp.sqrt(variance + epsilon)
  # var = σ^2
  # we want to divide by the standard deviation
  return normalized * w + b


