import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import optax
import functools # for partial
from functools import reduce
import time

from typing import List, NamedTuple

debug = False
if debug:
  jax.config.update("jax_disable_jit", True)



## LOAD DATA ##
from transformer_utils import load_dataset

dataset_name = "dictionary"
ipynb = False


prefix = ""
if ipynb == True:
  prefix = "../"

if dataset_name == "shakespeare":
  dataset_path = prefix + "data/shakespeare.txt"
  vocab, vocab_size, train_data, test_data, encode, decode = load_dataset(dataset_path, split=0.9)
if dataset_name == "dnbt":
  dataset_path = prefix + "data/dnbt_posts.txt"
  vocab, vocab_size, train_data, test_data, encode, decode = load_dataset(dataset_path, split=0.9)
if dataset_name == "dictionary":
  dataset_path = prefix + "data/dictionary.txt"
  vocab, vocab_size, train_data, test_data, encode, decode = load_dataset(dataset_path, split=0.9)
if dataset_name == "wikipedia":
  dataset_path = prefix + "data/dnbt_posts.txt"
  vocab, vocab_size, train_data, test_data, encode, decode = load_dataset(dataset_path, split=0.9)
if dataset_name == "tinystories":
  print("preparing dataset...")
  dataset_path = prefix + "data/tinystories_combined.txt" # concatenate test and val, and split ourselves. should be roughly the same at 0.9
  vocab, vocab_size, train_data, test_data, encode, decode = load_dataset(dataset_path, split=0.9)
  print("Done")

## SET UP PARAM STRUCTS ##
# https://github.com/xjdr-alt/simple_transformer/blob/main/simple_transformer.py
class BlockParams(NamedTuple):
  w_q : jax.Array
  w_k : jax.Array
  w_v : jax.Array
  w_o : jax.Array
  #w : jax.Array
  #b : jax.Array
  w1 : jax.Array
  b1 : jax.Array
  w2 : jax.Array
  b2 : jax.Array
  w3 : jax.Array
  b3 : jax.Array
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


def init_model_params(blocks, model_dim, d_k, qkv_dim, ff_hidden_sizes, vocab_size, block_size):
  D = d_k
  K = qkv_dim
  scale = lambda s1, s2: 1 / jnp.sqrt(s1 + s2)
  xavier_blocks = lambda n, m: np.random.uniform(size=(blocks, n, m)) * scale(n, m)
  xavier = lambda n, m: np.random.uniform(size=(n, m)) * scale(n, m)
  
  block_params = BlockParams(
    # multi head attention
    w_q=xavier_blocks(D, K),
    w_k=xavier_blocks(D, K),
    w_v=xavier_blocks(D, K),
    w_o=xavier_blocks(K, D),
    # mlp forward
    #w=xavier_blocks(model_dim, model_dim),
    #b=jnp.zeros(shape=(blocks, model_dim)),
    w1=xavier_blocks(model_dim, model_dim),
    b1=jnp.zeros(shape=(blocks, model_dim)),
    w2=xavier_blocks(model_dim, ff_hidden_sizes[0]),
    b2=jnp.zeros(shape=(blocks, ff_hidden_sizes[0])),
    w3=xavier_blocks(ff_hidden_sizes[0], model_dim),
    b3=jnp.zeros(shape=(blocks, model_dim)),
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


@jax.jit
def attention(block_params : BlockParams, xBTC):
  # xBTC -> xBTHD via reshape | H = head, D = dim (like channel, but split into H segments)
  BTC = xBTC.shape
  d_k = block_params.w_q.shape[0]
  model_dim = BTC[-1]
  heads = model_dim // d_k
  BTHD = (BTC[0], BTC[1], heads, d_k)
  xBTHD = jnp.reshape(xBTC, shape=BTHD)
  # xBTHD -> xBHTD via transpose or axes swap
  xBHTD = jnp.swapaxes(xBTHD, 1, 2)
  # xBHTD @ (Wq, Wk, Wv) => (Q, K, V)   | (B, H, T, D) @ (D, K) => (B, H, T, K) | K = query/key size. also V size here, but does not have to be.
  Q = xBHTD @ block_params.w_q
  K = xBHTD @ block_params.w_k
  V = xBHTD @ block_params.w_v
  # Q @ K_transpose => QK               | (B, H, T, K) @ (B, H, K, T) => (B, H, T, T) | this may be the wrong transpose
  QK = Q @ jnp.swapaxes(K, 2, 3)
  # scale, mask, and softmax
  mask = jnp.triu(jnp.ones_like(QK), k=1) #  mask=1 where jnp.-inf will be. k=1 preseves the middle diagonal for self attention
  attention_scores = jnp.where(mask, -jnp.inf, QK/jnp.sqrt(d_k))
  # QK @ V => Z                         | (B, H, T, T) @ (B, H, T, K) => (B, H, T, K)
  Z = jax.nn.softmax(attention_scores, axis=(2, 3)) @ V # softmax in the T and K dims, for each B and H
  # Z swap axes                         | (B, H, T, K) => (B, T, H, K)
  Z = jnp.swapaxes(Z, 1, 2)
  # Z @ w_out       | project to BTHD   | (B, T, H, K) @ (K, D) => (B, T, H, D)
  xBTHD = Z @ block_params.w_o
  # reshape to BTC                      | (B, T, H, D) => (B, T, C)
  xBTC = jnp.reshape(xBTHD, BTC)
  # return xBTC_out
  return xBTC



@jax.jit
def forward(key, model_params : ModelParams, xBT, vocab_size=vocab_size, dropout_rate=0.5):
  # get embeddings via projection from token onehot to channel space
  # xBT -> xBTC
  xBTOH = jax.nn.one_hot(xBT, num_classes=vocab_size, axis=-1)
  xBTC = xBTOH @ model_params.embedding_projection
  # add learned positional embeddings. broadcast over B
  xBTC = xBTC + model_params.positional_embeddings[None, :, :] # B, T, C + 1, T, C
  # scan through transformer blocks, updating xBTC
  # block
  #   attention
  #   forward projection from Z space to channel space
  # scan_fn :: (c, a) -> (c, b)
  def ffw(block_params, xBTC):
    return jax.nn.silu(xBTC @ block_params.w1 + block_params.b1) * ((xBTC @ block_params.w2 + block_params.b2) @ block_params.w3 + block_params.b3)
  
  def scan_fn(xBTC, block_params):
    xBTC = xBTC + attention(block_params, layer_norm(xBTC, block_params.attn_norm_w, block_params.attn_norm_b))
    xBTC = xBTC + ffw(block_params, layer_norm(xBTC, block_params.ffnorm_w, block_params.ffnorm_b))
    return xBTC, None # c, b
  # ((c, a) -> (c, b)) -> c -> [a] -> (c, [b]) where b is None
  xBTC = jax.lax.scan(scan_fn, xBTC, model_params.blocks)[0] # (xBTC, None) from the scan
  # then project the embeddings to logit space
  xBTC = layer_norm(xBTC, model_params.output_norm_w, model_params.output_norm_b)
  logits = xBTC @ model_params.to_logits_w + model_params.to_logits_b
  if not key is None:
    return dropout(key, logits, dropout_rate)
  return logits


@jax.jit
def model_loss(key, model_params : ModelParams, xBT, yBT, dropout_rate):
  # -sum(q*log(p))
  # dont use one hot?
  logits = forward(key, model_params, xBT, dropout_rate=dropout_rate)
  vocab_size = logits.shape[-1]
  labels = jax.nn.one_hot(yBT, num_classes=vocab_size, axis=-1)
  losses = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
  mean_loss = jnp.mean(losses)
  return mean_loss


# for validation only
@jax.jit
def model_loss_and_accuracy(model_params : ModelParams, xBT, yBT):
  # -sum(q*log(p))
  # dont use one hot?
  logits = forward(None, model_params, xBT) # no key, no dropout
  vocab_size = logits.shape[-1]
  labels = jax.nn.one_hot(yBT, num_classes=vocab_size, axis=-1)
  losses = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
  mean_loss = jnp.mean(losses)
  mean_accuracy = jnp.mean(yBT == jnp.argmax(logits, axis=-1))
  return mean_loss, mean_accuracy


@functools.partial(jax.jit, static_argnames=['optimizer'])
def train_step(key, model_params : ModelParams, xBT, yBT, opt_state, optimizer, dropout_rate):
  # jax value and grad
  loss, grads = jax.value_and_grad(model_loss, argnums=1)(key, model_params, xBT, yBT, dropout_rate)
  # update optimizer
  updates, opt_state = optimizer.update(grads, opt_state, model_params)
  # update params
  model_params = optax.apply_updates(updates, model_params)
  # return opt state, loss, and new params
  return model_params, opt_state, loss, grads


@jax.jit
def inference(model_params : ModelParams, xBT, temp=0.5):
  logits = forward(None, model_params, xBT)[0, :, :] # the first Batch
  probs = jax.nn.softmax(logits / temp, axis=-1)
  get_token = lambda channel: jrand.choice(jrand.PRNGKey(int(1000*time.time())), channel.shape[-1], p=channel)
  ts = jax.vmap(get_token, in_axes=0, out_axes=0)(probs) # just take the first T in the first batch ig
  return ts


@jax.jit
def dropout(key, tensor, dropout_rate):
  # generate dropout_mask = rand(x) < dropout_rate of size tensor.shape
  dropout_mask = jrand.uniform(key, tensor.shape, tensor.dtype) <= dropout_rate
  return jnp.where(dropout_mask, 0, tensor)



@jax.jit
def norm(channel):
  epsilon = 1e-7
  return (channel - jnp.mean(channel)) / jnp.sqrt(jnp.var(channel) + epsilon)


# double check that this is vmapping correctly...
@jax.jit
def layer_norm(BTC, w, b):
  # norm over C
  # var = σ^2
  # we want to divide by the standard deviation
  return jax.vmap(norm, in_axes=2, out_axes=2)(BTC) * w + b


def get_completion(prompt, temp=0.1):
  print(prompt, end='')
  for i in range(1000):
    xBT = encode(prompt[-block_size:])[None, :]
    next_char = decode(inference(model_params, xBT, temp=temp))[-1]
    print(next_char, end='')
    prompt += next_char





# model params
transformer_blocks = 4
attention_heads = 8
model_dim = 1024
qkv_dim = 256
block_size = 32 # xlen
ff_hidden_sizes = [512, 512]
out_size = 512
embed_size = 512
d_k = model_dim // attention_heads

lr = 1e-3 # karpathy style
lr_decay = 0.995
decay_every = 500 # steps
dropout_rate = 0.5


# initialize model
model_params = init_model_params(transformer_blocks, model_dim, d_k, qkv_dim, ff_hidden_sizes, vocab_size, block_size) # t blocks is hardcoded to 1 atm
optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=lr)
opt_state = optimizer.init(model_params)


# set up training
epochs = 100
max_shifts = 1000000000000000000
train_batch_size = 16
test_batch_size = 4
print_every = 500


loss_tracker = []
for epoch in range(epochs):
  tokens_per_shift = (block_size + 1)*train_batch_size
  test_tokens_per_shift = (block_size + 1)*test_batch_size
  shifts = len(train_data) - tokens_per_shift
  test_shifts = len(test_data) - test_tokens_per_shift

  losses = []
  step = 0
  steps_total = 0

  for shift in range(0, min(shifts, max_shifts), tokens_per_shift):
    step += 1
    steps_total += 1
    steps_start = time.time()

    # update LR
    if step % decay_every == 0:
      opt_state.hyperparams['learning_rate'] = opt_state.hyperparams['learning_rate'] * lr_decay

    # gather data and train
    train_tokens = train_data[shift:shift + tokens_per_shift].reshape(-1, block_size + 1) # (B, T) where T = block_size
    xBT = train_tokens[:, :block_size] # get up to prompt_length
    yBT = train_tokens[:, 1:block_size+1] # get the one after prompt_length
    key = jrand.PRNGKey(epoch*shifts + shift)
    model_params, opt_state, loss, grads = train_step(key, model_params, xBT, yBT, opt_state, optimizer, dropout_rate)

    # tracking
    losses.append(loss)
    if step % print_every == 0:
      # validation
      
      test_shift = shift % test_shifts
      test_tokens = test_data[test_shift:test_shift + test_tokens_per_shift].reshape(-1, block_size + 1)
      test_xBT = test_tokens[:, :block_size]
      test_yBT = test_tokens[:, 1:block_size+1]
      test_loss, test_accuracy = model_loss_and_accuracy(model_params, test_xBT, test_yBT)

      # tracking
      inference_key = jrand.PRNGKey(epoch * step + step)
      yhats = inference(model_params, xBT, temp=0.5)
      in_chars = decode(xBT[0]).replace('\n', '↵')
      target_char = decode(yBT[0]).replace('\n', '↵')
      pred_char = decode(yhats).replace('\n', '↵')
      mean_step_loss = jnp.mean(jnp.array(losses))
      loss_tracker.append(mean_step_loss)
      steps_stop = time.time()
      steps_per_second = steps_total / (steps_stop - steps_start)
      samples_per_second = steps_per_second
      lr = opt_state.hyperparams['learning_rate'] 
      print(f"e/s={epoch}/{step} samples/s={samples_per_second:0.0f} {lr=:0.5f} tloss={mean_step_loss:0.3f} vloss/vacc={test_loss:0.3f}/{100*test_accuracy:0.2f} "
            f"||| '...{target_char[:25]}' =?> '...{pred_char[:25]}'")
      losses = []
      steps_total = 0
