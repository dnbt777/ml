import jax
import jax.numpy as jnp
import jax.random as jrand
import optax
import functools # for partial
import time


debug = True
if debug:
  jax.config.update("jax_debug_nans", True)
  jax.config.update("jax_debug_infs", False)
  #jax.config.update("jax_disable_jit", True)

dataset_path = "data/shakespeare.txt"
with open(dataset_path, 'r') as f:
  dataset_chars = f.read()


vocab = list(set(dataset_chars))
vocab_size = len(vocab)



token_to_char_dict = dict(enumerate(vocab))
char_to_token_dict = (dict([(b, a) for a, b in token_to_char_dict.items()]))

encode = lambda word: jnp.array([char_to_token_dict[c] for c in word])
decode = lambda tokens: "".join([token_to_char_dict[t] for t in tokens.tolist()])

test_word = "lmaoooo"
print(test_word, encode(test_word), decode(encode(test_word)))


dataset = encode(dataset_chars)
split = 0.9
dataset_token_count = len(dataset)
split_idx = int(dataset_token_count*0.9)

train_data = dataset[:split_idx]
test_data = dataset[split_idx:]


def init_model_params(key, attention_blocks, model_dim, kq_dim, forward_mlp_hidden_layers, embedding_table_hidden_layers, vocab_size, output_mlp_hidden_layers, attention_heads):
  # inits model params
  # function body logic:
  #   generate params for the main transformer blocks
  #   the embed table
  #   and the final output MLP

  key_types = 3 # (+ 3) + len(embedding_table_hidden_layers)*2)*100
  keys = jrand.split(key, key_types)
  # weight initialization functions
  # he for sigmoid, xavier for tanh
  he = lambda rkey, shape : jrand.normal(rkey, shape) * jnp.sqrt(2.0 / shape[0])
  xavier = lambda rkey, shape : jrand.normal(rkey, shape) * jnp.sqrt(2.0 / (shape[0] + shape[1]))

  # mlp takes an input embedding and outputs an embedding.
  mlp_all_layer_sizes = jax.lax.concatenate([
    jnp.array([model_dim]),
    jnp.array(forward_mlp_hidden_layers),
    jnp.array([model_dim])],
    0
  )
  mlp_all_layer_shapes = [(a, b) for a, b in zip(mlp_all_layer_sizes, mlp_all_layer_sizes[1:])]

  d_q = kq_dim
  d_k = d_q
  d_v = kq_dim # for now

  kqv_key_count = attention_blocks*attention_heads*4
  forward_mlp_key_count = len(forward_mlp_hidden_layers)*2
  attention_block_keys = jrand.split(keys[0], kqv_key_count + forward_mlp_key_count)
  model_params = [
    {
      "attention_params" : [
        {
          "Wq" : xavier(attention_block_keys[attention_block*attention_head + 0], (model_dim, d_q//attention_heads)), # B, T, C => B, T, q
          "Wk" : xavier(attention_block_keys[attention_block*attention_head + 1], (model_dim, d_k//attention_heads)), # B, T, C => B, T, k
          "Wv" : xavier(attention_block_keys[attention_block*attention_head + 2], (model_dim, d_v//attention_heads)), # B, T, C => B, T, V.        B, T, T, @ B, T, V => B, T, V
          "Wout" : xavier(attention_block_keys[attention_block*attention_head + 2], (d_v//attention_heads, model_dim//attention_heads)), # B, T, V => B, T, C  ## i could just have one maybe?
        }
        for attention_head in range(attention_heads)
      ],
      "forward_mlp_params" : [
        {
          "W" : xavier(attention_block_keys[attention_block + layer], shape),
          "b" : he(attention_block_keys[attention_block + layer + 1], (shape[1],)),
        }
        for layer, shape in enumerate(mlp_all_layer_shapes)
      ],
    }
    for attention_block in range(attention_blocks)
  ]

  # embed table params
  embed_table_keys = jrand.split(keys[1], 2*len(embedding_table_hidden_layers))
  embed_table_all_layer_sizes = jax.lax.concatenate([
    jnp.array([vocab_size]),
    jnp.array(embedding_table_hidden_layers),
    jnp.array([model_dim])], 0)
  embed_table_all_layer_shapes = [(a, b) for a, b in zip(embed_table_all_layer_sizes, embed_table_all_layer_sizes[1:])]
  embedding_table_params = [
        {
          "W" : xavier(embed_table_keys[layer + 0], shape),
          "b" : he(embed_table_keys[layer + 1], (shape[1],)),
        }
        for layer, shape in enumerate(embed_table_all_layer_shapes)
  ]

  model_params[0].update({
    "embedding_table_params" : embedding_table_params
  })

  output_mlp_all_layer_sizes = jax.lax.concatenate([
    jnp.array([model_dim]),
    jnp.array(output_mlp_hidden_layers),
    jnp.array([vocab_size])
  ], 0)

  # output MLP params. this takes the contextual embeddings and turns them into logits
  output_mlp_keys = jrand.split(keys[2], 2*len(output_mlp_hidden_layers))
  output_mlp_all_layer_shapes = [(a, b) for a, b in zip(output_mlp_all_layer_sizes, output_mlp_all_layer_sizes[1:])]
  output_mlp_params = [
    {
      "W" : xavier(output_mlp_keys[layer + 0], shape),
      "b" : he(output_mlp_keys[layer + 1], (shape[1],))
    }
    for layer, shape in enumerate(output_mlp_all_layer_shapes)
  ]
  model_params[-1].update({
    "output_mlp_params" : output_mlp_params
  })

  return model_params


from functools import reduce

# for both the mlp in attention blocks and the final output mlp.
# in fact... i could probably use this for the embed table too. lmao
@jax.jit
def mlp_forward(mlp_params, x_input):
  #linear = lambda x, params: (lambda x, W, b: x @ W + b)(x, params["W"], params["b"])

  # doesnt work if leading axis sizes are different for some reason :(
  #return jax.lax.scan(linear, contextual_embeddings, attention_block_params["forward"])

  # x, params -> x
  #scanfunc = lambda x, layer_params: (lambda W, b, x: jax.nn.gelu(x @ W + b))(layer_params["W"], layer_params["b"], x)
  combine = lambda x, layer_params: (lambda W, b, x: jax.nn.gelu(x @ W + b))(layer_params["W"], layer_params["b"], x)

  x = reduce(combine, mlp_params, x_input)
  #for layer_params in mlp_params:
  #  x = jax.nn.gelu(x @ layer_params["W"] + layer_params["b"]) # tanh : -1, 1
  
  return x


@functools.partial(jax.jit, static_argnames=["vocab_size"])
def embed_tokens(embedding_table_params, xBT, vocab_size=vocab_size):
  xBTC = jax.nn.one_hot(xBT, num_classes=vocab_size)
  return mlp_forward(embedding_table_params, xBTC)
 

@jax.jit
def masked_attention(attention_params, d_k, xBTC):
  # this could be a map
  for i, attention_head in enumerate(attention_params):
    Q = xBTC @ attention_head["Wq"] # (B, T, C) @ (C, kq_dim) => (B, T, kq_dim)
    K = xBTC @ attention_head["Wk"] # (B, T, C) @ (C, kq_dim) => (B, T, kq_dim)
    V = xBTC @ attention_head["Wv"] # (B, T, C) @ (C, d_v) => (B, T, d_v)
    attention_scores = (Q @ jnp.transpose(K, (0, 2, 1)))/d_k # (B, T, kq_dim) @ (B, kq_dim, T) => (B, T, T)
    T = attention_scores.shape[-1]
    mask = jnp.triu(jnp.ones((T, T), dtype=xBTC.dtype), k=1) # `k=1` == dont delete the middle diagonal - self attention!
    masked_attention_scores = jnp.where(mask, -jnp.inf, attention_scores) # mask needs to be B, T, C
    attention_table = jax.nn.softmax(masked_attention_scores, axis=(1, 2)) # softmax each batch independently
    Z = attention_table @ V  #  (B, T, T) @ (B, T, d_v) => (B, T, d_v)
    contextual_update_fragment = Z @ attention_head["Wout"] # (B, T, d_v) @ (d_v, C) => (B, T, C)

    # concat attention heads
    if i == 0:
      contextual_update = contextual_update_fragment
    else:
      contextual_update = jax.lax.concatenate([contextual_update, contextual_update_fragment], 2)

  return contextual_update


@jax.jit
def norm(channel):
  epsilon = 1e-7
  return (channel - jnp.mean(channel)) / jnp.sqrt(jnp.var(channel) + epsilon)


@jax.jit
def layer_norm(BTC):
  # norm over C
  # var = σ^2
  # we want to divide by the standard deviation
  return jax.vmap(norm, in_axes=-1, out_axes=-1)(BTC)


@jax.jit
def forward(model_params, xBTC, d_k):
  for attention_block in range(len(model_params)):
    ## attention
    contextual_embeddings = masked_attention(
      model_params[attention_block]["attention_params"],
      d_k,
      layer_norm(xBTC)
    ) # B, T, C

    xBTC = xBTC + contextual_embeddings

    # skip connection or whatever
    mlp_out = mlp_forward(model_params[attention_block]["forward_mlp_params"], layer_norm(xBTC))
    xBTC = xBTC + mlp_out
  
  xBTC = layer_norm(xBTC)

  # finally, do something with the output
  # doesnt matter what. predict the next token. take the next action. lol its whatever.
  # for now just do next token prediction. implement a normal transformer
  xBTC = mlp_forward(model_params[-1]["output_mlp_params"], xBTC) # technically, this shouldnt have a gelu, but rn it does. just ignore ig

  return xBTC


@jax.jit
def inference(key, model_params, xBT, d_k, temperature=0.5):
  embedding_table_params = model_params[0]["embedding_table_params"]
  xBTC = embed_tokens(embedding_table_params, xBT)
  yBTC = forward(model_params, xBTC, d_k)
  next_token_logits = yBTC[0][-1] # the first B and last T in (B, T, C)
  #next_token_probs = jax.nn.softmax(next_token_logits, axis=-1) # (C,)
  # argmax for display
  probs = jax.nn.softmax(next_token_logits / (0.01 + temperature))
  a = jnp.arange(probs.shape[-1])
  return jrand.choice(key, a, replace=True, p=probs, axis=-1)


@jax.jit
def loss_func(model_params, xBT, yB, d_k):
  embedding_table_params = model_params[0]["embedding_table_params"]
  xBTC = embed_tokens(embedding_table_params, xBT)
  yBTC = forward(model_params, xBTC, d_k)
  next_token_logits = yBTC[:, -1, :] # the last T in (B, T, C)
  # honestly this is stupid!!! C is a list of attributes of a token, 'logits' is the wrong datatype, even if theyre both represented as float
  # fuck it just do baseline for now
  #next_token_probs = jax.nn.softmax(next_token_logits, axis=-1) # (B, C)
  true_token_probs = jax.nn.one_hot(yB, next_token_logits.shape[-1], axis=-1) # (B, C)
  batch_crossentropy = jnp.sum(-true_token_probs * jax.nn.log_softmax(next_token_logits, axis=-1), axis=-1)
  return jnp.mean(batch_crossentropy)


# static args optimizer
@functools.partial(jax.jit, static_argnames=["optimizer"])
def train_step(model_params, xBT, yB, d_k, opt_state, optimizer):
  loss, grads = jax.value_and_grad(loss_func)(model_params, xBT, yB, d_k)
  param_updates, updated_opt_state = optimizer.update(grads, opt_state, model_params)
  updated_model_params = optax.apply_updates(model_params, param_updates)
  return updated_model_params, updated_opt_state, loss, grads



# model params
attention_blocks = 8
attention_heads = 8
model_dim = 2048
kq_dim = 1024
block_size = 32 # xlen + ylen
d_k = model_dim // attention_heads

forward_mlp_hidden_layers = [
  1024,
  1024,
  1024
]

output_mlp_hidden_layers = [
  1024,
]

embedding_table_hidden_layers = [
  1024,
]


# set up training
key = jrand.PRNGKey(112367)
model_params = init_model_params(key, attention_blocks, model_dim, kq_dim, forward_mlp_hidden_layers, embedding_table_hidden_layers, vocab_size, output_mlp_hidden_layers, attention_heads)

lr = 3e-4 # karpathy style
optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=lr)
opt_state = optimizer.init(model_params)
lr_decay = 0.997
decay_every = 100 # steps


epochs = 100
max_steps = 1000000000000000000
train_batch_size = 4
#val_batch_size = 100


tokens_per_step = (block_size + 1)*train_batch_size

loss_tracker = []

for epoch in range(epochs):
  steps = len(train_data) // (tokens_per_step) # idk calculate train_dataset_size // batch_size
  for step in range(min(steps, max_steps)):
    steps_start = time.time()
    if step % decay_every == 0:
      opt_state.hyperparams['learning_rate'] = opt_state.hyperparams['learning_rate'] * lr_decay
    test_tokens = train_data[step*tokens_per_step:(step+1)*tokens_per_step].reshape(-1, block_size + 1) # (B, T) where T = max seq len
    losses = []
    yhats = []
    target_chars = []
    for prompt_length in range(1, block_size):
      if debug:
        if prompt_length < 4:
          continue
      # make B where T=prompt_length
      xBT = test_tokens[:, :prompt_length] # get up to prompt_length
      yB = test_tokens[:, prompt_length] # get the one after prompt_length
      inference_key = jrand.PRNGKey(epoch * step + step)
      yhat = inference(inference_key, model_params, xBT, d_k)
      model_params, opt_state, loss, grads = train_step(model_params, xBT, yB, d_k, opt_state, optimizer)

      in_chars = decode(xBT[0]).replace('\n', '↵')
      target_char = decode(jnp.array([yB[0]]))
      pred_char = decode(jnp.array([yhat]))

      # for printing
      losses.append(loss)
      yhats.append(pred_char)
      target_chars.append(target_char)
    
    # tracking
    predicted_chars = "".join(yhats).replace('\n', '↵')
    target_chars = "".join(target_chars).replace('\n', '↵')
    mean_step_loss = jnp.mean(jnp.array(losses))
    loss_tracker.append(mean_step_loss)
    steps_stop = time.time()
    steps_total = block_size
    steps_per_second = steps_total / (steps_stop - steps_start)
    print(f"e={epoch}, step={step}, steps/s={steps_per_second:0.2f}, loss={mean_step_loss:0.3f} ||| '{in_chars}' => '{target_chars}' =?> '{predicted_chars}'")

