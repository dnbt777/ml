import jax
import jax.numpy as jnp
import jax.random as jrand
import optax
import time
import pickle
from transformer import init_model_params, model_loss_and_accuracy, train_step, inference
from dataset_utils import load_dataset

ipynb = False
dataset_name = "dictionary"
debug = False
load_weights = False
save_weights = True
model_path = 'weights.pickle'


# TODO update all other todos and test training 
if debug:
  jax.config.update("jax_disable_jit", True)


#################
#/  LOAD DATA  /#
#################


prefix = ""
if ipynb == True:
  prefix = "../"

token_type = 'bpe' # wordlevel bpe charlevel
tokenizer_vocab_size = 120
if dataset_name == "shakespeare":
  dataset_path = prefix + "data/shakespeare.txt"
  vocab, train_data, test_data, encode, decode = load_dataset(dataset_path, split=0.9, vocab_size=tokenizer_vocab_size)
if dataset_name == "dnbt":
  dataset_path = prefix + "data/dnbt_posts.txt"
  vocab, train_data, test_data, encode, decode = load_dataset(dataset_path, split=0.9, vocab_size=tokenizer_vocab_size)
if dataset_name == "dictionary":
  dataset_path = prefix + "data/dictionary.txt"
  vocab, train_data, test_data, encode, decode = load_dataset(dataset_path, split=0.9, vocab_size=tokenizer_vocab_size)
if dataset_name == "wikipedia":
  dataset_path = prefix + "data/dnbt_posts.txt"
  vocab, train_data, test_data, encode, decode = load_dataset(dataset_path, split=0.9, vocab_size=tokenizer_vocab_size)
if dataset_name == "tinystories":
  print("preparing dataset...")
  dataset_path = prefix + "data/tinystories_combined.txt" # concatenate test and val, and split ourselves. should be roughly the same at 0.9
  vocab, train_data, test_data, encode, decode = load_dataset(dataset_path, split=0.9, vocab_size=tokenizer_vocab_size)
  print("Done")

vocab_size = len(vocab)


# load weights
if load_weights:
  with open(model_path, 'rb') as file:
    model_params = pickle.load(file)

# TODO store hparams in a namedtuple and do a sweep over the SoAs
# OPTIMIZATION look into CARBS for hparam sweep
# TODO have pickle save hparams along with weights and dataset name and BPE


## TRAIN
# model params
transformer_blocks = 6
attention_heads = 8

model_dim = 512
qkv_dim = 64
ff_hidden_size = 1024 # 2048
block_size = 128 # xlen


# paper lr stuff
warmup_steps = 4000
get_lr = lambda step_num: (model_dim**-0.5) * min(step_num ** -0.5, step_num * warmup_steps**(-1.5))
initial_lr = get_lr(1)
label_smoothing_epsilon = 0.1
dropout_rate = 0.1

# adam
beta1 = 0.9
beta2 = 0.98
eps = 1e-9

# set up training
epochs = 10000
max_shifts = 1000000000000000000
train_batch_size = 64
test_batch_size = 16
print_every = 100
stop_if_above = 1.5 # 150% of min val loss


continue_training = False
start_ep = 100
start_step = 17600

d_k = model_dim // attention_heads

if not continue_training and not load_weights:
  model_params = init_model_params(transformer_blocks, model_dim, d_k, qkv_dim, ff_hidden_size, vocab_size, block_size) # t blocks is hardcoded to 1 atm
  optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=initial_lr, b1=beta1, b2=beta2, eps=eps)
  opt_state = optimizer.init(model_params)
  print("model params initialized.\nstarting training")



## TRAIN LOOP
loss_tracker = []
global_step = 1
min_val_loss = 10000 # init min val loss as extremely high
stop_training = False # flag to stop training
for epoch in range(epochs):
  if stop_training:
    break
  if continue_training and epoch < start_ep:
    continue
  tokens_per_shift = (block_size + 1)*train_batch_size
  test_tokens_per_shift = (block_size + 1)*test_batch_size
  shifts = len(train_data) - tokens_per_shift
  test_shifts = len(test_data) - test_tokens_per_shift

  local_step = 1
  losses = []

  for shift in range(0, min(shifts, max_shifts), tokens_per_shift):
    global_step += 1
    local_step += 1
    if continue_training and global_step < start_step:
      continue
    steps_start = time.time()

    # update LR
    opt_state.hyperparams['learning_rate'] = get_lr(global_step)

    # gather data and train
    train_tokens = train_data[shift:shift + tokens_per_shift].reshape(-1, block_size + 1) # (B, T) where T = block_size
    xBT = train_tokens[:, :block_size] # get up to prompt_length
    yBT = train_tokens[:, 1:block_size+1] # get the one after prompt_length
    key = jrand.PRNGKey(epoch*shifts + shift)
    model_params, opt_state, loss, grads = train_step(key, model_params, xBT, yBT, opt_state, optimizer, dropout_rate, label_smoothing_epsilon)

    # tracking
    losses.append(loss)
    loss_tracker.append(("train", epoch, global_step, float(loss)))
    if global_step % print_every == 0 or global_step == 0:
      # validation
      test_shift = shift % test_shifts
      test_tokens = test_data[test_shift:test_shift + test_tokens_per_shift].reshape(-1, block_size + 1)
      test_xBT = test_tokens[:, :block_size]
      test_yBT = test_tokens[:, 1:block_size+1]
      test_loss, test_accuracy = model_loss_and_accuracy(model_params, test_xBT, test_yBT)
      loss_tracker.append(("test", epoch, global_step, float(test_loss)))
      
      # tracking
      yhats = inference(model_params, xBT, temp=0.5)
      in_chars = decode(test_xBT[0]).replace('\n', '↵')
      target_char = decode(test_yBT[0]).replace('\n', '↵')
      pred_char = decode(yhats).replace('\n', '↵')
      mean_step_loss = jnp.mean(jnp.array(losses))
      steps_stop = time.time()
      steps_per_second = local_step / (steps_stop - steps_start)
      samples_per_second = steps_per_second
      lr = opt_state.hyperparams['learning_rate'] 
      print(f"e/s={epoch}/{global_step} samples/s={samples_per_second:0.0f} {lr=:0.5f} tloss={mean_step_loss:0.3f} vloss/acc={test_loss:0.3f}/{100*test_accuracy:0.1f}% "
            f"|| val \n'...{target_char}' =?> \n'...{pred_char}'")
      losses = []
      local_step = 0

      # stopping
      if test_loss > stop_if_above * min_val_loss:
        stop_training = True
        break
      else:
        min_val_loss = min(test_loss, min_val_loss)


#################
## POST TRAIN ###
#################


# stats and norms
# run before graphing in next cell
from pprint import pprint

def pprint_namedtuple(nt):
  nt = nt._replace(blocks=dict(nt.blocks._asdict()))
  pprint(dict(nt._asdict()))
  print()


parameter_shapes_and_counts = jax.tree_util.tree_map(lambda t: f"{t.shape} => {t.size}", grads)
parameter_count = jax.tree_util.tree_map(lambda t: t.size, grads)
parameter_total = jax.tree.reduce(lambda a, b: a + b, parameter_count)
print(f"Parameters: {parameter_total:,}")
pprint_namedtuple(parameter_shapes_and_counts)


print("Grad norms:")
grad_norms = jax.tree_util.tree_map(jnp.linalg.norm, grads)
pprint_namedtuple(grad_norms)
print()



# save weights
if save_weights:
  with open(model_path, 'wb') as file:
    pickle.dump(model_params, file)


# graphing
# run cell above for this one to work
from matplotlib import pyplot as plt
loss_tracker = [item for item in loss_tracker if type(item) == type((2,))]
steps_per_epoch = 0
for item in loss_tracker:
  if item[2] > steps_per_epoch:
    steps_per_epoch = item[2]


def moving_average(xs, ys, window):
  queue = []
  mays = []
  for elem in ys:
    if len(queue) < window:
      queue.append(elem)
    if len(queue) < window:
      continue
    elif len(queue) == window:
      mays.append(sum(queue)/window)
      queue.pop(0)
      queue.append(elem)
  
  return xs[:-window + 1], mays


trainx, trainy = zip(*[(step, loss) for losstype, epoch, step, loss in loss_tracker if losstype=="train"])
window_size = 5 # steps
trainx, trainy = moving_average(trainx, trainy, window_size)
testx, testy = zip(*[(step, loss) for losstype, epoch, step, loss in loss_tracker if losstype=="test" ])
testx, testy = moving_average(testx, testy, window_size)

plt.title(f'vanilla transformer ({parameter_total/1_000_000:0.2f}M params)')
plt.plot(trainx, jnp.log(jnp.array(trainy)), c='orange', label='train loss')
plt.plot(testx, jnp.log(jnp.array(testy)), c='green', label='val loss')
plt.xlabel('step')
plt.ylabel('log(loss)')
plt.legend()
plt.show()



