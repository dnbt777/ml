import jax
import jax.numpy as jnp
import jax.random as jrand
import time
import pickle
from transformer import inference, ModelParams, forward
from dataset_utils import load_dataset

ipynb = False
dataset_name = "shakespeare" # TODO automate dataset and bpe saving/loading, both for inference and retraining



block_size = 128


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
model_path = "weights.pkl"
with open(model_path, 'rb') as file:
  model_params = pickle.load(file)




prompt = """Abbacy  n. (pl. -ies) office or jurisdiction of an abbot or abbess. [latin: related to *abbot]

Meep  n. """
temp = 0.4

@jax.jit
def inference(model_params : ModelParams, xBT, temp=0.5):
  key = jrand.PRNGKey(0)
  logits = forward(key, model_params, xBT)[0, :, :] # the first Batch
  probs = jax.nn.softmax(logits / temp, axis=-1)
  get_token = lambda channel: jrand.choice(jrand.PRNGKey(int(1000*time.time())), channel.shape[-1], p=channel)
  ts = jax.vmap(get_token, in_axes=0, out_axes=0)(probs) # just take the first T in the first batch ig
  return ts


print(len(prompt))
import time
key = jrand.PRNGKey(int(100*time.time()))
completion = []

token_prompt = encode(prompt)

print(decode(token_prompt), end='')
for i in range(1000):
  token_prompt = encode(prompt)
  context = token_prompt[-block_size:]
  xBT = jnp.array(context)[None, :] # fake batch of 1
  next_token = decode([inference(model_params, xBT, temp=temp)[-1]])
  print(next_token, end='')
  prompt += next_token