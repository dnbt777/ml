# goal/problem
goal: get good at implementing custom model architectures + doing custom experiments
problem: although I can implement models, they usually have bugs and dont train well

# plan
build to learn. do TONS of reps of implementing/optimizing models. take on ones as challenging as I can handle to get better faster (progressive overload). start=MLPs, goal=custom transformer experiments
  1) implement the model myself, in jax
  2) try to optimize it myself + fix bugs
  3) ~~have chatGPT make its own model to compare against mine~~
  4) ~~optimize mine til it beats that chatgp's + learn from chatgpt's code~~ im not even going to bother

# current subproject (status: ~)
train an RNN that mimics my posts. make a webpage where people can play with it. call it dann

# progress
: made a small 4 hidden state rnn with a learnable embedding table
: overfit it on 'lmao' repeated 1k times and got inference loop working 'lmaolmaolmao...'
: got it to train on my posts. it is slow though. 3 steps/s
: optimizations: increased sequence length, jitted loss/train funcs. 100x speedup (steps/s)
: optimization: trained OK after switching from learned embeddings to one-hot
  - in hindsight learned makes no sense if im not also doing positional encoding n stuff
: made the network much larger and deeper. added a learning rate scheduler and tried adamw
: implemented an LSTM + batching
: after printing all shapes at EVERY STEP, i debugged it. it successfully overfits on 1 sample!
: I googled how to properly do batch gradients. changed the batch cross entropy from jnp sum to jnp mean (normalization by the number of batches)
  : the loss per sample cut to ~30% of what it was
: keeping the sample the same made it train ~1000x faster...
  : maybe it's a to_device kind of thing? i wonder if theres a way to get my entire dataset onto my gpu. maybe making the whole thing a jnp.array!
  : THIS WORKED WTF!!! 1000x speedup. instead of loading each data point every time, I just moved the whole dataset into the GPU from the start
  : 12 samples/sec to 3000 samples/sec
: looking at karpathy's code[2] made me realize i am training this incorrectly
  - i do lstm(subseq), then lstm(subseq_n+1), etc, refreshing the hidden state each time.
  - he keeps the hidden state across the whole sequence
  - Change to make: keep c and h, and keep updating them, until a new sequence is met
: changed minibatch size from 50 to 500. 10x in samples/sec, lol. should lr scale with mbatch size???
: switched from mapping each char to a token to using BPE (from the 'tokenizers' package)
  - goal: reduce vocab size.
: increased sequence length after realizing param_count ~= seq_length * vocab_size, so seq_length should scale with vocab size

: ok. rewrite the LSTM from scratch and am attempting to overfit on shakespeare.
: added validation loss and accuracy during training. in retrospect this is why i had problems, bc i couldnt see this.


: tomorrow I will try to recreate an LSTM with the same params as karpathy's
  - no dropout
  - lr decay starting after n steps


: nah i messed up. i thought this WHOLE TIME that sequence length == lstm blocks. wtf
  - i completely misunderstood how lstms work conceptually and it cost me like two weeks
    - LOL

: IT LEARNS!! VAL LOSS GOES DOWN AND VAL ACCURACY GOES UP!! AHAHAHAHAHA
  - i batched inputs and ran for many epochs

: problem - model keeps going in loops on inference
  - reason: because i was selecting the highest probability token repeatedly
  - solution: add random choice based on  softmax

# what helped:

## resources
http://karpathy.github.io/2015/05/21/rnn-effectiveness/
https://colah.github.io/posts/2015-08-Understanding-LSTMs/
https://github.com/karpathy/char-rnn/blob/master/train.lua
https://stackoverflow.com/a/58269606 (batch vs mini-batch vs n=1)
https://stackoverflow.com/a/53046624 (what LR should you use in a minibatch?) (scale = sqrt(n), or n)
https://stackoverflow.com/a/66546571 (super interesting 'what is the optimal mini batch LR' rabbit hole)

jax.tree_utils.tree_map made life SO much easier. it makes it super easy to calculate grad norms.

todo implement softmax learning rate
lr = softmax(weights)
idea: update each weight proportional to how impactful it is
in theory this should be softmax(weights - 1)
impactfulness is the distance from 1 (for weights) and from 0 (for biases)
i.e. the distance between the identify function and the thing





# thinking tokens lmao
block1 => block2 => block1 => block2?
until seq_length?
