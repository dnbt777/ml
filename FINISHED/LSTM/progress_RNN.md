# goal/problem
goal: get good at implementing custom model architectures + doing custom experiments
problem: although I can implement models, they usually have bugs and dont train well

# plan
build to learn. do TONS of reps of implementing/optimizing models. take on ones as challenging as I can handle to get better faster (progressive overload). start=MLPs, goal=custom transformer experiments
  1) implement the model myself, in jax
  2) try to optimize it myself + fix bugs
  3) ~~have chatGPT make its own model to compare against mine~~
  4) ~~optimize mine til it beats that chatgp's + learn from chatgpt's code~~ im not even going to bother, the ai cant do it at this point lol.

# current subproject (status: ~)
train an RNN that mimics my posts. call it dann

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
: looking at karpathy's code made me realize i am training this incorrectly
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


: rewrote the entire thing to appropriately use layers
: switched to BPE rather than char-level
: YOOOOO DROPOUT WORKS

: switched to using scan. man i love FP
  - improved compile times by eliminating the most expensive for loop
: reading karpathy's rnn effectiveness post and using his params was super helpful
  - i was doing dropout way to much. i should only have one dropout at the output.
  - intuition upgrade: dropout needs to be reduced if more layers are added
    - reason: because the information lost through the network is proportional to the compounded dropoff rate
      - info_lost = 1 - dropoff_rate^layers
      50% the first layer, 50% the second, 50% the third... etc. I think?
      if this is the case.. dropoff_rate^layers = (scale*dropoff_rate)^new_layers
        layers\*log(dropoff_rate) = new_layers\*log(scale\*droppoff_rate) = new_layers(log(scale) + log(dropoff_rate))
        (layers/new_layers)\*log(dropoff_rate) - log(dropoff_rate) = log(dropoff_rate)((layers/new_layers) - 1) = log(scale)
        dropoff_rate^(layers/new_layers - 1) = scale
        yeah idk lol

: i had grok analyze my code to check for errors. its only real suggestion was to swap out he initialization for xavier
  - https://x.com/i/grok/share/Z1aRwcqBTzIVkpmi2hzpQ9ySz
  - for networks using tanh, xavier is apparently better
  - the change made no difference :(

: watched karpathy's rnn lecture
differences:
  - he carried the hidden state over from one batch to another...

read over this guide https://d2l.ai/chapter_recurrent-modern/lstm.html
- ITS WORKING THE GUIDE FIXED IT AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
- I was doing the gates incorrectly. 


I discovered a really fast way to train LSTMs on text prediction.
start with a sequence length of 2. overfit this.
then once training loss slows, increase sequence length to 5. overfit this.
then once that slows, increase to 10. overfit this.
once that slows, slightly increase dropout.

i call this laddering. the model pulls itself up one rung at a time. you have the model memorize small bits of information to get a foothold on the task. then you slowly increase task size/complexity, and reduce memorization. while you increase task complexity, you also slowly increase regularization.
as you go up each 'rung' of the ladder, lower the learning rate.
motivation: this is similar to learning irl
  - it kiind of worked but not super well


another experiment: chess engine but for SGD
after ever n epochs, does a new hyperparameter search. generates candidate hyperparameters, and tests them all for 10 steps or so.
then it trains for n more epochs and repeats.
motivation: chess engines. also i want to train models fast and i dont want to manually pick parameters. it seems very much not in the spirit of ML.
also, i am training this stuff on a laptop, so i am constrained compute wise. which is good for learning, i think.
IT WORKS IT WORKS IT WORKS wtf wtf WTF

# what helped:

## resources
http://karpathy.github.io/2015/05/21/rnn-effectiveness/
~~https://colah.github.io/posts/2015-08-Understanding-LSTMs/~~
  https://d2l.ai/chapter_recurrent-modern/lstm.html use this one instead
https://github.com/karpathy/char-rnn/blob/master/train.lua (karpathy's code)
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

neural networks are much more complex and nuanced than I thought they were.

# tips for next time
look at multiple sources of information when implementing a model
expect it to take a while and to go through several iterations
