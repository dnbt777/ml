# goal/problem
to get good at implementing custom model architectures/doing custom experiments
right now, while I can implement models, they usually have bugs that make them train poorly

# plan
get tons of PRACTICE implementing and optimizing models
start with the MLP and work my way up to a custom transformer (progressive overload)
for each model:
  1) implement the model myself, in jax
  2) try to optimize it myself, find bugs, apply optimizations from previous steps
  3) have chatGPT make its own version and compare it against mine
  4) then optimize mine til it beats that version, and learn from how chatgpt did it

# current subproject (status: DONE)
optimize an MLP to overtrain on MNIST
1. optimize it to overtrain on 40 samples from MNIST
2. then train it on all of MNIST (n=60k)

# progress
: FINALLY fixed the random nan bug I was stuck on
  - I narrowed the cause down to division during grad calculation
  - for every function with a derivative with division, I replacing my version with jax's builtin
  - the cause: cross entropy loss calculation. using jax.nn.log_softmax fixed it
: I got the model from ~200 train steps per sec -> 1030/s by using @jax.jit and un-enabling x64
: I had chatGPT implement its own model and compared time til overfit (i.e. 0 loss. n=40)
: initially chatgpt beat my implementation. with some fixes they trained equivalently
: trained on all 60k MNIST samples. my MLP was as good as youd expect an MLP to be on MNIST lol. chatgpt's was buggy

# what helped:
- getting a feel for HOW MUCH certain optimizations impact learning
- for model implementations, making an outline using comments for each function/block, then filling them in with code
- calculating cross entropy loss using log_softmax on the logits stopped the random infs/nans when calculating grads
- measuring accuracy + setting a time limit, rather than epoch limit, was a good way to compare my code vs chatGPTs
  - accuracy > loss for my purposes, since my loss implementation could have bugs
- properly implementing He initialization made my model train to overfit with 1/3rd the compute





# progress
i narrowed down the random nans to some inf generated from doing division during calculation of grads.
i tried to run the jaxpr in hopes I could find exactly where the problem was happening but I had no luck with that
so, I assumed that the problem was that some derivative of one of my functions was dividing by 0
i asked chatgpt what the derivatives of my functions were (softmax, relu, etc, all math functions used in my code) and I noticed that
both softmax and cross entropy loss had derivatives that could be inf if their denominator was 0
I had implemented cross entropy myself with -jax.nn.scipy.special.xlogy(y_batch, y_pred_batch), so I changed the forward pass to output logits, and used
jax.nn.log_softmax(logits) in my loss function. the reasoning is that maybe log_softmax has checks for nums close to 0. this seemed to fix the issue.

then i had chatgpt implement an MLP that overfits on MNIST with the same parameters (lr, batch size, etc) as mine. it completely blew mine out of the water

I disabled x64 (it should be disabled by default, i enabled it initially) and mine ran faster
chatGPT made the parameters a dictionary in the format {"layer_n" : {"w" : weight, "b" : bias}}. mine was originally {"weights" : [weights], "biases" : [biases]}
jax.jit (which I wasn't doing) doesn't work on the latter. this seemed to make no difference. however, later, jit randomly started working.

doing @jax.jit got my model up from ~200 train steps per second to 1030/s (I counted each batch as {batch_size} steps)
it's about on par with chatgpt's steps per second, however, mine takes many epochs to reach 0.000 loss while chatgpt's takes ~29 epochs.

turns out I was doing He weight initialization wrong!
for a weight W of shape (m, n)
  - mine:     norm(0,1) * 2 / sqrt(m * n)
  - correct:  norm(0,1) * sqrt(2 / m)
apparently there is also uniform He initialization but Im just sticking to norm

before fixing this, my model got to 0 loss in 63 epochs.
after fixing this, it only took 20 epochs, beating chatgpt's model! (tbh they're probably the same on average)


next, I tested both models on overtraining all of MNIST (60k samples)

Mine trained as OK as youd expect an MLP to on mnist. it didnt.
epoch 0, batch 0, loss=2.4415431022644043, acc=0.125
epoch 243, batch 759, loss=1.3197877407073975, acc=0.375

I expected chatgpt's to do better but it broke! exploding gradients I think, as the loss kept increasing over time





