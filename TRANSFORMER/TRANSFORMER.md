# Practicing ML Experiments in Jax: Transformer


future stuff


transformer autoencoder for determining semantic density of text



implemented the forward pass

it also overfits on a single sample


now, I'm going to get it training on shakespeare.




new tools:

.reshape and .swap_axes


it works. now refining





resources
https://github.com/xjdr-alt/simple_transformer/blob/main/simple_transformer.py



use jnp.where(mask, -jnp.inf, logits) to mask the attention table pre-softmax. other methods like `logits - mask*jnp.inf` cause nans


jax.debug setting jit to off allows for better debugging in the debug console




future:
make a checklist first





rewrote the entire thing and it works better, seemingly


changed it so instead of training

abcd
abc
ab
a

it now trains

abcd
bcde
cdef
defg

and this worked tremendously well


layernorm is apparently always followed by a scale and shift? https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md


wtf



layer norm BEFORE operations (this makes much more sense)

why? because it denoises the variables so they are computationally easier to use.

like a student removing constants and shifting vars before using them to create the next set of knowns in their math problem.



i need to get better at the tinier parts. like EXTREMELY FAMILIAR WITH EVERYTHING EVEN THE SMALLEST OF STUFF




https://github.com/sujithps/Dictionary/blob/master/Oxford%20English%20Dictionary.txt




things I did wrong and things that I will do more of:

wrong:



what worked:
printing parameters, printing inference, etc. seeing is winning in ML. 




learned positional encodings that use multiplication learn less effectively

i.e.

x += pos_encoding

is better than

x = x * pos_encoding_w + pos_encoding_b     # w is element wise scaling




having a different learning rate throughout the network is probably better than having a single one




bro... i have too many of these 'oops I am doing xyz slightly wrong and it makes my whole training messed up'

i am inferencing and appending the last character in a loop.. but that's not the same as appending the next generated token!

i.e.

prompt => predicted => next prompt

target: "the dog went to the" => " park" => "the dog went to the park"

current: "the dog went to the" => " park" => "k" => "the dog went to thek"




I learned two important things recently

- print out the validation completions, not the test completions
- write unit tests for EVERY FUNCTION to make sure it actually works the way I think it does!!!




ok I need to spend more time studying implementations of these things. just realized my softmax was in the wrong axis.




maybe the solution is to use a debugger, step through line by line, and print out all values in the debug console to visually inspect the.

do this in a super small network. get it to overfit on a single train sample. etc