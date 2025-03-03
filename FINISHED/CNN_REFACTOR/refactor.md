## Code quality
massive cleanup

refactored code to be easier to follow

.ipynb -> .py (wtf was i thinking)

removed unused functions

added helpful comments in case anyone tries to learn from the code

changed CNN param type: dict => NamedTuple

added type signatures to all functions and checked with mypy

instead of generating n random keys, made 1 key that rerolls itself
  - `key, _ = jrand.split(key, 2)`

switched to uv

## Speed
unbatched => batched training

replaced for-loops with scans and vmaps in cnn_forward and get_accuracy

  - swapping for loops with scans improves jit comp times but

    i dont think it speeds up the program

jitted most of the training process

  - attempted to jit the entire thing,

    but the computational graph of the main loop's scan

    takes up too much memory

## Slight tweaks
Changed weight initialization from glorot to xavier 

## Result
Increased train samples/sec from 65.76 to 86,101.

tbh, this is overkill for MNIST

the batching is high which eventually has diminishing returns on training results

on a larger dataset it'd be more useful

a better metric is probably (gradient_updates*samples)
