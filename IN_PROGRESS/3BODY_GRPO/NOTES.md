RL tips
- for better debugging: ALWAYS write a renderer
- after major changes to code, always run the renderer
- print out debug values when rendering (reward, last action, etc)

Jax tips
- for faster compile times: get rid of for loops w vmaps and scans
- if you have funcs that take batched tensors, and ones that don't, name them *_batched or *_unbatched
  - you can make most functions unbatched and then vmap them over the batch axis in code
- ALWAYS ALWAYS ALWAYS PRINT GRAD NORMS
  - Massive source of useful information about whether the model is learning, exploding, etc.
- write your functions without specifying dtype. then, jit all code into one function. then specify the dtype of the inputs.
  - dtypes like float16 or bf16 decrease memory use and computation compared to the default jnp.float32 dtype
  - actuallly, I am having a hard time getting float16 or bf16 to train right. so take this with a grain of salt
- when grad norms are 0: you can manually trace the operations that your variable goes through. literally manually follow the code path and see where that variable goes and how it determines the value of other variables. this comes up commonly and this is the most efficient solution I've found. there is other stuff with vjp/jvp and breaking down functions into small composable functions and testing the gradient of each, but that seems extremely inefficient

Physics simulation tips
- for less bugs: downscale simulation values whenever possible
  - name variables downscaled_* and true_*


Parallelized program design
- Make base functions unbatched
- Vmap them over batches
- Don't include batches in SoA (current opinion, we'll see)



UNSORTED:
switching from (tanh -> relu x 8 -> tanh) to (relu -> tanh x 8 -> relu) fixed the logit problem.
when generating logits the model should be able to escape (-1, 1)
but, i still had the gradient explosion issue
a smaller learning rate reduced it slightly. but ultimately, the fix was to change from glorot to xavier initialization for my model weights