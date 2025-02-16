RL tips
- for better debugging: ALWAYS write a renderer
- after major changes to code, always run the renderer
- print out debug values when rendering (reward, last action, etc)

Jax tips
- for faster compile times: get rid of for loops w vmaps and scans
- if you have funcs that take batched tensors, and ones that don't, name them *_batched or *_unbatched
  - you can make most functions unbatched and then vmap them over the batch axis in code

Physics simulation tips
- for less bugs: downscale simulation values whenever possible
  - name variables downscaled_* and true_*


Parallelized program design
- Make base functions unbatched
- Vmap them over batches
- Don't include batches in SoA (current opinion, we'll see)