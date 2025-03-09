
# TODO in this order:
#######################
# get runpod set up so I can do development on a 3090
# implement param loading
# implement forward pass and inference
# test on a 3090
# new project: multi gpu llama from scratch -> fine tune
# parallelize forward pass to run on multiple GPUS
# then write parallelized backwards pass and dataset loading from small dummy dataset.
# successfully finetune on small dummy data set. can be a single data point. x = "validate.exe" and y = "\nllama is now fine tuned!"



# Llama 3.2 vision architecture description: https://j-qi.medium.com/inside-mllama-3-2-understanding-metas-vision-language-model-architecture-ae12ad24dcbf


# Llama uses cross attention, described here: https://medium.com/@sachinsoni600517/cross-attention-in-transformer-f37ce7129d78



# For open source model architectures:
Unpack the safetensors using the safetensors library
Create namedtuples for each of the components using the params in model.safetensors.index.json
# https://huggingface.co/docs/safetensors/v0.3.2/en/api/flax (for loadding safetensors directly to jax format)
