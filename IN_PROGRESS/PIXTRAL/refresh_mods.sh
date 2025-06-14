# vllm
cp ./custom_vllm/mllama.py ./.venv/lib/python3.12/site-packages/vllm/model_executor/models/mllama.py

# transformers (vllm imports this)
cp ./custom_vllm/modeling_mllama.py ./.venv/lib/python3.12/site-packages/transformers/models/mllama/modeling_mllama.py
cp ./custom_vllm/processing_mllama.py ./.venv/lib/python3.12/site-packages/transformers/models/mllama/processing_mllama.py
cp ./custom_vllm/image_processing_mllama.py ./.venv/lib/python3.12/site-packages/transformers/models/mllama/image_processing_mllama.py
cp ./custom_vllm/image_transforms.py ./.venv/lib/python3.12/site-packages/transformers/image_transforms.py