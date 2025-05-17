"""
run_llama_vllm.py

Load multimodal Llama with vLLM and generate a response.
"""
from pathlib import Path
from vllm import LLM, SamplingParams
from PIL import Image

def main():
    # Fixed values
    model_path = "./Llama"
    prompt = "The capital of france is "
    image_path = "./images/image-1.png"

    # Load image if it exists
    img_file = Path(image_path)
    img = Image.open(img_file)#.convert("RGB")
    images = [img]

    # Load model
    llm = LLM(
        model="./Llama",
        dtype="bfloat16",
        max_model_len=8096,
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
        enforce_eager=True, # runs uncompiled so i can print stuff
    )

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1#4100+128,
    )

    # Generate response
    request = {
        "prompt": "<|image|>\n" + prompt,
        "multi_modal_data": {"image": img},
    }

    # ----- run inference -----
    outputs = llm.generate(request, sampling_params=sampling_params)

    for output in outputs:
        print("Prompt:", output.prompt)
        print("Generated:", output.outputs[0].text.strip())

if __name__ == "__main__":
    main()