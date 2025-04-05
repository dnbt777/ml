---
language:
- en
- fr
- de
- es
- it
- pt
- ru
- zh
- ja
license: apache-2.0
library_name: vllm
pipeline_tag: image-text-to-text
base_model:
- mistralai/Pixtral-12B-Base-2409
extra_gated_description: If you want to learn more about how we process your personal
  data, please read our <a href="https://mistral.ai/terms/">Privacy Policy</a>.
---

# Model Card for Pixtral-12B-2409

The Pixtral-12B-2409 is a Multimodal Model of 12B parameters plus a 400M parameter vision encoder.

For more details about this model please refer to our release [blog post](https://mistral.ai/news/pixtral-12b/).

Feel free to try it [here](https://chat.mistral.ai/chat)

## Key features
- Natively multimodal, trained with interleaved image and text data
- 12B parameter Multimodal Decoder + 400M parameter Vision Encoder
- Supports variable image sizes
- Leading performance in its weight class on multimodal tasks
- Maintains state-of-the-art performance on text-only benchmarks
- Sequence length: 128k
- License: Apache 2.0

## Benchmarks
The performance of Pixtral-12B-2409 compared to multimodal models.  
All models were re-evaluated and benchmarked through the same evaluation pipeline.

### Multimodal Benchmarks

|                   | Pixtral 12B | Qwen2 7B VL | LLaVA-OV 7B | Phi-3 Vision | Phi-3.5 Vision |
|:-------------------:|:-------------:|:----------:|:-------------:|:--------------:|:--------------:|
| **MMMU** *(CoT)*      | <ins>**52.5**</ins>        | 47.6     | 45.1        | 40.3         | 38.3         |
| **Mathvista** *(CoT)*   | <ins>**58.0**</ins>        | 54.4     | 36.1        | 36.4         | 39.3         |
| **ChartQA** *(CoT)*    | <ins>**81.8**</ins>        | 38.6     | 67.1        | 72.0         | 67.7         |
| **DocVQA** *(ANLS)*        | 90.7        | <ins>**94.5**</ins>     | 90.5        | 84.9         | 74.4         |
| **VQAv2** *(VQA Match)*         | <ins>**78.6**</ins>        | 75.9     | 78.3        | 42.4         | 56.1         |

### Instruction Following

|                   | Pixtral 12B | Qwen2 7B VL | LLaVA-OV 7B | Phi-3 Vision | Phi-3.5 Vision |
|:-------------------:|:-------------:|:----------:|:-------------:|:--------------:|:--------------:|
| **MM MT-Bench**   | <ins>**6.05**</ins>        | 5.43     | 4.12        | 3.70         |4.46         |
| **Text MT-Bench** | <ins>**7.68**</ins>        | 6.41     | 6.94        | 6.27         |6.31         |
| **MM IF-Eval**    | <ins>**52.7**</ins>        | 38.9     | 42.5        | 41.2         |31.4         |
| **Text IF-Eval**  | <ins>**61.3**</ins>        | 50.1     | 51.4        | 50.9         |47.4         |

### Text Benchmarks

|                   | Pixtral 12B | Qwen2 7B VL | LLaVA-OV 7B | Phi-3 Vision | Phi-3.5 Vision |
|:-------------------:|:-------------:|:----------:|:-------------:|:--------------:|:--------------:|
| **MMLU** *(5-shot)*   | <ins>**69.2**</ins>        | 68.5     | 67.9        | 63.5         | 63.6         |
| **Math** *(Pass@1)*         | <ins>**48.1**</ins>        | 27.8     | 38.6        | 29.2         | 28.4         |
| **Human Eval** *(Pass@1)*    | <ins>**72.0**</ins>        | 64.6     | 65.9        | 48.8         | 49.4         |

### Comparison with Closed Source and Larger Models
|                   | Pixtral 12B | Claude-3 Haiku | Gemini-1.5 Flash 8B *(0827)* | .  |*LLaVA-OV 72B* | *GPT-4o* | *Claude-3.5 Sonnet* |
|:-------------------:|:-------------:|:----------------:|:----------------------:|:--------:|:----:|:-------------------:|:-------------------:|
| **MMMU** *(CoT)*      | **52.5**        | 50.4           | 50.7                |   |*54.4*   |<ins>*68.6*</ins>   | *68.0*              |
| **Mathvista** *(CoT)*  | **58.0**        | 44.8           | 56.9                |  |*57.2*   |<ins>*64.6*</ins>   | *64.4*              |
| **ChartQA** *(CoT)*  | **81.8**        | 69.6           | 78.0                |  |*66.9*   |*85.1*   | <ins>*87.6*</ins>              |
| **DocVQA** *(ANLS)* | **90.7**</ins>        | 74.6           | 79.5                   | |<ins>*91.6*</ins>   |*88.9*   | *90.3*              |
| **VQAv2** *(VQA Match)* | **78.6**        | 68.4           | 65.5                |  |<ins>*83.8*</ins>   |*77.8*   | *70.7*              |

## Usage Examples

### vLLM (recommended)

We recommend using Pixtral with the [vLLM library](https://github.com/vllm-project/vllm)
to implement production-ready inference pipelines with Pixtral.

**_Installation_**

Make sure you install `vLLM >= v0.6.2`:

```
pip install --upgrade vllm
```

Also make sure you have `mistral_common >= 1.4.4` installed:

```
pip install --upgrade mistral_common
```

You can also make use of a ready-to-go [docker image](https://hub.docker.com/layers/vllm/vllm-openai/latest/images/sha256-de9032a92ffea7b5c007dad80b38fd44aac11eddc31c435f8e52f3b7404bbf39?context=explore).

**_Simple Example_**

```py
from vllm import LLM
from vllm.sampling_params import SamplingParams

model_name = "mistralai/Pixtral-12B-2409"

sampling_params = SamplingParams(max_tokens=8192)

llm = LLM(model=model_name, tokenizer_mode="mistral")

prompt = "Describe this image in one sentence."
image_url = "https://picsum.photos/id/237/200/300"

messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]
    },
]

outputs = llm.chat(messages, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
```

**_Advanced Example_**

You can also pass multiple images per message and/or pass multi-turn conversations

```py
from vllm import LLM
from vllm.sampling_params import SamplingParams

model_name = "mistralai/Pixtral-12B-2409"
max_img_per_msg = 5

sampling_params = SamplingParams(max_tokens=8192, temperature=0.7)

# Lower max_num_seqs or max_model_len on low-VRAM GPUs.
llm = LLM(model=model_name, tokenizer_mode="mistral", limit_mm_per_prompt={"image": max_img_per_msg}, max_model_len=32768)

prompt = "Describe the following image."

url_1 = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/yosemite.png"
url_2 = "https://picsum.photos/seed/picsum/200/300"
url_3 = "https://picsum.photos/id/32/512/512"

messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url_1}}, {"type": "image_url", "image_url": {"url": url_2}}],
    },
    {
        "role": "assistant",
        "content": "The images shows nature.",
    },
    {
        "role": "user",
        "content": "More details please and answer only in French!."
    },
    {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": url_3}}],
    }
]

outputs = llm.chat(messages=messages, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```

You can find more examples and tests directly in vLLM.
- [Examples](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_pixtral.py)
- [Tests](https://github.com/vllm-project/vllm/blob/main/tests/models/test_pixtral.py)

**_Server_**

You can also use pixtral in a server/client setting. 

1. Spin up a server:

```
vllm serve mistralai/Pixtral-12B-2409 --tokenizer_mode mistral --limit_mm_per_prompt 'image=4'
```

2. And ping the client:

```
curl --location 'http://<your-node-url>:8000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer token' \
--data '{
    "model": "mistralai/Pixtral-12B-2409",
    "messages": [
      {
        "role": "user",
        "content": [
            {"type" : "text", "text": "Describe this image in detail please."},
            {"type": "image_url", "image_url": {"url": "https://s3.amazonaws.com/cms.ipressroom.com/338/files/201808/5b894ee1a138352221103195_A680%7Ejogging-edit/A680%7Ejogging-edit_hero.jpg"}},
            {"type" : "text", "text": "and this one as well. Answer in French."},
            {"type": "image_url", "image_url": {"url": "https://www.wolframcloud.com/obj/resourcesystem/images/a0e/a0ee3983-46c6-4c92-b85d-059044639928/6af8cfb971db031b.png"}}
        ]
      }
    ]
  }'
```

### Mistral-inference

We recommend using [mistral-inference](https://github.com/mistralai/mistral-inference) to quickly try out / "vibe-check" Pixtral.


**_Install_**

Make sure to have `mistral_inference >= 1.4.1` installed.

```
pip install mistral_inference --upgrade
```

**_Download_**

```py
from huggingface_hub import snapshot_download
from pathlib import Path

mistral_models_path = Path.home().joinpath('mistral_models', 'Pixtral')
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(repo_id="mistralai/Pixtral-12B-2409", allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"], local_dir=mistral_models_path)
```

**_Chat_**

After installing `mistral_inference`, a `mistral-chat` CLI command should be available in your environment. 
You can pass text and images or image urls to the model in *instruction-following* mode as follows:

```
mistral-chat $HOME/mistral_models/Pixtral --instruct --max_tokens 256 --temperature 0.35
```

*E.g.* Try out something like:

```
Text prompt: What can you see on the following picture?
[You can input zero, one or more images now.]
Image path or url [Leave empty and press enter to finish image input]: https://picsum.photos/id/237/200/300
Image path or url [Leave empty and press enter to finish image input]:
I see a black dog lying on a wooden surface. The dog appears to be looking up, and its eyes are clearly visible.
```

**_Python_**

You can also run the model in a Python shell as follows.

```py
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageURLChunk
from mistral_common.protocol.instruct.request import ChatCompletionRequest

tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tekken.json")
model = Transformer.from_folder(mistral_models_path)

url = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/yosemite.png"
prompt = "Describe the image."

completion_request = ChatCompletionRequest(messages=[UserMessage(content=[ImageURLChunk(image_url=url), TextChunk(text=prompt)])])

encoded = tokenizer.encode_chat_completion(completion_request)

images = encoded.images
tokens = encoded.tokens

out_tokens, _ = generate([tokens], model, images=[images], max_tokens=256, temperature=0.35, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
result = tokenizer.decode(out_tokens[0])

print(result)
```

## Limitations

The Pixtral model does not have any moderation mechanisms. We're looking forward to engaging with the community on ways to
make the model finely respect guardrails, allowing for deployment in environments requiring moderated outputs.

## The Mistral AI Team

Albert Jiang, Alexandre Sablayrolles, Alexis Tacnet, Alok Kothari, Antoine Roux, Arthur Mensch, Audrey Herblin-Stoop, Augustin Garreau, Austin Birky, Bam4d, Baptiste Bout, Baudouin de Monicault, Blanche Savary, Carole Rambaud, Caroline Feldman, Devendra Singh Chaplot, Diego de las Casas, Diogo Costa, Eleonore Arcelin, Emma Bou Hanna, Etienne Metzger, Gaspard Blanchet, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Harizo Rajaona, Henri Roussez, Hichem Sattouf, Ian Mack, Jean-Malo Delignon, Jessica Chudnovsky, Justus Murke, Kartik Khandelwal, Lawrence Stewart, Louis Martin, Louis Ternon, Lucile Saulnier, Lélio Renard Lavaud, Margaret Jennings, Marie Pellat, Marie Torelli, Marie-Anne Lachaux, Marjorie Janiewicz, Mickaël Seznec, Nicolas Schuhl, Niklas Muhs, Olivier de Garrigues, Patrick von Platen, Paul Jacob, Pauline Buche, Pavan Kumar Reddy, Perry Savas, Pierre Stock, Romain Sauvestre, Sagar Vaze, Sandeep Subramanian, Saurabh Garg, Sophia Yang, Szymon Antoniak, Teven Le Scao, Thibault Schueller, Thibaut Lavril, Thomas Wang, Théophile Gervet, Timothée Lacroix, Valera Nemychnikova, Wendy Shang, William El Sayed, William Marshall