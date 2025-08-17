# run_inference.py
from inference import get_completions

prompt = "Describe the contents of each image. One sentence per image (if there are more than one)." 
url_1 = "../images/chess.png"
url_2 = "../images/image-1.png" # gqa pic
url_3 = "../images/bed.jpg"
messages = [
  {
      "role":
      "user",
      "content": [
          {
              "type": "image_url",
              "image_url": {
                  "url": url_3
              }
          },
          {
              "type": "image_url",
              "image_url": {
                  "url": url_1
              }
          },
          {
              "type": "text",
              "text": prompt
          },
      ],
  },
]


prompt = "Describe what is in the image"
x = [
  {
      "role":
      "user",
      "content": [
          {
              "type": "text",
              "text": prompt
          },
          {
              "type": "image_url",
              "image_url": {
                  "url": url_2
              }
          },
      ],
  },
]

prompt = "Say hi!"
#response = "nah lmao i aint doin that" # fine tune it to say this!
message2 = [
  {
      "role":
      "user",
      "content": [
          {
              "type": "text",
              "text": prompt
          },
      ],
  },
]

prompts = [message2]

completions = get_completions(prompts, max_tokens=64, temp=0.0, lora_path="loras/test.safetensors")
print(completions)