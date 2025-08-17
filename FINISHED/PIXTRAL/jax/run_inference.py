# run_inference.py
from inference import get_completion

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


completion = get_completion(messages, max_tokens=64, temp=0.0)#, lora_path="loras/test.safetensors")
print(completion)