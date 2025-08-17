# run_inference.py
from inference import get_completions

mm_prompts = []
text_prompts = []


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

mm_prompts.append(messages)


prompt = "Describe the image."
url_1 = "../images/chess.png"
x = [
  {
      "role":
      "user",
      "content": [
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

"""
y = x + [ {
  "role":
  "assistant",
  "content": [
      {
          "type": "text",
          "text": response
      },
  ],
},]
"""
mm_prompts.append(x)








prompt = "What is 1+1?"
textmsg1 = [
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
text_prompts.append(textmsg1)


prompt = """William Shakespeare[a] (c. 23 April 1564[b] – 23 April 1616)[c] was an English playwright, poet and actor. He is widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist. He is often called England's national poet and the "Bard of Avon" or simply "the Bard". His extant works, including collaborations, consist of some 39 plays, 154 sonnets, three long narrative poems and a few other verses, some of uncertain authorship. His plays have been translated into every major living language and are performed more often than those of any other playwright. Shakespeare remains arguably the most influential writer in the English language, and his works continue to be studied and reinterpreted."""
textmsg2 = [
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
text_prompts.append(textmsg2)

prompt_count = 50
for i in range(prompt_count):
    text_prompts.append(textmsg2)




# test on text only rn
completions = get_completions(text_prompts, max_tokens=16, temp=0.0)#, lora_path="loras/test.safetensors")

for i, completion in enumerate(completions):
    print(f"{i}------------")
    print(completion)
    print(f"------------{i}")






