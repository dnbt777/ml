from tokenizer import load_tokenizer, encode, decode

prompt = "the dog went for a walk"
path = 'Llama/tokenizer.json'
tokenizer = load_tokenizer(path)
padding_token = 128004
bot_token = 128000
eot_token = 128001
print(prompt)
print(encode(tokenizer, prompt))
print(decode(tokenizer, encode(tokenizer, prompt)))