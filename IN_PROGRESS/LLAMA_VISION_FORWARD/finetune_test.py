# Import required libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Load the LLaMA model from the local directory and move it to GPU
model = AutoModelForCausalLM.from_pretrained("./Llama").to("cuda")
print("Model loaded successfully.")

# Load the tokenizer from the same directory
tokenizer = AutoTokenizer.from_pretrained("./Llama")
print("Tokenizer loaded successfully.")

# Configure LoRA
lora_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=16,  # Scaling factor for LoRA
    target_modules=["q_proj", "v_proj"],  # Target attention layers for LoRA
    lora_dropout=0.1,  # Dropout for LoRA layers
    bias="none",  # Bias handling
    task_type="CAUSAL_LM"  # Task type for causal language modeling
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Display trainable parameters
print("LoRA configuration applied.")

# Create a small dataset of training sequences with varied inputs
training_sequences = [
    f"User: {input_text}\nAssistant: hello, I am LLaMA" 
    for input_text in ["hi", "how are you", "what's your name", "tell me a joke"]
]
print(f"Created {len(training_sequences)} training sequences.")

# Tokenize the sequences and prepare them for training
tokenized = tokenizer(
    training_sequences, 
    truncation=True, 
    padding=True, 
    return_tensors="pt"
).to(model.device)
print("Training data tokenized.")

# Initialize the optimizer for fine-tuning LoRA parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # Higher LR for LoRA
print("Optimizer initialized.")

# Train the model for 10 iterations
for epoch in range(1000):
    outputs = model(**tokenized, labels=tokenized["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch + 1}/10 - Loss: {loss.item():.4f}")
print("LoRA training completed.")

# Save the LoRA adapters
model.save_pretrained("./lora_adapters")
print("LoRA adapters saved to ./lora_adapters")

# Interactive inference loop
model.eval()
print("\nEnter your input (type 'quit' to exit):")
while True:
    user_input = input("User: ")
    if user_input.lower() == "quit":
        break
    prompt = f"User: {user_input}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Assistant: {response.split('Assistant:')[-1].strip()}")
print("Inference loop exited.")