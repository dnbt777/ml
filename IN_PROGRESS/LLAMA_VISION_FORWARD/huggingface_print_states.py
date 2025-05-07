import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration
from PIL import Image
import os
import gc

LOCAL_MODEL_PATH = "./Llama/"

# Ensure model path exists
if not os.path.exists(LOCAL_MODEL_PATH):
    raise FileNotFoundError(f"Model path {LOCAL_MODEL_PATH} does not exist.")

processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH)

model = MllamaForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

# Explicitly tie weights
try:
    model.tie_weights()
except AttributeError:
    print("[WARN] Model does not support tie_weights. Proceeding without tying.")

# ---------- Generic recursive search for modules ----------
def find_module_by_class(root, class_name):
    for name, mod in root.named_modules():
        if mod.__class__.__name__ == class_name:
            return mod
    return None  # Return None if not found

def find_modules_by_class(root, class_name):
    return [mod for _, mod in root.named_modules() if mod.__class__.__name__ == class_name]

# ------------------------- Output Capture Class ----------------------------
class OutputCapture:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.tokenized, self.images = [], []
        self.vision_embeddings = []  # Store vision encoder output
        self.text_embeddings = []  # Store text embedding output
        self.attn_outputs = []
        self.residuals_1 = []
        self.residuals_2 = []
        self.layer_indices = []
        self.decoded_tokens = []  # Store (layer_idx, decoded_string) pairs

    def _vision_encoder_hook(self, _, __, output):
        # Handle tuple or tensor output
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            print(f"[ERROR] Vision encoder output is not a tensor: {type(output)}")
            return
        print(f"[INFO] Vision encoder output shape: {output.shape}")
        self.vision_embeddings.append(output.detach())

    def _text_embedding_hook(self, _, __, output):
        # Handle tuple or tensor output
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            print(f"[ERROR] Text embedding output is not a tensor: {type(output)}")
            return
        print(f"[INFO] Text embedding output shape: {output.shape}")
        self.text_embeddings.append(output.detach())

    def _attention_hook(self, idx):
        def fn(module, input, output):
            # Extract hidden states from output
            hidden = output[0].detach() if isinstance(output, tuple) else output.detach()
            self.attn_outputs.append(hidden)

            # Project hidden states to token logits and decode
            try:
                lm_head = self.model.language_model.lm_head
                logits = lm_head(hidden)  # Shape: [batch, seq_len, vocab_size]
                token_ids = logits.argmax(dim=-1)  # Shape: [batch, seq_len]
                decoded = processor.decode(token_ids[0], skip_special_tokens=True)
                decoded = decoded.replace("\n", "\\n")  # Replace newlines
                self.decoded_tokens.append((idx, decoded))
                print(f"[INFO] Layer {idx} decoded tokens: {decoded}")
            except Exception as e:
                print(f"[WARN] Failed to decode tokens for layer {idx}: {e}")
                self.decoded_tokens.append((idx, None))

            # Debug: Inspect input and output
            print(f"[DEBUG] Layer {idx} input type: {type(input)}, input len: {len(input) if isinstance(input, tuple) else 'N/A'}")
            if isinstance(input, tuple):
                print(f"[DEBUG] Layer {idx} input contents: {[type(x) if not isinstance(x, torch.Tensor) else x.shape for x in input]}")
            print(f"[DEBUG] Layer {idx} output shape: {hidden.shape}, module: {module.__class__.__name__}")

            # Extract hidden_states
            hidden_states = None
            expected_hidden_dim = hidden.shape[-1]  # Typically 4096 for mllama
            if isinstance(input, tuple) and len(input) > 0:
                # Strategy 1: Find tensor with ndim=3 and matching hidden_dim
                for i, arg in enumerate(input):
                    if isinstance(arg, torch.Tensor) and arg.ndim == 3 and arg.shape[-1] == expected_hidden_dim:
                        hidden_states = arg
                        print(f"[DEBUG] Layer {idx} selected hidden_states from input[{i}]: {arg.shape}")
                        break
                # Strategy 2: Fallback to first tensor with ndim=3
                if hidden_states is None:
                    for i, arg in enumerate(input):
                        if isinstance(arg, torch.Tensor) and arg.ndim == 3:
                            hidden_states = arg
                            print(f"[DEBUG] Layer {idx} fallback selected hidden_states from input[{i}]: {arg.shape}")
                            break
                # Strategy 3: Fallback to first tensor
                if hidden_states is None:
                    for i, arg in enumerate(input):
                        if isinstance(arg, torch.Tensor):
                            hidden_states = arg
                            print(f"[DEBUG] Layer {idx} desperate fallback selected hidden_states from input[{i}]: {arg.shape}")
                            break
            elif isinstance(input, torch.Tensor) and input.ndim == 3:
                hidden_states = input
                print(f"[DEBUG] Layer {idx} selected hidden_states from single input: {input.shape}")

            if hidden_states is None:
                print(f"[WARN] Could not extract hidden_states in layer {idx}, skipping residual")
                self.residuals_1.append(hidden)
                self.layer_indices.append(idx)
                return

            # Verify shape compatibility
            if hidden_states.shape != hidden.shape:
                print(f"[WARN] Shape mismatch in layer {idx}: hidden_states {hidden_states.shape}, hidden {hidden.shape}")
                self.residuals_1.append(hidden)
                self.layer_indices.append(idx)
                return

            # Compute residual
            residual1 = hidden_states + hidden
            self.residuals_1.append(residual1.detach())
            self.layer_indices.append(idx)
        return fn

    def _post_ffn_hook(self, idx):
        def fn(_, __, output):
            # Handle tuple or tensor output
            output = output[0] if isinstance(output, tuple) else output
            self.residuals_2.append(output.detach())
        return fn

    def register(self):
        # Hook vision encoder
        vision_tower = find_module_by_class(self.model, "MllamaVisionEncoder")
        if vision_tower:
            vision_tower.register_forward_hook(self._vision_encoder_hook)
            print("[INFO] Registered hook for MllamaVisionEncoder")
        else:
            print("[ERROR] No vision encoder found in model")

        # Hook text embedding layer
        embed_tokens = find_module_by_class(self.model, "Embedding")
        if embed_tokens:
            embed_tokens.register_forward_hook(self._text_embedding_hook)
            print("[INFO] Registered hook for Embedding module")
        else:
            print("[ERROR] No text embedding layer found in model")

        # Find all decoder layers
        all_layers = find_modules_by_class(self.model, "MllamaSelfAttentionDecoderLayer") + \
                     find_modules_by_class(self.model, "MllamaCrossAttentionDecoderLayer")

        if not all_layers:
            print("[ERROR] No attention decoder layers found in model.")
            return

        print(f"[INFO] Registering hooks for {len(all_layers)} layers")
        for idx, layer in enumerate(all_layers):
            if hasattr(layer, "self_attn"):
                layer.self_attn.register_forward_hook(self._attention_hook(idx))
                print(f"[INFO] Registered self_attn hook for layer {idx}")
            elif hasattr(layer, "cross_attn"):
                layer.cross_attn.register_forward_hook(self._attention_hook(idx))
                print(f"[INFO] Registered cross_attn hook for layer {idx}")
            else:
                print(f"[WARN] No attention block in layer {idx}")
            layer.register_forward_hook(self._post_ffn_hook(idx))

    def store_inputs(self, inputs):
        self.tokenized.append(inputs["input_ids"].detach().clone())
        if "pixel_values" in inputs:
            self.images.append(inputs["pixel_values"].detach().clone())

# -------------------------- Run the Capture --------------------------------
wrapper = OutputCapture(model)
wrapper.register()

prompt = "<|image|> Describe the image."
image_path = "bed.jpg"

# Ensure image exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image {image_path} does not exist.")

image = Image.open(image_path)

# Process inputs
inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt",
    padding=True,
    truncation=True
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
wrapper.store_inputs(inputs)

# Debug: Inspect input tokens
print(f"[DEBUG] Input IDs: {inputs['input_ids'].shape}, {inputs['input_ids']}")
if "pixel_values" in inputs:
    print(f"[DEBUG] Pixel Values: {inputs['pixel_values'].shape}")

try:
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1,  # Generate only 1 token
            do_sample=False,
            temperature=None,
            top_p=None
        )
except RuntimeError as e:
    print(f"[ERROR] Model generation failed: {e}")
    exit(1)

# Decode and print output
print("\nGenerated Output:")
output_text = processor.decode(output_ids[0], skip_special_tokens=True)
output_text = output_text.replace("\n", "\\n")  # Replace newlines
print(output_text)

# ---------------------------- Debug Output --------------------------------
print("\n--- Captured States ---")
print("Input IDs:", wrapper.tokenized[0].shape if wrapper.tokenized else "None")
print("Vision Embeddings:", wrapper.vision_embeddings[0].shape if wrapper.vision_embeddings else "None")
print("Text Embeddings:", wrapper.text_embeddings[0].shape if wrapper.text_embeddings else "None")
print("Attention Output [0]:", wrapper.attn_outputs[0].shape if wrapper.attn_outputs else "None")
print("First Residual [0]:", wrapper.residuals_1[0].shape if wrapper.residuals_1 else "None")
print("Second Residual [0]:", wrapper.residuals_2[0].shape if wrapper.residuals_2 else "None")

# Print decoded tokens for each layer
print("\n--- Decoded Tokens per Layer ---")
for idx, decoded in sorted(wrapper.decoded_tokens, key=lambda x: x[0]):
    print(f"Layer {idx}: {decoded if decoded is not None else 'Failed to decode'}")

# Print output of the last layer
last_layer_idx = max(idx for idx, _ in wrapper.decoded_tokens) if wrapper.decoded_tokens else None
if last_layer_idx is not None:
    last_layer_decoded = next((decoded for idx, decoded in wrapper.decoded_tokens if idx == last_layer_idx), None)
    print(f"\nLast Layer (Layer {last_layer_idx}) Decoded Output: {last_layer_decoded if last_layer_decoded is not None else 'Failed to decode'}")
else:
    print("\nLast Layer Decoded Output: No layers captured")

# Clean up memory
del inputs, output_ids
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None