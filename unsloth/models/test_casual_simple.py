import torch
from unsloth.models import FastCasualModel

# Create a small model for testing
model, tokenizer = FastCasualModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B",
    max_seq_length=32,    # Small for testing
    hidden_size=64,       # Small for testing
    num_attention_heads=2,
    num_hidden_layers=2,
    intermediate_size=128,
    device_map="cpu"     # Use CPU for testing
)

# Print model configuration
config = model.config
print("\nModel Configuration:")
print(f"vocab_size: {config.vocab_size}")
print(f"hidden_size: {config.hidden_size}")
print(f"num_attention_heads: {config.num_attention_heads}")
print(f"num_hidden_layers: {config.num_hidden_layers}")
print(f"max_position_embeddings: {config.max_position_embeddings}")

# Try a simple forward pass
input_ids = torch.randint(0, config.vocab_size, (1, 16))
attention_mask = torch.ones_like(input_ids)

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
)

print("\nModel Output Shape:", outputs.logits.shape)

# Test training mode
model = FastCasualModel.for_training(model)
print("\nTraining Mode:", model.training)

# Test inference mode
model = FastCasualModel.for_inference(model)
print("Inference Mode:", model.training)
