import torch
from unsloth.models import FastCasualModel

# Create model with full configuration
model, tokenizer = FastCasualModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B",
    max_seq_length=2048,
    device_map="cpu",     # Use CPU for testing
    attention_bias=False,
    attention_dropout=0.0,
    hidden_act="silu",
    hidden_size=4096,
    initializer_range=0.02,
    intermediate_size=11008,
    num_attention_heads=32,
    num_hidden_layers=32,
    num_key_value_heads=32,
    pretraining_tp=1,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
)

# Print model configuration
config = model.config
print("\nModel Configuration:")
print(f"vocab_size: {config.vocab_size}")
print(f"hidden_size: {config.hidden_size}")
print(f"num_attention_heads: {config.num_attention_heads}")
print(f"num_hidden_layers: {config.num_hidden_layers}")
print(f"max_position_embeddings: {config.max_position_embeddings}")
print(f"intermediate_size: {config.intermediate_size}")
print(f"num_key_value_heads: {config.num_key_value_heads}")
print(f"rope_theta: {config.rope_theta}")

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
