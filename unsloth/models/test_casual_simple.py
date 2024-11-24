# Import necessary libraries
import torch
from unsloth.models.loader import FastCasualModel
from transformers import AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B")

# Create model with small architecture for testing
model, tokenizer = FastCasualModel.create_for_pretraining(
    tokenizer=tokenizer,
    max_seq_length=32,          # Reduced from 2048
    device_map="cpu",           # Use CPU for testing
    attention_bias=False,
    attention_dropout=0.0,
    hidden_act="silu",
    hidden_size=128,            # Reduced from 4096
    initializer_range=0.02,
    intermediate_size=256,      # Reduced from 11008
    num_attention_heads=4,      # Reduced from 32
    num_hidden_layers=2,        # Reduced from 32
    num_key_value_heads=4,      # Reduced from 32
    pretraining_tp=1,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
)

# Print model configuration
total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters:", total_params)
config = model.config
print("Model configuration:", config) 
print(model)