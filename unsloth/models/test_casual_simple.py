import torch
from unsloth.models.loader import FastCasualModel
from transformers import AutoTokenizer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_casual_model():
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B")

    logger.info("Creating model...")
    # Create model with small architecture for testing
    model, tokenizer = FastCasualModel.from_pretrained(
        tokenizer=tokenizer,
        max_seq_length=32,          # Reduced for testing
        device_map="cpu",           # Use CPU for testing
        # Llama model configuration - small size for testing
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

    # Test model creation
    assert model is not None, "Model creation failed"
    assert tokenizer is not None, "Tokenizer creation failed"

    # Print model configuration
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total number of parameters: {total_params}")
    logger.info(f"Model configuration: {model.config}")

    # Test basic forward pass
    logger.info("Testing forward pass...")
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, 16))  # batch_size=1, seq_len=16
    attention_mask = torch.ones_like(input_ids)
    
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logger.info("Forward pass successful")
        assert outputs.logits.shape == (1, 16, tokenizer.vocab_size), "Unexpected output shape"
    except Exception as e:
        logger.error(f"Forward pass failed: {str(e)}")
        raise

    # Test model training mode
    logger.info("Testing training mode...")
    try:
        model.train()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = outputs.loss
        loss.backward()
        logger.info("Training mode test successful")
    except Exception as e:
        logger.error(f"Training mode test failed: {str(e)}")
        raise

    logger.info("All tests passed successfully!")

if __name__ == "__main__":
    test_casual_model()