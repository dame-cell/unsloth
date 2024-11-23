# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    LlamaConfig,
)
from .llama import *
from ..kernels import post_patch_loss_function
from ._utils import __version__
from transformers import set_seed as transformers_set_seed
import os
import logging
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

__all__ = ["FastBaseCasualModel"]

logger = logging.getLogger(__name__)

def _wrap_fast_inference(generate, device_type, dtype, model):
    @torch.inference_mode()
    def _fast_generate(*args, **kwargs):
        kwargs.pop("token_type_ids", None)
        
        model_eos_token_id = getattr(model.config, "eos_token_id", None)
        if model_eos_token_id is not None and hasattr(model_eos_token_id, "__iter__"):
            model_eos_token_id = model_eos_token_id[0]
        
        with torch.autocast(device_type=device_type, dtype=dtype):
            return generate(*args, **kwargs)
    return _fast_generate

def _get_dtype(dtype):
    if dtype is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return dtype

class FastBaseCasualModel:
    @staticmethod
    def create_for_pretraining(
        tokenizer,
        max_seq_length            = 2048,
        dtype                     = None,
        device_map              = "sequential",
        rope_scaling            = None,
        trust_remote_code       = False,
        use_gradient_checkpointing = "unsloth",
        # Llama model configuration
        attention_bias          = False,
        attention_dropout       = 0.0,
        hidden_act             = "silu",
        hidden_size            = 4096,
        initializer_range      = 0.02,
        intermediate_size      = 11008,
        num_attention_heads    = 32,
        num_hidden_layers      = 32,
        num_key_value_heads    = 32,
        pretraining_tp         = 1,
        rms_norm_eps           = 1e-6,
        rope_theta             = 10000.0,
        tie_word_embeddings    = False,
        *args, **kwargs,
    ):
        """Create a new model from scratch for pretraining"""
        
        # Create custom LlamaConfig for pretraining
        model_config = LlamaConfig(
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            hidden_act=hidden_act,
            hidden_size=hidden_size,
            initializer_range=initializer_range,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_seq_length,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            pretraining_tp=pretraining_tp,
            rms_norm_eps=rms_norm_eps,
            rope_scaling=rope_scaling,
            rope_theta=rope_theta,
            tie_word_embeddings=tie_word_embeddings,
            vocab_size=len(tokenizer),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            head_dim=hidden_size // num_attention_heads,
            mlp_bias=False,
            model_type="llama",
            use_cache=True,
            **kwargs
        )

        # Create fresh model with random weights
        model = AutoModelForCausalLM.from_config(
            model_config,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        # Move model to device
        if device_map is not None:
            model = model.to(device_map)

        # Apply gradient checkpointing if requested
        if use_gradient_checkpointing == "unsloth":
            model.enable_gradient_checkpointing()

        # Print model size
        model_size = sum(t.numel() for t in model.parameters())
        logger.info(f"Model size: {model_size/1000**2:.1f}M parameters")

        return model, tokenizer

    @staticmethod
    def for_inference(model):
        """Prepare model for inference by enabling evaluation mode and fast generation."""
        model.eval()
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device_type == "cuda" else torch.float32
        
        if hasattr(model, "generate"):
            model.generate = _wrap_fast_inference(
                model.generate, device_type, dtype, model
            )
        return model

    @staticmethod
    def for_training(model, use_gradient_checkpointing=True):
        """Prepare model for training with optimizations."""
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.train()
        return model
