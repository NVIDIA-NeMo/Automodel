# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""FLUX.3 Qwen3-VL text-encoding helpers shared by the preprocessing processors.

FLUX.3 conditions on stacked Qwen3-VL hidden states: the prompt goes through the
chat template, is right-padded to a multiple of 80 tokens, and evenly spaced hidden
layers (derived from the checkpoint geometry) are concatenated per token.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor


class Flux3TextStack(TypedDict):
    """Frozen image-small text encoder, processor, and selected hidden layers."""

    text_encoder: Qwen3VLForConditionalGeneration
    tokenizer: Qwen3VLProcessor
    text_encoder_out_layers: tuple[int, ...]


logger = logging.getLogger(__name__)

# FLUX.3 modular pipeline text-encoding default (Flux3TextEncoderStep).
MAX_SEQUENCE_LENGTH = 8192
PAD_MULTIPLE = 80


def resolve_text_encoder_out_layers(
    joint_attention_dim: int, text_hidden_size: int, text_num_hidden_layers: int
) -> tuple[int, ...]:
    """Derive the text-encoder hidden layers to stack from checkpoint geometry.

    FLUX.3 conditions on ``joint_attention_dim / hidden_size`` stacked hidden
    layers, evenly spaced through the encoder depth. For the released
    checkpoint (Qwen3-VL-4B: 36 layers, hidden 2560, joint 20480) this yields
    the pipeline default (4, 8, ..., 32); for tiny test checkpoints it scales
    down consistently.

    Args:
        joint_attention_dim: Transformer's text-embedding channel width.
        text_hidden_size: Text encoder hidden size.
        text_num_hidden_layers: Text encoder layer count.

    Returns:
        Tuple of hidden-state indices to stack, in ascending order.

    Raises:
        ValueError: If the checkpoint geometry is inconsistent (dim not a
            multiple of hidden size, or more stacked layers than the encoder has).
    """
    if joint_attention_dim % text_hidden_size != 0:
        raise ValueError(
            f"joint_attention_dim ({joint_attention_dim}) is not a multiple of the text encoder "
            f"hidden size ({text_hidden_size}); cannot derive text_encoder_out_layers."
        )
    num_stack = joint_attention_dim // text_hidden_size
    step = text_num_hidden_layers // num_stack
    if step < 1:
        raise ValueError(
            f"Checkpoint expects {num_stack} stacked text-encoder layers but the encoder only has "
            f"{text_num_hidden_layers} layers."
        )
    return tuple(step * i for i in range(1, num_stack + 1))


def load_qwen3vl_text_stack(model_name: str, device: str) -> Flux3TextStack:
    """Load the FLUX.3 Qwen3-VL text stack from a resolved modular snapshot dir.

    Args:
        model_name: Resolved local snapshot dir of the FLUX.3 modular checkpoint.
        device: Device to load the text encoder on.

    Returns:
        Dict with 'text_encoder' (Qwen3VLForConditionalGeneration), 'tokenizer'
        (AutoProcessor), and 'text_encoder_out_layers' (tuple of hidden-state
        indices derived from the checkpoint geometry).
    """
    from diffusers import Flux3Transformer2DModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    logger.info("  Configuring Qwen3-VL text encoder...")
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    # FLUX.3 repos ship a copy of the model config.json inside tokenizer/ to
    # work around the AutoProcessor+subfolder bug in transformers 4.57.x.
    tokenizer = AutoProcessor.from_pretrained(model_name, subfolder="tokenizer")

    # The number of stacked hidden layers is fixed by the transformer's
    # joint_attention_dim; derive it so tiny test checkpoints work too.
    transformer_config = Flux3Transformer2DModel.load_config(model_name, subfolder="transformer")
    out_layers = resolve_text_encoder_out_layers(
        joint_attention_dim=transformer_config["joint_attention_dim"],
        text_hidden_size=text_encoder.config.text_config.hidden_size,
        text_num_hidden_layers=text_encoder.config.text_config.num_hidden_layers,
    )
    logger.info("  Text encoder out layers: %s", out_layers)
    return {"text_encoder": text_encoder, "tokenizer": tokenizer, "text_encoder_out_layers": out_layers}


def encode_text_qwen3vl(prompt: str, models: Flux3TextStack, device: str) -> torch.Tensor:
    """Encode a prompt with FLUX3's training-time text-conditioning semantics.

    Each prompt is right-padded to a multiple of 80 tokens. Explicit Qwen
    position IDs keep padding tokens at position 1, including with Transformers
    versions that would otherwise assign them position 0.

    Args:
        prompt: Caption used to condition image generation.
        models: Frozen text encoder, processor, and hidden-layer indices.
        device: Device used for text encoding.

    Returns:
        FP16 CPU tensor of shape [sequence, joint_attention_dim], where sequence
        is the prompt length rounded up to 80 tokens, capped at 8192. All rows,
        including padding, remain visible to the diffusion transformer.
    """
    text_encoder = models["text_encoder"]
    tokenizer = models["tokenizer"]

    with torch.no_grad():
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        unpadded = tokenizer.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQUENCE_LENGTH)
        prompt_length = unpadded["input_ids"].shape[1]
        padded_length = min(((prompt_length + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE, MAX_SEQUENCE_LENGTH)
        encoded = tokenizer.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            padding_side="right",
            truncation=True,
            max_length=padded_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        position_ids = torch.arange(padded_length, device=device)[None].masked_fill(attention_mask == 0, 1)
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        prompt_embeds = torch.cat(
            [output.hidden_states[layer] for layer in models["text_encoder_out_layers"]], dim=-1
        ).to(device=device, dtype=text_encoder.dtype)

    return prompt_embeds.squeeze(0).detach().cpu().to(torch.float16)
