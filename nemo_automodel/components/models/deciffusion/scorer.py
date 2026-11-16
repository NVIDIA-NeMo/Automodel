# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Scoring and checkpoint-contract checks for Deciffusion decision post-training."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
from torch.distributed.tensor import DTensor

SHARED_CONTEXT_FLAG = "deciffusion_shared_context"
ARCHITECTURE = "NemotronLabsDiffusionModel"
FULL_FINETUNE_PREFIXES = ("encoder.layers.", "encoder.norm.")
_WRAPPER_SEGMENTS = ("_checkpoint_wrapped_module.", "_orig_mod.")


def full_finetune_trainable(parameter_name: str) -> bool:
    """Whether full fine-tuning trains this parameter: decoder layers and final norm, not embeddings or the head.

    Activation-checkpoint and ``torch.compile`` wrapper segments in the name are ignored.
    """
    for segment in _WRAPPER_SEGMENTS:
        parameter_name = parameter_name.replace(segment, "")
    return parameter_name.startswith(FULL_FINETUNE_PREFIXES)


def require_bidirectional_text_config(config: Any) -> None:
    """Reject checkpoints whose runtime contract this recipe does not implement.

    The recipe trains one question per read with fully bidirectional attention over a text prompt. Block-causal
    checkpoints keep the same class and ``dlm_paradigm`` but need a different attention graph; vision and quantized
    variants need their own input/storage paths. None of them is run silently as an ordinary bidirectional model.

    Args:
        config: Hugging Face config (attribute access) of the checkpoint.

    Raises:
        ValueError: If the architecture, attention graph or storage format is unsupported.
    """
    architectures = list(getattr(config, "architectures", None) or [])
    if architectures != [ARCHITECTURE]:
        raise ValueError(f"expected architectures == [{ARCHITECTURE!r}], got {architectures}")
    if getattr(config, "dlm_paradigm", None) != "bidirectional":
        raise ValueError(f"dlm_paradigm must be 'bidirectional', got {getattr(config, 'dlm_paradigm', None)!r}")
    if getattr(config, "deciffusion_block_causal", None) or getattr(config, "deciffusion_bc_blocks", None):
        raise ValueError(
            "block-causal Deciffusion checkpoints are not supported: they need a block-causal attention mask"
        )
    for name in ("vision_config", "quantization_config"):
        if getattr(config, name, None):
            raise ValueError(f"{name} is set: vision and quantized Deciffusion variants are not supported")
    if getattr(config, "mask_token_id", None) is None:
        raise ValueError("the checkpoint config has no mask_token_id")


@torch.no_grad()
def gather_head_rows(model: nn.Module, token_ids: Sequence[int]) -> torch.Tensor:
    """Rows of the frozen, bias-free diffusion head for ``token_ids``.

    All ranks must call this when the head weight is a sharded ``DTensor``.

    Args:
        model: Model exposing the diffusion head through ``get_output_embeddings()``.
        token_ids: Vocabulary ids of the rows to gather, length ``T``.

    Returns:
        Tensor of shape ``[T, hidden]`` in the head's dtype (a copy).
    """
    head = model.get_output_embeddings()
    if head is None:
        raise ValueError("model exposes no output embeddings (diffusion head)")
    if getattr(head, "bias", None) is not None:
        raise ValueError("the diffusion head has a bias; restricted-row scoring would drop it")
    weight = head.weight.full_tensor() if isinstance(head.weight, DTensor) else head.weight
    index = torch.tensor(list(token_ids), dtype=torch.long, device=weight.device)
    return weight.index_select(0, index).clone()


def option_logits(
    model: nn.Module,
    input_ids: Sequence[int],
    code_token_ids: Sequence[int],
    *,
    mask_token_id: int,
    rows: dict[int, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Option logits of one question: one bidirectional read of ``prompt + [mask]``, head restricted to the codes.

    Equivalent to ``diffusion_head(hidden[:, -1])[code_token_ids]``; the frozen head rows are used directly so the
    head never has to run under a sharded wrapper.

    Args:
        model: Backbone returning the last hidden state for ``input_ids`` (``output_last_hidden_states_only=True``).
        input_ids: Prompt token ids, length ``L`` (no mask).
        code_token_ids: Single-token code of each option, length ``N``.
        mask_token_id: Diffusion mask token appended as the answer slot.
        rows: Frozen head rows by vocabulary id, each of shape ``[hidden]``.
        device: Device of the token ids.

    Returns:
        fp32 tensor of shape ``[N]``.
    """
    ids = torch.tensor([[*input_ids, mask_token_id]], dtype=torch.long, device=device)
    output = model(input_ids=ids, output_last_hidden_states_only=True, use_cache=False)
    hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
    last = hidden[0, -1]
    weight = torch.stack([rows[int(token)] for token in code_token_ids]).to(device=last.device, dtype=last.dtype)
    return (weight @ last).float()
