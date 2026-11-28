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

"""Scoring and checkpoint-contract checks for Nemotron Decision post-training."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from nemo_automodel.components.models.nemotron_decision.model import ARCHITECTURE
from nemo_automodel.components.models.nemotron_decision.packed import PackedRead, allowed_attention

FULL_FINETUNE_PREFIXES = ("encoder.layers.", "encoder.norm.")
_WRAPPER_SEGMENTS = ("_checkpoint_wrapped_module.", "_orig_mod.")


def full_finetune_trainable(parameter_name: str) -> bool:
    """Whether backbone fine-tuning trains this parameter: decoder layers and final norm, not embeddings or the head.

    Activation-checkpoint and ``torch.compile`` wrapper segments in the name are ignored.
    """
    for segment in _WRAPPER_SEGMENTS:
        parameter_name = parameter_name.replace(segment, "")
    return parameter_name.startswith(FULL_FINETUNE_PREFIXES)


def require_decision_config(config: Any) -> None:
    """Reject checkpoints whose runtime contract this recipe does not implement.

    The recipe trains Nemotron 3.5 Decision checkpoints: a text Ministral 3 encoder read with bidirectional attention,
    the decision head at each question's mask token, one question per read or (``packed_read``) several questions of
    one state per read. Packed reads need a bidirectional checkpoint; a ``read_layout`` block in the config, when
    present, must name the checkpoint's own mask token. Vision and quantized variants need their own input and storage
    paths and are not run silently as the text model.

    Args:
        config: Hugging Face config (attribute access) of the checkpoint.

    Raises:
        ValueError: If the architecture, attention, storage format or declared read layout is unsupported.
    """
    architectures = list(getattr(config, "architectures", None) or [])
    if architectures != [ARCHITECTURE]:
        raise ValueError(f"expected architectures == [{ARCHITECTURE!r}], got {architectures}")
    if getattr(config, "model_type", None) != "ministral3":
        raise ValueError(f"expected model_type 'ministral3', got {getattr(config, 'model_type', None)!r}")
    if getattr(config, "is_causal", False):
        raise ValueError("decision checkpoints are read bidirectionally; is_causal must be false")
    for name in ("vision_config", "quantization_config"):
        if getattr(config, name, None):
            raise ValueError(f"{name} is set: vision and quantized decision variants are not supported")
    if getattr(config, "mask_token_id", None) is None:
        raise ValueError("the checkpoint config has no mask_token_id")
    layout = getattr(config, "read_layout", None)
    if layout is not None:
        if not isinstance(layout, dict):
            raise ValueError("read_layout must be a mapping")
        if layout.get("mask_token_id", config.mask_token_id) != config.mask_token_id:
            raise ValueError("read_layout.mask_token_id differs from the checkpoint's mask_token_id")
        if layout.get("prompt_format", "json-sep") not in ("json", "json-sep"):
            raise ValueError(f"unsupported read_layout.prompt_format {layout.get('prompt_format')!r}")


@torch.no_grad()
def gather_head_rows(model: nn.Module, token_ids: Sequence[int]) -> torch.Tensor:
    """Rows of the frozen, bias-free decision head for ``token_ids``.

    All ranks must call this when the head weight is a sharded ``DTensor``.

    Args:
        model: Model exposing the decision head through ``get_output_embeddings()``.
        token_ids: Vocabulary ids of the rows to gather, length ``T``.

    Returns:
        Tensor of shape ``[T, hidden]`` in the head's dtype (a copy).
    """
    head = model.get_output_embeddings()
    if head is None:
        raise ValueError("model exposes no output embeddings (decision head)")
    if getattr(head, "bias", None) is not None:
        raise ValueError("the decision head has a bias; restricted-row scoring would drop it")
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
        mask_token_id: Mask token appended as the answer slot.
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


def packed_option_logits(
    model: nn.Module,
    read: PackedRead,
    code_token_ids: Sequence[Sequence[int]],
    *,
    rows: dict[int, torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    """Option logits of every question of one packed read (see ``packed.py`` for the layout).

    One forward over the packed ids with their positions and the dense boolean attention mask ``[1, 1, L, L]``
    (``True`` = attend: state -> state; question -> state + itself); question ``k`` is read at ``read.slots[k]`` with
    the head restricted to its codes.

    Args:
        model: Backbone returning the last hidden state (``output_last_hidden_states_only=True``) and accepting a 4D
            boolean ``attention_mask`` and ``position_ids`` (``NemotronForDecision`` with SDPA or eager attention).
        read: The packed read.
        code_token_ids: Single-token codes of each question's options, one sequence per slot.
        rows: Frozen head rows by vocabulary id, each of shape ``[hidden]``.
        device: Device of the token ids.

    Returns:
        One fp32 tensor of shape ``[options]`` per question, in slot order.
    """
    if len(code_token_ids) != len(read.slots):
        raise ValueError("code_token_ids needs one entry per read slot")
    ids = torch.tensor([read.input_ids], dtype=torch.long, device=device)
    positions = torch.tensor([read.position_ids], dtype=torch.long, device=device)
    mask = allowed_attention(torch.tensor([read.groups], device=device))[:, None]  # [1, 1, L, L], True = attend
    output = model(
        input_ids=ids, attention_mask=mask, position_ids=positions, output_last_hidden_states_only=True, use_cache=False
    )
    hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
    out = []
    for slot, codes in zip(read.slots, code_token_ids):
        state = hidden[0, slot]
        weight = torch.stack([rows[int(token)] for token in codes]).to(device=state.device, dtype=state.dtype)
        out.append((weight @ state).float())
    return out
