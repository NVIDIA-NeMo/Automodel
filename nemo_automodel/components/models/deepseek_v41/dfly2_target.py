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

"""DFly2 target features of DeepSeek V4.1: the paired layers' keys and key lists.

A ratio-1 CSA2 layer attends over ``[its FP8 sliding-window KV | the shared 1:1 compressed
KV]``: row ``r`` reads the window keys ``r - 127 .. r`` and the compressed rows its indexer
selected, offset past the window keys. The draft layer paired with it reads exactly those
keys, so this capture returns, per paired layer, that concatenated key tensor and every
row's key list in the same layout.

The keys are taken during the frozen forward by a hook on each paired attention module,
while its parameters are still gathered under FSDP: the window keys are recomputed from
the layer's input with the layer's own projections (bitwise the values the layer used),
and the compressed KV and selection are read from the attention state it publishes.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist
import torch.nn as nn

from nemo_automodel.components.models.deepseek_v41.attention import (
    DeepseekV41Attention,
    DeepseekV41AttentionOutput,
    DeepseekV41AttentionState,
    _apply_rope,
)
from nemo_automodel.components.models.deepseek_v41.quantization import quantize_cache
from nemo_automodel.components.speculative.dfly2.common import DFly2TargetFeatures


def _window_key_indices(valid: torch.Tensor, window: int) -> torch.Tensor:
    """List each row's sliding-window keys: the last ``window`` tokens up to and including the row.

    Args:
        valid: Boolean tensor of shape [batch, sequence], false at right padding.
        window: Window width.

    Returns:
        Long tensor of shape [batch, sequence, min(window, sequence)] of token rows, ``-1``
        where the slot falls before the sequence start or on padding.
    """
    batch, sequence = valid.shape
    positions = torch.arange(sequence, device=valid.device)
    slots = (positions - window + 1).clamp_min(0).unsqueeze(-1) + torch.arange(
        min(window, sequence), device=valid.device
    )
    visible = slots <= positions.unsqueeze(-1)
    safe = slots.clamp_max(sequence - 1)
    visible = visible.unsqueeze(0) & valid.gather(1, safe.flatten().expand(batch, -1)).view(batch, *slots.shape)
    return torch.where(visible, slots.unsqueeze(0), -1)


def dfly2_target_features(
    model: nn.Module, input_ids: torch.Tensor, *, attention_mask: torch.Tensor, layer_ids: Sequence[int]
) -> DFly2TargetFeatures:
    """Run the frozen V4.1 target and capture the DFly2 features of the paired layers.

    Args:
        model: ``DeepseekV41ForCausalLM`` target, text only, without context parallelism.
        input_ids: Integer tensor of shape [batch, sequence].
        attention_mask: Binary right-padding tensor of shape [batch, sequence].
        layer_ids: Paired ratio-1 layers, in draft-layer order.

    Returns:
        Features whose ``context_keys[i]`` has shape [batch, 2 * sequence, head_dim] (the
        window KV, then the compressed KV) and ``context_indices[i]`` shape [batch,
        sequence, window + index_topk] (window rows, then selected rows offset by
        ``sequence``), plus the final normalized hidden state [batch, sequence, hidden].

    Raises:
        ValueError: A paired layer is not a ratio-1 layer.
    """
    text_config = model.config.text_config
    layers = model.model.layers
    captured: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    last_hidden: list[torch.Tensor] = []

    def _capture(layer_id: int):
        def _hook(
            module: DeepseekV41Attention,
            args: tuple[torch.Tensor, ...],
            kwargs: dict[str, torch.Tensor | DeepseekV41AttentionState | dist.ProcessGroup | bool | None],
            output: DeepseekV41AttentionOutput,
        ) -> None:
            """Record the keys of a layer whose input has shape [batch, sequence, hidden]."""
            hidden_states = args[0] if args else kwargs["hidden_states"]
            angles = module.rotary_emb(kwargs["position_ids"])
            window = quantize_cache(
                _apply_rope(module.kv_norm(module.wkv(hidden_states)), angles), format="fp8", block_size=32
            )
            captured[layer_id] = (window, output.state.compressed_kv, output.state.topk_indices)

        return _hook

    handles = []
    for layer_id in layer_ids:
        if text_config.compress_ratios[layer_id] != 1:
            raise ValueError(f"DFly2 pairs ratio-1 target layers; layer {layer_id} is not one")
        handles.append(layers[str(layer_id)].attn.register_forward_hook(_capture(layer_id), with_kwargs=True))
    handles.append(model.model.norm.register_forward_hook(lambda _module, _inputs, output: last_hidden.append(output)))
    try:
        model(input_ids, attention_mask=attention_mask, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()

    sequence = input_ids.shape[1]
    window_indices = _window_key_indices(attention_mask.bool(), text_config.sliding_window)
    context_keys, context_indices = [], []
    for layer_id in layer_ids:
        window, compressed, selected = captured[layer_id]
        context_keys.append(torch.cat((window, compressed.to(window.dtype)), dim=1))
        picks = torch.where(selected >= 0, selected + sequence, -1)
        context_indices.append(torch.cat((window_indices, picks), dim=-1).to(torch.int32))
    return DFly2TargetFeatures(
        context_keys=tuple(context_keys), context_indices=tuple(context_indices), last_hidden_states=last_hidden[0]
    )


__all__ = ["dfly2_target_features"]
