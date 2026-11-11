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

"""Weight-name conversion between the Praxis checkpoint and the Automodel module tree.

The checkpoint (``model-*.safetensors`` with ``model.safetensors.index.json``) stores the
trunk under ``backbone.{encoder,decoder}.`` and the expert under ``action_expert.``. The
action expert names are identical on both sides. The trunk differs in these places:

* projections: ``t_proj.VideoTensor`` -> ``time_proj``; ``input_projections.TextEncodingTensor.proj.{0,2,3}``
  -> ``text_in.{fc1,fc2,norm}``; ``input_projections.VideoTensor.proj`` -> ``video_in``;
  ``output_projections.VideoTensor.proj`` -> ``video_out`` (encoder) or
  ``output_projections.VideoTensor.proj.{0,1}`` -> ``video_out.{norm,linear}`` (decoder);
* attention: ``mha.in_proj.{q,k,v}_weight`` -> ``attn.{q,k,v}_proj.weight`` and the other
  ``mha.*`` names -> ``attn.*``;
* dense feed-forward: ``ffn.0.linear.{a,b}_weight`` -> ``ffn.{gate,up}_proj.weight`` and
  ``ffn.1.weight`` -> ``ffn.down_proj.weight``;
* mixture of experts: ``ffn.router.weight`` -> ``ffn.gate.weight``, ``ffn.expert_bias`` ->
  ``ffn.gate.e_score_correction_bias``, ``ffn.routed.{gate_up,down}_weight`` ->
  ``ffn.experts.{gate_and_up_projs,down_projs}`` (same ``[experts, in, out]`` layout), and the
  packed shared expert ``ffn.shared.gate_up`` of shape [hidden, 2 * inter] is split and
  transposed into ``ffn.shared_experts.{gate,up}_proj.weight`` of shape [inter, hidden];
  ``ffn.shared.down`` of shape [inter, hidden] is transposed into ``down_proj.weight``.
"""

from __future__ import annotations

import re

import torch
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter

_PREFIX = re.compile(r"^(?P<tower>backbone\.(?:encoder|decoder)\.)(?P<rest>.+)$")
_BLOCK = re.compile(r"^(?P<block>core\.blocks\.\d+\.)(?P<leaf>.+)$")

_ENCODER_LEAVES = {
    "t_proj.VideoTensor.weight": "time_proj.weight",
    "t_proj.VideoTensor.bias": "time_proj.bias",
    "input_projections.TextEncodingTensor.proj.0.weight": "text_in.fc1.weight",
    "input_projections.TextEncodingTensor.proj.0.bias": "text_in.fc1.bias",
    "input_projections.TextEncodingTensor.proj.2.weight": "text_in.fc2.weight",
    "input_projections.TextEncodingTensor.proj.2.bias": "text_in.fc2.bias",
    "input_projections.TextEncodingTensor.proj.3.weight": "text_in.norm.weight",
    "input_projections.VideoTensor.proj.weight": "video_in.weight",
    "input_projections.VideoTensor.proj.bias": "video_in.bias",
    "output_projections.VideoTensor.proj.weight": "video_out.weight",
    "output_projections.VideoTensor.proj.bias": "video_out.bias",
}
_DECODER_LEAVES = {
    "t_proj.VideoTensor.weight": "time_proj.weight",
    "t_proj.VideoTensor.bias": "time_proj.bias",
    "input_projections.VideoTensor.proj.weight": "video_in.weight",
    "input_projections.VideoTensor.proj.bias": "video_in.bias",
    "output_projections.VideoTensor.proj.0.weight": "video_out.norm.weight",
    "output_projections.VideoTensor.proj.1.weight": "video_out.linear.weight",
    "output_projections.VideoTensor.proj.1.bias": "video_out.linear.bias",
}
_BLOCK_LEAVES = {
    "mha.in_proj.q_weight": "attn.q_proj.weight",
    "mha.in_proj.k_weight": "attn.k_proj.weight",
    "mha.in_proj.v_weight": "attn.v_proj.weight",
    "mha.gate_proj.weight": "attn.gate_proj.weight",
    "mha.q_norm.weight": "attn.q_norm.weight",
    "mha.k_norm.weight": "attn.k_norm.weight",
    "mha.out_proj.weight": "attn.out_proj.weight",
    "mha.out_proj.bias": "attn.out_proj.bias",
    "ffn.0.linear.a_weight": "ffn.gate_proj.weight",
    "ffn.0.linear.b_weight": "ffn.up_proj.weight",
    "ffn.1.weight": "ffn.down_proj.weight",
    "ffn.router.weight": "ffn.gate.weight",
    "ffn.expert_bias": "ffn.gate.e_score_correction_bias",
    "ffn.routed.gate_up_weight": "ffn.experts.gate_and_up_projs",
    "ffn.routed.down_weight": "ffn.experts.down_projs",
}
_SHARED_GATE_UP = "ffn.shared.gate_up"
_SHARED_DOWN = "ffn.shared.down"
_SHARED_GATE = "ffn.shared_experts.gate_proj.weight"
_SHARED_UP = "ffn.shared_experts.up_proj.weight"
_SHARED_DOWN_NATIVE = "ffn.shared_experts.down_proj.weight"

_MODULATOR_HF = re.compile(r"^core\.modulators\.(?P<i>\d+)\.(?P<p>weight|bias)$")
_MODULATOR_NATIVE = re.compile(r"^core\.blocks\.(?P<i>\d+)\.modulator\.(?P<p>weight|bias)$")

_LEAVES_TO_HF = {v: k for k, v in _BLOCK_LEAVES.items()}
_ENCODER_TO_HF = {v: k for k, v in _ENCODER_LEAVES.items()}
_DECODER_TO_HF = {v: k for k, v in _DECODER_LEAVES.items()}


def _split_key(key: str) -> tuple[str, str, str] | None:
    """Split a trunk key into ``(tower prefix, block prefix or '', leaf)``; ``None`` for other keys."""
    m = _PREFIX.match(key)
    if m is None:
        return None
    tower, rest = m.group("tower"), m.group("rest")
    b = _BLOCK.match(rest)
    if b is None:
        return tower, "", rest
    return tower, b.group("block"), b.group("leaf")


class PraxisStateDictAdapter(StateDictAdapter):
    """Convert between checkpoint keys (``from_hf``) and module keys (``to_hf``)."""

    def __init__(self, shared_expert_inter_dim: int) -> None:
        self.shared_expert_inter_dim = shared_expert_inter_dim

    def from_hf(
        self, hf_state_dict: dict[str, Tensor], device_mesh: DeviceMesh | None = None, **kwargs
    ) -> dict[str, Tensor]:
        """Map checkpoint tensors to module tensors.

        Args:
            hf_state_dict: Checkpoint tensors keyed by their checkpoint names.

        Returns:
            Tensors keyed by module names. Shared-expert weights are new contiguous tensors; every
            other tensor is passed through.
        """
        out: dict[str, Tensor] = {}
        for key, value in hf_state_dict.items():
            parts = _split_key(key)
            if parts is None:
                out[key] = value
                continue
            tower, block, leaf = parts
            modulator = _MODULATOR_HF.match(leaf) if not block else None
            if modulator is not None:
                out[f"{tower}core.blocks.{modulator.group('i')}.modulator.{modulator.group('p')}"] = value
            elif block:
                if leaf == _SHARED_GATE_UP:
                    gate, up = value.chunk(2, dim=1)
                    out[tower + block + _SHARED_GATE] = gate.t().contiguous()
                    out[tower + block + _SHARED_UP] = up.t().contiguous()
                elif leaf == _SHARED_DOWN:
                    out[tower + block + _SHARED_DOWN_NATIVE] = value.t().contiguous()
                else:
                    out[tower + block + _BLOCK_LEAVES.get(leaf, leaf)] = value
            else:
                table = _ENCODER_LEAVES if tower.endswith("encoder.") else _DECODER_LEAVES
                out[tower + table.get(leaf, leaf)] = value
        return out

    def to_hf(self, state_dict: dict[str, Tensor], exclude_key_regex: str | None = None, **kwargs) -> dict[str, Tensor]:
        """Map module tensors back to checkpoint tensors (the inverse of :meth:`from_hf`).

        Args:
            state_dict: Module tensors keyed by module names.

        Returns:
            Tensors keyed by checkpoint names; the two shared-expert input projections of a block
            are packed into one ``[hidden, 2 * inter]`` tensor.
        """
        if exclude_key_regex is not None:
            state_dict = {k: v for k, v in state_dict.items() if not re.match(exclude_key_regex, k)}
        out: dict[str, Tensor] = {}
        pending: dict[str, dict[str, Tensor]] = {}
        for key, value in state_dict.items():
            parts = _split_key(key)
            if parts is None:
                out[key] = value
                continue
            tower, block, leaf = parts
            if block and leaf in ("modulator.weight", "modulator.bias"):
                index = block.split(".")[2]
                out[f"{tower}core.modulators.{index}.{leaf.split('.')[1]}"] = value
            elif block:
                if leaf in (_SHARED_GATE, _SHARED_UP):
                    pending.setdefault(tower + block, {})[leaf] = value
                elif leaf == _SHARED_DOWN_NATIVE:
                    out[tower + block + _SHARED_DOWN] = value.t().contiguous()
                else:
                    out[tower + block + _LEAVES_TO_HF.get(leaf, leaf)] = value
            else:
                table = _ENCODER_TO_HF if tower.endswith("encoder.") else _DECODER_TO_HF
                out[tower + table.get(leaf, leaf)] = value
        for prefix, halves in pending.items():
            if set(halves) != {_SHARED_GATE, _SHARED_UP}:
                raise KeyError(f"{prefix}: both shared-expert projections are needed to pack gate_up")
            out[prefix + _SHARED_GATE_UP] = torch.cat(
                [halves[_SHARED_GATE].t(), halves[_SHARED_UP].t()], dim=1
            ).contiguous()
        return out

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Tensor, **kwargs) -> list[tuple[str, Tensor]]:
        """Convert one module tensor to its checkpoint tensor(s); see :meth:`to_hf`."""
        return list(self.to_hf({fqn: tensor}).items())
