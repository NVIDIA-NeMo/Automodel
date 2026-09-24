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

"""Shared construction of identical HF and native Beam models for parity tests."""

from __future__ import annotations

import glob
import os
import tempfile
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import nn

from nemo_automodel.components.models.beam.model import BeamForCausalLM
from nemo_automodel.components.models.common import BackendConfig

if TYPE_CHECKING:
    from transformers.models.beam import BeamConfig


def hf_model_and_checkpoint(cfg: BeamConfig, device: torch.device) -> tuple[nn.Module, dict[str, torch.Tensor]]:
    """Create HF weights with nontrivial gate biases and save the per-expert checkpoint layout."""
    from safetensors.torch import load_file
    from transformers.models.beam import BeamForCausalLM as HFBeam

    torch.manual_seed(0)
    hf = HFBeam(cfg)
    with torch.no_grad():
        for layer in hf.model.layers:
            layer.self_attn.gate_bias.normal_(std=0.5)
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.e_score_correction_bias.normal_(std=0.1)
    with tempfile.TemporaryDirectory() as d:
        hf.save_pretrained(d, safe_serialization=True)
        sd = {}
        for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            sd.update(load_file(f))
    return hf.to(device).eval(), sd


def native_from_hf(
    cfg: BeamConfig, backend_config: BackendConfig, sd: dict[str, torch.Tensor], device: torch.device
) -> BeamForCausalLM:
    """Load an HF checkpoint into a native model.

    Args:
        cfg: HF model configuration.
        backend_config: Native backend selection.
        sd: HF tensors: per-expert gate/up weights [moe_intermediate, hidden], down
            weights [hidden, moe_intermediate], and unchanged non-expert layouts.
        device: Target device.

    Returns:
        Native model in evaluation mode with the same parameter values.
    """
    model = BeamForCausalLM(cfg, backend=backend_config).to(device)
    native_sd = model.state_dict_adapter.from_hf(dict(sd))
    missing, unexpected = model.load_state_dict(native_sd, strict=False)
    assert not unexpected, unexpected
    # TE modules carry non-parameter ``_extra_state`` (FP8 metadata) entries; RoPE caches are buffers.
    assert not [k for k in missing if "rotary" not in k and not k.endswith("_extra_state")], missing
    return model.eval()


class RopeCall(NamedTuple):
    """One attention-boundary observation, retaining no tensor storage."""

    x_dtype: torch.dtype
    dtypes: tuple[torch.dtype, torch.dtype]
    storages: tuple[int, int]
    exact: bool


def capture_rope_calls(model: BeamForCausalLM) -> list[RopeCall]:
    """Record the dtype, exact values and shared storage of each RoPE attention call."""
    from nemo_automodel.components.models.beam.layers import hf_rope_tables
    from nemo_automodel.components.models.gpt_oss.rope_utils import position_ids_to_freqs_cis

    calls = []

    def hook(attn, args, kwargs):
        """Inspect the attention boundary without retaining tensors.

        Args:
            attn: Beam attention module.
            args: First tensor is hidden states [batch, sequence, hidden] or [tokens, hidden].
            kwargs: FP32 rope_tables (cos, sin), each [..., 1, head_dim], and integer
                position_ids matching the hidden tensor's leading axes; optional qkv_format.
        """
        cos, sin = kwargs["rope_tables"]
        freqs = position_ids_to_freqs_cis(
            model.model.rotary_emb[attn.layer_type],
            kwargs["position_ids"],
            qkv_format=kwargs.get("qkv_format", "bshd"),
            for_fused_rope=False,
        )
        exp_cos, exp_sin = hf_rope_tables(freqs)
        calls.append(
            RopeCall(
                x_dtype=args[0].dtype,
                dtypes=(cos.dtype, sin.dtype),
                storages=(cos.untyped_storage().data_ptr(), sin.untyped_storage().data_ptr()),
                exact=torch.equal(cos, exp_cos) and torch.equal(sin, exp_sin),
            )
        )

    for layer in model.model.layers.values():
        if layer.self_attn.use_rope:
            layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True)
    return calls
