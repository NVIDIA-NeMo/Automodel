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

"""Unsharded control coverage for the M3 post-FSDP router precision fix."""

import torch

from nemo_automodel.components.models.minimax_m3_vl.layers import MiniMaxM3RMSNorm
from nemo_automodel.components.models.minimax_m3_vl.model import (
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
)
from nemo_automodel.components.moe.layers import Gate


def _check_initialization(model: MiniMaxM3SparseForCausalLM | MiniMaxM3SparseForConditionalGeneration) -> None:
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)
    gates = [module for name, module in model.named_modules() if name.split(".")[-1] == "gate"]
    assert gates
    assert all(isinstance(gate, Gate) for gate in gates)
    for gate in gates:
        assert gate.weight.dtype == torch.float32
        assert gate.weight.std() > 0
        assert not torch.equal(gate.weight, gate.weight.bfloat16().float())
        assert gate.e_score_correction_bias.dtype == torch.float32
    norms = [module for module in model.modules() if isinstance(module, MiniMaxM3RMSNorm)]
    assert norms
    for norm in norms:
        assert norm.weight.dtype == torch.bfloat16
        assert torch.count_nonzero(norm.weight) == 0
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.model.rotary_emb._compute_concentration_and_inv_freq()[1].dtype == torch.float32


def test_unsharded_text_router_initialization(model: MiniMaxM3SparseForCausalLM) -> None:
    _check_initialization(model)


def test_unsharded_vl_router_initialization(vlm_model: MiniMaxM3SparseForConditionalGeneration) -> None:
    _check_initialization(vlm_model)


def test_unsharded_mtp_router_initialization(mtp_model: MiniMaxM3SparseForCausalLM) -> None:
    _check_initialization(mtp_model)
    assert any("mtp" in name and isinstance(module, Gate) for name, module in mtp_model.named_modules())
