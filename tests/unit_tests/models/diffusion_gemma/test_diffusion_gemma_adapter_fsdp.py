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

"""State-dict adapter behaviour for pure-FSDP2 (ep_size=1, no device mesh).

Guards that the expert-tensor mapping is unchanged on the plain-tensor path
(parity / single-rank) and that, with no MoE mesh, the adapter preserves an
FSDP-sharded expert tensor (rather than collapsing it to a local shard, which
would break the DCP global-shape match).
"""

import torch

from nemo_automodel.components.models.diffusion_gemma.state_dict_adapter import (
    DiffusionGemmaStateDictAdapter,
)


def _make_adapter() -> DiffusionGemmaStateDictAdapter:
    moe_config = type("MoE", (), {"n_routed_experts": 4})()
    # Backend / config are unused by the helpers under test.
    return DiffusionGemmaStateDictAdapter(config=None, moe_config=moe_config, backend=None, dtype=torch.float32)


def test_fold_per_expert_scale_plain_matches_direct_multiply():
    torch.manual_seed(0)
    down = torch.randn(4, 6, 8)  # [E, inter, hidden]
    scale = torch.randn(4)  # [E]

    folded = DiffusionGemmaStateDictAdapter._fold_per_expert_scale(down, scale)

    assert torch.equal(folded, down * scale[:, None, None])


def test_gather_expert_tensor_without_mesh_preserves_plain_tensor():
    adapter = _make_adapter()
    tensor = torch.randn(4, 8, 6)

    out = adapter._gather_expert_tensor(tensor, device_mesh=None, n_experts=4)

    # Same object, untouched — DCP reads/writes the global tensor itself.
    assert out is tensor


def test_from_hf_without_mesh_folds_scale_and_keeps_all_experts():
    """``from_hf`` (no mesh) maps stacked experts with the scale folded in."""
    adapter = _make_adapter()
    n_experts = 4
    hidden, inter = 8, 6

    gate_up = torch.randn(n_experts, 2 * inter, hidden)  # [E, 2*inter, hidden]
    down = torch.randn(n_experts, hidden, inter)  # [E, hidden, inter]
    scale = torch.randn(n_experts)

    hf_sd = {
        # The tied embedding (maps to model.embed_tokens.weight; the adapter
        # reconstructs the absent lm_head from it). A real checkpoint always
        # carries this; include it so from_hf exercises the full path.
        "model.decoder.embed_tokens.weight": torch.randn(16, hidden),
        "model.decoder.layers.0.experts.gate_up_proj": gate_up,
        "model.decoder.layers.0.experts.down_proj": down,
        "model.decoder.layers.0.router.per_expert_scale": scale,
    }

    native = adapter.from_hf(hf_sd, device_mesh=None)

    gate_and_up = native["model.layers.0.moe.experts.gate_and_up_projs"]
    down_projs = native["model.layers.0.moe.experts.down_projs"]

    assert gate_and_up.shape == (n_experts, hidden, 2 * inter)
    assert down_projs.shape == (n_experts, inter, hidden)
    # Scale folded into the (transposed) down projection.
    expected_down = (down.transpose(-2, -1) * scale[:, None, None]).to(torch.float32)
    assert torch.allclose(down_projs, expected_down, atol=1e-5)


def test_from_hf_lora_adapter_only_skips_lm_head_reconstruction():
    """A PEFT/LoRA checkpoint holds only adapter tensors (no base embedding), so
    ``from_hf`` must NOT try to rebuild the tied lm_head — it should pass the
    renamed adapter keys through without raising. Regression for LoRA resume.
    """
    adapter = _make_adapter()
    hidden, rank = 8, 4
    # Adapter-only state dict (decoder.* prefix, as a real checkpoint stores it);
    # no embed_tokens / lm_head / expert tensors.
    hf_sd = {
        "model.decoder.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(rank, hidden),
        "model.decoder.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(hidden, rank),
    }

    native = adapter.from_hf(hf_sd, device_mesh=None)  # must not raise

    # No lm_head is fabricated for an adapter-only checkpoint.
    assert "lm_head.weight" not in native
    # Adapter keys are renamed decoder.* -> model.* and preserved.
    assert "model.layers.0.self_attn.q_proj.lora_A.weight" in native
    assert "model.layers.0.self_attn.q_proj.lora_B.weight" in native
