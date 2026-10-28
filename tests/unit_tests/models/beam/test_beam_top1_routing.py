# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Top-1 routing normalization (HF normalization): ``BeamGate`` follows HF ``BeamTopKRouter`` (normalize whenever
``norm_topk_prob``) where the shared ``Gate`` normalizes only for top-k > 1; ``BeamMoE`` installs it."""

import copy

import pytest
import torch

from nemo_automodel.components.models.beam.layers import BeamGate, BeamMoE
from nemo_automodel.components.models.beam.model import BeamForCausalLM, beam_moe_config
from nemo_automodel.components.moe.layers import Gate, MoE

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture
def top1_config(beam_config):
    cfg = copy.deepcopy(beam_config)
    cfg.num_experts, cfg.num_experts_per_tok, cfg.n_group, cfg.topk_group = 4, 1, 1, 1
    return cfg


def _hf_router(cfg, seed, device, weight_scale=1.0):
    from transformers.models.beam.modeling_beam import BeamTopKRouter

    g = torch.Generator().manual_seed(seed)
    router = BeamTopKRouter(cfg)
    with torch.no_grad():
        router.weight.copy_(torch.randn(router.weight.shape, generator=g) * weight_scale)
        router.e_score_correction_bias.copy_(torch.randn(cfg.num_experts, generator=g) * 0.1)
    return router.to(device)


def _native_gate(cfg, hf_router, device, gate_cls=BeamGate):
    gate = gate_cls(beam_moe_config(cfg, torch.bfloat16), gate_precision=torch.float32).to(device)
    with torch.no_grad():
        gate.weight.copy_(hf_router.weight)
        gate.e_score_correction_bias.copy_(hf_router.e_score_correction_bias)
    return gate


class TestBeamGateTop1:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    @pytest.mark.parametrize("norm_topk_prob", [True, False])
    @pytest.mark.parametrize("route_scale", [2.5, 1.0])
    def test_matches_hf_weights_indices_and_gradients(self, top1_config, device, dtype, norm_topk_prob, route_scale):
        """Identical fp32 router weights/bias and activations: same expert, same routing weight (HF's normalized
        weight times the route scale) and the same input / router-weight gradients, bit for bit."""
        cfg = top1_config
        cfg.norm_topk_prob, cfg.routed_scaling_factor = norm_topk_prob, route_scale
        hf = _hf_router(cfg, 3, device, weight_scale=0.1)  # logits within ~+-8: scores >= 3e-4, so w/(w+1e-20) == 1
        gate = _native_gate(cfg, hf, device)
        x = (torch.randn(512, cfg.hidden_size, generator=torch.Generator().manual_seed(4)) * 2).to(device, dtype)
        upstream = torch.randn(512, 1, generator=torch.Generator().manual_seed(5)).to(device)
        x_hf, x_nat = x.clone().requires_grad_(), x.clone().requires_grad_()

        _, hf_w, hf_idx = hf(x_hf)
        hf_scaled = hf_w * route_scale  # HF applies routed_scaling_factor to the aggregated output
        nat_w, nat_idx, _ = gate(x_nat, torch.ones(512, dtype=torch.bool, device=device), None)

        assert torch.equal(nat_idx, hf_idx)
        assert nat_w.dtype == torch.float32 and torch.equal(nat_w, hf_scaled)
        if norm_topk_prob:  # ordinary scores: HF's normalized top-1 weight is exactly 1.0 (times the route scale)
            assert torch.equal(nat_w, torch.full_like(nat_w, route_scale))
        (hf_scaled * upstream).sum().backward()
        (nat_w * upstream).sum().backward()
        assert torch.equal(x_nat.grad, x_hf.grad)
        assert torch.equal(gate.weight.grad, hf.weight.grad)

    @pytest.mark.parametrize("device", DEVICES)
    def test_tiny_scores_keep_hf_epsilon_semantics(self, top1_config, device):
        """Logits around -40 give sigmoid scores ~4e-18, where w / (w + 1e-20) is ~0.998 (the epsilon is material)
        and the router receives a non-zero gradient: still bit-identical to HF."""
        cfg = top1_config
        cfg.norm_topk_prob, cfg.routed_scaling_factor = True, 2.5
        hf = _hf_router(cfg, 6, device, weight_scale=0.0)
        with torch.no_grad():  # constant negative logits: scores = sigmoid(-40 - eps)
            hf.weight.fill_(-40.0 / cfg.hidden_size)
            hf.e_score_correction_bias.zero_()
        gate = _native_gate(cfg, hf, device)
        x = torch.ones(64, cfg.hidden_size, device=device) + 1e-3 * torch.randn(
            64, cfg.hidden_size, generator=torch.Generator().manual_seed(7)
        ).to(device)
        x_hf, x_nat = x.clone().requires_grad_(), x.clone().requires_grad_()
        _, hf_w, hf_idx = hf(x_hf)
        nat_w, nat_idx, _ = gate(x_nat, torch.ones(64, dtype=torch.bool, device=device), None)
        assert (hf_w < 0.999).all(), "test setup: scores must be tiny enough for the epsilon to matter"
        assert torch.equal(nat_idx, hf_idx) and torch.equal(nat_w, hf_w * 2.5)
        (hf_w * 2.5).sum().backward()
        nat_w.sum().backward()
        assert gate.weight.grad.abs().max() > 0
        assert torch.equal(gate.weight.grad, hf.weight.grad) and torch.equal(x_nat.grad, x_hf.grad)

    def test_top2_unchanged_versus_shared_gate(self, beam_config):
        """For top-k > 1 the override is inert: BeamGate equals the shared Gate bit for bit."""
        hf = _hf_router(beam_config, 8, "cpu")
        beam_gate, shared_gate = (
            _native_gate(beam_config, hf, "cpu"),
            _native_gate(beam_config, hf, "cpu", gate_cls=Gate),
        )
        x = torch.randn(256, beam_config.hidden_size).to(torch.bfloat16)
        mask = torch.ones(256, dtype=torch.bool)
        w1, i1, _ = beam_gate(x, mask, None)
        w2, i2, _ = shared_gate(x, mask, None)
        assert torch.equal(w1, w2) and torch.equal(i1, i2)

    def test_routing_core_path_matches_eager(self, top1_config):
        """The CUDA-graph routing core calls gate._route_scores, so it sees the override too."""
        cfg = top1_config
        hf = _hf_router(cfg, 9, "cpu")
        gate = _native_gate(cfg, hf, "cpu")
        x = torch.randn(128, cfg.hidden_size)
        mask = torch.ones(128, dtype=torch.bool)
        w_eager, i_eager, _ = gate(x, mask, None)
        gate.use_routing_core = True
        w_core, i_core, _ = gate(x, mask, None)
        assert torch.equal(w_eager, w_core) and torch.equal(i_eager, i_core)
        assert torch.allclose(w_core, torch.full_like(w_core, cfg.routed_scaling_factor))


class TestBeamMoEInstallation:
    def test_block_builds_beam_moe_with_beam_gate(self, beam_config, backend_config):
        model = BeamForCausalLM(beam_config, backend=backend_config)
        model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)
        for name, layer in model.model.layers.items():
            if name == "0":
                continue
            assert isinstance(layer.mlp, BeamMoE) and isinstance(layer.mlp, MoE)
            assert isinstance(layer.mlp.gate, BeamGate) and isinstance(layer.mlp.gate, Gate)
            assert torch.isfinite(layer.mlp.gate.weight).all() and layer.mlp.gate.weight.std() > 0
            assert layer.mlp.gate.use_routing_core is False
            assert layer.mlp.gate.weight.dtype == torch.float32 and layer.mlp.gate.gate_precision == torch.float32
        keys = set(model.state_dict().keys())
        assert {"model.layers.1.mlp.gate.weight", "model.layers.1.mlp.gate.e_score_correction_bias"} <= keys

    def test_load_balance_tracking_sees_beam_gates(self, beam_config, backend_config):
        """The MoE load-balance metrics enable tracking via isinstance(Gate): every BeamGate records its expert load."""
        from nemo_automodel.components.moe.load_balance_metrics import enable_load_balance_tracking

        model = BeamForCausalLM(beam_config, backend=backend_config)
        enable_load_balance_tracking(model)
        gates = [layer.mlp.gate for name, layer in model.model.layers.items() if name != "0"]
        assert gates and all(isinstance(g, BeamGate) and g._track_load_balance for g in gates)
        ids = torch.randint(0, beam_config.vocab_size, (2, 16), generator=torch.Generator().manual_seed(12))
        with torch.no_grad():
            model(ids)
        for g in gates:
            assert g._last_expert_load is not None
            assert int(g._last_expert_load.sum()) == 2 * 16 * beam_config.num_experts_per_tok

    def test_fake_balanced_gate_is_left_alone(self, beam_config, backend_config):
        from dataclasses import replace

        model = BeamForCausalLM(beam_config, backend=replace(backend_config, fake_balanced_gate=True))
        assert not isinstance(model.model.layers["1"].mlp.gate, BeamGate)
