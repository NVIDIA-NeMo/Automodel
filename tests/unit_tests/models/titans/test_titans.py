# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Unit + functional tests for the Titans model (linear *and* deep memory).

Two key correctness proofs:

* **Linear reduction check** -- with momentum disabled the linear (``mem_depth=1``)
  Titans recurrence is numerically identical to Gated DeltaNet (fla's
  ``chunk_gated_delta_rule``).
* **Deep parity check** -- the chunked deep (``mem_depth>=2``) test-time-GD path at
  ``chunk_size=1`` reproduces an independent slow per-token reference recurrence
  (gradients taken with ``torch.autograd``, so it shares no code with the kernel's
  analytic backward) to fp64 tolerance; ``chunk_size>1`` re-anchors the gradient.

This file can be run directly as a script (``python test_titans.py``) on a CUDA
box to print all validation numbers, or collected by pytest (CUDA-only tests skip
on CPU).
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from nemo_automodel.components.distributed.parallelizer import _apply_titans_activation_checkpointing
from nemo_automodel.components.models.titans.config import TitansConfig
from nemo_automodel.components.models.titans.layers import (
    NeuralMemory,
    titans_delta_rule_recurrence,
)
from nemo_automodel.components.models.titans.model import TitansForCausalLM
from nemo_automodel.shared.import_utils import safe_import

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="fla GDN kernel requires CUDA")
HAVE_ACCELERATED_SCAN, accelerated_scan = safe_import(
    "nemo_automodel.components.models.titans.accelerated_scan"
)
accelerated_scan_only = pytest.mark.skipif(
    not (CUDA and HAVE_ACCELERATED_SCAN),
    reason="Titans accelerated scan requires CUDA and Triton",
)


def _tiny_config(**overrides):
    cfg = dict(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=32,
        intermediate_size=256,
        max_position_embeddings=128,
        torch_dtype="float32",
    )
    cfg.update(overrides)
    c = TitansConfig(**cfg)
    c.architectures = ["TitansForCausalLM"]
    return c


def test_titans_activation_checkpointing_wraps_neural_memory():
    model = TitansForCausalLM(_tiny_config(mem_depth=2))

    _apply_titans_activation_checkpointing(model)

    assert all(isinstance(layer.memory, CheckpointWrapper) for layer in model.model.layers)


def test_persistent_memory_is_prepended_without_changing_logit_length():
    config = _tiny_config(
        mem_depth=2,
        chunk_size=4,
        memory_batch_size=12,
        num_persistent_memory_tokens=4,
    )
    model = TitansForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    output = model(input_ids).logits
    output.sum().backward()

    assert output.shape == (2, 8, config.vocab_size)
    assert model.model.persistent_memory.shape == (4, config.hidden_size)
    assert model.model.persistent_memory.grad is not None


def test_persistent_memory_token_count_must_be_nonnegative():
    with pytest.raises(ValueError, match="num_persistent_memory_tokens must be non-negative"):
        _tiny_config(num_persistent_memory_tokens=-1)


# --------------------------------------------------------------------------- #
# (c) Reduction check: Titans (eta=0) == Gated DeltaNet (fla)
# --------------------------------------------------------------------------- #
@cuda_only
def test_reduction_recurrence_matches_fla_gdn():
    """titans_delta_rule_recurrence(eta=0) == fla.chunk_gated_delta_rule."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(0)
    B, T, H, K, V = 2, 48, 3, 16, 16
    dev = "cuda"
    q = torch.randn(B, T, H, K, device=dev)
    k = torch.randn(B, T, H, K, device=dev)
    v = torch.randn(B, T, H, V, device=dev)
    beta = torch.rand(B, T, H, device=dev)
    g = -F.softplus(torch.randn(B, T, H, device=dev))  # g <= 0
    qn = F.normalize(q.float(), dim=-1)
    kn = F.normalize(k.float(), dim=-1)

    o_gdn, _ = chunk_gated_delta_rule(qn, kn, v.float(), g.float(), beta.float(), use_qk_l2norm_in_kernel=False)
    eta0 = torch.zeros(B, T, H, device=dev)
    o_titans = titans_delta_rule_recurrence(qn, kn, v.float(), g.float(), beta.float(), eta0)

    max_diff = (o_titans - o_gdn).abs().max().item()
    assert max_diff < 1e-2, f"reduction diff too large: {max_diff}"

    # And momentum must actually change the output.
    eta1 = torch.full((B, T, H), 0.5, device=dev)
    o_mom = titans_delta_rule_recurrence(qn, kn, v.float(), g.float(), beta.float(), eta1)
    assert (o_mom - o_gdn).abs().max().item() > 1e-2
    return max_diff


@cuda_only
def test_reduction_module_level():
    """NeuralMemory momentum path with eta->0 matches the momentum-off (fla GDN) path."""
    torch.manual_seed(0)
    dim, mem_dim, heads = 128, 32, 4
    mem = NeuralMemory(dim, mem_dim=mem_dim, num_heads=heads, momentum=True, dtype=torch.float32).cuda()
    mem.init_weights()
    x = torch.randn(2, 40, dim, device="cuda")

    # momentum-off path = fla Gated DeltaNet
    mem.momentum = False
    out_gdn = mem(x)

    # momentum-on path, but force eta = sigmoid(-30) ~= 0  -> must match GDN
    class _ConstNeg(torch.nn.Module):
        def __init__(self, h):
            super().__init__()
            self.h = h

        def forward(self, t):
            return torch.full((*t.shape[:-1], self.h), -30.0, device=t.device, dtype=t.dtype)

    mem.momentum = True
    orig = mem.m_proj
    mem.m_proj = _ConstNeg(heads)
    out_reduced = mem(x)
    mem.m_proj = orig

    max_diff = (out_reduced - out_gdn).abs().max().item()
    assert max_diff < 1e-2, f"module reduction diff too large: {max_diff}"

    # real momentum changes the output
    out_mom = mem(x)
    assert (out_mom - out_gdn).abs().max().item() > 1e-3
    return max_diff


# --------------------------------------------------------------------------- #
# (a) Build + forward shape
# --------------------------------------------------------------------------- #
@cuda_only
def test_autoconfig_build_and_forward():
    from transformers import AutoModelForCausalLM

    import nemo_automodel.components.models.titans  # noqa: F401  (HF registration)

    cfg = _tiny_config()
    model = AutoModelForCausalLM.from_config(cfg).cuda()
    assert model.lm_head.weight is model.model.embed_tokens.weight  # tied
    assert model.model.layers[0].memory.A_log.dtype == torch.float32
    x = torch.randint(0, cfg.vocab_size, (2, 32), device="cuda")
    out = model(input_ids=x, labels=x)
    assert out.logits.shape == (2, 32, cfg.vocab_size)
    assert torch.isfinite(out.loss)


# --------------------------------------------------------------------------- #
# (b) Training: loss decreases
# --------------------------------------------------------------------------- #
@cuda_only
def test_training_loss_decreases():
    from transformers import AutoModelForCausalLM

    import nemo_automodel.components.models.titans  # noqa: F401

    torch.manual_seed(0)
    cfg = _tiny_config(num_hidden_layers=2)
    model = AutoModelForCausalLM.from_config(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # Memorizable task: a fixed repeated sequence.
    seq = torch.randint(0, cfg.vocab_size, (4, 64), device="cuda")
    losses = []
    for _ in range(30):
        opt.zero_grad()
        out = model(input_ids=seq, labels=seq)
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss))
    assert losses[-1] < losses[0] - 0.5, f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
    return losses


# --------------------------------------------------------------------------- #
# State-dict adapter round-trip
# --------------------------------------------------------------------------- #
def test_state_dict_adapter_roundtrip():
    from nemo_automodel.components.models.titans.state_dict_adapter import TitansStateDictAdapter

    cfg = _tiny_config()
    adapter = TitansStateDictAdapter(cfg)
    sd = {
        "model.embed_tokens.weight": torch.randn(256, 128),
        "model.layers.0.memory._fp32_params.A_log": torch.zeros(4, dtype=torch.bfloat16),
        "model.layers.0.memory._fp32_params.dt_bias": torch.ones(4, dtype=torch.bfloat16),
        "model.layers.0.memory.q_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
    }
    hf = adapter.to_hf(sd)
    assert "model.layers.0.memory.A_log" in hf
    assert "model.layers.0.memory.dt_bias" in hf
    back = adapter.from_hf(hf)
    assert set(back.keys()) == set(sd.keys())
    # decay-gate params upcast to fp32, others preserved
    assert back["model.layers.0.memory._fp32_params.A_log"].dtype == torch.float32
    assert back["model.layers.0.memory._fp32_params.dt_bias"].dtype == torch.float32
    assert back["model.layers.0.memory.q_proj.weight"].dtype == torch.bfloat16
    forced = adapter.forced_hf_dtype_mapping(sd)
    assert forced == {
        "model.layers.0.memory._fp32_params.A_log": "F32",
        "model.layers.0.memory._fp32_params.dt_bias": "F32",
    }


# --------------------------------------------------------------------------- #
# Deep memory (mem_depth>=2): build + train
# --------------------------------------------------------------------------- #
@cuda_only
def test_deep_build_and_train():
    """AutoModelForCausalLM.from_config(mem_depth=2) builds, forwards, and trains."""
    from transformers import AutoModelForCausalLM

    import nemo_automodel.components.models.titans  # noqa: F401

    torch.manual_seed(0)
    cfg = _tiny_config(mem_depth=2, chunk_size=8)
    model = AutoModelForCausalLM.from_config(cfg).cuda()
    assert model.lm_head.weight is model.model.embed_tokens.weight  # tied
    # deep memory builds per-head MLP weight params (the inner-loop initial state)
    assert len(model.model.layers[0].memory.mem_weights) == cfg.mem_depth

    x = torch.randint(0, cfg.vocab_size, (2, 32), device="cuda")
    out = model(input_ids=x, labels=x)
    assert out.logits.shape == (2, 32, cfg.vocab_size)
    assert torch.isfinite(out.loss)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    seq = torch.randint(0, cfg.vocab_size, (4, 64), device="cuda")
    losses = []
    for _ in range(30):
        opt.zero_grad()
        o = model(input_ids=seq, labels=seq)
        o.loss.backward()
        opt.step()
        losses.append(float(o.loss))
    # the memory MLP initial weights must actually receive a gradient
    assert model.model.layers[0].memory.mem_weights[0].grad is not None
    assert losses[-1] < losses[0] - 0.5, f"deep loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
    return losses


def test_deep_memory_uses_configured_paper_expansion():
    """The public config builds the paper's expanded two-layer memory MLP."""
    from transformers import AutoModelForCausalLM

    import nemo_automodel.components.models.titans  # noqa: F401

    cfg = _tiny_config(mem_depth=2, memory_expansion_factor=4)
    model = AutoModelForCausalLM.from_config(cfg)
    state = model.state_dict()

    assert cfg.memory_expansion_factor == 4
    assert state["model.layers.0.memory.mem_weights.0"].shape == (4, 32, 128)
    assert state["model.layers.0.memory.mem_weights.1"].shape == (4, 128, 32)


def test_deep_memory_exposes_residual_norm_fast_weight():
    """The deep memory checkpoint includes the paper's residual-norm weight."""
    from transformers import AutoModelForCausalLM

    import nemo_automodel.components.models.titans  # noqa: F401

    cfg = _tiny_config(mem_depth=2, memory_residual_norm=True)
    state = AutoModelForCausalLM.from_config(cfg).state_dict()

    assert cfg.memory_residual_norm is True
    assert state["model.layers.0.memory.mem_norm_weight"].shape == (4, 32)


def test_neural_memory_builds_causal_qkv_convolutions():
    """The paper's width-four Q/K/V convolutions are part of the checkpoint."""
    from transformers import AutoModelForCausalLM

    import nemo_automodel.components.models.titans  # noqa: F401

    cfg = _tiny_config(mem_depth=2, qkv_conv_kernel_size=4)
    model = AutoModelForCausalLM.from_config(cfg)
    state = model.state_dict()

    assert cfg.qkv_conv_kernel_size == 4
    for projection in ("q", "k", "v"):
        assert state[f"model.layers.0.memory.{projection}_conv.weight"].shape == (128, 1, 4)


def test_deep_memory_state_carry_matches_concatenated_forward():
    """Chunk-aligned streaming preserves convolution and memory state."""
    torch.manual_seed(7)
    memory = NeuralMemory(
        dim=32,
        mem_dim=8,
        num_heads=4,
        mem_depth=2,
        expansion_factor=2,
        chunk_size=4,
        qkv_conv_kernel_size=4,
        dtype=torch.float64,
    ).double()
    x = torch.randn(2, 8, 32, dtype=torch.float64)

    with torch.no_grad():
        expected = memory(x)
        first, state = memory(x[:, :4], return_state=True)
        second, _ = memory(x[:, 4:], past_state=state, return_state=True)

    torch.testing.assert_close(torch.cat((first, second), dim=1), expected, rtol=1e-10, atol=1e-10)


def test_170m_lmm_recipe_matches_paper_scale():
    """The first full reproduction recipe pins the paper's 170M setup."""
    recipe_path = Path(__file__).parents[4] / "examples/llm_pretrain/titans_170m_lmm.yaml"
    recipe = yaml.safe_load(recipe_path.read_text())
    model = recipe["model"]["config"]

    assert recipe["step_scheduler"]["global_batch_size"] * recipe["dataset"]["seq_len"] == 524_288
    assert recipe["step_scheduler"]["max_steps"] == 28_610
    assert model["vocab_size"] == 32_000
    assert model["num_hidden_layers"] == 12
    assert model["hidden_size"] == 768
    assert model["num_attention_heads"] == 16
    assert model["mem_depth"] == 2
    assert model["memory_expansion_factor"] == 4
    assert model["qkv_conv_kernel_size"] == 4
    assert model["chunk_size"] == 16
    assert model["deep_memory_backend"] == "titans_pytorch"
    assert model["memory_batch_size"] == 4224
    assert model["num_persistent_memory_tokens"] == 128
    assert model["torch_dtype"] == "float32"
    assert recipe["dataset"]["seq_len"] == 4096

    config_values = {key: value for key, value in model.items() if key not in {"_target_", "architectures"}}
    with torch.device("meta"):
        instantiated = TitansForCausalLM(TitansConfig(**config_values))
    assert sum(parameter.numel() for parameter in instantiated.parameters()) == 173_695_680


def test_multinode_ablation_runner_preserves_paper_batch_and_uses_c10d():
    root = Path(__file__).parents[4]
    runner = (root / "examples/llm_pretrain/slurm/cwdfw_titans_170m_ablation.sbatch").read_text()
    submitter = (root / "tools/submit_titans_ablation_cwdfw.sh").read_text()

    assert "--standalone" not in runner
    assert "--rdzv-backend=c10d" in runner
    assert "--nnodes=\\$SLURM_NNODES" in runner
    assert "WORLD_SIZE=$((NODES * 8))" in runner
    assert "128 % WORLD_SIZE" in runner
    assert "--gpus 8" in submitter
    assert "checkouts/$LOCAL_SHA" in submitter
    for variant in ("no_persistent", "no_convolution", "no_momentum", "no_weight_decay", "depth3", "depth4"):
        assert f"{variant})" in runner


# --------------------------------------------------------------------------- #
# Deep memory parity: chunked test-time GD vs slow per-token reference
# --------------------------------------------------------------------------- #
def _deep_mlp(weights, x):
    """Differentiable single-token memory MLP. ``x``: [N, mem_dim]; weights: [N, in, out]."""
    residual = x
    norm_weight = None
    if weights[-1].ndim == 2:
        *weights, norm_weight = weights

    h = x
    for i, w in enumerate(weights):
        if i > 0:
            h = F.gelu(h)
        h = torch.einsum("nm,nmo->no", h, w)
    if norm_weight is not None:
        h = residual + F.rms_norm(h, (h.shape[-1],), eps=1e-6) * norm_weight
    return h


def _deep_per_token_reference(mem, x, chunk_size):
    """Slow per-token reference for the deep recurrence (autograd surprise).

    Mirrors ``deep-memory/test_correctness.py``: retrieve every token in a chunk
    with the chunk-anchor weights, then apply the per-token momentum + forget
    update with the gradient evaluated at the (fixed) anchor. ``chunk_size=1`` is
    exact per-token GD (anchor == running weights). Returns folded ``[B*H, S, D]``.
    """
    B, S, _ = x.shape
    H, D = mem.num_heads, mem.mem_dim
    assert S % chunk_size == 0
    # Same projections the deep forward uses (no L2-norm on the deep path).
    q = mem.q_proj(x).view(B, S, H, D)
    k = mem.k_proj(x).view(B, S, H, D)
    v = mem.v_proj(x).view(B, S, H, D)
    beta = mem.b_proj(x).sigmoid()
    g = mem._decay_gate(mem.a_proj(x))
    eta = mem.m_proj(x).sigmoid() if mem.momentum else None
    dtype = q.dtype

    def fold(t):
        tt = t.transpose(1, 2)  # [B,H,S,...]
        return tt.reshape(B * H, *tt.shape[2:]).to(dtype)

    qf = F.normalize(fold(q), dim=-1)
    kf = F.normalize(fold(k), dim=-1)
    vf = fold(v)
    thetaf = fold(beta)
    keepf = fold(g).exp()
    etaf = fold(eta) if eta is not None else torch.zeros_like(thetaf)

    weights = mem._deep_init_weights(B, dtype)
    momentum = [torch.zeros_like(w) for w in weights]
    retrieved = [None] * S
    for ci in range(S // chunk_size):
        anchor = [w.detach().clone() for w in weights]
        for t in range(chunk_size):
            idx = ci * chunk_size + t
            retrieved[idx] = _deep_mlp(anchor, qf[:, idx])
        for t in range(chunk_size):
            idx = ci * chunk_size + t
            aw = [w.detach().clone().requires_grad_(True) for w in anchor]
            pred = _deep_mlp(aw, kf[:, idx])
            loss = (pred - vf[:, idx]).pow(2).mean(dim=-1).sum()
            grads = torch.autograd.grad(loss, aw)
            surprise = [-thetaf[:, idx].reshape(gr.shape[0], *((1,) * (gr.ndim - 1))) * gr for gr in grads]
            if mem.momentum:
                momentum = [
                    etaf[:, idx].reshape(s.shape[0], *((1,) * (s.ndim - 1))) * s0 + s
                    for s0, s in zip(momentum, surprise)
                ]
            else:
                momentum = surprise
            weights = [
                keepf[:, idx].reshape(w.shape[0], *((1,) * (w.ndim - 1))) * w + s for w, s in zip(weights, momentum)
            ]
    return torch.stack(retrieved, dim=1)  # [N, S, D]


def _deep_public_chunk_reference(mem, q, k, v, beta, g, eta, memory_batch_size):
    """Sequential reference for titans-pytorch's chunk-aggregated update semantics."""
    B, S, H, D = q.shape
    c = mem.chunk_size
    assert S % c == 0
    assert memory_batch_size % c == 0

    def fold(t):
        tt = t.transpose(1, 2)
        return tt.reshape(B * H, *tt.shape[2:]).double()

    qf = F.normalize(fold(q), dim=-1)
    kf = F.normalize(fold(k), dim=-1)
    vf = fold(v)
    thetaf = fold(beta)
    keepf = fold(g).exp()
    etaf = fold(eta) if eta is not None else torch.zeros_like(thetaf)
    weights = mem._deep_init_weights(B, torch.float64)
    momentum = [torch.zeros_like(w) for w in weights]
    retrieved = []

    for segment_start in range(0, S, memory_batch_size):
        segment_end = min(segment_start + memory_batch_size, S)
        anchor = [w.clone() for w in weights]
        surprises = [[] for _ in anchor]
        chunk_etas = []
        chunk_keeps = []
        updates = [[weight] for weight in weights]
        for start in range(segment_start, segment_end, c):
            stop = start + c
            grads = mem._mem_grads(kf[:, start:stop], vf[:, start:stop], anchor)
            for layer_surprises, grad in zip(surprises, grads):
                shape = (grad.shape[0], c, *((1,) * (grad.ndim - 2)))
                layer_surprises.append((-thetaf[:, start:stop].reshape(shape) * grad).sum(dim=1))
            chunk_etas.append(etaf[:, start:stop].mean(dim=1))
            chunk_keeps.append(keepf[:, start:stop].mean(dim=1))

        for chunk_index in range(len(chunk_etas)):
            for layer_index, surprise_by_chunk in enumerate(surprises):
                surprise = surprise_by_chunk[chunk_index]
                gate_shape = (surprise.shape[0], *((1,) * (surprise.ndim - 1)))
                if mem.momentum:
                    momentum[layer_index] = (
                        chunk_etas[chunk_index].reshape(gate_shape) * momentum[layer_index] + surprise
                    )
                else:
                    momentum[layer_index] = surprise
                weights[layer_index] = (
                    chunk_keeps[chunk_index].reshape(gate_shape) * weights[layer_index] + momentum[layer_index]
                )
                updates[layer_index].append(weights[layer_index])

        for offset in range(segment_end - segment_start):
            update_index = (offset + 1) // c
            token_weights = [layer_updates[update_index] for layer_updates in updates]
            start = segment_start + offset
            retrieved.append(mem._mem_forward(qf[:, start : start + 1], token_weights))

    return torch.cat(retrieved, dim=1), (weights, momentum)


@cuda_only
def test_deep_public_backend_matches_sequential_chunk_reference():
    torch.manual_seed(2026)
    B, S, H, D = 2, 32, 2, 8
    mem = (
        NeuralMemory(
            dim=H * D,
            mem_dim=D,
            num_heads=H,
            mem_depth=2,
            chunk_size=4,
            momentum=True,
            forget=True,
            deep_memory_backend="titans_pytorch",
            memory_batch_size=16,
        )
        .cuda()
        .double()
    )
    x = torch.randn(B, S, H * D, device="cuda", dtype=torch.float64)
    q = mem.q_proj(x).view(B, S, H, D)
    k = mem.k_proj(x).view(B, S, H, D)
    v = mem.v_proj(x).view(B, S, H, D)
    beta = mem.b_proj(x).sigmoid()
    g = mem._decay_gate(mem.a_proj(x))
    eta = mem.m_proj(x).sigmoid()

    with torch.no_grad():
        actual, actual_state = mem._deep_recurrence(q, k, v, beta, g, eta, return_state=True)
        expected, expected_state = _deep_public_chunk_reference(mem, q, k, v, beta, g, eta, 16)

    actual = actual.transpose(1, 2).reshape(B * H, S, D)
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-8)
    for actual_group, expected_group in zip(actual_state, expected_state):
        for actual_tensor, expected_tensor in zip(actual_group, expected_group):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-8, atol=1e-8)


@cuda_only
def test_deep_public_backend_has_finite_backward():
    torch.manual_seed(2027)
    memory = (
        NeuralMemory(
            dim=32,
            mem_dim=8,
            num_heads=4,
            mem_depth=2,
            chunk_size=4,
            deep_memory_backend="titans_pytorch",
            memory_batch_size=16,
            dtype=torch.float32,
        )
        .cuda()
        .float()
    )
    x = torch.randn(2, 32, 32, device="cuda", requires_grad=True)

    memory(x).square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(weight.grad is not None and torch.isfinite(weight.grad).all() for weight in memory.mem_weights)


@accelerated_scan_only
def test_accelerated_scan_256_step_forward_and_backward():
    torch.manual_seed(2028)
    gates = torch.sigmoid(torch.randn(2, 17, 256, device="cuda", requires_grad=True))
    inputs = torch.randn(2, 17, 256, device="cuda", requires_grad=True)
    output_weights = torch.randn_like(inputs)

    actual = accelerated_scan.scan(gates.contiguous(), inputs.contiguous())
    states = []
    state = torch.zeros_like(inputs[..., 0])
    for index in range(inputs.shape[-1]):
        state = gates[..., index] * state + inputs[..., index]
        states.append(state)
    expected = torch.stack(states, dim=-1)
    actual_grads = torch.autograd.grad((actual * output_weights).sum(), (gates, inputs), retain_graph=True)
    expected_grads = torch.autograd.grad((expected * output_weights).sum(), (gates, inputs))

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-4, atol=2e-4)


def test_deep_public_backend_state_matches_aligned_calls():
    torch.manual_seed(2029)
    memory = NeuralMemory(
        dim=32,
        mem_dim=8,
        num_heads=4,
        mem_depth=2,
        chunk_size=4,
        deep_memory_backend="titans_pytorch",
        memory_batch_size=8,
        qkv_conv_kernel_size=4,
        dtype=torch.float64,
    ).double()
    x = torch.randn(2, 16, 32, dtype=torch.float64)

    with torch.no_grad():
        expected = memory(x)
        first, state = memory(x[:, :8], return_state=True)
        second, _ = memory(x[:, 8:], past_state=state, return_state=True)

    torch.testing.assert_close(torch.cat((first, second), dim=1), expected, rtol=1e-10, atol=1e-10)


@cuda_only
def test_deep_chunked_matches_per_token_reference():
    """Deep chunked recurrence == slow autograd per-token reference (fp64)."""
    dim, mem_dim, heads, B, S = 48, 16, 3, 2, 64
    settings = [
        ("momentum+forget", True, True),
        ("forget-only", False, True),
        ("momentum-only", True, False),
        ("plain", False, False),
    ]
    worst = 0.0
    results = {}
    for depth in (2, 3):
        for name, mom, forg in settings:
            torch.manual_seed(depth * 100 + len(name))
            mem = (
                NeuralMemory(dim, mem_dim=mem_dim, num_heads=heads, mem_depth=depth, momentum=mom, forget=forg)
                .cuda()
                .double()
            )
            x = torch.randn(B, S, dim, device="cuda", dtype=torch.float64)
            # Re-project once and feed identical q/k/v to the kernel.
            q = mem.q_proj(x).view(B, S, heads, mem_dim)
            k = mem.k_proj(x).view(B, S, heads, mem_dim)
            v = mem.v_proj(x).view(B, S, heads, mem_dim)
            beta = mem.b_proj(x).sigmoid()
            gg = mem._decay_gate(mem.a_proj(x))
            eta = mem.m_proj(x).sigmoid() if mem.momentum else None

            for c in (1, 2, 4, 8, 16):
                mem.chunk_size = c
                with torch.no_grad():
                    core = mem._deep_recurrence(q, k, v, beta, gg, eta)  # [B,S,H,D]
                kernel = core.transpose(1, 2).reshape(B * heads, S, mem_dim)
                ref = _deep_per_token_reference(mem, x, c)
                err = (kernel - ref).abs().max().item()
                worst = max(worst, err)
                scale = max(1.0, ref.abs().max().item())
                assert err < 1e-8 * scale, (
                    f"deep parity FAIL depth={depth} {name} chunk={c}: max|diff|={err:.3e}, scale={scale:.3e}"
                )
                if c == 1:
                    results[(depth, name)] = err

            # chunk_size>1 must actually re-anchor (differ from exact per-token GD)
            mem.chunk_size = 8
            with torch.no_grad():
                c8 = mem._deep_recurrence(q, k, v, beta, gg, eta).transpose(1, 2).reshape(B * heads, S, mem_dim)
            mem.chunk_size = 1
            with torch.no_grad():
                c1 = mem._deep_recurrence(q, k, v, beta, gg, eta).transpose(1, 2).reshape(B * heads, S, mem_dim)
            assert (c8 - c1).abs().max().item() > 1e-6, f"chunk size had no effect for {name}"
    return worst, results


if __name__ == "__main__":
    print("=" * 70)
    print("Titans memory validation (linear + deep)")
    print("=" * 70)
    if not CUDA:
        print("CUDA not available; only CPU adapter test will run.")
        test_state_dict_adapter_roundtrip()
        print("[adapter] round-trip OK")
        raise SystemExit(0)

    d = test_reduction_recurrence_matches_fla_gdn()
    print(f"(c) reduction [recurrence]: titans(eta=0) vs fla GDN max|diff| = {d:.2e}  (PASS)")
    d2 = test_reduction_module_level()
    print(f"(c) reduction [module]:     momentum-off vs eta->0  max|diff| = {d2:.2e}  (PASS)")
    test_autoconfig_build_and_forward()
    print("(a) AutoModelForCausalLM.from_config build + forward: PASS")
    losses = test_training_loss_decreases()
    print(f"(b) training loss: {losses[0]:.3f} -> {losses[-1]:.3f} over {len(losses)} steps (PASS)")
    test_state_dict_adapter_roundtrip()
    print("    state-dict adapter round-trip + fp32 contract: PASS")

    print("-" * 70)
    print("Deep memory (mem_depth>=2)")
    dl = test_deep_build_and_train()
    print(f"(d) deep build + train loss: {dl[0]:.3f} -> {dl[-1]:.3f} over {len(dl)} steps (PASS)")
    worst, results = test_deep_chunked_matches_per_token_reference()
    for (depth, name), err in results.items():
        print(f"(e) deep parity depth={depth:<2} {name:<16} chunk=1 vs per-token GD: max|diff| = {err:.2e}  (PASS)")
    print(f"(e) deep parity worst max|diff| across all configs/chunks = {worst:.2e}  (PASS)")
    print("=" * 70)
    print("ALL CHECKS PASSED")
