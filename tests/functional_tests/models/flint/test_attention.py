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

"""Real TE and Magi learned-sink attention gates: CUDA/FSDP/CP gradients, FP16 boundary and packed isolation."""

from __future__ import annotations

import datetime
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import socket
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed import DistributedSetup, FSDP2Config, ParallelismSizes
from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.flint.config import FlintConfig
from nemo_automodel.components.models.flint.layers import FlintAttention
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from nemo_automodel.shared.import_utils import safe_import
from tests.functional_tests.models.flint.reference import full_native_tensors, rms, rope


def te_config(*, depth: int = 3, dtype: torch.dtype = torch.bfloat16) -> FlintConfig:
    """Build asymmetric tiny widths with the released model's TE head dimension."""
    kinds = ["full_attention", "sliding_attention", "full_attention"][:depth]
    return FlintConfig(
        vocab_size=31,
        hidden_size=32,
        intermediate_size=48,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=24,
        num_hidden_layers=depth,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[0],
        layer_types=kinds,
        no_rope_layer_indices=[i for i, k in enumerate(kinds) if k == "full_attention"],
        sliding_window=513,
        max_position_embeddings=2048,
        router_aux_loss_coef=0.0,
        dtype=dtype,
        residual_dtype="fp32",
        pad_token_id=2,
        eos_token_id=[2, 3, 4],
    )


def attention_backend(name: str) -> BackendConfig:
    """Select actual attention with shared Torch projection/norm/expert operators."""
    return BackendConfig(
        attn=name,
        linear="torch",
        rms_norm="torch_fp32",
        rope="torch",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        gate_precision=None,
        enable_fsdp_optimizations=False,
    )


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Describe same-shape floating CUDA values without changing comparison math.

    Args:
        actual: Candidate tensor, in native parameter/activation layout.
        expected: Same-layout reference tensor; both are promoted to FP64 here.

    Returns:
        Maximum absolute error, relative L2 error and flattened cosine similarity.
        Two zero vectors have cosine one; exactly one zero vector has cosine zero.
        No absolute epsilon changes directions for deliberately tiny gradients.
    """
    a, b = actual.detach().double().flatten(), expected.detach().double().flatten()
    norm_a, norm_b = a.norm(), b.norm()
    if norm_a == 0 or norm_b == 0:
        cosine = float(norm_a == norm_b)
    else:
        cosine = float(((a @ b) / (norm_a * norm_b)).clamp(-1, 1))
    return {
        "max_abs": float((a - b).abs().max()),
        "relative_l2": float((a - b).norm() / b.norm().clamp_min(1e-20)),
        "cosine": cosine,
    }


def _provenance() -> dict:
    """Capture native source bytes and actual CUDA dependency versions."""
    root = Path(__file__).resolve().parents[4]
    files = ["config.py", "model.py", "layers.py", "parallelization.py", "packing.py"]
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformer_engine": importlib.metadata.version("transformer_engine"),
        "device": torch.cuda.get_device_name(),
        "source_sha256": {
            name: hashlib.sha256((root / "nemo_automodel/components/models/flint" / name).read_bytes()).hexdigest()
            for name in files
        },
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _record(name: str, values: dict) -> None:
    directory = Path(os.environ.get("FLINT_BACKEND_EVIDENCE", "/tmp/flint-backends"))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps({"provenance": _provenance(), **values}, indent=2) + "\n")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("default_dtype", [torch.float32, torch.bfloat16])
def test_te_sink_stays_fp32_under_public_loader_default_dtype(default_dtype: torch.dtype) -> None:
    """A BF16 loader default cannot quantize the protected trainable TE offset."""
    torch.manual_seed(437526)
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        with torch.device("cuda"):
            attention = FlintAttention(te_config(), 0, attention_backend("te"))
    finally:
        torch.set_default_dtype(original)
    attention.init_weights(torch.device("cuda"))
    assert attention.q_proj.weight.dtype == torch.bfloat16
    assert attention._fp32_params.softmax_offset.dtype == torch.float32
    assert attention._fp32_params.softmax_offset.requires_grad
    x = torch.randn(1, 7, attention.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    calls = []
    handle = attention._fp32_params.fused_attention.register_forward_hook(
        lambda module, args, output: calls.append(True)
    )
    output = attention(x)
    (output.float() * torch.randn_like(output, dtype=torch.float32)).sum().backward()
    handle.remove()
    gradient = attention._fp32_params.softmax_offset.grad
    assert calls == [True]
    assert gradient is not None and gradient.dtype == torch.float32
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) == attention.num_heads
    _record(
        f"te_sink_loader_default_{default_dtype}",
        {
            "sink_dtype": str(gradient.dtype),
            "sink_gradient_nonzero": int(torch.count_nonzero(gradient)),
            "default_dtype": str(default_dtype),
        },
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("layer_index", [0, 1])
@pytest.mark.parametrize("padded", [False, True])
def test_te_actual_learned_sink_forward_and_every_gradient(layer_index: int, padded: bool) -> None:
    """Compare BF16 [2,528,32] CUDA attention, including source-width SWA513."""
    torch.manual_seed(4375 + layer_index)
    config = te_config()
    te = FlintAttention(config, layer_index, attention_backend("te")).cuda()
    te.init_weights(torch.device("cuda"))
    sdpa = FlintAttention(config, layer_index, attention_backend("sdpa")).cuda()
    state = {name: value for name, value in te.state_dict().items() if not name.endswith("_extra_state")}
    sdpa.load_state_dict(state, strict=True)
    with torch.no_grad():
        values = torch.tensor([-0.7, 0.3, 0.8, -0.1], device="cuda")
        te._fp32_params.softmax_offset.copy_(values)
        sdpa._fp32_params.softmax_offset.copy_(values)
    valid = torch.ones(2, 528, dtype=torch.bool, device="cuda")
    if padded:
        valid[0, 515:] = False
        valid[1, 521:] = False
    original = torch.randn(2, 528, 32, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn_like(original) / original.numel() ** 0.5
    x_te, x_sdpa = original.clone().requires_grad_(), original.clone().requires_grad_()
    actual, expected = te(x_te, attention_mask=valid), sdpa(x_sdpa, attention_mask=valid)
    # Padded query values have no target meaning in TE; exclude them explicitly.
    upstream = upstream * valid.unsqueeze(-1)
    (actual * upstream).sum().backward()
    (expected * upstream).sum().backward()
    metrics = {"output": _errors(actual[valid], expected[valid]), "gradients": {}}
    torch.testing.assert_close(actual[valid], expected[valid], atol=2e-3, rtol=0.12)
    for name, parameter in te.named_parameters():
        reference = dict(sdpa.named_parameters())[name]
        assert parameter.grad is not None and reference.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        metrics["gradients"][name] = _errors(parameter.grad, reference.grad)
        torch.testing.assert_close(parameter.grad, reference.grad, atol=2e-3, rtol=0.15, msg=name)
        assert metrics["gradients"][name]["relative_l2"] < 0.08, (name, metrics["gradients"][name])
    metrics["input_gradient"] = _errors(x_te.grad, x_sdpa.grad)
    torch.testing.assert_close(x_te.grad, x_sdpa.grad, atol=2e-3, rtol=0.15)
    assert te._fp32_params.softmax_offset.dtype == torch.float32
    assert te._fp32_params.softmax_offset.grad.dtype == torch.float32
    assert torch.count_nonzero(te._fp32_params.softmax_offset.grad) == config.num_attention_heads
    _record(f"te_attention_{layer_index}_padding_{padded}", metrics)


def _parallel_worker(rank: int, port: int, cp: bool, evidence: str) -> None:
    """Compare a real two-rank native TE update against unsharded TE.

    Args:
        rank: Local CUDA/NCCL rank in a two-process job.
        port: Free localhost rendezvous port.
        cp: Select CP2 ownership rather than DP2 replicated sequences.
        evidence: Rank-zero JSON output path; all ranks enter every collective.
    """
    from datetime import timedelta

    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from nemo_automodel import NeMoAutoModelForCausalLM
    from nemo_automodel.components.distributed import DistributedSetup, FSDP2Config, ParallelismSizes
    from tests.functional_tests.models.flint.reference import full_native_tensors

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank)
    )
    torch.cuda.set_device(rank)
    torch.manual_seed(74375)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, timeout=timedelta(minutes=8))
    provenance = _provenance()
    try:
        config = te_config(dtype=torch.float32)
        setup = DistributedSetup.build(
            strategy=FSDP2Config(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    output_dtype=torch.bfloat16,
                    cast_forward_inputs=False,
                ),
                enable_fsdp2_prefetch=False,
            ),
            parallelism_sizes=ParallelismSizes(dp_size=1 if cp else 2, cp_size=2 if cp else 1),
            world_size=2,
        )
        model = NeMoAutoModelForCausalLM.from_config(
            config,
            dtype=torch.float32,
            backend=attention_backend("te"),
            attn_implementation="eager",
            use_liger_kernel=False,
            use_sdpa_patching=False,
            distributed_setup=setup,
        ).train()
        original = full_native_tensors(model)
        reference = te_config().build(backend=attention_backend("te")).to(device).train()
        reference.load_state_dict(original, strict=False)
        global_tokens = 1040 if cp else 32
        tokens = (torch.arange(global_tokens, device=device)[None] * 7 + 2) % config.vocab_size
        targets = (tokens + 1) % config.vocab_size
        if cp:
            tokens[:, 525:528] = tokens[:, 1037:1040] = 2
            targets[:, 525:528] = targets[:, 1037:1040] = -100
        global_supervised = int((targets != -100).sum())
        batch = {"input_ids": tokens, "labels": targets}
        if cp:
            lengths = torch.tensor([[525, 509]], device=device, dtype=torch.int32)
            padded_lengths = torch.tensor([[528, 512]], device=device, dtype=torch.int32)
            source_loss_mask = torch.arange(global_tokens, device=device, dtype=torch.float32)[None] / global_tokens
            batch.update(
                qkv_format="thd",
                seq_lens=lengths,
                seq_lens_padded=padded_lengths,
                position_ids=torch.cat((torch.arange(528, device=device), torch.arange(512, device=device)))[None],
                loss_mask=source_loss_mask,
            )
            prepared = model.prepare_model_inputs_for_cp(batch)
            _, local, layout = prepared["cp_sharder"].shard_batch(model.cp_mesh, None, batch, padding_token_id=2)
            prepared_reference = reference.prepare_model_inputs_for_cp(batch)
            _, reference_batch, _ = prepared_reference["cp_sharder"].shard_batch(None, None, batch, padding_token_id=2)
            indices = layout.local_token_global_indices
            torch.testing.assert_close(
                local["loss_mask"], source_loss_mask.index_select(1, indices.long()), atol=0, rtol=0
            )
            torch.testing.assert_close(reference_batch["loss_mask"], source_loss_mask, atol=0, rtol=0)
        else:
            local, reference_batch, indices = dict(batch), dict(batch), None
        labels = local.pop("labels")
        reference_labels = reference_batch.pop("labels")
        for key in ("seq_lens", "seq_lens_padded", "loss_mask"):
            local.pop(key, None)
            reference_batch.pop(key, None)
        captured = []
        handles = [
            layer.register_forward_hook(lambda module, args, output: captured.append(output.dtype))
            for layer in model.model.layers
        ]
        actual = model(**local).logits
        expected = reference(**reference_batch).logits
        assert actual.dtype == expected.dtype == torch.float32
        if cp:
            expected_local = expected.index_select(1, indices.long())
        else:
            expected_local = expected
        valid_queries = labels != -100
        torch.testing.assert_close(actual[valid_queries], expected_local[valid_queries], atol=4e-3, rtol=0.1)
        actual_loss = F.cross_entropy(actual.flatten(0, 1), labels.flatten(), reduction="sum") / global_supervised
        expected_loss = F.cross_entropy(expected.flatten(0, 1), reference_labels.flatten(), reduction="mean")
        # FSDP averages its CP/DP mesh. CP ranks own disjoint target numerators;
        # DP ranks deliberately repeat one sample for this storage-lifetime gate.
        (actual_loss * (2 if cp else 1)).backward()
        expected_loss.backward()
        gradients = full_native_tensors(model, gradients=True)
        metrics = {
            "provenance": provenance,
            "cp_size": 2 if cp else 1,
            "dp_size": 1 if cp else 2,
            "output": _errors(actual[valid_queries], expected_local[valid_queries]),
            "gradients": {},
            "global_physical_tokens": global_tokens,
            "global_supervised_tokens": global_supervised,
        }
        if cp:
            metrics["transports"] = [layer.self_attn.attn_module.cp_comm_type for layer in model.model.layers]
            assert metrics["transports"] == ["a2a"] * config.num_hidden_layers
        references = dict(reference.named_parameters())
        metrics["status"] = "comparisons_pending"
        for name, gradient in gradients.items():
            expected_gradient = references[name].grad
            assert expected_gradient is not None
            metrics["gradients"][name] = _errors(gradient, expected_gradient)
            denominator = expected_gradient.double().square().sum().clamp_min(1e-30)
            metrics["gradients"][name]["least_squares_scale"] = float(
                (gradient.double() * expected_gradient.double()).sum() / denominator
            )
        if rank == 0:
            Path(evidence).write_text(json.dumps(metrics, indent=2) + "\n")
        for name, gradient in gradients.items():
            expected_gradient = references[name].grad
            assert torch.isfinite(gradient).all(), name
            torch.testing.assert_close(gradient, expected_gradient.float(), atol=3e-3, rtol=0.15, msg=name)
            assert metrics["gradients"][name]["relative_l2"] < 0.12, (name, metrics["gradients"][name])
            if name.endswith("softmax_offset"):
                assert gradient.dtype == torch.float32 and torch.count_nonzero(gradient) == config.num_attention_heads
        actual_norm = torch.stack([gradient.double().square().sum() for gradient in gradients.values()]).sum().sqrt()
        expected_norm = (
            torch.stack([parameter.grad.double().square().sum() for parameter in references.values()]).sum().sqrt()
        )
        torch.testing.assert_close(actual_norm, expected_norm, atol=1e-5, rtol=0.02)
        metrics["global_gradient_norm"] = float(actual_norm)
        metrics["reference_global_gradient_norm"] = float(expected_norm)
        assert captured and all(dtype == torch.float32 for dtype in captured)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.step()
        updated = full_native_tensors(model)
        metrics["update_max_abs"] = max(
            float((updated[name] - (original[name] - 0.01 * references[name].grad.float())).abs().max())
            for name in updated
        )
        assert metrics["update_max_abs"] < 3e-5
        for handle in handles:
            handle.remove()
        metrics["status"] = "passed"
        if rank == 0:
            Path(evidence).write_text(json.dumps(metrics, indent=2) + "\n")
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("cp", [False, True], ids=["fsdp2", "fsdp2_cp2"])
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_te_fsdp_protected_sink_and_cp_gradients(cp: bool) -> None:
    """Exercise actual TE saved-offset lifetime, FP32 residuals and one SGD update."""
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    directory = Path(os.environ.get("FLINT_BACKEND_EVIDENCE", "/tmp/flint-backends"))
    directory.mkdir(parents=True, exist_ok=True)
    mp.spawn(_parallel_worker, args=(port, cp, str(directory / f"te_parallel_cp_{cp}.json")), nprocs=2, join=True)


def _reference_attention(
    x: torch.Tensor, weights: dict[str, torch.Tensor], config: FlintConfig, layer_index: int
) -> torch.Tensor:
    """Compose independent FP32 projections and FP16 QKV/output roundings.

    Args:
        x: FP32 CUDA input [batch, sequence, hidden], with an independent gradient.
        weights: Independent FP32 leaves; projection matrices [out,in], per-head
            normalization scales [head_dim], and sink logits [query_heads].
        config: Architecture and explicit attention boundary policy.
        layer_index: Full/no-RoPE or sliding/RoPE layer in the tiny source schedule.

    Returns:
        FP32 [batch, sequence, hidden] output with preserved autograd history.
        This uses no production projection, attention, normalization or rope helper.
    """
    b, s, _ = x.shape
    qh, kh, d = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    q = rms(F.linear(x, weights["q_proj.weight"]).view(b, s, qh, d), weights["q_norm.weight"], config.rms_norm_eps)
    k = rms(F.linear(x, weights["k_proj.weight"]).view(b, s, kh, d), weights["k_norm.weight"], config.rms_norm_eps)
    v = F.linear(x, weights["v_proj.weight"]).view(b, s, kh, d)
    if layer_index not in config.no_rope_layer_indices:
        q, k = rope(q, config.rope_parameters["rope_theta"]), rope(k, config.rope_parameters["rope_theta"])
    q, k, v = q.half(), k.half(), v.half()
    q = q.transpose(1, 2).float()
    k, v = [tensor.repeat_interleave(qh // kh, 2).transpose(1, 2).float() for tensor in (k, v)]
    distance = torch.arange(s, device=x.device)[:, None] - torch.arange(s, device=x.device)[None]
    allowed = distance >= 0
    if config.layer_types[layer_index] == "sliding_attention":
        allowed &= distance < config.sliding_window
    scores = (q @ k.transpose(-1, -2) / math.sqrt(d)).masked_fill(~allowed, -torch.inf)
    sink = weights["_fp32_params.softmax_offset"].view(1, qh, 1, 1).expand(b, qh, s, 1)
    probability = torch.softmax(torch.cat((scores, sink), -1), -1)[..., :-1]
    attended = (probability @ v).transpose(1, 2).contiguous().view(b, s, qh * d).half().float()
    return F.linear(attended, weights["o_proj.weight"])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("layer_index", [0, 1])
@pytest.mark.parametrize("upstream_kind", ["normal", "zero", "small_scaled"])
def test_actual_fp16_te_boundary_outputs_and_all_gradients(
    layer_index: int, upstream_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check FP32 parameters/input/output and genuine fused TE forward/backward.

    Small gradients use explicit loss scaling, then unscale FP32 adjoints. This
    tests a differentiable precision boundary, without endorsing unscaled FP16
    training or claiming the failed BF16 model passed its full-vocabulary gate.
    """
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
    monkeypatch.setenv("NVTE_FLASH_ATTN", "0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(164375 + layer_index)
    config = FlintConfig(
        **{**te_config(dtype=torch.float32).to_dict(), "attention_compute_dtype": "fp16", "sliding_window": 7}
    )
    native = FlintAttention(config, layer_index, attention_backend("te")).cuda()
    native.init_weights(torch.device("cuda"))
    with torch.no_grad():
        native._fp32_params.softmax_offset.copy_(torch.tensor([-0.7, 0.3, 0.8, -0.1], device="cuda"))
    weights = {name: parameter.detach().clone().requires_grad_() for name, parameter in native.named_parameters()}
    original = torch.randn(2, 20, config.hidden_size, device="cuda")
    x, expected_x = original.clone().requires_grad_(), original.clone().requires_grad_()
    upstream = torch.randn_like(original) / original.numel() ** 0.5
    scale = 1.0
    if upstream_kind == "zero":
        upstream.zero_()
    elif upstream_kind == "small_scaled":
        upstream *= 1e-8
        # The retained scale=65536 diagnostic loses subnormal FP16 adjoints.
        # A larger explicit scale keeps this deliberately tiny objective away
        # from that boundary; compare the unscaled result to normal gradients.
        scale = float(2**24)
    calls = []
    handle = native._fp32_params.fused_attention.register_forward_hook(lambda module, args, output: calls.append(True))
    actual = native(x)
    expected = _reference_attention(expected_x, weights, config, layer_index)
    actual_gradients = torch.autograd.grad((actual * upstream).sum() * scale, (x, *native.parameters()))
    expected_gradients = torch.autograd.grad((expected * upstream).sum() * scale, (expected_x, *weights.values()))
    handle.remove()
    assert calls == [True], "the actual fused TE module must execute"
    assert actual.dtype == torch.float32 and all(parameter.dtype == torch.float32 for parameter in native.parameters())
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0.03)
    metrics = {
        "output": _errors(actual, expected),
        "loss_scale": scale,
        "gradients": {},
        "policy_harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    names = ["input", *dict(native.named_parameters())]
    for name, gradient, reference in zip(names, actual_gradients, expected_gradients, strict=True):
        gradient, reference = gradient / scale, reference / scale
        assert gradient.dtype == torch.float32 and torch.isfinite(gradient).all(), name
        metrics["gradients"][name] = _errors(gradient, reference)
        if upstream_kind == "zero":
            assert torch.count_nonzero(gradient) == torch.count_nonzero(reference) == 0, name
        else:
            assert torch.count_nonzero(gradient) > 0 and torch.count_nonzero(reference) > 0, name
            torch.testing.assert_close(gradient, reference, atol=1e-4 / scale, rtol=0.08, msg=name)
            assert metrics["gradients"][name]["relative_l2"] < 0.03, (name, metrics["gradients"][name])
    if upstream_kind == "small_scaled":
        normal_input = original.clone().requires_grad_()
        normal_gradients = torch.autograd.grad(
            (native(normal_input) * (upstream / 1e-8)).sum(), (normal_input, *native.parameters())
        )
        metrics["loss_scaling_recovery"] = {}
        for probe_scale in (1.0, 65536.0, scale):
            if probe_scale == scale:
                gradients = actual_gradients
            else:
                probe_input = original.clone().requires_grad_()
                gradients = torch.autograd.grad(
                    (native(probe_input) * upstream).sum() * probe_scale, (probe_input, *native.parameters())
                )
            recovery = {}
            for name, gradient, normal in zip(names, gradients, normal_gradients, strict=True):
                # Compare at ordinary gradient magnitude, avoiding cosine's
                # default absolute epsilon hiding tiny-vector directions.
                recovered = gradient / (probe_scale * 1e-8)
                recovery[name] = {
                    **_errors(recovered, normal),
                    "nonzero": int(torch.count_nonzero(gradient)),
                    "zero": gradient.numel() - int(torch.count_nonzero(gradient)),
                }
                if probe_scale == scale:
                    assert recovery[name]["relative_l2"] < 0.03, (name, recovery[name])
            metrics["loss_scaling_recovery"][str(int(probe_scale))] = recovery
    _record(f"fp16_te_boundary_{layer_index}_{upstream_kind}", metrics)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("layer_index", [0, 1])
def test_real_te_packed_document_padding_and_window_jacobian(layer_index: int) -> None:
    """A selected query sees only its own document and its allowed causal window."""
    torch.manual_seed(437500 + layer_index)
    config = te_config()
    attention = FlintAttention(config, layer_index, attention_backend("te")).cuda().train()
    length, stride = 514, 576
    tokens = 2 * stride
    hidden = torch.randn(tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    padding = torch.ones(tokens, device="cuda", dtype=torch.bool)
    padding[:length] = padding[stride : stride + length] = False
    positions = torch.arange(stride, device="cuda").repeat(2)
    kwargs = {
        "position_ids": positions,
        "qkv_format": "thd",
        "cu_seqlens": torch.tensor([0, length, 2 * length], device="cuda", dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([0, stride, tokens], device="cuda", dtype=torch.int32),
        "max_seqlen": length,
        "padding_mask": padding,
    }
    query = stride + length - 1
    output = attention(hidden, **kwargs)
    upstream = torch.linspace(-1.3, 0.9, config.hidden_size, device="cuda", dtype=torch.float32)
    gradient, sink_gradient = torch.autograd.grad(
        (output[query].float() * upstream).sum(), (hidden, attention._fp32_params.softmax_offset)
    )
    assert torch.isfinite(gradient).all() and torch.isfinite(sink_gradient).all()
    assert torch.count_nonzero(sink_gradient) == config.num_attention_heads
    assert sink_gradient.dtype == torch.float32
    cross_document = gradient[:length]
    assert torch.count_nonzero(cross_document) == 0
    assert torch.count_nonzero(gradient[padding]) == 0
    if layer_index == 1:
        # Query position513 sees positions1..513: exactly513 tokens including itself.
        assert torch.count_nonzero(gradient[stride]) == 0
    else:
        assert torch.count_nonzero(gradient[stride]) > 0
    assert torch.count_nonzero(gradient[query - 1]) > 0
    assert torch.count_nonzero(gradient[query]) > 0

    forbidden = padding.clone()
    forbidden[:length] = True
    if layer_index == 1:
        forbidden[stride] = True
    altered = hidden.detach().clone()
    altered[forbidden] = torch.randn_like(altered[forbidden]) * 7 + 11
    with torch.no_grad():
        isolated = attention(altered, **kwargs)
        torch.testing.assert_close(isolated[query], output[query], atol=0, rtol=0)
        allowed = hidden.detach().clone()
        allowed[query - 1] += 10
        changed = attention(allowed, **kwargs)
    assert not torch.equal(changed[query], output[query]), "Allowed preceding token must affect the selected query"
    result = {
        "status": "passed",
        "scope": "Actual TE THD forward and input Jacobian; no backend stand-in or output masking in the test",
        "provenance": _provenance(),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "layer_index": layer_index,
        "physical_tokens": tokens,
        "document_lengths": [length, length],
        "padded_lengths": [stride, stride],
        "query_document_position": length - 1,
        "sliding_window_including_query": 513 if layer_index == 1 else None,
        "cross_document_gradient_nonzero": int(torch.count_nonzero(cross_document)),
        "padding_gradient_nonzero": int(torch.count_nonzero(gradient[padding])),
        "outside_window_gradient_nonzero": int(torch.count_nonzero(gradient[stride])) if layer_index == 1 else None,
        "forbidden_perturbation_query_output_exact": True,
        "allowed_perturbation_changes_output": True,
        "all_sink_gradients_nonzero_fp32": True,
    }
    directory = Path(os.environ.get("FLINT_BACKEND_EVIDENCE", "/tmp/flint-packed-isolation"))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"te_packed_isolation_{layer_index}.json").write_text(json.dumps(result, indent=2) + "\n")


def _global_max(value: torch.Tensor) -> float:
    """Reduce one scalar CUDA value over the two attention context ranks."""
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value)


def _worker(rank: int, port: int, directory: str) -> None:
    """Compare replicated CP1 with actual CP2 attention on local [T/2,H] rows.

    Args:
        rank: CUDA/context rank in a two-rank process group.
        port: Task-owned rendezvous port on localhost.
        directory: Evidence destination outside the checkpoint directory.
    """
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE="2")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=8))
    destination = Path(directory) / f"te_long_document_cp_rank{rank}.json"
    result: dict[str, Any] = {"status": "running", "rank": rank, "scope": __doc__}
    try:
        torch.manual_seed(4375513)
        torch.cuda.manual_seed_all(4375513)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        config = FlintConfig(dtype=torch.bfloat16, router_aux_loss_coef=0.0)
        attention = FlintAttention(config, 2, attention_backend("te")).cuda().train()
        attention.init_weights(torch.device("cuda"))
        assert (attention.hidden_size, attention.num_heads, attention.num_kv_heads, attention.head_dim) == (
            2048,
            32,
            4,
            128,
        )
        available, tex = safe_import("transformer_engine_torch")
        assert available, "Actual TE extension is required"
        device = torch.device("cuda", rank)
        lengths = [512, 513, 514, 1027]
        spans = [(length + 3) // 4 * 4 for length in lengths]
        padded_cu = torch.tensor([0] + spans, device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        real_cu = torch.tensor([0] + lengths, device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        valid = torch.cat([torch.arange(span, device=device) < length for length, span in zip(lengths, spans)])
        positions = torch.cat(
            [torch.cat((torch.arange(length), torch.zeros(span - length))) for length, span in zip(lengths, spans)]
        ).to(device=device, dtype=torch.long)
        hidden = torch.randn(sum(spans), config.hidden_size, device=device, dtype=torch.bfloat16)
        upstream = torch.randn_like(hidden) * valid[:, None] / (sum(lengths) * config.hidden_size) ** 0.5
        reference_kwargs = {
            "position_ids": positions,
            "qkv_format": "thd",
            "cu_seqlens": real_cu,
            "cu_seqlens_padded": padded_cu,
            "max_seqlen": max(lengths),
            "padding_mask": ~valid,
        }
        source = hidden.detach().clone().requires_grad_(True)
        expected = attention(source, **reference_kwargs)
        (expected.float() * upstream.float()).sum().backward()
        expected_output = expected.detach()
        expected_input_gradient = source.grad.detach().clone()
        expected_parameters = {
            name: parameter.grad.detach().clone() for name, parameter in attention.named_parameters()
        }
        assert torch.isfinite(expected_output).all() and torch.isfinite(expected_input_gradient).all()
        assert all(torch.isfinite(value).all() for value in expected_parameters.values())
        attention.zero_grad(set_to_none=True)
        del source, expected
        indices = tex.thd_get_partitioned_indices(padded_cu, sum(spans), 2, rank).long()
        attention.attn_module.set_context_parallel_group(
            dist.group.WORLD, [0, 1], torch.cuda.Stream(), cp_comm_type="a2a"
        )
        kwargs = {
            "position_ids": positions[indices],
            "qkv_format": "thd",
            "cu_seqlens": padded_cu,
            "cu_seqlens_padded": padded_cu,
            "max_seqlen": max(spans),
            "padding_mask": ~valid[indices],
        }
        x = hidden[indices].detach().requires_grad_(True)
        output = attention(x, **kwargs)
        (output.float() * upstream[indices].float()).sum().backward()
        assert attention.attn_module.cp_comm_type == "a2a"
        assert torch.isfinite(output).all() and torch.isfinite(x.grad).all()
        selected = valid[indices]
        result.update(
            provenance=_provenance(),
            test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            checkpoint_payload_read=False,
            dimensions={"hidden": 2048, "query_heads": 32, "kv_heads": 4, "head_dim": 128},
            window_including_query=513,
            document_lengths=lengths,
            padded_lengths=spans,
            output=_errors(output[selected], expected_output[indices][selected]),
            input_gradient=_errors(x.grad, expected_input_gradient[indices]),
            parameter_gradients={},
            tolerances={"output_relative_l2": 1e-3, "input_and_parameter_gradient_relative_l2": 2e-2},
            sink_reduction="TE A2A returns the complete sink adjoint on each rank; ordinary weight contributions sum",
        )
        destination.write_text(json.dumps(result, indent=2) + "\n")
        for key, tolerance in (("output", 1e-3), ("input_gradient", 2e-2)):
            error = _global_max(torch.tensor(result[key]["relative_l2"], device=device))
            result[key]["worst_rank_relative_l2"] = error
            assert error < tolerance, result
        for name, parameter in attention.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            gradient = parameter.grad.detach().float().clone()
            if name != "_fp32_params.softmax_offset":
                dist.all_reduce(gradient)
            else:
                assert gradient.dtype == torch.float32 and torch.count_nonzero(gradient) == config.num_attention_heads
            metrics = _errors(gradient, expected_parameters[name])
            metrics["worst_rank_relative_l2"] = _global_max(torch.tensor(metrics["relative_l2"], device=device))
            result["parameter_gradients"][name] = metrics
            assert metrics["worst_rank_relative_l2"] < 2e-2, (name, metrics)

        # A selected final query has a sparse exact Jacobian under the real CP
        # communication and attention backward, independent of parity tolerances.
        attention.zero_grad(set_to_none=True)
        probe = hidden[indices].detach().requires_grad_(True)
        probe_output = attention(probe, **kwargs)
        last_start, query = sum(spans[:-1]), sum(spans[:-1]) + lengths[-1] - 1
        is_query = indices == query
        direction = torch.linspace(-0.9, 1.3, config.hidden_size, device=device)
        (probe_output.float() * is_query[:, None] * direction).sum().backward()
        forbidden = (~valid[indices]) | (indices < last_start) | (indices < query - 512)
        forbidden_count = torch.count_nonzero(probe.grad[forbidden]).to(torch.long)
        dist.all_reduce(forbidden_count)
        allowed_count = torch.count_nonzero(probe.grad[(indices == query) | (indices == query - 1)]).to(torch.long)
        dist.all_reduce(allowed_count)
        assert int(forbidden_count) == 0 and int(allowed_count) > 0
        result["jacobian"] = {
            "forbidden_gradient_nonzero": int(forbidden_count),
            "allowed_gradient_nonzero": int(allowed_count),
        }

        with torch.no_grad():
            altered = hidden.clone()
            altered[:last_start] = -altered[:last_start]
            altered[~valid] = 11
            changed = attention(altered[indices], **kwargs)
            last = selected & (indices >= last_start)
            document_leak = _global_max((changed[last] - output.detach()[last]).abs().max().float())
            altered = hidden.clone()
            altered[last_start] = -altered[last_start]
            changed = attention(altered[indices], **kwargs)
            far = last & (indices >= last_start + 513)
            far_delta = _global_max((changed[far] - output.detach()[far]).abs().max().float())
            first = indices == last_start
            near_delta = torch.zeros((), device=device)
            if first.any():
                near_delta = (changed[first] - output.detach()[first]).abs().max().float()
            near_delta = _global_max(near_delta)
        result["perturbations"] = {
            "other_document_and_padding_max_effect": document_leak,
            "outside_window_max_effect": far_delta,
            "changed_token_max_effect": near_delta,
        }
        assert document_leak == far_delta == 0 and near_delta > 0, result
        result["status"] = "passed"
    except BaseException:
        result.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        destination.write_text(json.dumps(result, indent=2) + "\n")
        dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_source_dimension_swa513_cp2_long_document_parity_and_isolation() -> None:
    """Exercise real CP2 [T/2,2048] attention against replicated CP1 [T,2048]."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    directory = Path(os.environ.get("FLINT_BACKEND_EVIDENCE", "/tmp/flint-long-document-cp"))
    directory.mkdir(parents=True, exist_ok=True)
    mp.spawn(_worker, args=(port, str(directory)), nprocs=2, join=True)


def _packed_batch(device: torch.device) -> dict[str, torch.Tensor | str]:
    """Freeze real windows512/513/514 plus ignored-document and padding sentinels.

    Args:
        device: Runtime CUDA device.

    Returns:
        Global IDs, shifted labels and positions [1,2048], real/padded document
        lengths [1,4], and qkv_format='thd'. ID29 belongs only to the ignored
        document; ID30 only to physical padding. Valid EOS2 remains trainable.
    """
    physical, real = [512, 576, 576, 384], [512, 513, 514, 13]
    ids = torch.full((2048,), 30, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    positions = torch.cat([torch.arange(n, device=device) for n in physical])[None]
    offset = 0
    for index, (padded, length) in enumerate(zip(physical, real, strict=True)):
        values = (
            (torch.arange(length, device=device) * 7 + 2) % 28
            if index < 3
            else torch.full((length,), 29, device=device)
        )
        ids[offset : offset + length] = values
        if index < 3:
            labels[offset : offset + length - 1] = values[1:]
        offset += padded
    return {
        "input_ids": ids[None],
        "labels": labels[None],
        "position_ids": positions,
        "seq_lens": torch.tensor([real], device=device, dtype=torch.int32),
        "seq_lens_padded": torch.tensor([physical], device=device, dtype=torch.int32),
        "qkv_format": "thd",
    }


def _magi_worker(rank: int, port: int, evidence: str) -> None:
    """Hold FP32 masters/BF16 compute fixed while changing only context degree."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank)
    )
    os.environ["MAGI_ATTENTION_DSINK_ALL_REDUCE_OP"] = "none"
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, timeout=timedelta(minutes=10))
    try:
        torch.manual_seed(94375)
        config = te_config(dtype=torch.float32)
        source_model = config.build(backend=attention_backend("eager")).to(device)
        source = {name: value.detach().clone() for name, value in source_model.state_dict().items()}
        del source_model
        results = []
        for cp in [1, 2]:
            setup = DistributedSetup.build(
                strategy=FSDP2Config(
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32,
                        output_dtype=torch.bfloat16,
                        cast_forward_inputs=False,
                    ),
                    enable_fsdp2_prefetch=False,
                ),
                parallelism_sizes=ParallelismSizes(dp_size=2 // cp, cp_size=cp),
                world_size=2,
            )
            model = config.build(backend=attention_backend("magi")).to(device)
            model.load_state_dict(source, strict=True)
            model = model.parallelizer.parallelize(model, setup.mesh_context)
            model.cp_mesh = setup.mesh_context.device_mesh["cp"]
            batch = _packed_batch(device)
            sharder = ContextParallelSharder(model, setup.mesh_context.device_mesh, batch, padding_token_id=30)
            _, local = sharder.shard(batch)
            labels = local.pop("labels")
            for name in ["seq_lens", "seq_lens_padded", "loss_mask"]:
                local.pop(name, None)
            output = model(**local).logits
            logits = sharder.gather_token_tensor(output.detach(), seq_dim=1)
            loss = F.cross_entropy(output.flatten(0, 1), labels.flatten(), reduction="sum") / 1536
            (loss * cp).backward()
            gradients = full_native_tensors(model, gradients=True)
            if cp == 2:
                dist.all_reduce(loss)
            assert torch.count_nonzero(gradients["model.embed_tokens.weight"][29:31]) == 0
            assert all(torch.isfinite(value).all() for value in gradients.values())
            sinks = {name: value for name, value in gradients.items() if name.endswith("softmax_offset")}
            assert len(sinks) == 3 and all(
                value.dtype == torch.float32 and torch.count_nonzero(value) == 4 for value in sinks.values()
            )
            # Use one global reference norm so CP replication cannot hide scaling.
            norm = torch.linalg.vector_norm(torch.stack([value.float().norm() for value in gradients.values()]))
            reported_norm = scale_grads_and_clip_grad_norm(
                float("inf"),
                [model],
                device_mesh=setup.mesh_context.device_mesh,
                foreach=True,
                grad_norm_backend="triton",
            )
            if isinstance(reported_norm, DTensor):
                reported_norm = reported_norm.full_tensor()
            torch.testing.assert_close(reported_norm.float(), norm, atol=1e-5, rtol=1e-5)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            optimizer.step()
            updated = full_native_tensors(model)
            current = {
                "loss": float(loss.detach()),
                "global_gradient_norm": float(norm),
                "reported_global_gradient_norm": float(reported_norm),
                "logits": logits.cpu(),
                "gradients": {n: v.cpu() for n, v in gradients.items()},
                "updated": {n: v.cpu() for n, v in updated.items()},
            }
            results.append(current)
            del model, optimizer, gradients, updated, output, logits, loss, sharder
            gc.collect()
            torch.cuda.empty_cache()
        baseline, candidate = results
        torch.testing.assert_close(
            torch.tensor(candidate["loss"]), torch.tensor(baseline["loss"]), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            torch.tensor(candidate["global_gradient_norm"]),
            torch.tensor(baseline["global_gradient_norm"]),
            atol=3e-3,
            rtol=0.02,
        )
        valid = torch.zeros(2048, dtype=torch.bool)
        for start, length in [(0, 512), (512, 513), (1088, 514), (1664, 13)]:
            valid[start : start + length] = True
        left, right = baseline["logits"][:, valid], candidate["logits"][:, valid]
        difference = (left - right).abs()
        cosine = float(F.cosine_similarity(left.flatten(), right.flatten(), dim=0))
        assert float(difference.max()) < 1e-2 and cosine > 0.9999
        errors = {}
        for name, expected in baseline["gradients"].items():
            actual = candidate["gradients"][name]
            torch.testing.assert_close(actual, expected, atol=3e-3, rtol=0.15, msg=name)
            relative = float((actual - expected).norm() / expected.norm().clamp_min(1e-20))
            assert relative < 0.12, (name, relative)
            errors[name] = {"max_abs": float((actual - expected).abs().max()), "relative_l2": relative}
            torch.testing.assert_close(
                candidate["updated"][name], baseline["updated"][name], atol=3e-5, rtol=0, msg=name
            )
        if rank == 0:
            Path(evidence).write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "backend": "actual Magi calc_attn sink API",
                        "magi_kernel_backend": os.environ.get("MAGI_ATTENTION_KERNEL_BACKEND", "ffa"),
                        "control": "same source tensors, packed inputs, FP32 masters, BF16 compute; CP1/CP2 only",
                        "cp1_loss": baseline["loss"],
                        "cp2_loss": candidate["loss"],
                        "logit_max_abs": float(difference.max()),
                        "logit_cosine": cosine,
                        "all_parameter_gradient_errors": errors,
                        "sink_gradients_finite_nonzero": True,
                        "padding_and_ignored_document_embedding_gradient_exact_zero": True,
                        "window_document_lengths": [512, 513, 514],
                        "physical_tokens": 2048,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_actual_magi_context_parallel_learned_sink_training(tmp_path: Path) -> None:
    """Run the real optional distributed kernel; absence of Magi is an explicit failure."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    _directory = Path(os.environ.get("FLINT_MAGI_EVIDENCE", str(tmp_path)))
    _directory.mkdir(parents=True, exist_ok=True)
    mp.spawn(_magi_worker, args=(port, str(_directory / "magi_cp1_cp2.json")), nprocs=2, join=True)
