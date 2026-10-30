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

"""CPU training gates against independent oracles, plus checks of the test-owned SDPA reference and metrics."""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.datasets.utils import default_collater
from nemo_automodel.components.loss.loss import MaskedCrossEntropyConfig
from nemo_automodel.components.loss.utils import calculate_loss
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.flint.config import FlintConfig
from nemo_automodel.components.models.flint.model import FlintForCausalLM
from nemo_automodel.components.optim.optimizer import AdamWConfig
from nemo_automodel.components.training.rng import StatefulRNG
from tests.functional_tests.models.flint.sdpa_reference import (
    FlintSDPAReference,
    LogitAccumulator,
    ReferenceConfig,
    compare_hidden,
    compare_logits,
    sdpa_sink_attention,
)
from tests.unit_tests.models.flint.conftest import checkpoint_style_config, cpu_loader

# DCP's first import and real filesystem checkpoint round trips exceed the usual
# per-test five-second budget on CPU; no distributed/GPU work runs here.


def _tiny_config() -> FlintConfig:
    """Keep full/no-RoPE, sliding/RoPE, dense, and routed/shared MoE in one model."""
    return FlintConfig(
        vocab_size=31,
        hidden_size=12,
        intermediate_size=20,
        moe_intermediate_size=5,
        shared_expert_intermediate_size=7,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[0],
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        no_rope_layer_indices=[0, 2],
        sliding_window=3,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        router_aux_loss_coef=0.0,
        pad_token_id=2,
        eos_token_id=[2, 3, 4],
        tie_word_embeddings=False,
        dtype=torch.float32,
    )


def _new_model(seed: int = 41) -> FlintForCausalLM:
    """Build through the actual public CPU loader before capturing optimizer identities."""
    torch.manual_seed(seed)
    backend = BackendConfig(
        attn="eager", linear="torch", rms_norm="torch_fp32", rope_fusion=False, experts="torch", dispatcher="torch"
    )
    with cpu_loader():
        model = NeMoAutoModelForCausalLM.from_config(
            _tiny_config(),
            dtype=torch.float32,
            attn_implementation="eager",
            backend=backend,
            use_liger_kernel=False,
            use_sdpa_patching=False,
        )
    assert isinstance(model, FlintForCausalLM)
    model.train()
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
    assert model.get_input_embeddings().weight is not model.get_output_embeddings().weight
    return model


def _shifted_batch() -> dict[str, torch.Tensor]:
    """Collate a shifted batch with real EOS=pad and ragged positional padding.

    Targets are shifted exactly once before the production collator/loss. The
    boolean padding mask describes positions, independently of token value 2.

    Returns:
        Newly allocated CPU tensors with batch B=2 and padded sequence S=5:
        input_ids and labels are int64 [B,S]; labels use -100 for ignored
        targets. attention_mask is int64 [B,S], one for valid tokens.
        padding_mask is boolean [B,S], True only for positional padding.
    """
    sequences = [[5, 2, 9, 4, 7, 3], [8, 6, 2, 3]]
    examples = [{"input_ids": s[:-1], "labels": s[1:], "attention_mask": [1] * (len(s) - 1)} for s in sequences]
    examples[0]["___PAD_TOKEN_IDS___"] = {"input_ids": 2, "labels": -100, "attention_mask": 0}
    return default_collater(examples)


def _rms(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize the final feature axis in the independent CPU reference.

    Args:
        x: FP32 CPU tensor [...,D], with arbitrary leading axes and feature
            width D. It retains autograd history and is not mutated.
        gamma: FP32 CPU trainable scale [D], broadcast over x's leading axes.
        eps: Positive scalar added to the mean squared feature value.

    Returns:
        FP32 CPU tensor [...,D] with autograd history and new storage,
        aliasing neither x nor gamma.
    """
    return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps) * gamma


def _rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    """Apply NeoX half-pair RoPE at contiguous zero-based sequence positions.

    Args:
        x: FP32 CPU tensor [B,S,N,D], ordered batch, sequence, attention
            heads, and even head width. The input is not mutated.
        theta: Positive rotary frequency base.

    Returns:
        FP32 CPU rotated tensor [B,S,N,D], with new storage and the input's
        autograd history. Channels in the first and second halves are paired.
    """
    half = x.shape[-1] // 2
    frequencies = theta ** (-torch.arange(half, dtype=x.dtype) / half)
    phases = torch.arange(x.shape[1], dtype=x.dtype)[:, None] * frequencies[None, :]
    cos, sin = phases.cos()[None, :, None, :], phases.sin()[None, :, None, :]
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


def _oracle_logits(
    weights: dict[str, torch.Tensor], config: FlintConfig, batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Compute logits with explicit query-head and expert-ID reference loops.

    Args:
        weights: Independent FP32 CPU trainable leaves keyed by expanded HF
            names. Embedding/head matrices are [V,H]; Q/K/V projections are
            [Nq*D,H]/[Nkv*D,H]/[Nkv*D,H]; attention output is [H,Nq*D].
            Residual/final norm scales are [H], Q/K norm scales [D], and sink
            logits [Nq]. Each dense/expert/shared MLP uses [I,H] gate/up and
            [H,I] down matrices with that MLP's own intermediate width I.
            V is vocabulary, H residual width, Nq/Nkv query/KV head count,
            and D explicit head width. These tensors do not alias the native
            parameters and are never mutated by this function.
        config: Architecture settings defining those widths and layer schedules.
        batch: CPU mapping containing int64 input_ids [B,S] and boolean,
            int64, or FP32 binary attention_mask [B,S], ordered batch and sequence. Extra fields
            are ignored. Masks select real keys; padded-query logits are
            unspecified and excluded from all reference objectives.

    Returns:
        FP32 CPU logits [B,S,V] in new storage, retaining gradients to every
        consumed weight tensor. The inputs and their storage are unchanged.
    """
    x = F.embedding(batch["input_ids"], weights["model.embed_tokens.weight"])
    b, s, _ = x.shape
    q_heads, kv_heads, d = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    positions = torch.arange(s)
    causal = positions[:, None] >= positions[None, :]
    valid_keys = batch["attention_mask"].bool()[:, None, :]
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        attn = f"{prefix}.self_attn"
        h = _rms(x, weights[f"{prefix}.input_layernorm.weight"], config.rms_norm_eps)
        q = F.linear(h, weights[f"{attn}.q_proj.weight"]).reshape(b, s, q_heads, d)
        k = F.linear(h, weights[f"{attn}.k_proj.weight"]).reshape(b, s, kv_heads, d)
        v = F.linear(h, weights[f"{attn}.v_proj.weight"]).reshape(b, s, kv_heads, d)
        q = _rms(q, weights[f"{attn}.q_norm.weight"], config.rms_norm_eps)
        k = _rms(k, weights[f"{attn}.k_norm.weight"], config.rms_norm_eps)
        if layer not in config.no_rope_layer_indices:
            theta = float(config.rope_parameters["rope_theta"])
            q, k = _rope(q, theta), _rope(k, theta)
        allowed = causal
        if config.layer_types[layer] == "sliding_attention":
            allowed = allowed & ((positions[:, None] - positions[None, :]) < config.sliding_window)
        allowed = allowed[None, :, :] & valid_keys
        heads = []
        for head in range(q_heads):
            kv = head // (q_heads // kv_heads)
            logits = q[:, :, head] @ k[:, :, kv].transpose(-1, -2) / math.sqrt(d)
            logits = logits.masked_fill(~allowed, -torch.inf)
            sink = weights[f"{attn}.softmax_offset"][head].expand(b, s, 1)
            probabilities = torch.softmax(torch.cat((logits, sink), dim=-1), dim=-1)[..., :-1]
            heads.append(probabilities @ v[:, :, kv])
        attended = torch.stack(heads, dim=2).reshape(b, s, q_heads * d)
        x = x + F.linear(attended, weights[f"{attn}.o_proj.weight"])
        h = _rms(x, weights[f"{prefix}.post_attention_layernorm.weight"], config.rms_norm_eps)
        mlp = f"{prefix}.mlp"

        def swiglu(name: str, values: torch.Tensor) -> torch.Tensor:
            """Apply one independently named dense, routed, or shared SwiGLU.

            Args:
                name: Prefix of [I,H] gate/up and [H,I] down matrices in
                    the enclosing FP32 CPU weights mapping.
                values: FP32 CPU activations [...,H], with arbitrary leading
                    axes and residual width H. Values are not mutated.

            Returns:
                FP32 CPU activations [...,H] in new storage with gradients
                to values and the three named weight matrices.
            """
            gate = F.linear(values, weights[f"{name}.gate_proj.weight"])
            up = F.linear(values, weights[f"{name}.up_proj.weight"])
            return F.linear(F.silu(gate) * up, weights[f"{name}.down_proj.weight"])

        if layer in config.mlp_only_layers:
            output = swiglu(mlp, h)
        else:
            flat = h.reshape(-1, config.hidden_size)
            scores = torch.sigmoid(F.linear(flat, weights[f"{mlp}.gate.weight"]).float())
            selected, ids = torch.topk(scores, config.num_experts_per_tok, dim=-1, sorted=False)
            routes = selected / (selected.sum(dim=-1, keepdim=True) + 1e-20)
            output = torch.zeros_like(flat)
            for expert in range(config.num_experts):
                # Explicit full expert evaluation makes the independent gradient
                # oracle simple; selection contributes zero for unused experts.
                value = swiglu(f"{mlp}.experts.{expert}", flat)
                coefficient = ((ids == expert) * routes).sum(dim=-1, keepdim=True)
                output = output + value * coefficient
            output = (output + swiglu(f"{mlp}.shared_expert", flat)).reshape_as(h)
        x = x + output
    return F.linear(_rms(x, weights["model.norm.weight"], config.rms_norm_eps), weights["lm_head.weight"]).float()


def _native_loss(model: FlintForCausalLM, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Evaluate already shifted labels through production model/loss dispatch.

    Args:
        model: Native CPU model with FP32 trainable parameters and output head.
        batch: CPU mapping with int64 input_ids/labels [B,S], ordered batch
            and sequence, and boolean/int64/FP32 binary attention_mask [B,S]. Labels are shifted
            once and use -100 for ignored positions; the mapping is not mutated.

    Returns:
        FP32 CPU scalar CE averaged once over non-ignored targets. It owns its
        storage and retains gradients to the model's trainable parameters.
    """
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    assert logits.dtype == torch.float32
    return calculate_loss(
        MaskedCrossEntropyConfig().build(),
        logits=logits,
        labels=batch["labels"],
        num_label_tokens=(batch["labels"] != -100).sum(),
    )


@pytest.mark.timeout(90)
@pytest.mark.parametrize("max_norm", [float("inf"), 0.1], ids=["unclipped", "clipped"])
def test_ce_gradients_global_norm_and_adamw_update_match_independent_oracle(max_norm: float) -> None:
    """All HF tensor gradients and one real optimizer update match independent math."""
    model, batch = _new_model(), _shifted_batch()
    assert batch["input_ids"][0, 1] == 2 and not batch["padding_mask"][0, 1]
    assert batch["labels"].tolist() == [[2, 9, 4, 7, 3], [6, 2, 3, -100, -100]]
    assert batch["padding_mask"].tolist() == [[False] * 5, [False, False, False, True, True]]
    hf_weights = model.state_dict_adapter.to_hf(dict(model.named_parameters()))
    oracle = {name: value.detach().clone().requires_grad_() for name, value in hf_weights.items()}
    expected = _oracle_logits(oracle, model.config, batch)
    actual = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    valid = batch["attention_mask"].bool()
    torch.testing.assert_close(actual[valid], expected[valid], rtol=2e-5, atol=2e-6)
    actual_loss = _native_loss(model, batch)
    expected_loss = F.cross_entropy(expected.reshape(-1, 31), batch["labels"].reshape(-1), ignore_index=-100)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-6, atol=2e-6)
    actual_loss.backward()
    expected_loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    native_grads = model.state_dict_adapter.to_hf(
        {name: p.grad if p.grad is not None else torch.zeros_like(p) for name, p in model.named_parameters()}
    )
    assert native_grads.keys() == oracle.keys()
    for name, p in oracle.items():
        assert p.grad is not None, name
        torch.testing.assert_close(native_grads[name], p.grad, rtol=2e-4, atol=3e-7, msg=name)
    for name, gradient in native_grads.items():
        if any(
            fragment in name for fragment in ("softmax_offset", "q_norm", "k_norm", "shared_expert", ".mlp.gate.weight")
        ):
            assert torch.count_nonzero(gradient) > 0, name
    actual_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    expected_norm = torch.nn.utils.clip_grad_norm_(list(oracle.values()), max_norm)
    torch.testing.assert_close(actual_norm, expected_norm, rtol=2e-5, atol=2e-6)
    optimizer = AdamWConfig(lr=0.003, weight_decay=0.01, eps=1e-6).build(model)[0]
    reference_optimizer = torch.optim.AdamW(list(oracle.values()), lr=0.003, weight_decay=0.01, eps=1e-6)
    optimizer.step()
    reference_optimizer.step()
    updated = model.state_dict_adapter.to_hf(model.state_dict())
    for name, p in oracle.items():
        torch.testing.assert_close(updated[name], p, rtol=3e-5, atol=5e-7, msg=name)


@pytest.mark.timeout(90)
def test_tiny_shifted_next_token_loss_decreases() -> None:
    """An untied model learns a fixed, correctly shifted ragged batch on CPU."""
    model, batch = _new_model(), _shifted_batch()
    optimizer = AdamWConfig(lr=0.02, weight_decay=0.0).build(model)[0]
    losses = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss = _native_loss(model, batch)
        assert torch.isfinite(loss)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    assert losses[-1] < losses[0] * 0.8, losses


def _random_batch() -> dict[str, torch.Tensor]:
    """Sample next-token data while consuming Python, NumPy, and PyTorch RNGs.

    Returns:
        CPU mapping with int64 input_ids/labels [B=2,S=5] and FP32 binary
        attention_mask [B,S]. Labels are shifted once and contiguous; inputs
        view a fresh internal [B,S+1] token sample. Returned fields do not
        alias one another and require no gradients.
    """
    tokens = (torch.randint(0, 31, (2, 6)) + random.randrange(31) + int(np.random.randint(31))) % 31
    return {"input_ids": tokens[:, :-1], "labels": tokens[:, 1:].contiguous(), "attention_mask": torch.ones(2, 5)}


def _step(model: FlintForCausalLM, optimizer: torch.optim.Optimizer, scheduler: Any) -> torch.Tensor:
    """Run one CPU model/CE/update/scheduler step on fresh random data.

    Args:
        model: Native model owning FP32 CPU trainable tensors.
        optimizer: Real optimizer over those same parameter objects; its
            gradients and state are updated in place.
        scheduler: LR scheduler attached to optimizer, advanced once per update.

    Returns:
        Detached FP32 CPU scalar pre-update loss, cloned into independent
        storage after backward and the optimizer/scheduler updates complete.
    """
    optimizer.zero_grad(set_to_none=True)
    loss = _native_loss(model, _random_batch())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    return loss.detach().clone()


def _assert_state_equal(actual: Any, expected: Any) -> None:
    """Compare model, optimizer, scheduler, and RNG snapshots exactly.

    Args:
        actual: Nested mappings/lists/tuples of scalar state or CPU tensors.
            Model and optimizer tensors retain their native parameter layout;
            RNG tensors are one-dimensional engine-specific byte/key arrays.
            Tensor dtype and shape must match the corresponding expected value.
        expected: Snapshot in the identical nested layout, including arbitrary
            FP32 model/moment shapes, scalar step tensors, or integer RNG arrays.
            Neither input, its nested objects, nor its tensor storage is mutated.
    """
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_state_equal(a, e)
    else:
        assert actual == expected


@pytest.mark.timeout(90)
def test_real_cpu_checkpoint_resumes_model_adam_rng_and_scheduler(tmp_path: Path) -> None:
    """Native torch-save DCP resumes exact FP32 training without HF export casting."""
    rng = StatefulRNG(123)
    model = _new_model()
    optimizer = AdamWConfig(lr=0.004, weight_decay=0.02).build(model)[0]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
    for _ in range(2):
        _step(model, optimizer, scheduler)
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_scheduler = copy.deepcopy(scheduler.state_dict())
    saved_rng = copy.deepcopy(rng.state_dict())
    checkpoint = CheckpointingConfig(
        checkpoint_dir=tmp_path,
        model_save_format="torch_save",
        save_consolidated=False,
        is_async=False,
    ).build(dp_rank=0, tp_rank=0, pp_rank=0)
    path = str(tmp_path / "step_2")
    try:
        checkpoint.save_model(model, path)
        checkpoint.save_optimizer(optimizer, model, path, scheduler)
        checkpoint.save_on_global_ranks(rng, "rng", path)
        continuous_losses = [_step(model, optimizer, scheduler) for _ in range(3)]
        continuous_rng = copy.deepcopy(rng.state_dict())

        restored = _new_model(seed=999)
        restored_optimizer = AdamWConfig(lr=0.004, weight_decay=0.02).build(restored)[0]
        restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=2, gamma=0.7)
        checkpoint.load_model(restored, str(Path(path) / "model"))
        checkpoint.load_optimizer(restored_optimizer, restored, path, restored_scheduler)
        checkpoint.load_on_global_ranks(rng, "rng", path)
        _assert_state_equal(restored_optimizer.state_dict(), saved_optimizer)
        _assert_state_equal(restored_scheduler.state_dict(), saved_scheduler)
        _assert_state_equal(rng.state_dict(), saved_rng)
        resumed_losses = [_step(restored, restored_optimizer, restored_scheduler) for _ in range(3)]
        for actual, expected in zip(resumed_losses, continuous_losses):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _assert_state_equal(restored.state_dict(), model.state_dict())
        _assert_state_equal(restored_optimizer.state_dict(), optimizer.state_dict())
        _assert_state_equal(restored_scheduler.state_dict(), scheduler.state_dict())
        _assert_state_equal(rng.state_dict(), continuous_rng)
    finally:
        checkpoint.close()


def _source_config() -> dict[str, object]:
    return checkpoint_style_config()


def _tiny_source() -> dict[str, object]:
    raw = _source_config()
    raw.update(
        vocab_size=31,
        hidden_size=12,
        intermediate_size=20,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=5,
        shared_expert_intermediate_size=7,
        sliding_window=3,
        max_position_embeddings=64,
        use_cache=False,
        pad_token_id=2,
        bos_token_id=1,
        eos_token_id=[2, 3, 4],
    )
    return raw


def _weights(config: ReferenceConfig) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(748)
    result = {}
    for name, shape in config.expected_shapes().items():
        tensor = torch.randn(shape, generator=generator) * 0.12
        if "layernorm.weight" in name or "_norm.weight" in name or name == "model.norm.weight":
            tensor += 1
        result[name] = tensor
    return result


def test_packed_lengths_preserve_real_eos_and_already_shifted_labels() -> None:
    from tests.functional_tests.models.flint.run_packed_cp_logits import pack_valid_mask

    batch = {
        "input_ids": torch.tensor([[5, 2, 2, 2, 9, 2, 2, 2]]),
        "labels": torch.tensor([[2, 7, -100, -100, 2, 4, 1, -100]]),
        "position_ids": torch.tensor([[0, 1, 0, 0, 0, 1, 2, 0]]),
        "seq_lens": torch.tensor([[2, 3]], dtype=torch.int32),
        "seq_lens_padded": torch.tensor([[4, 4]], dtype=torch.int32),
    }
    labels = batch["labels"].clone()
    assert pack_valid_mask(batch).tolist() == [[True, True, False, False, True, True, True, False]]
    assert torch.equal(batch["labels"], labels)
    batch["labels"][0, 3] = 1
    with pytest.raises(ValueError, match="padding"):
        pack_valid_mask(batch)


def test_real_document_boundary_crops_keep_source_shift_and_physical_extent() -> None:
    from tests.functional_tests.models.flint.run_packed_cp_logits import boundary_batch, pack_valid_mask, tensor_stats

    dataset = []
    for index, length in enumerate([3858, 3299, 2365, 900, 600, 514]):
        ids = (torch.arange(4096) + index) % 31
        labels = (ids + 1) % 31
        labels[length - 1 :] = -100
        positions = torch.zeros(4096, dtype=torch.int64)
        positions[:length] = torch.arange(length)
        dataset.append(
            {
                "input_ids": ids,
                "labels": labels,
                "position_ids": positions,
                "seq_lens": torch.tensor([length]),
                "seq_lens_padded": torch.tensor([4096]),
            }
        )
    batch, receipts = boundary_batch(dataset, padding_id=2)
    assert batch["input_ids"].shape == (1, 8192)
    assert batch["seq_lens"].tolist() == [[512, 513, 514, 3294, 2364, 832]]
    assert int(pack_valid_mask(batch).sum()) == 8029
    offset = 0
    for receipt, extent in zip(receipts, batch["seq_lens_padded"][0].tolist()):
        length = receipt["length"]
        source = dataset[receipt["row"]]
        assert torch.equal(batch["input_ids"][0, offset : offset + length], source["input_ids"][:length])
        assert torch.equal(batch["labels"][0, offset : offset + length - 1], source["labels"][: length - 1])
        assert batch["labels"][0, offset + length - 1] == -100
        offset += extent
    stats = tensor_stats(torch.tensor([2.0, 3.0]), torch.tensor([1.0, 5.0]))
    assert stats == {"reference_ss": 26.0, "actual_ss": 13.0, "error_ss": 5.0, "dot": 17.0, "max_abs": 2.0}


def test_packed_archive_compares_every_valid_logit_and_rejects_tampering(tmp_path: Path) -> None:
    from tests.functional_tests.models.flint.run_packed_cp_logits import (
        archive_receipt,
        compare_or_save_logits,
        verify_archive,
    )

    generator = torch.Generator().manual_seed(4375)
    logits = torch.randn(1, 5, 31, generator=generator)
    valid = torch.tensor([[True, True, False, True, True]])
    archive = tmp_path / "logits.safetensors"
    compare_or_save_logits(logits, valid, archive, None)
    receipt = archive_receipt(archive)
    assert verify_archive(tmp_path, receipt) == archive
    shifted = logits + 10
    result = compare_or_save_logits(shifted, valid, tmp_path / "unused", archive)
    assert result["acceptance"]["passed"] and result["compared_logit_values"] == 4 * 31
    shifted[0, -1, -1] += 15
    result = compare_or_save_logits(shifted, valid, tmp_path / "unused", archive)
    assert not result["acceptance"]["passed"] and result["max_kl_position"]["position"] == 4
    with archive.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_archive(tmp_path, receipt)


def test_complete_gradient_archive_uses_analytic_global_statistics(tmp_path: Path) -> None:
    from tests.functional_tests.models.flint.run_packed_cp_logits import capture_gradients, tensor_stats

    model = torch.nn.Linear(2, 3)
    (model.weight.sum() * 2 + model.bias.sum() * 3).backward()
    archive = tmp_path / "gradients.safetensors"
    original = capture_gradients(model, archive, None, rank=0)
    assert original["elements"] == 9 and original["parameter_tensor_count"] == 2
    assert original["actual_global_norm"] == pytest.approx(math.sqrt(51))
    model.weight.grad.add_(0.001)
    result = capture_gradients(model, tmp_path / "unused", archive, rank=0)
    delta = float(model.weight.grad[0, 0]) - 2
    assert result["relative_l2"] == pytest.approx(math.sqrt(6 * delta**2 / 51))
    assert result["per_parameter"]["bias"]["max_abs_error"] == 0
    assert result["per_parameter"]["weight"]["max_abs_error"] == pytest.approx(delta)
    with pytest.raises(ValueError, match="nonfinite"):
        tensor_stats(torch.ones(2), torch.tensor([1.0, float("nan")]))


def test_cp_precision_observers_reject_ignored_mixed_precision_policy() -> None:
    from tests.functional_tests.models.flint.run_packed_cp_logits import observe_compute_precision

    class DPA(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.softmax_offset = torch.nn.Parameter(torch.zeros(1))

        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return (q + k + v) / 3

    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self, name, torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16))
            self.attn_module = DPA()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            q, k, v = [
                getattr(self, name)(x.to(getattr(self, name).weight.dtype)) for name in ("q_proj", "k_proj", "v_proj")
            ]
            return self.o_proj(self.attn_module(q, k, v))

    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = Attention()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.self_attn(x).float()

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([Layer()])
    model.lm_head = torch.nn.Linear(2, 3, dtype=torch.float32)
    records = {"active_case": "test"}
    handles = observe_compute_precision(model, records)
    try:
        output = model.lm_head(model.model.layers[0](torch.ones(1, 3, 2)))
        assert output.dtype == torch.float32
        assert next(item for item in records["calls"] if item["module"].endswith("dpa.input"))["qkv_dtypes"] == [
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.bfloat16",
        ]
        model.float()
        with pytest.raises(AssertionError, match="actual TE Q/K/V must be BF16"):
            model.model.layers[0](torch.ones(1, 3, 2))
    finally:
        for handle in handles:
            handle.remove()


def test_reference_prefix_preserves_literal_released_schedule() -> None:
    config = ReferenceConfig.from_checkpoint(_source_config())
    assert config.source_num_hidden_layers == 49 and config.num_hidden_layers == 4
    assert config.layer_types == ("full_attention", "full_attention", "sliding_attention", "sliding_attention")
    assert config.no_rope_layer_indices == (0, 1) and config.mlp_only_layers == (0, 1)
    assert len(config.expected_shapes()) == 1589
    assert config.head_dim * config.num_attention_heads != config.hidden_size


def test_sdpa_sink_toy_output_and_analytic_gradient() -> None:
    query = torch.zeros(1, 3, 1, 2)
    key = torch.zeros_like(query)
    value = torch.tensor([2.0, 4.0, 6.0]).reshape(1, 3, 1, 1).expand(-1, -1, -1, 2)
    sink = torch.tensor([0.0], requires_grad=True)
    output = sdpa_sink_attention(query, key, value, sink, attention_mask=torch.ones(1, 3), sliding_window=None)
    assert output[0, 2, 0, 0].item() == 3.0
    output[0, 2, 0, 0].backward()
    torch.testing.assert_close(sink.grad, torch.tensor([-0.75]), rtol=0, atol=1e-7)


def test_sdpa_gqa_matches_independent_per_head_softmax_loop() -> None:
    generator = torch.Generator().manual_seed(12)
    query = torch.randn(2, 7, 4, 4, generator=generator)
    key = torch.randn(2, 7, 2, 4, generator=generator)
    value = torch.randn(2, 7, 2, 4, generator=generator)
    sinks = torch.tensor([-1.3, 0.7, 2.1, -0.2])
    valid = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    actual = sdpa_sink_attention(query, key, value, sinks, attention_mask=valid, sliding_window=3, query_chunk_size=2)
    expected = torch.zeros_like(actual)
    for batch in range(2):
        for position in range(7):
            for head in range(4):
                positions = [
                    k for k in range(max(0, position - 2), position + 1) if valid[batch, k] and valid[batch, position]
                ]
                scores = torch.stack(
                    [(query[batch, position, head] * key[batch, k, head // 2]).sum() / 2 for k in positions]
                    + [sinks[head]]
                )
                probabilities = scores.softmax(0)[:-1]
                for probability, k in zip(probabilities, positions):
                    expected[batch, position, head] += probability * value[batch, k, head // 2]
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)


def test_sdpa_window_513_includes_self_and_masks_all_real_keys_safely() -> None:
    query = torch.zeros(1, 514, 2, 2)
    key = torch.zeros(1, 514, 1, 2)
    value = torch.zeros_like(key)
    value[:, 0] = 1
    actual = sdpa_sink_attention(
        query,
        key,
        value,
        torch.full((2,), -100.0),
        attention_mask=torch.ones(1, 514),
        sliding_window=513,
        query_chunk_size=64,
    )
    torch.testing.assert_close(actual[0, 512], torch.full((2, 2), 1 / 513), atol=1e-8, rtol=0)
    assert torch.count_nonzero(actual[0, 513]) == 0
    masked = sdpa_sink_attention(
        query[:, :3],
        key[:, :3],
        value[:, :3],
        torch.tensor([80.0, -80.0]),
        attention_mask=torch.zeros(1, 3),
        sliding_window=None,
    )
    assert torch.isfinite(masked).all() and torch.count_nonzero(masked) == 0


@pytest.mark.parametrize("attention_backend", ["eager", "sdpa"])
def test_full_fp32_oracle_matches_native_on_identical_independent_tiny_weights(attention_backend: str) -> None:
    from nemo_automodel import NeMoAutoModelForCausalLM
    from nemo_automodel.components.models.flint.config import FlintConfig
    from tests.functional_tests.models.flint.reference import torch_backend

    raw = _tiny_source()
    reference_config = ReferenceConfig.from_checkpoint(raw)
    weights = _weights(reference_config)
    reference = FlintSDPAReference(reference_config, weights, query_chunk_size=2)
    native_data = copy.deepcopy(raw)
    native_data.pop("torch_dtype")
    native_data.update(
        num_hidden_layers=4,
        schedule_num_hidden_layers=49,
        router_aux_loss_coef=0.0,
        dtype="float32",
    )
    backend = torch_backend()
    backend.attn = attention_backend
    with cpu_loader():
        native = NeMoAutoModelForCausalLM.from_config(
            FlintConfig(**native_data),
            dtype=torch.float32,
            attn_implementation=attention_backend,
            backend=backend,
            use_liger_kernel=False,
            use_sdpa_patching=False,
        ).eval()
    native.load_state_dict(native.state_dict_adapter.from_hf(dict(weights)), strict=True)
    assert native.model.embed_tokens.weight.data_ptr() != weights["model.embed_tokens.weight"].data_ptr()
    ids = torch.tensor([[5, 2, 9, 4, 7], [8, 6, 2, 2, 2]])
    valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    layer_outputs = []
    handles = [
        layer.register_forward_hook(lambda module, inputs, output: layer_outputs.append(output.detach().clone()))
        for layer in native.model.layers
    ]
    try:
        with torch.inference_mode():
            actual = native(input_ids=ids, attention_mask=valid).logits
            states = reference.forward_hidden(ids, attention_mask=valid)
            expected = reference.head(states.final_norm)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(actual[valid], expected[valid], atol=1e-5, rtol=0)
    assert len(layer_outputs) == 4
    for observed, target in zip(layer_outputs, states.layers):
        torch.testing.assert_close(observed[valid], target[valid], atol=1e-5, rtol=0)
    metrics = compare_logits(expected, actual, attention_mask=valid, rows_per_chunk=2)
    assert metrics["evaluated_positions"] == 8 and metrics["vocab_size"] == 31
    assert metrics["acceptance"]["passed"]


def test_reference_does_not_import_native_model_or_adapter() -> None:
    import ast

    from tests.functional_tests.models.flint import sdpa_reference

    tree = ast.parse(Path(sdpa_reference.__file__).read_text())
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name.startswith(("nemo_automodel", "transformers")) for name in imports)


def test_reference_rejects_missing_extra_and_wrong_dtype_weights() -> None:
    config = ReferenceConfig.from_checkpoint(_tiny_source())
    weights = _weights(config)
    missing = dict(weights)
    missing.pop("model.layers.3.self_attn.softmax_offset")
    with pytest.raises(ValueError, match="missing"):
        FlintSDPAReference(config, missing)
    with pytest.raises(ValueError, match="extra"):
        FlintSDPAReference(config, {**weights, "unexpected": torch.ones(1)})
    bad = dict(weights)
    bad["lm_head.weight"] = bad["lm_head.weight"].bfloat16()
    with pytest.raises(ValueError, match="FP32"):
        FlintSDPAReference(config, bad)


def test_no_rope_prefix_is_position_invariant_but_sliding_layers_use_positions() -> None:
    raw = _tiny_source()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    first = torch.arange(5).unsqueeze(0)
    perturbed = torch.tensor([[0, 7, 2, 12, 4]])
    full_config = ReferenceConfig.from_checkpoint(raw, num_hidden_layers=2)
    full = FlintSDPAReference(full_config, _weights(full_config))
    a = full.forward_hidden(ids, attention_mask=valid, position_ids=first).final_norm
    b = full.forward_hidden(ids, attention_mask=valid, position_ids=perturbed).final_norm
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    mixed_config = ReferenceConfig.from_checkpoint(raw)
    mixed = FlintSDPAReference(mixed_config, _weights(mixed_config))
    a = mixed.forward_hidden(ids, attention_mask=valid, position_ids=first).final_norm
    b = mixed.forward_hidden(ids, attention_mask=valid, position_ids=perturbed).final_norm
    assert (a - b).abs().max().item() > 1e-5


def test_real_prefix_loader_checks_full_index_but_never_reads_omitted_layers(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    from tests.functional_tests.models.flint.run_four_layer_logits import load_reference_weights

    raw = _tiny_source()
    config = ReferenceConfig.from_checkpoint(raw)
    full_shapes = ReferenceConfig.from_checkpoint(raw, num_hidden_layers=49).expected_shapes()
    raw_weights = {name: tensor.bfloat16() for name, tensor in _weights(config).items()}
    save_file(raw_weights, tmp_path / "prefix.safetensors")
    index = {
        "weight_map": {
            name: "prefix.safetensors" if name in raw_weights else "intentionally-not-present.safetensors"
            for name in full_shapes
        }
    }
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(json.dumps(index))
    weights, receipt = load_reference_weights(tmp_path, raw, config, torch.device("cpu"))
    assert len(receipt) == len(raw_weights) and set(weights) == set(raw_weights)
    for name, tensor in weights.items():
        assert tensor.dtype == torch.float32
        torch.testing.assert_close(tensor, raw_weights[name].float(), rtol=0, atol=0)
    assert all(row["source_dtype"] == "torch.bfloat16" for row in receipt)
    index["weight_map"]["unknown.source.weight"] = "intentionally-not-present.safetensors"
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="index keys"):
        load_reference_weights(tmp_path, raw, config, torch.device("cpu"))


def test_distribution_metrics_use_reference_to_candidate_kl_and_natural_jsd() -> None:
    reference = torch.tensor([[[math.log(0.8), math.log(0.2)]]])
    candidate = torch.tensor([[[math.log(0.5), math.log(0.5)]]])
    result = compare_logits(reference, candidate)
    p, q = reference.double().softmax(-1), candidate.double().softmax(-1)
    expected_kl = (p * (p.log() - q.log())).sum().item()
    midpoint = 0.5 * (p + q)
    expected_jsd = 0.5 * ((p * (p.log() - midpoint.log())).sum() + (q * (q.log() - midpoint.log())).sum()).item()
    assert result["mean_kl"] == pytest.approx(expected_kl, abs=1e-7)
    assert result["mean_jsd"] == pytest.approx(expected_jsd, abs=1e-7)
    reverse = compare_logits(candidate, reference)
    assert abs(result["mean_kl"] - reverse["mean_kl"]) > 0.01
    assert result["acceptance"] == {"metric": "mean_kl", "operator": "<", "threshold": 1e-3, "passed": False}


def test_metrics_include_late_position_and_unprobed_vocabulary_entry() -> None:
    reference = torch.zeros(1, 4096, 19)
    candidate = reference.clone()
    candidate[0, 4095, 18] = 10
    result = compare_logits(reference, candidate, rows_per_chunk=31)
    assert result["evaluated_positions"] == 4096 and result["compared_logit_values"] == 4096 * 19
    assert result["max_kl_position"] == {"batch_index": 0, "position": 4095}
    assert result["max_kl"] > 6 and result["p95_kl"] == 0
    assert result["top1_agreement"] == 4095 / 4096
    assert not result["acceptance"]["passed"]


def test_primary_gate_is_distributional_not_the_old_max_absolute_error_gate() -> None:
    generator = torch.Generator().manual_seed(444)
    reference = torch.randn(2, 5, 17, generator=generator)
    candidate = reference + 100
    result = compare_logits(reference, candidate)
    assert result["max_abs_logit_difference"] > 99
    assert abs(result["mean_kl"]) < 1e-5
    assert result["acceptance"]["passed"]


def test_metrics_chunking_mask_and_hidden_statistics() -> None:
    generator = torch.Generator().manual_seed(511)
    reference = torch.randn(2, 5, 17, generator=generator)
    candidate = reference + 0.02 * torch.randn(2, 5, 17, generator=generator)
    valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    first = compare_logits(reference, candidate, attention_mask=valid, rows_per_chunk=1)
    second = compare_logits(reference, candidate, attention_mask=valid, rows_per_chunk=7)
    assert first["evaluated_positions"] == 8
    for key in ("mean_kl", "p95_kl", "mean_jsd", "logit_cosine", "mean_abs_logit_difference"):
        assert first[key] == pytest.approx(second[key], abs=1e-12)
    hidden = compare_hidden(torch.ones(1, 2, 3), torch.full((1, 2, 3), 2.0))
    assert hidden == {
        "elements": 6,
        "mean_abs_difference": 1.0,
        "max_abs_difference": 1.0,
        "relative_rmse": 1.0,
        "cosine": 1.0,
    }


def test_metrics_reject_nonfinite_or_duplicate_positions() -> None:
    accumulator = LogitAccumulator(3)
    accumulator.update(torch.zeros(1, 3), torch.zeros(1, 3), positions=[(0, 7)])
    with pytest.raises(ValueError, match="twice"):
        accumulator.update(torch.zeros(1, 3), torch.zeros(1, 3), positions=[(0, 7)])
    with pytest.raises(FloatingPointError):
        accumulator.update(torch.zeros(1, 3), torch.full((1, 3), torch.inf))
