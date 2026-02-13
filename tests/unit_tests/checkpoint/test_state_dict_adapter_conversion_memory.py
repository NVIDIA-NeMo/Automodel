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

"""Test that state dict adapter HF <-> native conversion does not double GPU memory.

Regression test: conversion should stay within model weights + a small margin (~5--10%),
not allocate a full second copy of the weights (double memory).

To trace where peak memory occurs, run with:
  NEMO_TRACE_STATE_DICT_MEM=1 pytest tests/unit_tests/checkpoint/test_state_dict_adapter_conversion_memory.py -v -s
"""

from __future__ import annotations

import gc
import os
from types import SimpleNamespace

import pytest
import torch

from transformers import LlamaConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.llama.state_dict_adapter import LlamaStateDictAdapter
from nemo_automodel.components.models.qwen3_moe.state_dict_adapter import Qwen3MoeStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _payload_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    return sum(_tensor_bytes(t) for t in state_dict.values())


def _skip_if_low_free_cuda_mem(required_bytes: int, safety_factor: float = 2.0) -> None:
    free, _ = torch.cuda.mem_get_info()
    if free < int(required_bytes * safety_factor):
        pytest.skip(
            f"Not enough free CUDA memory for this test. "
            f"Need ~{required_bytes * safety_factor / (1024**3):.2f} GiB, have {free / (1024**3):.2f} GiB."
        )


def _reset_cuda_mem_stats() -> None:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def _trace_gpu_mem(label: str, always_print: bool = False) -> tuple[int, int]:
    """Sync, get current and peak allocated, print if NEMO_TRACE_STATE_DICT_MEM=1 or always_print. Returns (current, peak)."""
    torch.cuda.synchronize()
    current = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    if always_print or os.environ.get("NEMO_TRACE_STATE_DICT_MEM") == "1":
        print(f"[mem] {label}: current={current / (1024**2):.2f} MiB  peak={peak / (1024**2):.2f} MiB")
    return (current, peak)


def _make_adapter(dim: int, moe_inter_dim: int, n_experts: int = 4, dtype: torch.dtype = torch.float16):
    cfg = SimpleNamespace()
    moe_config = MoEConfig(
        n_routed_experts=n_experts,
        n_shared_experts=0,
        n_activated_experts=min(2, n_experts),
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=dim,
        inter_dim=dim * 2,
        moe_inter_dim=moe_inter_dim,
        norm_topk_prob=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
        dtype=dtype,
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )
    return Qwen3MoeStateDictAdapter(config=cfg, moe_config=moe_config, backend=backend, dtype=dtype)


def _make_llama_fused_qkv_adapter(
    num_layers: int = 2,
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    intermediate_size: int = 128,
):
    """Tiny Llama config for fused QKV (qkv_proj + gate_up_proj) memory test."""
    config = LlamaConfig(
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=256,
        max_position_embeddings=128,
    )
    return LlamaStateDictAdapter(config)


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("adapter_kind", ["qwen3_moe", "llama_fused_qkv"])
def test_state_dict_adapter_conversion_peak_gpu_memory_within_model_weights_plus_margin(
    adapter_kind: str,
):
    """Round-trip conversion must not exceed model weights + margin.

    - Qwen3 MoE: lazy/JIT conversion keeps peak <= baseline + 10%.
    - Llama fused QKV: eager combined-projection adapter; peak <= baseline + 50%.
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    dtype = torch.float16

    # margin_ratio is the important parameter of this test. It determines the extra
    # space over the baseline that we allow it to use during conversion.
    margin_ratio = 0.10

    if adapter_kind == "qwen3_moe":
        n_experts = 4
        dim = 4096
        moe_inter_dim = 1024
        adapter = _make_adapter(dim=dim, moe_inter_dim=moe_inter_dim, n_experts=n_experts, dtype=dtype)

        def make_native_state_dict():
            return {
                "model.layers.0.mlp.experts.gate_and_up_projs": torch.empty(
                    (n_experts, dim, 2 * moe_inter_dim), device=device, dtype=dtype
                ),
                "model.layers.0.mlp.experts.down_projs": torch.empty(
                    (n_experts, moe_inter_dim, dim), device=device, dtype=dtype
                ),
            }

        expected_native_keys = (
            "model.layers.0.mlp.experts.gate_and_up_projs",
            "model.layers.0.mlp.experts.down_projs",
        )
    else:
        # Llama with fused QKV: qkv_proj + gate_up_proj per layer (larger so margin is meaningful)
        num_layers = 4
        hidden_size = 1024
        num_attention_heads = 8
        num_key_value_heads = 8
        intermediate_size = 2048
        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        kv_size = num_key_value_heads * head_dim
        adapter = _make_llama_fused_qkv_adapter(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
        )

        def make_native_state_dict():
            sd = {}
            for layer_idx in range(num_layers):
                prefix = f"model.layers.{layer_idx}"
                # qkv_proj: (q + k + v) x hidden
                sd[f"{prefix}.self_attn.qkv_proj.weight"] = torch.empty(
                    q_size + 2 * kv_size, hidden_size, device=device, dtype=dtype
                )
                # gate_up_proj: (gate + up) x hidden, gate/up each intermediate_size
                sd[f"{prefix}.mlp.gate_up_proj.weight"] = torch.empty(
                    2 * intermediate_size, hidden_size, device=device, dtype=dtype
                )
            return sd

        expected_native_keys = (
            "model.layers.0.self_attn.qkv_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
        )

    native_sd = make_native_state_dict()
    payload_bytes = _payload_bytes(native_sd)
    _skip_if_low_free_cuda_mem(required_bytes=payload_bytes * 3)

    _reset_cuda_mem_stats()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    _trace_gpu_mem(f"after baseline ({adapter_kind}, native_sd created)", always_print=True)

    # Round-trip: native -> HF -> native (Qwen3 MoE uses lazy dicts; Llama uses eager).
    hf_sd = adapter.to_hf(native_sd, inplace=True)
    _trace_gpu_mem(f"after to_hf ({adapter_kind})", always_print=True)

    native_again = adapter.from_hf(hf_sd, inplace=True)
    _trace_gpu_mem(f"after from_hf ({adapter_kind}, before consuming)", always_print=True)

    # Consume one key at a time without holding all tensors.
    for i, k in enumerate(native_again.keys()):
        _ = native_again[k]
        _trace_gpu_mem(f"after native_again item {i} ({k})", always_print=True)
        del _

    _trace_gpu_mem("after full consumption", always_print=True)

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    extra_bytes = peak - baseline

    allowed_extra = int(payload_bytes * margin_ratio)
    assert extra_bytes <= allowed_extra, (
        f"State dict adapter round-trip ({adapter_kind}) used too much peak GPU memory.\n"
        f"payload_bytes (model weights)={payload_bytes} ({payload_bytes / (1024**2):.2f} MiB)\n"
        f"baseline_allocated={baseline} ({baseline / (1024**2):.2f} MiB)\n"
        f"peak_allocated={peak} ({peak / (1024**2):.2f} MiB)\n"
        f"extra_bytes={extra_bytes} ({extra_bytes / (1024**2):.2f} MiB)\n"
        f"allowed_extra (weights * {margin_ratio})={allowed_extra} ({allowed_extra / (1024**2):.2f} MiB)\n"
        f"Peak must be <= baseline + {margin_ratio * 100:.0f}%."
    )

    for key in expected_native_keys:
        assert key in native_again, f"Expected key {key} in native_again"
