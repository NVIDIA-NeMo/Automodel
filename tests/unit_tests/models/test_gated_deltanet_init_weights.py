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


"""Checkpoint-free GatedDeltaNet initialization: production ``initialize_weights``
must leave no parameter uninitialized and support a finite first forward/backward
through the real convolution and recurrence (torch fallbacks)."""

from __future__ import annotations

import inspect

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5_moe import cp_linear_attn
from nemo_automodel.components.moe.layers import MoEConfig

_GDN = dict(
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_conv_kernel_dim=4,
)


def _backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def _moe_config(config) -> MoEConfig:
    return MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.hidden_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.num_experts,
        n_shared_experts=1,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
        shared_expert_gate=True,
        shared_expert_inter_dim=config.shared_expert_intermediate_size,
    )


def _moe_text_kwargs() -> dict:
    return dict(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        tie_word_embeddings=False,
        layer_types=["linear_attention"],
        **_GDN,
    )


def _build_qwen3_next():
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

    from nemo_automodel.components.models.qwen3_next.model import Qwen3NextForCausalLM

    config = Qwen3NextConfig(**_moe_text_kwargs())
    return Qwen3NextForCausalLM.from_config(config, moe_config=_moe_config(config), backend=_backend()), {}


def _build_qwen3_5_moe():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForCausalLM

    config = Qwen3_5MoeTextConfig(**_moe_text_kwargs())
    model = Qwen3_5MoeForCausalLM.from_config(config, moe_config=_moe_config(config), backend=_backend())
    return model, {"position_ids": torch.arange(6).view(1, 1, 6).expand(3, 2, 6)}


def _build_qwen3_8_flash_next():
    from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration
    from tests.unit_tests.models.qwen3_8_flash_next.test_qwen3_8_flash_next_model import (
        _tiny_config,
        _tiny_moe_config,
    )

    config = _tiny_config()
    for name, value in _GDN.items():
        setattr(config.text_config, name, value)
    config.text_config.layer_types = ["linear_attention"]
    model = Qwen3_8_FlashNextForConditionalGeneration.from_config(
        config, moe_config=_tiny_moe_config(config.text_config), backend=_backend()
    )
    return model, {}


def _use_torch_fallbacks(model: torch.nn.Module, monkeypatch: pytest.MonkeyPatch) -> int:
    """Route every GatedDeltaNet through the pure-torch convolution and chunked recurrence."""
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf_qwen3_5_moe
    from transformers.models.qwen3_next import modeling_qwen3_next as hf_qwen3_next

    monkeypatch.setattr(cp_linear_attn, "causal_conv1d_fn", inspect.unwrap(hf_qwen3_5_moe.causal_conv1d_fn))
    monkeypatch.setattr(
        cp_linear_attn,
        "torch_chunk_gated_delta_rule",
        inspect.unwrap(hf_qwen3_5_moe.torch_chunk_gated_delta_rule),
    )
    n = 0
    for module in model.modules():
        if hasattr(module, "conv1d") and hasattr(module, "chunk_gated_delta_rule"):
            module.causal_conv1d_fn = None
            hf = hf_qwen3_5_moe if isinstance(module, cp_linear_attn.CPAwareGatedDeltaNet) else hf_qwen3_next
            module.chunk_gated_delta_rule = inspect.unwrap(hf.torch_chunk_gated_delta_rule)
            n += 1
    return n


@pytest.mark.cpu_only
@pytest.mark.parametrize("build", [_build_qwen3_5_moe, _build_qwen3_8_flash_next])
def test_initialize_weights_from_poisoned_memory_runs_forward_backward(build, monkeypatch: pytest.MonkeyPatch) -> None:
    _check_initialize_weights(build, torch.device("cpu"), monkeypatch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Qwen3-Next places its rotary embedding on CUDA")
def test_qwen3_next_initialize_weights_from_poisoned_memory_runs_forward_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _check_initialize_weights(_build_qwen3_next, torch.device("cuda"), monkeypatch)


def _check_initialize_weights(build, device: torch.device, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    with device:
        model, forward_kwargs = build()
    # Stand-in for uninitialized memory of a checkpoint-free (meta / to_empty) build.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))
    model.initialize_weights(buffer_device=device, dtype=torch.float32)
    uninitialized = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    assert not uninitialized, uninitialized
    # HF-style normal(0, initializer_range=0.02) conv init, not torch's kaiming default (std about 0.29 for kernel 4)
    for name, parameter in model.named_parameters():
        if name.endswith("linear_attn.conv1d.weight"):
            assert 0.01 < parameter.std().item() < 0.04, (name, parameter.std().item())

    assert _use_torch_fallbacks(model, monkeypatch) == 1
    model.train()
    forward_kwargs = {k: v.to(device) for k, v in forward_kwargs.items()}
    output = model(input_ids=torch.randint(1, 64, (2, 6), device=device), **forward_kwargs)
    logits = getattr(output, "logits", output)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()

    grads = {n: p.grad for n, p in model.named_parameters()}
    conv = [n for n in grads if n.endswith("linear_attn.conv1d.weight")]
    embed = [n for n in grads if n.endswith("embed_tokens.weight")]
    assert len(conv) == 1 and len(embed) == 1, (conv, embed)
    for name in conv + embed:
        assert grads[name] is not None, name
        assert torch.isfinite(grads[name]).all(), name
        assert grads[name].abs().sum() > 0, name
