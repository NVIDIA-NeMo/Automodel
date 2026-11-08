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

"""Qwen3-MoE fused rotary position, precision and padded-storage regressions."""

import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3MoeConfig

from nemo_automodel.components.attention.utils import preprocess_args_and_kwargs_for_attn
from nemo_automodel.components.distributed.context_parallel.utils import make_cp_batch_for_te
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding, position_ids_to_freqs_cis
from nemo_automodel.components.models.qwen3_moe.layers import Qwen3MoeAttention
from nemo_automodel.components.models.qwen3_moe.model import Qwen3MoeModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual TE rotary CUDA")


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Compare complete tensors rather than sampled elements.

    Args:
        actual: Candidate tensor with arbitrary axes.
        expected: Reference tensor with the same shape and dtype.

    Returns:
        FP64 relative L2 error over all tensor elements.
    """
    a, b = actual.detach().double(), expected.detach().double()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _model(fusion: bool, factor: float, dtype: torch.dtype) -> Qwen3MoeModel:
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=512,
        intermediate_size=768,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        num_experts=2,
        num_experts_per_tok=1,
        mlp_only_layers=[0],
        max_position_embeddings=int(32768 * factor),
        torch_dtype=dtype,
        rope_theta=1_000_000.0,
        attention_dropout=0.0,
        rope_scaling=(
            {
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": 32768,
                "beta_slow": 1.0,
                "beta_fast": 32.0,
            }
            if factor != 1.0
            else None
        ),
    )
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch",
        dispatcher="torch",
        rope_fusion=fusion,
        enable_hf_state_dict_adapter=False,
    )
    model = Qwen3MoeModel(config, backend).cuda()
    model.init_weights(torch.device("cuda"))
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("case", ["contiguous", "different_positions", "yarn"])
def test_qwen_rotary_fusion_outputs_and_parameter_gradients(
    case: str, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing only fusion preserves Qwen outputs and all dense-block gradients."""
    pytest.importorskip("transformer_engine")
    # This FP32 component test isolates rotary arithmetic from TF32 GEMM rounding.
    # The full BF16 distributed validation retains its production runtime settings.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(4358)
    factor = 4.0 if case == "yarn" else 1.0
    fused, eager = _model(True, factor, dtype), _model(False, factor, dtype)
    eager.load_state_dict(fused.state_dict())
    positions = torch.arange(37, device="cuda")[None].expand(2, -1)
    if case == "different_positions":
        positions = torch.stack((positions[0] + 8192, positions[1] * 2 + 16384))
    tokens = torch.randint(0, 128, (2, 37), device="cuda")
    with (
        sdpa_kernel(SDPBackend.MATH),
        patch(
            "nemo_automodel.components.models.qwen3_moe.layers.preprocess_args_and_kwargs_for_attn",
            wraps=preprocess_args_and_kwargs_for_attn,
        ) as prepare_attention,
    ):
        actual = fused(tokens, position_ids=positions)
        expected = eager(tokens, position_ids=positions)
        upstream = torch.randn_like(actual)
        (actual * upstream).sum().backward()
        (expected * upstream).sum().backward()
    # Inspect actual post-rotation Q/K delivered to attention, in addition to
    # whole-block results. The old BF16 staged rounding fails this tighter gate.
    assert len(prepare_attention.call_args_list) == 2
    for index in (0, 1):
        actual_rotary = prepare_attention.call_args_list[0].args[index]
        expected_rotary = prepare_attention.call_args_list[1].args[index]
        if dtype == torch.bfloat16:
            torch.testing.assert_close(actual_rotary, expected_rotary, rtol=0, atol=0)
        else:
            assert _relative_error(actual_rotary, expected_rotary) < 1e-6
    tolerance = 5e-5 if dtype == torch.float32 else 5e-3
    assert _relative_error(actual, expected) < tolerance
    for (name, parameter), (reference_name, reference_parameter) in zip(
        fused.named_parameters(), eager.named_parameters()
    ):
        assert name == reference_name
        assert parameter.grad is not None and reference_parameter.grad is not None
        assert _relative_error(parameter.grad, reference_parameter.grad) < tolerance, name


def test_qwen_cp1_padded_documents_preserve_valid_outputs_and_gradients() -> None:
    """Native CP1 physical padding must not change valid-token rotary positions."""
    pytest.importorskip("transformer_engine")
    torch.manual_seed(4358)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0, 4, 5, 6, 7, 8, 0, 0, 0]], device="cuda"),
        "labels": torch.tensor([[2, 3, 4, -100, 5, 6, 7, 8, 9, -100, -100, -100]], device="cuda"),
        "position_ids": torch.tensor([[0, 1, 2, 0, 0, 1, 2, 3, 4, 0, 0, 0]], device="cuda"),
        "seq_lens": torch.tensor([[3, 5]], device="cuda"),
        "seq_lens_padded": torch.tensor([[4, 8]], device="cuda"),
        "qkv_format": "thd",
    }
    prepared = make_cp_batch_for_te(None, batch)
    assert prepared["cu_seqlens"].tolist() == [0, 3, 8]
    assert prepared["cu_seqlens_padded"].tolist() == [0, 4, 12]
    valid = ~prepared["padding_mask"]
    config = Qwen3MoeConfig(
        hidden_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        torch_dtype=torch.bfloat16,
        attention_dropout=0.0,
    )
    backend = BackendConfig(
        attn="te",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch",
        dispatcher="torch",
        rope_fusion=True,
    )
    fused = Qwen3MoeAttention(config, backend).cuda()
    fused.init_weights(torch.device("cuda"))
    eager = copy.deepcopy(fused)
    eager.backend = replace(backend, rope_fusion=False)
    rotary = RotaryEmbedding(128, 1_000_000, torch.float32, device=torch.device("cuda"))
    kwargs = {
        name: prepared[name]
        for name in ("cu_seqlens", "cu_seqlens_padded", "max_seqlen", "pad_between_seqs", "qkv_format")
    }
    x = torch.randn(12, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xe = x.detach().clone().requires_grad_()
    actual = fused(x, freqs_cis=position_ids_to_freqs_cis(rotary, prepared["position_ids"], "thd", True), **kwargs)
    expected = eager(xe, freqs_cis=position_ids_to_freqs_cis(rotary, prepared["position_ids"], "thd", False), **kwargs)
    upstream = torch.randn_like(actual)
    upstream[~valid] = 0
    actual.backward(upstream)
    expected.backward(upstream)
    assert _relative_error(actual[valid], expected[valid]) < 5e-3
    assert _relative_error(x.grad[valid], xe.grad[valid]) < 5e-3
    for (name, parameter), (_, reference) in zip(fused.named_parameters(), eager.named_parameters()):
        assert _relative_error(parameter.grad, reference.grad) < 5e-3, name
