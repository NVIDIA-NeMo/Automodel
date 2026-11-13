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

"""Compiled Inkling attention and the fused output loss: CUDA forward/backward gates."""

import copy
from collections.abc import Callable, Iterator

import pytest
import torch
from torch.nn import functional as F

from nemo_automodel.components.loss.utils import calculate_loss
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.attention import AttentionSources, Inkling2Attention
from nemo_automodel.components.models.inkling2.loss import Inkling2FusedLinearCrossEntropy
from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM
from tests.unit_tests.models.inkling2.test_precision import initialize_nonuniform_queries
from tests.unit_tests.models.inkling2.test_reference import tiny_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Compiled FlexAttention requires CUDA")


@pytest.fixture(autouse=True)
def precision_context() -> Iterator[None]:
    # Each case creates a different model geometry; isolate Dynamo guards so
    # the matrix does not exhaust the cache shared by PyTorch code objects.
    torch._dynamo.reset()
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def compare_gradients(native: torch.Tensor, expected: torch.Tensor, *, bf16: bool, name: str) -> float:
    """Compare every value in a gradient tensor.

    Args:
        native: Gradient tensor with arbitrary shape.
        expected: Reference gradient tensor with the same shape.
        bf16: Whether to apply the established BF16 reduction tolerances.
        name: Parameter name reported on failure.

    Returns:
        Relative L2 difference as a Python scalar.
    """
    assert torch.isfinite(native).all() and torch.isfinite(expected).all(), name
    difference = native.float() - expected.float()
    relative = (difference.norm() / expected.float().norm().clamp_min(5e-4)).item()
    if bf16:
        assert difference.abs().max() <= 5e-4 or relative <= 0.08, (name, relative)
    else:
        torch.testing.assert_close(native, expected, rtol=3e-4, atol=2e-5, msg=name)
    return relative


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layer", [0, 5])
def test_attention_padding_window_relative_gradients_and_repeat(
    dtype: torch.dtype, layer: int, record_property: Callable[[str, float], None]
) -> None:
    torch.manual_seed(471 + layer)
    cfg = tiny_config()
    cfg.num_attention_heads = cfg.swa_num_attention_heads = 6
    cfg.num_key_value_heads = cfg.swa_num_key_value_heads = 1  # Released model's 6:1 GQA ratio.
    cfg.head_dim = cfg.swa_head_dim = 32
    cfg.sliding_window_size, cfg.rel_extent = 67, 35
    cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold = 0.7, 0.15
    eager = Inkling2Attention(cfg, layer, dtype=dtype).cuda()
    eager.reset_parameters(0.2)
    native = Inkling2Attention(cfg, layer, dtype=dtype, backend="flex").cuda()
    native.load_state_dict(eager.state_dict())
    valid = torch.zeros(3, 139, device="cuda", dtype=torch.bool)
    valid[0, :135], valid[1, 7:137] = True, True  # Last row entirely padded.
    positions = (valid.long().cumsum(-1) - 1).clamp_min(0) + 21
    base = [torch.randn(3, 139, 16, dtype=dtype, device="cuda") for _ in range(1 if layer == 0 else 3)]
    expected_inputs = [x.clone().requires_grad_() for x in base]
    wanted = eager(AttentionSources(*(expected_inputs * 3 if layer == 0 else expected_inputs)), valid, positions)
    upstream = torch.randn_like(wanted)
    wanted.backward(upstream)
    repeats = []
    for repeat in range(2):
        native.zero_grad(set_to_none=True)
        inputs = [x.clone().requires_grad_() for x in base]
        actual = native(AttentionSources(*(inputs * 3 if layer == 0 else inputs)), valid, positions)
        torch.testing.assert_close(
            actual,
            wanted,
            rtol=0.02 if dtype == torch.bfloat16 else 3e-5,
            atol=0.003 if dtype == torch.bfloat16 else 2e-6,
        )
        assert torch.count_nonzero(actual[2]) == 0
        actual.backward(upstream)
        errors = []
        for (name, parameter), (_, other) in zip(native.named_parameters(), eager.named_parameters(), strict=True):
            assert parameter.grad is not None and other.grad is not None, name
            errors.append(compare_gradients(parameter.grad, other.grad, bf16=dtype == torch.bfloat16, name=name))
        for actual_input, expected_input in zip(inputs, expected_inputs, strict=True):
            compare_gradients(actual_input.grad, expected_input.grad, bf16=dtype == torch.bfloat16, name="input")
        repeats.append({name: p.grad.clone() for name, p in native.named_parameters()})
        if repeat == 0:
            record_property("output_max_abs", (actual.float() - wanted.float()).abs().max().item())
            record_property("gradient_max_relative_l2", max(errors))
    mismatches = [name for name in repeats[0] if not torch.equal(repeats[0][name], repeats[1][name])]
    assert not mismatches, mismatches


@pytest.mark.parametrize("seed", [17, 41, 93])
@pytest.mark.parametrize("dtype,experts", [("float32", "torch"), ("bfloat16", "torch"), ("bfloat16", "torch_mm")])
def test_thirteen_layer_training_parity(
    seed: int, dtype: str, experts: str, record_property: Callable[[str, float], None]
) -> None:
    torch.manual_seed(seed)
    cfg = tiny_config(dtype=dtype)
    if experts == "torch_mm":
        cfg.hidden_size, cfg.intermediate_size, cfg.n_routed_experts = 64, 64, 16
    eager = Inkling2ForCausalLM(cfg).cuda()
    initialize_nonuniform_queries(eager)
    backend = BackendConfig(attn="flex", linear="torch", rms_norm="torch", experts=experts, dispatcher="torch")
    native = Inkling2ForCausalLM(cfg, backend=backend).cuda()
    native.load_state_dict(eager.state_dict())
    ids = torch.randint(0, 37, (1, 9), device="cuda")
    expected = eager(ids)
    out = native(ids, output_hidden_states=True)
    a, b = out.logits[..., :37], expected[..., :37]
    difference = (a - b).abs().detach().flatten()
    assert difference.max() < (0.01 if dtype == "bfloat16" else 1e-5)
    assert F.cosine_similarity(a.detach().flatten(), b.detach().flatten(), dim=0) > 0.9999
    targets = torch.roll(ids, -1, 1)
    targets[:, -1] = -100
    if dtype == "bfloat16":
        loss_fn = Inkling2FusedLinearCrossEntropy(37)
        loss = calculate_loss(loss_fn, hidden_states=out.hidden_states, labels=targets, model=native, num_label_tokens=8)
    else:
        # The fused kernel takes BF16 operands only; FP32 runs keep the model's own FP32 projection.
        loss = F.cross_entropy(out.logits.flatten(0, 1), targets.flatten())
    expected_loss = F.cross_entropy(expected.flatten(0, 1), targets.flatten())
    torch.testing.assert_close(loss, expected_loss, rtol=0.002 if dtype == "bfloat16" else 2e-6, atol=1e-5)
    loss.backward()
    expected_loss.backward()
    errors = []
    for (name, parameter), (_, other) in zip(native.named_parameters(), eager.named_parameters(), strict=True):
        errors.append(compare_gradients(parameter.grad, other.grad, bf16=dtype == "bfloat16", name=name))
    record_property("logit_max_abs", difference.max().item())
    record_property("logit_mean_abs", difference.mean().item())
    record_property("logit_p95_abs", difference.quantile(0.95).item())
    record_property("gradient_max_relative_l2", max(errors))
    record_property("parameter_tensors_checked", len(list(native.parameters())))


@pytest.mark.parametrize("empty", [False, True])
def test_fused_loss_takes_shifted_labels_token_weights_and_one_normalization(empty: bool) -> None:
    torch.manual_seed(9)
    model = Inkling2ForCausalLM(tiny_config(dtype="bfloat16")).cuda()
    reference = copy.deepcopy(model)
    # This is the training batch contract: targets are shifted before model/loss.
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [0, 0, 7, 8, 9, 10]], device="cuda")
    ids, targets = tokens[:, :-1], tokens[:, 1:].clone()
    valid = torch.tensor([[True] * 5, [False, False, True, True, True]], device="cuda")
    targets[1, :2] = -100
    if empty:
        targets.fill_(-100)
    out = model(ids, attention_mask=valid, logits_to_keep=1, output_hidden_states=True)
    assert out.logits.shape == (2, 1, 40) and out.hidden_states.shape == (2, 5, 16)
    assert out.hidden_states.dtype == torch.bfloat16
    assert out.logits.dtype == torch.float32 and torch.isneginf(out.logits[..., 37:]).all()
    loss_fn = Inkling2FusedLinearCrossEntropy(37)
    count = int((targets != -100).sum())
    multipliers = torch.linspace(0.5, 1.5, 10, device="cuda").reshape(2, 5)
    actual = calculate_loss(
        loss_fn,
        hidden_states=out.hidden_states,
        labels=targets,
        model=model,
        num_label_tokens=count,
        loss_weights=multipliers,
    )
    full = reference(ids, attention_mask=valid)
    per_token = F.cross_entropy(full.flatten(0, 1), targets.flatten(), reduction="none").reshape(2, 5)
    expected = (per_token * multipliers).sum() / max(count, 1)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    actual.backward()
    expected.backward()
    for (name, parameter), (_, other) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            compare_gradients(parameter.grad, other.grad, bf16=True, name=name)


def test_mask_changes_and_activation_recomputation() -> None:
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(82)
    cfg = tiny_config(dtype="bfloat16")
    native = Inkling2Attention(cfg, 5, dtype=torch.bfloat16, backend="flex").cuda()
    native.reset_parameters(0.1)
    eager = Inkling2Attention(cfg, 5, dtype=torch.bfloat16).cuda()
    eager.load_state_dict(native.state_dict())
    source = torch.randn(1, 17, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    positions = torch.arange(17, device="cuda").unsqueeze(0)
    for start, end in [(0, 17), (5, 17), (0, 11), (0, 17)]:
        valid = (positions >= start) & (positions < end)
        eager.zero_grad(set_to_none=True)
        expected = eager(AttentionSources(source, source, source), valid, positions)
        native.zero_grad(set_to_none=True)
        actual = checkpoint(native, AttentionSources(source, source, source), valid, positions, use_reentrant=False)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
        upstream = torch.randn_like(actual)
        actual.backward(upstream)
        expected.backward(upstream)
        for (name, parameter), (_, other) in zip(native.named_parameters(), eager.named_parameters(), strict=True):
            compare_gradients(parameter.grad, other.grad, bf16=True, name=name)
