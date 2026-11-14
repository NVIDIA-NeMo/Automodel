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

"""BF16 compute policy, independent model/gradient comparisons and dtype lifecycle."""

from collections.abc import Callable, Iterator

import pytest
import torch
from torch.nn import functional as F

from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.inkling2.layers import ResidualState
from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM

from . import reference
from .test_reference import tiny_config

# Fixed before measuring: BF16 kernel shape/reduction changes may round gradients
# differently. Compare each parameter, with a small absolute floor for near-zero tensors.
LOGIT_MAX_ABS = 1e-2
LOGIT_MIN_COSINE = 0.9999
GRADIENT_RELATIVE_L2 = 0.08
GRADIENT_ABSOLUTE_FLOOR = 5e-4
# A BF16 residual stream moves the FP32 router's selection scores by up to about
# 4e-4 in these tiny models, while the reference's top-k cutoff gaps are typically
# 5e-3. Only two experts closer than this at the cutoff may swap.
ROUTING_TIE_GAP = 5e-4
DEVICES = [
    pytest.param("cpu", marks=pytest.mark.cpu_only),
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")),
]


@pytest.fixture(autouse=True)
def precision_context() -> Iterator[None]:
    previous = torch.get_num_threads()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.set_num_threads(previous)
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32


def initialize_nonuniform_queries(model: Inkling2ForCausalLM) -> None:
    """Exercise learned depth weights and nontrivial router ordering."""
    with torch.no_grad():
        model.output_query.weight.normal_(std=0.2)
        for layer in model.layers.values():
            if layer.attn_query is not None:
                layer.attn_query.weight.normal_(std=0.2)
            layer.mlp_query.weight.normal_(std=0.2)
            layer.mlp.gate.e_score_correction_bias.uniform_(-0.03, 0.03)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", [17, 41, 93])
def test_bf16_outputs_routes_and_all_gradients(
    seed: int, device: str, record_property: Callable[[str, float], None]
) -> None:
    torch.manual_seed(seed)
    cfg = tiny_config(dtype="bfloat16")
    model = Inkling2ForCausalLM(cfg).to(device=device)
    initialize_nonuniform_queries(model)
    parameters = dict(model.named_parameters())
    oracle = {k: v.detach().clone().requires_grad_(k in parameters) for k, v in model.state_dict().items()}
    ids = torch.randint(0, cfg.unpadded_vocab_size, (1, 9), device=device)
    partials, routes = [], []

    def capture_partial(
        module: torch.nn.Module,
        inputs: tuple[ResidualState, torch.Tensor, torch.Tensor],
        output: ResidualState,
    ) -> None:
        partials.append(output.partial.detach())

    def capture_route(
        module: torch.nn.Module, inputs: tuple[torch.Tensor], output: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        routes.append(output[1].detach())

    handles = [layer.register_forward_hook(capture_partial) for layer in model.layers.values()]
    handles += [layer.mlp.gate.register_forward_hook(capture_route) for layer in model.layers.values()]
    try:
        actual = model(ids)
    finally:
        for handle in handles:
            handle.remove()
    expected = reference.forward(cfg, oracle, ids)
    a, b = actual[..., :37], expected.logits[..., :37]
    error = (a - b).detach().abs().flatten()
    cosine = F.cosine_similarity(a.detach().flatten(), b.detach().flatten(), dim=0).item()
    assert actual.dtype == torch.float32 and all(t.dtype == torch.bfloat16 for t in partials)
    assert error.max() < LOGIT_MAX_ABS
    assert cosine > LOGIT_MIN_COSINE
    for index, (left, right) in enumerate(zip(partials, expected.partials, strict=True)):
        torch.testing.assert_close(left, right, rtol=0.03, atol=0.003, msg=f"layer {index}")
    flips = 0
    for index, (left, right, choice) in enumerate(
        zip(routes, expected.routing_ids, expected.routing_scores, strict=True)
    ):
        # Routing selects a set of experts; the ordering within that set is not
        # semantic. Every token must select the reference experts, except that the
        # BF16 residual stream can swap the two experts straddling the top-k cutoff
        # when their reference selection scores are closer than the noise BF16 inputs
        # add to the FP32 router. Logits and every parameter gradient are checked below.
        for token, (selected, wanted) in enumerate(zip(left.tolist(), right.tolist(), strict=True)):
            added, dropped = sorted(set(selected) - set(wanted)), sorted(set(wanted) - set(selected))
            if not added and not dropped:
                continue
            assert len(added) == len(dropped) == 1, (index, token, added, dropped)
            gap = (choice[token, dropped[0]] - choice[token, added[0]]).item()
            assert 0.0 <= gap <= ROUTING_TIE_GAP, (index, token, added, dropped, gap)
            flips += 1
    upstream = torch.randn_like(a)
    (a * upstream).sum().backward()
    (b * upstream).sum().backward()
    relative_errors, maximum_errors = [], []
    for name, parameter in parameters.items():
        native, wanted = parameter.grad, oracle[name].grad
        assert native is not None and wanted is not None, name
        assert torch.isfinite(native).all() and torch.isfinite(wanted).all(), name
        difference = native.float() - wanted.float()
        denominator = wanted.float().norm().clamp_min(GRADIENT_ABSOLUTE_FLOOR)
        relative = (difference.norm() / denominator).item()
        assert difference.abs().max() <= GRADIENT_ABSOLUTE_FLOOR or relative <= GRADIENT_RELATIVE_L2, (name, relative)
        relative_errors.append(relative)
        maximum_errors.append(difference.abs().max().item())
    record_property("logit_max_abs", error.max().item())
    record_property("logit_mean_abs", error.mean().item())
    record_property("logit_p95_abs", error.quantile(0.95).item())
    record_property("logit_cosine", cosine)
    record_property("gradient_max_abs", max(maximum_errors))
    record_property("gradient_max_relative_l2", max(relative_errors))
    record_property("parameter_tensors_checked", len(parameters))
    record_property("routing_tie_flips", flips)


@pytest.mark.cpu_only
def test_dtype_policy_survives_shared_cast_and_meta_materialization() -> None:
    model = Inkling2ForCausalLM(tiny_config())
    initialize_nonuniform_queries(model)
    protected = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if any(
            part in k for part in ("attn_query", "mlp_query", "output_query", "q_l2_clip", "rel_l2_clip", ".mlp.gate.")
        )
    }
    cast_model_to_dtype(model, torch.bfloat16)
    for name, value in model.state_dict().items():
        assert value.dtype == (torch.float32 if name in protected else torch.bfloat16), name
        if name in protected:
            torch.testing.assert_close(value, protected[name], rtol=0, atol=0)
    with torch.device("meta"):
        empty = Inkling2ForCausalLM(tiny_config(dtype="bfloat16"))
    empty.to_empty(device="cpu")
    empty.initialize_weights()
    assert all(torch.isfinite(p).all() for p in empty.parameters())
    assert {k: v.dtype for k, v in model.state_dict().items()} == {k: v.dtype for k, v in empty.state_dict().items()}
    assert torch.isfinite(empty(torch.tensor([[1, 2, 3]]))[..., :37]).all()


@pytest.mark.parametrize("device", DEVICES)
def test_outer_autocast_and_padding_preserve_explicit_precision(device: str) -> None:
    torch.manual_seed(51)
    model = Inkling2ForCausalLM(tiny_config(dtype="bfloat16")).to(device=device)
    ids = torch.tensor([[1, 3, 5, 7, 9]], device=device)
    plain = model(ids)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        under_autocast = model(ids)
    torch.testing.assert_close(plain, under_autocast, rtol=0, atol=0)
    padded = torch.tensor([[39, 39, 1, 3, 5, 7, 9, 38]], device=device)
    valid = torch.tensor([[False, False, True, True, True, True, True, False]], device=device)
    actual = model(padded, attention_mask=valid)
    torch.testing.assert_close(actual[:, 2:7], plain, rtol=0.02, atol=0.002)
    assert torch.isneginf(actual[..., 37:]).all()
