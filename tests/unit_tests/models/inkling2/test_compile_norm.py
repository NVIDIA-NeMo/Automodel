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

"""Compiled FP32 chains of Inkling2 (BackendConfig.compile_norm) against their eager functions."""

import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from functools import partial

import pytest
import torch
from torch import nn
from torch._dynamo.eval_frame import _debug_get_cache_entry_list, reset_code

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2 import layers
from nemo_automodel.components.models.inkling2.config import Inkling2TextConfig
from nemo_automodel.components.models.inkling2.layers import (
    L2Clip,
    MessageController,
    ResidualQuery,
    ResidualState,
    ShortConvolution,
    accumulate,
    capped_swiglu,
    close_block,
    gate_context,
    normalize,
    value_norm,
)
from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="The chains are compiled for CUDA")

# Module-level names that compile_cores rebinds, with the flags that record it.
CHAINS = (
    "_mix",
    "_mix_normalize",
    "_control",
    "_normalize",
    "_value_norm",
    "_gate",
    "_accumulate",
    "_close",
    "_swiglu",
    "_dynamic_convolution",
    "_clip",
    "_scaled_clip",
    "_CORES_COMPILED",
    "_ACTIVATION_COMPILED",
)
BATCH, SEQUENCE, HIDDEN = 2, 19, 48
Leaves = list[torch.Tensor]
Outputs = torch.Tensor | tuple[torch.Tensor, ...]


@pytest.fixture(autouse=True)
def exact_reference_products() -> Iterator[None]:
    """Keep TF32 out of the eager reference's matrix products."""
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = before


@pytest.fixture
def restore_chains() -> Iterator[None]:
    """Restore the module-level chains after a test, so that later tests run the eager arithmetic."""
    eager = {name: vars(layers)[name] for name in CHAINS}
    yield
    vars(layers).update(eager)


@pytest.fixture
def eager_chains(restore_chains: None) -> Callable[[], AbstractContextManager[None]]:
    """Compile the chains for one test; the returned context manager runs its body on the eager ones."""
    eager = {name: vars(layers)[name] for name in CHAINS}
    layers.compile_cores()
    compiled = {name: vars(layers)[name] for name in CHAINS}

    @contextmanager
    def use_eager() -> Iterator[None]:
        vars(layers).update(eager)
        try:
            yield
        finally:
            vars(layers).update(compiled)

    return use_eager


def _randn(*shape: int, seed: int, scale: float = 1.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    values = scale * torch.randn(*shape, generator=torch.Generator().manual_seed(seed))
    return values.to("cuda", dtype).requires_grad_(True)


def _documents() -> tuple[torch.Tensor, torch.Tensor]:
    """Return validity and document positions [BATCH, SEQUENCE]: two documents, then one with padding."""
    valid = torch.ones(BATCH, SEQUENCE, dtype=torch.bool)
    positions = torch.arange(SEQUENCE).repeat(BATCH, 1)
    positions[0, 8:] -= 8
    valid[1, 13:] = False
    positions[1, 13:] = 0
    return valid.cuda(), positions.cuda()


def _outputs_and_gradients(run: Callable[[], Outputs], leaves: Leaves) -> tuple[Leaves, Leaves]:
    """Return the outputs of run and the gradients of one weighted sum of them."""
    result = run()
    outputs = list(result) if isinstance(result, tuple) else [result]
    generator = torch.Generator().manual_seed(11)
    loss = sum((out.float() * torch.randn(out.shape, generator=generator).to(out.device)).sum() for out in outputs)
    gradients = torch.autograd.grad(loss, leaves)
    return [out.detach() for out in outputs], list(gradients)


def _assert_compiled_matches_eager(
    run: Callable[[], Outputs],
    leaves: Leaves,
    use_eager: Callable[[], AbstractContextManager[None]],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> None:
    """Compare outputs and gradients of run on the compiled chains with the eager ones.

    Args:
        run: Calls the public layer or function under test on the leaves.
        leaves: Tensors whose gradients are compared.
        use_eager: Context manager of the eager_chains fixture.
        rtol: Relative tolerance.
        atol: Absolute tolerance as a fraction of the largest eager value of each tensor.
    """
    assert layers._CORES_COMPILED
    # One chain serves many source counts, query counts and dtypes; the model's layers call the
    # chains with the same allowance of compiled variants.
    with layers.compile_variants():
        outputs, gradients = _outputs_and_gradients(run, leaves)
    with use_eager():
        assert not layers._CORES_COMPILED
        expected_outputs, expected_gradients = _outputs_and_gradients(run, leaves)
    for got, wanted in zip(outputs + gradients, expected_outputs + expected_gradients, strict=True):
        assert got.dtype == wanted.dtype and got.shape == wanted.shape
        scale = wanted.float().abs().max().clamp_min(1e-6).item()
        torch.testing.assert_close(got.float(), wanted.float(), rtol=rtol, atol=atol * scale)


def test_compile_norm_defaults_to_false_and_chains_are_eager() -> None:
    assert BackendConfig().compile_norm is False
    assert layers._mix is layers._mix_core and layers._normalize is layers._normalize_core
    assert layers._swiglu is layers._swiglu_core
    assert not layers._CORES_COMPILED and not layers._ACTIVATION_COMPILED


def test_compile_norm_rebinds_the_chains_once(restore_chains: None) -> None:
    config = Inkling2TextConfig(
        hidden_size=16,
        num_hidden_layers=13,
        vocab_size=40,
        unpadded_vocab_size=37,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        d_rel=3,
        rel_extent=4,
        sliding_window_size=7,
        n_routed_experts=12,
        num_experts_per_tok=8,
        intermediate_size=8,
        log_scaling_n_floor=4,
        dtype="float32",
        architectures=["Inkling2ForCausalLM"],
    )

    def backend(compile_norm: bool) -> BackendConfig:
        return BackendConfig(
            attn="eager",
            linear="torch",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
            compile_norm=compile_norm,
        )

    Inkling2ForCausalLM(config, backend=backend(False))
    assert layers._mix is layers._mix_core and not layers._CORES_COMPILED
    Inkling2ForCausalLM(config, backend=backend(True))
    assert layers._CORES_COMPILED and layers._ACTIVATION_COMPILED
    rebound = {name: vars(layers)[name] for name in CHAINS}
    assert all(rebound[name] is not vars(layers)[name + "_core"] for name in CHAINS if name.islower())
    # A second model shares the first one's compiled chains.
    Inkling2ForCausalLM(config, backend=backend(True))
    assert all(vars(layers)[name] is rebound[name] for name in CHAINS)


def test_depth_mixture_compiles_the_variants_of_a_48_layer_model(
    restore_chains: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One chain holds more than 64 variants: the chains raise at the limit instead of running eagerly.

    A 48-layer model mixes 0 to 23 completed blocks, and every count is one variant of the chain
    in each class of calls; the three classes here differ in gradient mode and value normalization.
    Dynamo alone traces and guards the variants; no kernel is generated.
    """
    monkeypatch.setattr(torch, "compile", partial(torch.compile, backend="eager"))
    layers.compile_cores()
    norm = nn.RMSNorm(8, eps=1e-6)
    blocks = [torch.randn(1, 2, 8, generator=torch.Generator().manual_seed(seed)) for seed in range(24)]
    code = layers._mix_normalize_core.__code__
    # Start from an empty cache, and leave none of these variants to the tests that follow.
    reset_code(code)
    try:
        with layers.compile_variants():
            for gradients, normalize_values in ((True, False), (False, False), (True, True)):
                query = ResidualQuery(8, 1, normalize_values=normalize_values)
                with torch.set_grad_enabled(gradients):
                    for count in range(len(blocks)):
                        (mixed,) = query(ResidualState(tuple(blocks[:count]), blocks[-1]), (norm,))
                        assert mixed.shape == (1, 2, 8) and torch.isfinite(mixed).all()
        assert len(_debug_get_cache_entry_list(code)) == 3 * len(blocks)
    finally:
        reset_code(code)


@needs_gpu
@pytest.mark.parametrize("fused_norm", [False, True])
@pytest.mark.parametrize("normalize_values", [False, True])
@pytest.mark.parametrize("queries", [1, 3])
@pytest.mark.parametrize("blocks", [0, 1, 4])
def test_depth_mixture(
    blocks: int,
    queries: int,
    normalize_values: bool,
    fused_norm: bool,
    eager_chains: Callable[[], AbstractContextManager[None]],
) -> None:
    query = ResidualQuery(HIDDEN, queries, normalize_values=normalize_values).cuda()
    with torch.no_grad():
        query.weight.copy_(_randn(queries, HIDDEN, seed=1, scale=0.3))
    norms = tuple(nn.RMSNorm(HIDDEN, eps=1e-6).cuda() for _ in range(queries))
    for index, norm in enumerate(norms):
        with torch.no_grad():
            norm.weight.copy_(1 + _randn(HIDDEN, seed=20 + index, scale=0.1))
    state = ResidualState(
        tuple(_randn(BATCH, SEQUENCE, HIDDEN, seed=2 + index) for index in range(blocks)),
        _randn(BATCH, SEQUENCE, HIDDEN, seed=40),
    )
    leaves = [*state.blocks, state.partial, query.weight]
    if fused_norm:
        leaves += [norm.weight for norm in norms]
    _assert_compiled_matches_eager(lambda: query(state, norms if fused_norm else None), leaves, eager_chains)


@needs_gpu
def test_depth_mixture_with_bf16_storage(eager_chains: Callable[[], AbstractContextManager[None]]) -> None:
    query = ResidualQuery(HIDDEN, 3, normalize_values=False).cuda()
    with torch.no_grad():
        query.weight.copy_(_randn(3, HIDDEN, seed=1, scale=0.3))
    norms = tuple(nn.RMSNorm(HIDDEN, eps=1e-6, dtype=torch.bfloat16).cuda() for _ in range(3))
    state = ResidualState(
        tuple(_randn(BATCH, SEQUENCE, HIDDEN, seed=2 + index, dtype=torch.bfloat16) for index in range(4)),
        _randn(BATCH, SEQUENCE, HIDDEN, seed=40),
    )
    leaves = [*state.blocks, state.partial, query.weight, *(norm.weight for norm in norms)]
    # Outputs and block gradients are rounded to BF16: allow two units in the last place.
    _assert_compiled_matches_eager(lambda: query(state, norms), leaves, eager_chains, rtol=2e-2, atol=1e-2)


def _normalize_case(dtype: torch.dtype) -> tuple[Callable[[], Outputs], Leaves]:
    norm = nn.RMSNorm(HIDDEN, eps=1e-6, dtype=dtype).cuda()
    with torch.no_grad():
        norm.weight.copy_(1 + _randn(HIDDEN, seed=1, scale=0.1))
    x = _randn(BATCH, SEQUENCE, HIDDEN, seed=2)
    return (lambda: normalize(x, norm)), [x, norm.weight]


def _value_norm_case(dtype: torch.dtype) -> tuple[Callable[[], Outputs], Leaves]:
    value = _randn(BATCH, SEQUENCE, 2, 8, seed=3, dtype=dtype)
    return (lambda: value_norm(value, 1e-6)), [value]


def _gate_case(dtype: torch.dtype) -> tuple[Callable[[], Outputs], Leaves]:
    context = _randn(BATCH, SEQUENCE, 4, 8, seed=4)
    gate = _randn(BATCH, SEQUENCE, 32, seed=5, dtype=dtype)
    valid, _ = _documents()
    return (lambda: gate_context(context, gate, valid)), [context, gate]


def _accumulate_case(dtype: torch.dtype) -> tuple[Callable[[], Outputs], Leaves]:
    partial = _randn(BATCH, SEQUENCE, HIDDEN, seed=6)
    update = _randn(BATCH, SEQUENCE, HIDDEN, seed=7, dtype=dtype)
    return (lambda: accumulate(partial, update)), [partial, update]


def _close_case(dtype: torch.dtype) -> tuple[Callable[[], Outputs], Leaves]:
    partial = _randn(BATCH, SEQUENCE, HIDDEN, seed=8)
    update = _randn(BATCH, SEQUENCE, HIDDEN, seed=9, dtype=dtype)
    return (lambda: close_block(partial, update)), [partial, update]


def _controller_case(dtype: torch.dtype) -> tuple[Callable[[], Outputs], Leaves]:
    controller = MessageController(HIDDEN, 4, dtype=dtype).cuda()
    controller.reset_parameters(0.3)
    destination = _randn(BATCH, SEQUENCE, HIDDEN, seed=10)
    message = _randn(BATCH, SEQUENCE, HIDDEN, seed=11)
    return (lambda: controller(destination, message)), [destination, message, *controller.parameters()]


def _clip_case(learned_scale: bool, with_tau: bool) -> Callable[[torch.dtype], tuple[Callable[[], Outputs], Leaves]]:
    def build(dtype: torch.dtype) -> tuple[Callable[[], Outputs], Leaves]:
        clip = L2Clip(4, 8, 2.5, learned_scale=learned_scale).cuda()
        # Head vectors of length about 2.8: some are clipped at 2.5 and some are not.
        x = _randn(BATCH, SEQUENCE, 4, 8, seed=12, dtype=dtype)
        tau = 1 + torch.rand(BATCH, SEQUENCE, generator=torch.Generator().manual_seed(13)).cuda() if with_tau else None
        leaves = [x]
        if learned_scale:
            with torch.no_grad():
                clip.scale.copy_(1 + _randn(4, 8, seed=14, scale=0.2))
            leaves.append(clip.scale)
        lengths = torch.linalg.vector_norm(x.detach().float(), dim=-1)
        assert (lengths > 2.6).any() and (lengths < 2.4).any()
        return (lambda: clip(x, tau=tau)), leaves

    return build


def _static_convolution_case(dtype: torch.dtype) -> tuple[Callable[[], Outputs], Leaves]:
    convolution = ShortConvolution(16, 4, dtype=dtype, heads=2).cuda()
    convolution.reset_parameters(0.3)
    valid, positions = _documents()
    x = _randn(BATCH, SEQUENCE, 16, seed=15, dtype=dtype)
    return (lambda: convolution(x, valid, positions=positions)), [x, *convolution.parameters()]


def _dynamic_convolution_case(dtype: torch.dtype) -> tuple[Callable[[], Outputs], Leaves]:
    convolution = ShortConvolution(16, 4, dtype=dtype, heads=2, dynamic_input_size=HIDDEN).cuda()
    convolution.reset_parameters(0.3)
    valid, positions = _documents()
    x = _randn(BATCH, SEQUENCE, 16, seed=16, dtype=dtype)
    dynamic_input = _randn(BATCH, SEQUENCE, HIDDEN, seed=17, dtype=dtype)

    def run() -> torch.Tensor:
        return convolution(x, valid, dynamic_input=dynamic_input, positions=positions)

    return run, [x, dynamic_input, *convolution.parameters()]


@needs_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "build",
    [
        _normalize_case,
        _value_norm_case,
        _gate_case,
        _accumulate_case,
        _close_case,
        _controller_case,
        _clip_case(learned_scale=False, with_tau=False),
        _clip_case(learned_scale=True, with_tau=True),
        _static_convolution_case,
        _dynamic_convolution_case,
    ],
    ids=[
        "normalize",
        "value_norm",
        "gate",
        "accumulate",
        "close",
        "controller",
        "clip",
        "scaled_clip_tau",
        "static_convolution",
        "dynamic_convolution",
    ],
)
def test_chain(
    build: Callable[[torch.dtype], tuple[Callable[[], Outputs], Leaves]],
    dtype: torch.dtype,
    eager_chains: Callable[[], AbstractContextManager[None]],
) -> None:
    run, leaves = build(dtype)
    # BF16 storage rounds outputs and gradients to BF16: allow two units in the last place.
    rtol, atol = (1e-4, 1e-5) if dtype == torch.float32 else (2e-2, 1e-2)
    _assert_compiled_matches_eager(run, leaves, eager_chains, rtol=rtol, atol=atol)


def _plain_capped_swiglu(gate_up: torch.Tensor, weights: torch.Tensor, gate_cap: float, up_cap: float) -> torch.Tensor:
    """The activation written out for autograd: capped gate times sigmoid gate times capped up, times the weight."""
    gate, up = gate_up[..., 0::2].float(), gate_up[..., 1::2].float()
    capped_gate = gate_cap * torch.tanh(gate / gate_cap)
    capped_up = up_cap * torch.tanh(up / up_cap)
    return (capped_gate * torch.sigmoid(gate) * capped_up * weights.float()).to(gate_up.dtype)


def _assert_activation_matches_autograd(device: str, dtype: torch.dtype, rtol: float, atol: float) -> None:
    generator = torch.Generator().manual_seed(21)
    # Gates and ups beyond their caps of 4 and 25 as well as below them.
    scale = torch.tensor([3.0, 20.0]).repeat(24)
    gate_up = (scale * torch.randn(37, 48, generator=generator)).to(device, dtype).requires_grad_(True)
    weights = torch.rand(37, 1, generator=generator).to(device).requires_grad_(True)
    upstream = torch.randn(37, 24, generator=generator).to(device)
    results = []
    for activation in (
        lambda: capped_swiglu(gate_up, weights, gate_softcap=4.0, up_softcap=25.0),
        lambda: _plain_capped_swiglu(gate_up, weights, 4.0, 25.0),
    ):
        out = activation()
        results.append((out.detach(), *torch.autograd.grad((out.float() * upstream).sum(), [gate_up, weights])))
    for got, wanted in zip(*results, strict=True):
        assert got.dtype == wanted.dtype
        limit = atol * wanted.float().abs().max().item()
        torch.testing.assert_close(got.float(), wanted.float(), rtol=rtol, atol=limit)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_activation_gradient_matches_autograd_of_the_plain_formula(dtype: torch.dtype) -> None:
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (2e-2, 1e-2)
    _assert_activation_matches_autograd("cpu", dtype, rtol, atol)


@needs_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_activation_gradient_matches_autograd_of_the_plain_formula(
    dtype: torch.dtype, eager_chains: Callable[[], AbstractContextManager[None]]
) -> None:
    assert layers._ACTIVATION_COMPILED
    rtol, atol = (1e-4, 1e-5) if dtype == torch.float32 else (2e-2, 1e-2)
    _assert_activation_matches_autograd("cuda", dtype, rtol, atol)


def _call_in_thread(call: Callable[[], Outputs], count: int) -> None:
    """Run call in a thread with the given intra-op thread count, then restore the caller's count as the default."""
    default = torch.get_num_threads()

    def body() -> None:
        torch.set_num_threads(count)
        call()
        torch.set_num_threads(default)

    thread = threading.Thread(target=body)
    thread.start()
    thread.join()


@needs_gpu
def test_second_thread_reuses_the_compiled_variant(eager_chains: Callable[[], AbstractContextManager[None]]) -> None:
    def variants() -> int:
        return len(_debug_get_cache_entry_list(layers._value_norm_core.__code__))

    count = torch.get_num_threads()
    value = _randn(BATCH, SEQUENCE, 2, 8, seed=3).detach()
    # The thread count is part of a compiled variant's guards: without the alignment of the
    # chain wrapper, a thread with another count compiles its own variant.
    unaligned = torch.compile(layers._value_norm_core, fullgraph=True)
    unaligned(value, 1e-6)
    before = variants()
    _call_in_thread(lambda: unaligned(value, 1e-6), count + 1)
    assert variants() == before + 1
    value_norm(value, 1e-6)
    before = variants()
    _call_in_thread(lambda: value_norm(value, 1e-6), count + 2)
    assert variants() == before
    assert torch.get_num_threads() == count
