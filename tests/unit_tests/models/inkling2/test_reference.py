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

"""Observable semantic, gradient, configuration and lifecycle checks on CPU."""

import copy
import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.attention import (
    AttentionSources,
    Inkling2Attention,
)
from nemo_automodel.components.models.inkling2.config import (
    Inkling2Config,
    Inkling2TextConfig,
)
from nemo_automodel.components.models.inkling2.layers import (
    L2Clip,
    ResidualState,
    ShortConvolution,
    capped_swiglu,
    residual_mix,
)
from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM
from nemo_automodel.components.moe.layers import Gate

from . import reference

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tiny_config(*, dtype: str = "float32") -> Inkling2TextConfig:
    return Inkling2TextConfig(
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
        dtype=dtype,
        architectures=["Inkling2ForCausalLM"],
    )


@pytest.mark.parametrize("seed", [17, 41, 93])
def test_thirteen_layer_outputs_and_all_parameter_gradients(
    seed: int, record_property: Callable[[str, float], None]
) -> None:
    torch.manual_seed(seed)
    config = tiny_config()
    model = Inkling2ForCausalLM(config)
    # Nonzero pseudo-queries exercise learned selection rather than only uniform mixtures.
    with torch.no_grad():
        model.output_query.weight.normal_(std=0.2)
        for layer in model.layers.values():
            if layer.attn_query is not None:
                layer.attn_query.weight.normal_(std=0.2)
            layer.mlp_query.weight.normal_(std=0.2)
            layer.mlp.gate.e_score_correction_bias.uniform_(-0.03, 0.03)
    parameters = dict(model.named_parameters())
    oracle_weights = {
        name: value.detach().clone().requires_grad_(name in parameters) for name, value in model.state_dict().items()
    }
    ids = torch.randint(0, config.unpadded_vocab_size, (1, 9))
    partials: list[torch.Tensor] = []

    def capture_partial(
        module: torch.nn.Module,
        inputs: tuple[ResidualState, torch.Tensor, torch.Tensor],
        output: ResidualState,
    ) -> None:
        """Capture each [batch, sequence, hidden] partial for reference comparison."""
        partials.append(output.partial.detach())

    handles = [layer.register_forward_hook(capture_partial) for layer in model.layers.values()]
    try:
        actual = model(ids)
    finally:
        for handle in handles:
            handle.remove()
    expected = reference.forward(config, oracle_weights, ids)
    torch.testing.assert_close(actual, expected.logits, rtol=2e-4, atol=1e-5)
    for index, (native, wanted) in enumerate(zip(partials, expected.partials, strict=True)):
        torch.testing.assert_close(native, wanted, rtol=2e-4, atol=1e-5, msg=f"layer {index}")
    record_property("partial_max_abs", max((a - b).abs().max().item() for a, b in zip(partials, expected.partials)))
    error = (actual[..., :37] - expected.logits[..., :37]).abs().detach().flatten()
    record_property("logit_mean_abs", error.mean().item())
    record_property("logit_p95_abs", error.quantile(0.95).item())
    record_property("logit_max_abs", error.max().item())
    assert [i for i, layer in enumerate(model.layers.values()) if not layer.is_local] == [5, 11]
    assert [i for i, layer in enumerate(model.layers.values()) if layer.is_boundary] == [0, 6, 12]
    upstream = torch.randn_like(actual[..., :37])
    (actual[..., :37] * upstream).sum().backward()
    (expected.logits[..., :37] * upstream).sum().backward()
    max_gradient_error = 0.0
    for name, parameter in parameters.items():
        wanted = oracle_weights[name].grad
        assert wanted is not None and parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, wanted, rtol=3e-4, atol=2e-5, msg=name)
        max_gradient_error = max(max_gradient_error, (parameter.grad - wanted).abs().max().item())
    record_property("gradient_max_abs", max_gradient_error)
    assert model.layers["5"].attn.k_sconv.weight_proj.weight.grad.abs().sum() > 0
    assert model.layers["11"].attn.v_sconv.weight_proj.weight.grad.abs().sum() > 0
    assert model.layers["0"].mlp.experts.gate_and_up_projs.grad.abs().sum() > 0


def test_capped_interleaved_activation_and_analytic_gradients() -> None:
    pairs = torch.tensor(
        [
            [-100.0, 60.0],
            [-7.0, -40.0],
            [-1.0, 2.0],
            [0.0, 1.0],
            [1.0, -3.0],
            [7.0, 40.0],
            [100.0, 60.0],
        ]
    )
    gate_up = pairs.reshape(1, -1).requires_grad_()
    weight = torch.tensor([[0.7]], requires_grad=True)
    actual = capped_swiglu(gate_up, weight, gate_softcap=4, up_softcap=25)
    g, u = pairs[:, 0].double(), pairs[:, 1].double()
    tg, tu, sg = torch.tanh(g / 4), torch.tanh(u / 25), torch.sigmoid(g)
    expected = 4 * tg * sg * 25 * tu * 0.7
    torch.testing.assert_close(actual[0].double(), expected, rtol=2e-6, atol=1e-5)
    upstream = torch.tensor([[0.1, -0.5, 0.7, 1.3, -1.0, 0.2, 0.3]])
    (actual * upstream).sum().backward()
    dg = ((1 - tg**2) * sg + 4 * tg * sg * (1 - sg)) * 25 * tu * 0.7
    du = 4 * tg * sg * (1 - tu**2) * 0.7
    derivatives = torch.stack([dg, du], -1).reshape(1, -1)
    wanted = derivatives * upstream.double().repeat_interleave(2, -1)
    torch.testing.assert_close(gate_up.grad.double(), wanted, rtol=3e-6, atol=1e-5)
    torch.testing.assert_close(
        weight.grad.double(),
        (expected / 0.7 * upstream.double()[0]).sum().reshape(1, 1),
        rtol=3e-6,
        atol=1e-5,
    )


@pytest.mark.parametrize("dynamic", [False, True])
def test_convolution_tap_order_and_gradient_parity(dynamic: bool) -> None:
    torch.manual_seed(4)
    module = ShortConvolution(4, 4, dtype=torch.float32, dynamic_input_size=3 if dynamic else 0, heads=2)
    module.reset_parameters(0.3)
    x = torch.randn(2, 6, 4, requires_grad=True)
    valid = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    context = torch.randn(2, 6, 3, requires_grad=True) if dynamic else None
    clone_x = x.detach().clone().requires_grad_()
    clone_context = context.detach().clone().requires_grad_() if context is not None else None
    weights = {"conv." + name: p.detach().clone().requires_grad_() for name, p in module.named_parameters()}
    actual = module(x, valid, dynamic_input=context)
    expected = reference.convolution(
        clone_x,
        weights,
        "conv",
        width=4,
        valid=valid,
        dynamic_input=clone_context,
        heads=2,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    upstream = torch.randn_like(actual)
    (actual * upstream).sum().backward()
    (expected * upstream).sum().backward()
    torch.testing.assert_close(x.grad, clone_x.grad, rtol=2e-6, atol=2e-6)
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(parameter.grad, weights["conv." + name].grad, rtol=3e-6, atol=3e-6)
    if context is not None:
        torch.testing.assert_close(context.grad, clone_context.grad, rtol=3e-6, atol=3e-6)
    # An impulse distinguishes lag-zero from oldest-first serialized coefficients.
    impulse = torch.zeros(1, 5, 4)
    impulse[:, 0] = 1
    mask = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        if dynamic:
            module.weight_proj.weight.zero_()
            module.static_w.copy_(torch.arange(1.0, 5.0)[:, None].expand(4, 4))
        else:
            module.weight.copy_(torch.arange(1.0, 5.0).expand(4, 1, 4))
    y = module(impulse, mask, dynamic_input=torch.zeros(1, 5, 3) if dynamic else None)
    values = [2.0, 2.0, 3.0, 4.0, 0.0] if dynamic else [5.0, 3.0, 2.0, 1.0, 0.0]
    expected_impulse = torch.tensor(values) / (3 if dynamic else 5**0.5)
    torch.testing.assert_close(y[0, :, 0], expected_impulse)


def test_common_router_bias_is_only_for_selection() -> None:
    model = Inkling2ForCausalLM(tiny_config())
    gate = model.layers["0"].mlp.gate
    assert type(gate) is Gate
    assert model.layers["0"].mlp.shared_experts is None
    x = torch.ones(2, 16)
    with torch.no_grad():
        gate.weight.copy_(torch.linspace(-0.1, 0.1, 12)[:, None].expand(12, 16))
        gate.e_score_correction_bias.copy_(torch.linspace(1.0, -1.0, 12))
    actual, indices, _ = gate(x, torch.ones(2, dtype=torch.bool), None)
    scores = (x @ gate.weight.T).sigmoid()
    selected = (scores + gate.e_score_correction_bias).argsort(-1, descending=True)[:, :8]
    torch.testing.assert_close(indices, selected)
    wanted = scores.gather(-1, selected)
    torch.testing.assert_close(actual, 8 * wanted / wanted.sum(-1, keepdim=True))
    torch.testing.assert_close(actual.sum(-1), torch.full((2,), 8.0))


def test_l2_clipping_then_length_scaling_and_scale_gradient() -> None:
    module = L2Clip(1, 2, 5, learned_scale=True)
    x = torch.tensor([[[[6.0, 8.0]], [[0.3, 0.4]]]], requires_grad=True)
    tau = torch.tensor([[2.0, 3.0]])
    actual = module(x, tau=tau)
    torch.testing.assert_close(actual, torch.tensor([[[[6.0, 8.0]], [[0.9, 1.2]]]]), rtol=1e-6, atol=1e-6)
    actual[..., 0].sum().backward()
    assert torch.isfinite(x.grad).all() and module.scale.grad.abs().sum() > 0


@pytest.mark.parametrize("normalize", [False, True])
def test_residual_mixture_is_token_local_and_immutable(normalize: bool) -> None:
    torch.manual_seed(12)
    sources = [torch.randn(2, 5, 7, requires_grad=True) for _ in range(3)]
    copies = [x.detach().clone() for x in sources]
    queries = torch.randn(3, 7, requires_grad=True)
    state = ResidualState(tuple(sources[:-1]), sources[-1])
    actual = residual_mix(state, queries, normalize_values=normalize)
    expected = torch.stack([reference.mix(sources, q, normalize) for q in queries])
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    gradients = torch.autograd.grad(actual[:, 0, 2].square().sum(), sources)
    for source, original, gradient in zip(sources, copies, gradients):
        assert torch.equal(source, original)
        assert torch.count_nonzero(gradient[1]) == 0
        assert torch.count_nonzero(gradient[0, :2]) == 0
        assert torch.count_nonzero(gradient[0, 3:]) == 0


@pytest.mark.parametrize("layer", [0, 5])
def test_attention_window_relative_extent_and_gradients(layer: int) -> None:
    torch.manual_seed(53)
    cfg = tiny_config()
    module = Inkling2Attention(cfg, layer, dtype=torch.float32)
    module.reset_parameters(0.15)
    inputs = [torch.randn(1, 10, 16, requires_grad=True) for _ in range(3)]
    if layer == 0:
        inputs = [inputs[0]] * 3
    valid = torch.ones(1, 10, dtype=torch.bool)
    positions = torch.arange(10)[None]
    native = module(AttentionSources(*inputs), valid, positions)
    weights = {"attn." + name: p for name, p in module.named_parameters()}
    expected = reference.attention(cfg, weights, "attn", inputs, local=layer == 0, valid=valid, positions=positions)
    torch.testing.assert_close(native, expected, rtol=2e-5, atol=2e-6)
    upstream = torch.randn_like(native)
    params = list(module.parameters())
    actual_grad = torch.autograd.grad((native * upstream).sum(), params, retain_graph=True)
    expected_grad = torch.autograd.grad((expected * upstream).sum(), params)
    for actual, wanted in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, wanted, rtol=3e-5, atol=3e-6)
    # Alter a future token and verify causality through both attention and convolution.
    changed = [x.detach().clone() for x in inputs]
    for x in changed:
        x[:, -1] += 100
    altered = module(AttentionSources(*changed), valid, positions)
    torch.testing.assert_close(native[:, :-1], altered[:, :-1], rtol=0, atol=0)


def test_padding_vocab_mask_and_native_checkpoint_resume(
    tmp_path: Path, record_property: Callable[[str, float], None]
) -> None:
    torch.manual_seed(77)
    cfg = tiny_config()
    model = Inkling2ForCausalLM(cfg)
    tokens = torch.randint(0, 37, (1, 8))
    base = model(tokens)
    padded = torch.cat((torch.full((1, 2), 39), tokens, torch.full((1, 3), 38)), -1)
    mask = torch.tensor([[0, 0] + [1] * 8 + [0] * 3])
    actual = model(padded, attention_mask=mask)
    torch.testing.assert_close(actual[:, 2:10], base, rtol=2e-5, atol=2e-6)
    assert torch.isneginf(actual[..., 37:]).all()
    assert actual[..., :37].abs().max() < 30
    all_padding = model(torch.zeros(1, 3, dtype=torch.long), attention_mask=torch.zeros(1, 3))
    torch.testing.assert_close(all_padding[..., :37], torch.zeros(1, 3, 37))
    all_padding[..., :37].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)

    def step(current: Inkling2ForCausalLM, optim: torch.optim.Optimizer) -> float:
        optim.zero_grad(set_to_none=True)
        logits = current(tokens)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten())
        loss.backward()
        optim.step()
        return loss.detach().item()

    initial = step(model, optimizer)
    cfg.save_pretrained(tmp_path)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        tmp_path / "native.pt",
    )
    restored_cfg = AutoConfig.from_pretrained(tmp_path)
    restored = Inkling2ForCausalLM.from_config(restored_cfg)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.005)
    saved = torch.load(tmp_path / "native.pt", weights_only=True)
    restored.load_state_dict(saved["model"], strict=True)
    restored_optimizer.load_state_dict(saved["optimizer"])
    torch.testing.assert_close(restored(tokens), model(tokens), rtol=0, atol=0)
    assert restored.embed_tokens.weight is not restored.lm_head.weight
    next_loss = step(model, optimizer)
    assert step(restored, restored_optimizer) == next_loss
    for left, right in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for _ in range(3):
        final = step(model, optimizer)
    assert final < initial
    record_property("initial_tiny_batch_loss", initial)
    record_property("final_tiny_batch_loss", final)


def test_checkpoint_style_config_resolution_and_capabilities(tmp_path: Path) -> None:
    raw = json.loads((Path(__file__).parent / "checkpoint_config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(raw))
    config = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(config, Inkling2Config)
    text = config.text_config
    assert isinstance(text, Inkling2TextConfig)
    assert (text.n_routed_experts, text.num_experts_per_tok, text.n_shared_experts) == (
        384,
        8,
        0,
    )
    assert text.attn_res_normalize_values and text.use_value_norm_no_gamma
    assert not text.pre_norm_pre_affine and not text.final_norm_pre_affine
    assert (
        text.vocab_size,
        text.unpadded_vocab_size,
        text.final_logit_softcapping,
    ) == (201024, 200058, 30)
    assert config.vision_config.native_patch_size == 20
    assert config.mtp_config["target_layer_ids"][-1] == 48
    assert ModelRegistry.has_custom_model("Inkling2ForCausalLM")
    assert ModelRegistry.has_custom_model("Inkling2ForConditionalGeneration")
    caps = Inkling2ForCausalLM.ModelCapabilities()
    assert caps.supports_ep
    assert caps.supports_pp
    assert caps.supports_cp and not caps.supports_tp


def test_materialization_and_rejected_contracts() -> None:
    cfg = tiny_config()
    with torch.device("meta"):
        model = Inkling2ForCausalLM(cfg)
    model.to_empty(device="cpu")
    model.initialize_weights(torch.device("cpu"))
    result = model(torch.ones(1, 3, dtype=torch.long))
    assert torch.isfinite(result[..., :37]).all()
    with pytest.raises(ValueError, match="eager/flex attention"):
        Inkling2ForCausalLM(cfg, backend=BackendConfig(attn="sdpa"))
    with pytest.raises(ValueError, match="increase by one"):
        model(torch.ones(1, 3, dtype=torch.long), attention_mask=torch.tensor([[1, 0, 1]]))
    with pytest.raises(ValueError, match="increase by one"):
        model(torch.ones(1, 3, dtype=torch.long), position_ids=torch.tensor([[0, 2, 3]]))
    tied = copy.deepcopy(cfg)
    tied.tie_word_embeddings = True
    with pytest.raises(NotImplementedError, match="separate input and output embeddings"):
        Inkling2ForCausalLM(tied)
    with pytest.raises(ValueError, match="requires float32 or bfloat16"):
        Inkling2ForCausalLM(tiny_config(dtype="float16"))
    alias = Inkling2TextConfig(model_max_length=512)
    assert alias.max_position_embeddings == 512


@pytest.mark.parametrize("layer", [0, 5])
@pytest.mark.parametrize("head_dim", [4, 16])
def test_relative_only_attention_probabilities_do_not_depend_on_head_width(layer: int, head_dim: int) -> None:
    """With zero content, relative logits log(3), log(2), sink 0 give weights 3:2:1."""
    cfg = tiny_config()
    cfg.head_dim = cfg.swa_head_dim = head_dim
    module = Inkling2Attention(cfg, layer, dtype=torch.float32)
    q = torch.zeros(1, 2, module.heads, head_dim)
    k = torch.zeros(1, 2, module.kv_heads, head_dim)
    v = torch.zeros_like(k)
    v[:, 0, :, 0] = 1
    profiles = torch.zeros(1, module.heads, 2, module.rel_extent)
    profiles[:, :, 1, 0] = torch.log(torch.tensor(2.0))
    profiles[:, :, 1, 1] = torch.log(torch.tensor(3.0))
    profiles.requires_grad_()
    valid = torch.ones(1, 2, dtype=torch.bool)
    positions = torch.arange(2)[None]
    result = module._eager(
        q,
        k,
        v,
        profiles=profiles,
        sink=torch.zeros(1, 2, module.heads),
        valid=valid,
        positions=positions,
        key_valid=valid,
        key_positions=positions,
    )
    torch.testing.assert_close(result[0, 1, :, 0], torch.full((module.heads,), 0.5))
    gradient = torch.autograd.grad(result[0, 1, 0, 0], profiles)[0]
    torch.testing.assert_close(gradient[0, 0, 1, :2], torch.tensor([-1 / 6, 1 / 4]))


@pytest.mark.parametrize(
    ("fixture", "model_type", "scaling_floor"),
    [("checkpoint_config.json", "inkling2_mm_model", 8192), ("released_config.json", "inkling2", 128000)],
)
def test_checkpoint_autoconfig_round_trip(tmp_path: Path, fixture: str, model_type: str, scaling_floor: int) -> None:
    """Load actual checkpoint metadata through AutoConfig without downloading weights."""
    metadata = (Path(__file__).parent / fixture).read_text()
    (tmp_path / "config.json").write_text(metadata)
    config = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(config, Inkling2Config)
    assert config.model_type == model_type
    assert config.architectures == ["Inkling2ForConditionalGeneration"]
    assert config.text_config.log_scaling_n_floor == scaling_floor
    assert config.text_config.n_layers_per_attn_res_block == 6
    assert config.text_config.n_layers_per_attnres_block == 6
    assert config.text_config.max_position_embeddings == 1048576
    config.save_pretrained(tmp_path / "saved")
    restored = AutoConfig.from_pretrained(tmp_path / "saved")
    assert type(restored) is type(config)
    assert restored.text_config.to_dict() == config.text_config.to_dict()


@pytest.mark.parametrize("field", ["n_layers_per_attn_res_block", "n_layers_per_attnres_block"])
def test_residual_block_config_alias(field: str) -> None:
    """A non-default block size must reach model boundaries through either spelling."""
    config = Inkling2TextConfig(**{field: 4})
    assert config.n_layers_per_attn_res_block == config.n_layers_per_attnres_block == 4
    serialized = config.to_dict()
    assert serialized["n_layers_per_attn_res_block"] == 4
    assert "n_layers_per_attnres_block" not in serialized
    assert Inkling2TextConfig.from_dict(serialized).n_layers_per_attnres_block == 4
    with pytest.raises(ValueError, match="Conflicting attention residual block sizes"):
        Inkling2TextConfig(n_layers_per_attn_res_block=4, n_layers_per_attnres_block=6)
