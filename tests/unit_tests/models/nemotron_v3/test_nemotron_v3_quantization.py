# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import types
from collections import OrderedDict
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch
from torch import nn

from nemo_automodel.components.models.common.utils import (
    HAVE_TE,
    BackendConfig,
    TENvfp4Config,
    _forward_outside_fp8,
)
from nemo_automodel.components.models.nemotron_v3 import quantization as nemotron_quantization
from nemo_automodel.components.models.nemotron_v3.quantization import (
    NemotronSuperNvfp4Config,
    TEModulePrecision,
    TEPrecisionPlan,
    apply_precision_plan,
    resolve_precision_plan,
    te_gemm_modules,
)
from nemo_automodel.shared.import_utils import is_te_min_version

# The layout of ``NemotronSuperNvfp4Config`` against the 88-layer pattern of Nemotron 3 Super / 3.5 Super VL.
#
# The expected precision of every module is written out here from the layer pattern with plain loops; the test
# never asks the code under test which layer or class a name belongs to. Module names are the ones the model
# registers (``test_names_match_a_built_model`` builds a small model to pin them).


# ``layers_block_type`` of nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 (llm_config) as one symbol per
# layer: M = Mamba, E = MoE, * = attention.
SUPER_PATTERN = "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
TAIL = 14

ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
MOE = {
    "experts.gate_up_linear": "routed_experts",
    "experts.down_linear": "routed_experts",
    "shared_experts.up_proj": "shared_experts",
    "shared_experts.down_proj": "shared_experts",
    "fc1_latent_proj": "latent_proj",
    "fc2_latent_proj": "latent_proj",
}
RECIPE = {
    "mamba_in_proj": "nvfp4",
    "mamba_out_proj": "mxfp8",
    "routed_experts": "nvfp4",
    "shared_experts": "nvfp4",
    "attention": "high_precision",
    "latent_proj": "high_precision",
    "mtp": "high_precision",
}


def _set_layout(monkeypatch: pytest.MonkeyPatch, **changes: str | int) -> None:
    """Give the module under test another layout for this test: the published one with ``changes``."""
    layout = {"high_precision_tail_layers": TAIL, **RECIPE, **changes}
    monkeypatch.setattr(nemotron_quantization, "HIGH_PRECISION_TAIL_LAYERS", layout.pop("high_precision_tail_layers"))
    monkeypatch.setattr(nemotron_quantization, "PRECISION_BY_CLASS", layout)


def _expected(
    pattern: str, tail: int, prefix: str, block_infix: str, recipe: dict[str, str], moe_infix: str = ""
) -> dict[str, tuple[str, int | None, str, str]]:
    """GEMM module name -> (module class, layer, precision, scope), in ``named_modules`` order.

    ``block_infix`` is the checkpoint wrapper of every block (``activation_checkpointing: true``), ``moe_infix``
    the one inside every MoE mixer (``activation_checkpointing: moe``).
    """
    expected: dict[str, tuple[str, int | None, str, str]] = {}

    def add(name: str, module_class: str, layer: int | None, scope: str) -> None:
        in_tail = layer is not None and layer >= len(pattern) - tail
        expected[name] = (module_class, layer, "high_precision" if in_tail else recipe[module_class], scope)

    for layer, symbol in enumerate(pattern):
        mixer = f"{prefix}model.layers.{layer}.{block_infix}mixer"
        if symbol == "M":
            add(f"{mixer}.in_proj", "mamba_in_proj", layer, f"{mixer}.in_proj")
            add(f"{mixer}.out_proj", "mamba_out_proj", layer, f"{mixer}.out_proj")
        elif symbol == "*":
            for projection in ATTENTION:
                add(f"{mixer}.{projection}", "attention", layer, mixer)
        else:
            for module, module_class in MOE.items():
                add(f"{mixer}.{moe_infix}{module}", module_class, layer, f"{mixer}.{moe_infix}{module.split('.')[0]}")
    # One MTP depth of [attention, moe], with the fusion projection on its first block.
    attention_mixer = f"{prefix}mtp.layers.0.{block_infix}mixer"
    for projection in ATTENTION:
        add(f"{attention_mixer}.{projection}", "mtp", None, attention_mixer)
    add(f"{prefix}mtp.layers.0.{block_infix}eh_proj", "mtp", None, f"{prefix}mtp.layers.0.{block_infix}eh_proj")
    moe_mixer = f"{prefix}mtp.layers.1.{block_infix}mixer"
    for module in MOE:
        add(f"{moe_mixer}.{moe_infix}{module}", "mtp", None, f"{moe_mixer}.{moe_infix}{module.split('.')[0]}")
    return expected


def _resolved(plan: TEPrecisionPlan) -> dict[str, tuple[str, int | None, str, str]]:
    return {m.name: (m.module_class, m.layer, m.precision, m.scope) for m in plan.modules}


WRAPPER = "_checkpoint_wrapped_module."


@pytest.mark.cpu_only
@pytest.mark.parametrize("prefix", ["", "language_model."])
@pytest.mark.parametrize(("block_infix", "moe_infix"), [("", ""), (WRAPPER, ""), ("", WRAPPER)])
def test_published_layout_resolves_every_module(prefix: str, block_infix: str, moe_infix: str):
    """Text model and vision-language wrapper; no checkpoint wrappers, around the blocks, around the MoE mixers."""
    expected = _expected(SUPER_PATTERN, TAIL, prefix, block_infix, RECIPE, moe_infix)
    plan = resolve_precision_plan(list(expected), list(expected))
    assert _resolved(plan) == expected
    assert (plan.num_layers, plan.high_precision_tail_layers) == (88, 14)
    # Module counts, from the published layout: 34 + 6 Mamba, 33 + 7 MoE (two GEMMs each), 8 attention.
    assert plan.counts() == {
        "mamba_in_proj": {"nvfp4": 34, "mxfp8": 0, "high_precision": 6},
        "mamba_out_proj": {"nvfp4": 0, "mxfp8": 34, "high_precision": 6},
        "routed_experts": {"nvfp4": 66, "mxfp8": 0, "high_precision": 14},
        "shared_experts": {"nvfp4": 66, "mxfp8": 0, "high_precision": 14},
        "latent_proj": {"nvfp4": 0, "mxfp8": 0, "high_precision": 80},
        "attention": {"nvfp4": 0, "mxfp8": 0, "high_precision": 32},
        "mtp": {"nvfp4": 0, "mxfp8": 0, "high_precision": 11},
    }
    summary = plan.summary()
    assert "decoder layers: 88; layers 0-73 follow the precision of their module class" in summary
    assert "layers 74-87 run in high precision" in summary
    assert [line.split() for line in summary.splitlines() if line.startswith(("mamba_out_proj", "total"))] == [
        ["mamba_out_proj", "0", "34", "6"],
        ["total", "166", "34", "163"],
    ]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "changes, differing",
    [
        # A tail one layer short or long moves exactly the quantized GEMMs of layer 74 (MoE: routed and
        # shared experts) or of layer 73 (Mamba: both projections).
        ({"high_precision_tail_layers": 13}, 4),
        ({"high_precision_tail_layers": 15}, 2),
        ({"high_precision_tail_layers": 0}, 6 * 2 + 7 * 4),
        ({"mamba_out_proj": "nvfp4"}, 34),
        ({"mamba_in_proj": "high_precision"}, 34),
        ({"attention": "nvfp4"}, 7 * 4),
        ({"latent_proj": "nvfp4"}, 33 * 2),
        ({"routed_experts": "mxfp8"}, 33 * 2),
        ({"mtp": "nvfp4"}, 11),
    ],
)
def test_a_different_layout_is_told_apart_from_the_published_one(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, str | int], differing: int
):
    """Negative control: every wrong tail or module class changes the expected number of modules."""
    expected = _expected(SUPER_PATTERN, TAIL, "", "", RECIPE)
    _set_layout(monkeypatch, **changes)
    resolved = _resolved(resolve_precision_plan(list(expected), list(expected)))
    assert sum(resolved[name][2] != expected[name][2] for name in expected) == differing


@pytest.mark.cpu_only
def test_scopes_are_the_modules_that_select_the_precision():
    expected = _expected(SUPER_PATTERN, TAIL, "", "", RECIPE)
    scopes = resolve_precision_plan(list(expected), list(expected)).scopes()
    assert scopes["model.layers.0.mixer.in_proj"] == "nvfp4"
    assert scopes["model.layers.0.mixer.out_proj"] == "mxfp8"
    assert scopes["model.layers.1.mixer.experts"] == "nvfp4"
    assert scopes["model.layers.1.mixer.shared_experts"] == "nvfp4"
    assert scopes["model.layers.7.mixer"] == "high_precision"
    assert scopes["model.layers.73.mixer.out_proj"] == "mxfp8"
    assert scopes["model.layers.74.mixer.experts"] == "high_precision"
    assert scopes["model.layers.75.mixer.out_proj"] == "high_precision"
    assert scopes["model.layers.87.mixer.experts"] == "high_precision"
    assert scopes["mtp.layers.1.mixer.experts"] == "high_precision"
    # 80 Mamba projections, 8 attention mixers, 40 x (experts, shared expert, 2 latent), 6 in the MTP head.
    assert len(scopes) == 80 + 8 + 40 * 4 + 6
    # With the MoE mixers inside checkpoint wrappers the scopes are the modules the recompute calls again.
    wrapped = _expected(SUPER_PATTERN, TAIL, "", "", RECIPE, WRAPPER)
    wrapped_scopes = resolve_precision_plan(list(wrapped), list(wrapped)).scopes()
    assert wrapped_scopes["model.layers.1.mixer._checkpoint_wrapped_module.experts"] == "nvfp4"
    assert wrapped_scopes["model.layers.1.mixer._checkpoint_wrapped_module.shared_experts"] == "nvfp4"
    assert wrapped_scopes["model.layers.1.mixer._checkpoint_wrapped_module.fc2_latent_proj"] == "high_precision"
    assert wrapped_scopes["model.layers.7.mixer"] == "high_precision"
    assert len(wrapped_scopes) == len(scopes)
    assert not any(
        scope.endswith(("mixer", "_checkpoint_wrapped_module")) for scope in wrapped_scopes if "layers.1." in scope
    )


@pytest.mark.cpu_only
def test_models_it_does_not_understand_are_rejected(monkeypatch: pytest.MonkeyPatch):
    names = list(_expected("ME*M", 1, "", "", RECIPE))
    # The published tail of 14 layers does not fit a model of 4.
    with pytest.raises(ValueError, match="last 14 decoder layers in high precision, which exceeds the 4 decoder"):
        resolve_precision_plan(names, names)
    # The other checks, with a tail that fits the four layers.
    _set_layout(monkeypatch, high_precision_tail_layers=1)
    assert len(resolve_precision_plan(names, names).modules) == len(names)
    foreign = ["model.language_model.layers.0.mlp.experts.gate_up_linear", "lm_head"]
    with pytest.raises(ValueError, match="found no module named 'model.layers"):
        resolve_precision_plan(foreign, foreign)
    with pytest.raises(ValueError, match="found no module named 'model.layers"):
        resolve_precision_plan([], [])
    # ``mixer.up_proj`` is a projection of a dense MLP block (block type "mlp"), which the layout has no class for.
    extra = [*names, "model.layers.1.mixer.up_proj", "vision_projector.0"]
    with pytest.raises(
        ValueError, match=r"does not know 2 .*model.layers.1.mixer.up_proj, vision_projector.0\. .*dense MLP block"
    ):
        resolve_precision_plan(extra, extra)
    # A pipeline stage holds layers 2-3 only: the last layers of the model cannot be located.
    stage = [name for name in names if "layers.0." not in name and "layers.1." not in name]
    with pytest.raises(ValueError, match="whole decoder stack"):
        resolve_precision_plan(stage, stage)


@pytest.mark.cpu_only
def test_the_output_projection_has_no_class(monkeypatch: pytest.MonkeyPatch):
    """``lm_head`` is a torch linear: it is no entry of the plan, and a Transformer Engine one is not guessed."""
    names = list(_expected("ME*M", 1, "", "", RECIPE))
    _set_layout(monkeypatch, high_precision_tail_layers=1)
    for head in ("lm_head", "language_model.lm_head"):
        plan = resolve_precision_plan([*names, head], names)
        assert [module.name for module in plan.modules] == names
        with pytest.raises(ValueError, match=rf"does not know 1 .*: {head}\. "):
            resolve_precision_plan([*names, head], [*names, head])


@pytest.mark.cpu_only
def test_quantizing_a_module_without_transformer_engine_is_rejected(monkeypatch: pytest.MonkeyPatch):
    """A quantized class whose modules are plain torch modules would silently run in high precision."""
    names = list(_expected("ME*M", 1, "", "", RECIPE))
    _set_layout(monkeypatch, high_precision_tail_layers=1)
    # backend.mamba_linear="torch" (the default): the Mamba projections exist but are not Transformer Engine
    # modules. The error names the setting that builds them as such.
    torch_mamba = [name for name in names if not name.endswith(("in_proj", "out_proj"))]
    with pytest.raises(
        ValueError, match=r"quantizes 2 .*backend.mamba_linear='te'.*layers.0.mixer.in_proj \(mamba_in_proj: nvfp4\)"
    ):
        resolve_precision_plan(names, torch_mamba)
    # An expert backend without Transformer Engine grouped GEMMs: the container is there, its projections are not.
    torch_experts = [name for name in names if ".experts." not in name]
    modules = [*torch_experts, "model.layers.1.mixer.experts", "mtp.layers.1.mixer.experts"]
    with pytest.raises(ValueError, match=r"quantizes 1 .*model.layers.1.mixer.experts \(routed_experts: nvfp4\)"):
        resolve_precision_plan(modules, torch_experts)
    # The same container inside the checkpoint wrapper of its mixer.
    wrapped_names = list(_expected("ME*M", 1, "", "", RECIPE, WRAPPER))
    wrapped_torch_experts = [name for name in wrapped_names if ".experts." not in name]
    wrapped_modules = [*wrapped_torch_experts, f"model.layers.1.mixer.{WRAPPER}experts"]
    with pytest.raises(ValueError, match=r"quantizes 1 .*mixer._checkpoint_wrapped_module.experts \(routed_experts"):
        resolve_precision_plan(wrapped_modules, wrapped_torch_experts)
    # The tail keeps a module in high precision whatever backend built it.
    tail_mamba = [name for name in names if not name.startswith("model.layers.3.")]
    assert len(resolve_precision_plan(names, tail_mamba).modules) == len(names) - 2


@pytest.mark.cpu_only
@pytest.mark.parametrize("checkpointing", ["none", "blocks", "moe"])
def test_names_match_a_built_model(monkeypatch: pytest.MonkeyPatch, checkpointing: str):
    """The names the test spells out are the ones a Nemotron 3 model registers, also after ``apply_ac``."""
    from nemo_automodel.components.models.common import LMHead
    from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM
    from nemo_automodel.components.moe.parallelizer import apply_ac
    from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mtp import MockNemotronV3Config

    config = MockNemotronV3Config(
        num_hidden_layers=4,
        layers_block_type=["mamba", "moe", "attention", "mamba"],
        moe_latent_size=64,
        num_nextn_predict_layers=1,
        mtp_hybrid_override_pattern="*E",
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    model = NemotronHForCausalLM(config, backend)
    # The output projection is not built from ``backend.linear``, so it is never a Transformer Engine module.
    assert type(model.lm_head) is LMHead
    block_infix = moe_infix = ""
    if checkpointing != "none":
        # The production wrapping of the expert-parallel path; the MTP head here has one depth, so its MoE
        # block is wrapped like the decoder blocks.
        apply_ac(
            model,
            hidden_size=config.hidden_size,
            num_experts=config.n_routed_experts,
            moe_only=checkpointing == "moe",
        )
        block_infix, moe_infix = (WRAPPER, "") if checkpointing == "blocks" else ("", WRAPPER)
    built = {name for name, _ in model.named_modules()}
    # The torch expert backend keeps its weights on ``mixer.experts`` itself; Transformer Engine experts add
    # the two grouped projections below it.
    spelled_out = _expected("ME*M", 1, "", block_infix, RECIPE, moe_infix)
    expected = {name.replace(".gate_up_linear", "").replace(".down_linear", "") for name in spelled_out}
    assert expected <= built, sorted(expected - built)
    # The layout accepts exactly these names: every module of the built model that would be a Transformer
    # Engine GEMM module with ``backend.linear="te"`` gets a precision, in all three wrappings. The model has
    # four layers and torch experts, so the test shortens the tail and keeps the experts in high precision.
    gemm_names = [name for name in spelled_out if ".experts." not in name]
    _set_layout(monkeypatch, high_precision_tail_layers=1, routed_experts="high_precision")
    plan = resolve_precision_plan([name for name, _ in model.named_modules()], gemm_names)
    assert [module.name for module in plan.modules] == gemm_names
    assert "lm_head" in built and "lm_head" not in gemm_names
    assert set(plan.scopes()) <= built


# The forwards a plan binds, with a stand-in for Transformer Engine.


@contextmanager
def _fake_te_mxfp8_autocast():
    """Stand in for the Transformer Engine autocast, its global state, ``MXFP8BlockScaling`` and the GEMM modules.

    Yields the ``(enabled, recipe name)`` of every autocast entered. ``FP8GlobalStateManager.is_fp8_enabled()``
    follows the innermost autocast. ``transformer_engine.pytorch.Linear`` is a ``torch.nn.Linear`` subclass, so
    a test model can hold modules that count as Transformer Engine GEMM modules.
    """
    entered: list[tuple[bool, str | None]] = []
    active: list[bool] = []

    class MXFP8BlockScaling:
        pass

    @contextmanager
    def autocast(enabled: bool = True, recipe: MXFP8BlockScaling | None = None):
        entered.append((enabled, None if recipe is None else type(recipe).__name__))
        active.append(enabled)
        try:
            yield
        finally:
            active.pop()

    class FP8GlobalStateManager:
        @staticmethod
        def is_fp8_enabled() -> bool:
            return bool(active) and active[-1]

    class Linear(nn.Linear):
        pass

    class GroupedLinear(nn.Module):
        pass

    class LayerNormLinear(nn.Module):
        pass

    class LayerNormMLP(nn.Module):
        pass

    quantization = types.ModuleType("transformer_engine.pytorch.quantization")
    quantization.autocast = autocast
    quantization.FP8GlobalStateManager = FP8GlobalStateManager
    te_pytorch = types.ModuleType("transformer_engine.pytorch")
    te_pytorch.quantization = quantization
    te_pytorch.Linear = Linear
    te_pytorch.GroupedLinear = GroupedLinear
    te_pytorch.LayerNormLinear = LayerNormLinear
    te_pytorch.LayerNormMLP = LayerNormMLP
    recipe_module = types.ModuleType("transformer_engine.common.recipe")
    recipe_module.MXFP8BlockScaling = MXFP8BlockScaling
    te_common = types.ModuleType("transformer_engine.common")
    te_common.recipe = recipe_module
    te_root = types.ModuleType("transformer_engine")
    te_root.pytorch = te_pytorch
    te_root.common = te_common
    modules = {
        "transformer_engine": te_root,
        "transformer_engine.pytorch": te_pytorch,
        "transformer_engine.pytorch.quantization": quantization,
        "transformer_engine.common": te_common,
        "transformer_engine.common.recipe": recipe_module,
    }
    with patch("nemo_automodel.components.models.common.utils.HAVE_TE", True), patch.dict("sys.modules", modules):
        yield entered


def _plan_by_last_name(precisions: dict[str, str], gemm_module_names: list[str]) -> TEPrecisionPlan:
    """Test plan: precision by the last part of a GEMM module name; ``inner`` uses its parent as scope."""
    modules = tuple(
        TEModulePrecision(
            name=name,
            module_class=name.rsplit(".", 1)[-1],
            layer=None,
            precision=precisions[name.rsplit(".", 1)[-1]],
            scope=name.rsplit(".", 1)[0] if name.endswith(".inner") else name,
        )
        for name in gemm_module_names
    )
    return TEPrecisionPlan(modules=modules, num_layers=0, high_precision_tail_layers=0)


def test_scopes_get_the_forward_of_their_precision(caplog: pytest.LogCaptureFixture):
    x = torch.randn(2, 4)
    with _fake_te_mxfp8_autocast() as entered:
        from transformer_engine.pytorch import Linear
        from transformer_engine.pytorch.quantization import autocast

        model = nn.ModuleDict(
            {
                "quantized": Linear(4, 4),
                "eight_bit": Linear(4, 4),
                "block": nn.Sequential(OrderedDict(inner=Linear(4, 4), norm=nn.LayerNorm(4))),
                "torch_linear": nn.Linear(4, 4),
            }
        )
        expected = model.block(model.eight_bit(model.quantized(x)))
        plan = _plan_by_last_name(
            {"quantized": "nvfp4", "eight_bit": "mxfp8", "inner": "high_precision"},
            [name for name, _ in te_gemm_modules(model)],
        )
        with caplog.at_level("INFO", logger="nemo_automodel.components.models.nemotron_v3.quantization"):
            apply_precision_plan(model, plan)
            apply_precision_plan(model, plan)  # idempotent: one wrapper per scope
        with autocast(enabled=True):
            torch.testing.assert_close(model.block(model.eight_bit(model.quantized(x))), expected)
        # NVFP4 modules keep the outer recipe; MXFP8 and high-precision scopes switch it once each.
        assert entered == [(True, None), (True, "MXFP8BlockScaling"), (False, None)]
        # The binding follows the instance through a copy.
        clone = copy.deepcopy(model)
        assert clone.eight_bit.forward.__self__ is clone.eight_bit
        assert clone.block.forward.__self__ is clone.block
    assert "forward" in vars(model.block) and "forward" in vars(model.eight_bit)
    for untouched in (model.quantized, model.block.inner, model.block.norm, model.torch_linear):
        assert "forward" not in vars(untouched)
    summary = caplog.records[0].getMessage().splitlines()
    assert [line.split() for line in summary[1:]] == [
        ["module", "class", "nvfp4", "mxfp8", "high_precision"],
        ["quantized", "1", "0", "0"],
        ["eight_bit", "0", "1", "0"],
        ["inner", "0", "0", "1"],
        ["total", "1", "1", "1"],
    ]


def test_mxfp8_scopes_quantize_only_inside_an_enabled_autocast():
    """A forward without the Transformer Engine autocast, or inside a disabled one, stays in high precision."""
    x = torch.randn(2, 4)
    with _fake_te_mxfp8_autocast() as entered:
        from transformer_engine.pytorch import Linear
        from transformer_engine.pytorch.quantization import autocast

        model = nn.ModuleDict({"mlp": Linear(4, 4)})
        expected = model.mlp(x)
        apply_precision_plan(model, _plan_by_last_name({"mlp": "mxfp8"}, ["mlp"]))
        assert "forward" in vars(model.mlp)
        torch.testing.assert_close(model.mlp(x), expected)
        with autocast(enabled=False):
            torch.testing.assert_close(model.mlp(x), expected)
        assert entered == [(False, None)]
        with autocast(enabled=True):
            torch.testing.assert_close(model.mlp(x), expected)
        assert entered == [(False, None), (True, None), (True, "MXFP8BlockScaling")]


def test_plan_rejects_inconsistent_scopes():
    def module(name: str, precision: str, scope: str) -> TEModulePrecision:
        return TEModulePrecision(name=name, module_class="linear", layer=None, precision=precision, scope=scope)

    with pytest.raises(ValueError, match="not 'a.b' or one of its ancestors"):
        module("a.b", "nvfp4", "a.c")
    with pytest.raises(ValueError, match="different precisions"):
        TEPrecisionPlan(
            modules=(module("a.b", "nvfp4", "a"), module("a.c", "mxfp8", "a")),
            num_layers=0,
            high_precision_tail_layers=0,
        )
    with pytest.raises(ValueError, match="lies inside scope 'a'"):
        TEPrecisionPlan(
            modules=(module("a.b", "nvfp4", "a"), module("a.c.d", "nvfp4", "a.c")),
            num_layers=0,
            high_precision_tail_layers=0,
        )


# ``NemotronSuperNvfp4Config``: the ``te_nvfp4`` config that carries the layout.


def test_config_is_a_te_nvfp4_config_without_layout_settings():
    config = NemotronSuperNvfp4Config(stochastic_rounding=False)
    assert isinstance(config, TENvfp4Config) and config.name == "te_nvfp4"
    assert config.recipe_kwargs() == TENvfp4Config(stochastic_rounding=False).recipe_kwargs()
    # The layout keeps modules in high precision itself: no list of excluded names, not even the default one.
    assert config.exclude_fqns == ()
    with pytest.raises(
        ValueError, match=r"fixed precision layout and takes no te_nvfp4.exclude_fqns, got \['lm_head'\]"
    ):
        NemotronSuperNvfp4Config(exclude_fqns=["lm_head"])
    with pytest.raises(TypeError, match="mamba_out_proj"):
        NemotronSuperNvfp4Config(mamba_out_proj="nvfp4")
    backend = BackendConfig(te_nvfp4=config, experts="te", dispatcher="deepep")
    assert backend.te_quantization is config


def test_exclude_modules_applies_the_layout():
    """One Mamba layer below the tail of 14: NVFP4 input projection, MXFP8 output projection, the rest excluded."""
    with _fake_te_mxfp8_autocast():
        from transformer_engine.pytorch import Linear

        layers = nn.ModuleDict(
            {
                str(index): nn.ModuleDict({"mixer": nn.ModuleDict({"in_proj": Linear(4, 4), "out_proj": Linear(4, 4)})})
                for index in range(15)
            }
        )
        model = nn.ModuleDict({"model": nn.ModuleDict({"layers": layers})})
        config = NemotronSuperNvfp4Config()
        excluded = config.exclude_modules(model)
        assert config.exclude_modules(model) == excluded  # idempotent
    assert excluded == [
        f"model.layers.{index}.mixer.{projection}" for index in range(1, 15) for projection in ("in_proj", "out_proj")
    ]
    first = layers["0"].mixer
    assert "forward" not in vars(first.in_proj)
    assert vars(first.out_proj)["forward"].__func__ is nemotron_quantization._forward_in_mxfp8
    for index in range(1, 15):
        for projection in (layers[str(index)].mixer.in_proj, layers[str(index)].mixer.out_proj):
            assert vars(projection)["forward"].__func__ is _forward_outside_fp8


def test_exclude_modules_needs_transformer_engine_gemm_modules():
    with patch("nemo_automodel.components.models.common.utils.HAVE_TE", False):
        assert NemotronSuperNvfp4Config().exclude_modules(nn.Linear(2, 2)) == []
    with _fake_te_mxfp8_autocast():
        with pytest.raises(ValueError, match="has no Transformer Engine GEMM module"):
            NemotronSuperNvfp4Config().exclude_modules(nn.Linear(2, 2))


# ``NemotronSuperNvfp4Config`` forwards through real Transformer Engine modules (Blackwell GPUs only).


requires_te_nvfp4 = pytest.mark.skipif(
    not (
        HAVE_TE
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (10, 0)
        and is_te_min_version("2.19.0")
    ),
    reason="requires Transformer Engine 2.19+ and an NVFP4 GPU",
)


def _relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float((value.double() - reference.double()).norm() / reference.double().norm())


@requires_te_nvfp4
def test_modules_a_plan_gives_mxfp8_run_the_mxfp8_recipe_inside_the_nvfp4_autocast():
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch.quantization import autocast

    def build() -> nn.ModuleDict:
        # Transformer Engine modules keep the quantizers of their first forward, so every recipe gets its own copy.
        torch.manual_seed(0)
        layers = OrderedDict(
            (str(index), te.Linear(512, 512, bias=False, params_dtype=torch.bfloat16)) for index in range(2)
        )
        return nn.ModuleDict({"layers": nn.ModuleDict(layers)}).to("cuda")

    model, mxfp8_model = build(), build()
    x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    # ``layers.0`` in NVFP4, ``layers.1`` in MXFP8.
    plan = TEPrecisionPlan(
        modules=tuple(
            TEModulePrecision(name=name, module_class="linear", layer=None, precision=precision, scope=name)
            for name, precision in (("layers.0", "nvfp4"), ("layers.1", "mxfp8"))
        ),
        num_layers=0,
        high_precision_tail_layers=0,
    )
    apply_precision_plan(model, plan)
    with torch.no_grad():
        with TENvfp4Config().maybe_te_autocast():
            outputs = [model.layers[index](x) for index in ("0", "1")]
        with autocast(enabled=True, recipe=MXFP8BlockScaling()):
            mxfp8_outputs = [mxfp8_model.layers[index](x) for index in ("0", "1")]
        exact = [x.float() @ model.layers[index].weight.float().T for index in ("0", "1")]
    assert torch.equal(outputs[1], mxfp8_outputs[1])
    # The other module ran NVFP4: a different result with a larger error than MXFP8.
    assert not torch.equal(outputs[0], mxfp8_outputs[0])
    assert _relative_error(outputs[0], exact[0]) > 2 * _relative_error(mxfp8_outputs[0], exact[0])
