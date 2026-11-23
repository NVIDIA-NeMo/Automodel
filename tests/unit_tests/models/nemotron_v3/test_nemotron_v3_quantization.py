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

"""``NemotronV3PrecisionMap`` against the 88-layer pattern of Nemotron 3 Super / 3.5 Super VL.

The expected precision of every module is written out here from the layer pattern with plain loops; the test
never asks the map which layer or class a name belongs to. Module names are the ones the model registers
(``test_names_match_a_built_model`` builds a small model to pin them).
"""

from collections import Counter

import pytest

from nemo_automodel.components.models.common.te_precision import TEPrecisionPlan
from nemo_automodel.components.models.common.utils import TENvfp4Config
from nemo_automodel.components.models.nemotron_v3.quantization import NemotronV3PrecisionMap

pytestmark = pytest.mark.cpu_only

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


def _recipe_map(**changes: str | int) -> NemotronV3PrecisionMap:
    return NemotronV3PrecisionMap(**{"high_precision_tail_layers": TAIL, **RECIPE, **changes})


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


def test_pattern_is_the_published_layout():
    """40 Mamba, 40 MoE and 8 attention layers, attention at the published positions."""
    assert len(SUPER_PATTERN) == 88
    assert Counter(SUPER_PATTERN) == {"M": 40, "E": 40, "*": 8}
    assert [i for i, symbol in enumerate(SUPER_PATTERN) if symbol == "*"] == [7, 16, 25, 36, 47, 58, 69, 78]
    # Below the tail: 34 Mamba, 33 MoE, 7 attention. In the last 14 layers: 6 Mamba, 7 MoE, 1 attention.
    assert Counter(SUPER_PATTERN[:74]) == {"M": 34, "E": 33, "*": 7}
    assert Counter(SUPER_PATTERN[74:]) == {"M": 6, "E": 7, "*": 1}


WRAPPER = "_checkpoint_wrapped_module."


@pytest.mark.parametrize("prefix", ["", "language_model."])
@pytest.mark.parametrize(("block_infix", "moe_infix"), [("", ""), (WRAPPER, ""), ("", WRAPPER)])
def test_super_recipe_resolves_every_module(prefix: str, block_infix: str, moe_infix: str):
    """Text model and vision-language wrapper; no checkpoint wrappers, around the blocks, around the MoE mixers."""
    expected = _expected(SUPER_PATTERN, TAIL, prefix, block_infix, RECIPE, moe_infix)
    plan = _recipe_map().resolve(list(expected), list(expected))
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
def test_a_different_map_is_told_apart_from_the_recipe(changes: dict[str, str | int], differing: int):
    """Negative control: every wrong tail or module class changes the expected number of modules."""
    expected = _expected(SUPER_PATTERN, TAIL, "", "", RECIPE)
    resolved = _resolved(_recipe_map(**changes).resolve(list(expected), list(expected)))
    assert sum(resolved[name][2] != expected[name][2] for name in expected) == differing


def test_scopes_are_the_modules_that_select_the_precision():
    expected = _expected(SUPER_PATTERN, TAIL, "", "", RECIPE)
    scopes = _recipe_map().resolve(list(expected), list(expected)).scopes()
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
    wrapped_scopes = _recipe_map().resolve(list(wrapped), list(wrapped)).scopes()
    assert wrapped_scopes["model.layers.1.mixer._checkpoint_wrapped_module.experts"] == "nvfp4"
    assert wrapped_scopes["model.layers.1.mixer._checkpoint_wrapped_module.shared_experts"] == "nvfp4"
    assert wrapped_scopes["model.layers.1.mixer._checkpoint_wrapped_module.fc2_latent_proj"] == "high_precision"
    assert wrapped_scopes["model.layers.7.mixer"] == "high_precision"
    assert len(wrapped_scopes) == len(scopes)
    assert not any(
        scope.endswith(("mixer", "_checkpoint_wrapped_module")) for scope in wrapped_scopes if "layers.1." in scope
    )


def test_models_it_does_not_understand_are_rejected():
    names = list(_expected("ME*M", 1, "", "", RECIPE))
    recipe = _recipe_map(high_precision_tail_layers=1)
    assert len(recipe.resolve(names, names).modules) == len(names)
    foreign = ["model.language_model.layers.0.mlp.experts.gate_up_linear", "lm_head"]
    with pytest.raises(ValueError, match="found no module named 'model.layers"):
        recipe.resolve(foreign, foreign)
    with pytest.raises(ValueError, match="found no module named 'model.layers"):
        recipe.resolve([], [])
    # ``mixer.up_proj`` is a projection of a dense MLP block (block type "mlp"), which the map has no class for.
    extra = [*names, "model.layers.1.mixer.up_proj", "vision_projector.0"]
    with pytest.raises(
        ValueError, match=r"does not know 2 .*model.layers.1.mixer.up_proj, vision_projector.0\. .*dense MLP block"
    ):
        recipe.resolve(extra, extra)
    # A pipeline stage holds layers 2-3 only: the last layers of the model cannot be located.
    stage = [name for name in names if "layers.0." not in name and "layers.1." not in name]
    with pytest.raises(ValueError, match="whole decoder stack"):
        recipe.resolve(stage, stage)
    with pytest.raises(ValueError, match="exceeds the 4 decoder layers"):
        _recipe_map(high_precision_tail_layers=5).resolve(names, names)


def test_the_output_projection_has_no_class():
    """``lm_head`` is a torch linear: it is no entry of the plan, and a Transformer Engine one is not guessed."""
    names = list(_expected("ME*M", 1, "", "", RECIPE))
    recipe = _recipe_map(high_precision_tail_layers=1)
    for head in ("lm_head", "language_model.lm_head"):
        plan = recipe.resolve([*names, head], names)
        assert [module.name for module in plan.modules] == names
        with pytest.raises(ValueError, match=rf"does not know 1 .*: {head}\. "):
            recipe.resolve([*names, head], [*names, head])
    with pytest.raises(TypeError, match="lm_head"):
        NemotronV3PrecisionMap(**{"high_precision_tail_layers": 1, **RECIPE, "lm_head": "high_precision"})


def test_quantizing_a_module_without_transformer_engine_is_rejected():
    """A quantized class whose modules are plain torch modules would silently run in high precision."""
    names = list(_expected("ME*M", 1, "", "", RECIPE))
    recipe = _recipe_map(high_precision_tail_layers=1)
    # backend.mamba_linear="torch" (the default): the Mamba projections exist but are not Transformer Engine
    # modules. The error names the setting that builds them as such.
    torch_mamba = [name for name in names if not name.endswith(("in_proj", "out_proj"))]
    with pytest.raises(
        ValueError, match=r"quantizes 2 .*backend.mamba_linear='te'.*layers.0.mixer.in_proj \(mamba_in_proj: nvfp4\)"
    ):
        recipe.resolve(names, torch_mamba)
    plan = _recipe_map(
        high_precision_tail_layers=1, mamba_in_proj="high_precision", mamba_out_proj="high_precision"
    ).resolve(names, torch_mamba)
    assert [module.name for module in plan.modules] == torch_mamba
    assert plan.num_layers == 4  # the Mamba layers still count, although they hold no GEMM module of the plan
    # An expert backend without Transformer Engine grouped GEMMs: the container is there, its projections are not.
    torch_experts = [name for name in names if ".experts." not in name]
    modules = [*torch_experts, "model.layers.1.mixer.experts", "mtp.layers.1.mixer.experts"]
    with pytest.raises(ValueError, match=r"quantizes 1 .*model.layers.1.mixer.experts \(routed_experts: nvfp4\)"):
        recipe.resolve(modules, torch_experts)
    _recipe_map(high_precision_tail_layers=1, routed_experts="high_precision").resolve(modules, torch_experts)
    # The same container inside the checkpoint wrapper of its mixer.
    wrapped_names = list(_expected("ME*M", 1, "", "", RECIPE, WRAPPER))
    wrapped_torch_experts = [name for name in wrapped_names if ".experts." not in name]
    wrapped_modules = [*wrapped_torch_experts, f"model.layers.1.mixer.{WRAPPER}experts"]
    with pytest.raises(ValueError, match=r"quantizes 1 .*mixer._checkpoint_wrapped_module.experts \(routed_experts"):
        recipe.resolve(wrapped_modules, wrapped_torch_experts)
    # The tail keeps a module in high precision whatever backend built it.
    tail_mamba = [name for name in names if not name.startswith("model.layers.3.")]
    assert len(recipe.resolve(names, tail_mamba).modules) == len(names) - 2


@pytest.mark.parametrize(
    "changes", [{"mamba_out_proj": "bf16"}, {"high_precision_tail_layers": -1}, {"high_precision_tail_layers": "14"}]
)
def test_invalid_settings_are_rejected(changes: dict[str, str | int]):
    with pytest.raises(ValueError, match="must be"):
        _recipe_map(**changes)


def test_exclude_fqns_must_agree_with_the_map():
    """``exclude_fqns`` keeps working next to the map, but may not contradict it."""
    names = list(_expected("ME*M", 1, "", "", RECIPE))
    gemm_modules = [(name, "GroupedLinear" if ".experts." in name else "Linear") for name in names]
    recipe = _recipe_map(high_precision_tail_layers=1)
    agreeing = TENvfp4Config(precision_map=recipe, exclude_fqns=["lm_head", "q_proj"])
    assert _resolved(agreeing.resolve_precision_plan(names, gemm_modules)) == _expected("ME*M", 1, "", "", RECIPE)
    with pytest.raises(ValueError, match="mixer.q_proj' the precision high_precision, .*precision_map gives it nvfp4"):
        TENvfp4Config(
            precision_map=_recipe_map(high_precision_tail_layers=1, attention="nvfp4"), exclude_fqns=["q_proj"]
        ).resolve_precision_plan(names, gemm_modules)
    with pytest.raises(ValueError, match="mixer.in_proj' the precision high_precision, .*precision_map gives it nvfp4"):
        TENvfp4Config(precision_map=recipe, exclude_fqns=["in_proj"]).resolve_precision_plan(names, gemm_modules)


@pytest.mark.parametrize("checkpointing", ["none", "blocks", "moe"])
def test_names_match_a_built_model(checkpointing: str):
    """The names the test spells out are the ones a Nemotron 3 model registers, also after ``apply_ac``."""
    from nemo_automodel.components.models.common import BackendConfig, LMHead
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
    # The map accepts exactly these names: every module of the built model that would be a Transformer Engine
    # GEMM module with ``backend.linear="te"`` gets a precision, in all three layouts.
    gemm_names = [name for name in spelled_out if ".experts." not in name]
    plan = _recipe_map(high_precision_tail_layers=1, routed_experts="high_precision").resolve(
        [name for name, _ in model.named_modules()], gemm_names
    )
    assert [module.name for module in plan.modules] == gemm_names
    assert "lm_head" in built and "lm_head" not in gemm_names
    assert set(plan.scopes()) <= built
