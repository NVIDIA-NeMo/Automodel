# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from nemo_automodel.components.models.common.te_precision import TEModulePrecision, TEPrecisionMap, TEPrecisionPlan
from nemo_automodel.components.models.common.utils import (
    BackendConfig,
    TEFp8Config,
    TENvfp4Config,
    generation_config_from_model_config,
    get_is_first_microbatch,
    get_is_optim_step,
    get_rope_config,
    initialize_linear_module,
    initialize_rms_norm_module,
    load_pretrained_generation_config,
    restore_pretrained_generation_config,
    set_is_first_microbatch,
    set_is_optim_step,
)
from nemo_automodel.components.models.nemotron_v3.quantization import apply_precision_plan, te_gemm_modules


class TestQuackBackend:
    def test_quack_linear_preserves_linear_contract(self):
        class FakeQuackLinear(nn.Linear):
            pass

        with patch(
            "nemo_automodel.components.models.common.utils.safe_import_from",
            return_value=(True, FakeQuackLinear),
        ):
            module = initialize_linear_module("quack", 8, 16, bias=True, device="cpu", dtype=torch.float32)

        assert isinstance(module, FakeQuackLinear)
        assert module.weight.shape == (16, 8)
        assert module.bias.shape == (16,)

    def test_quack_rms_norm_preserves_rms_norm_contract(self):
        class FakeQuackRMSNorm(nn.RMSNorm):
            pass

        with patch(
            "nemo_automodel.components.models.common.utils.safe_import_from",
            return_value=(True, FakeQuackRMSNorm),
        ):
            module = initialize_rms_norm_module("quack", 8, eps=1e-6, device="cpu", dtype=torch.float32)

        assert isinstance(module, FakeQuackRMSNorm)
        assert module.weight.shape == (8,)
        assert module.eps == 1e-6

    @pytest.mark.parametrize(
        ("initializer", "args", "match"),
        [
            (initialize_linear_module, ("quack", 8, 16), "linear='quack'"),
            (initialize_rms_norm_module, ("quack", 8), "rms_norm='quack'"),
        ],
    )
    def test_quack_backend_reports_missing_optional_dependency(self, initializer, args, match):
        with (
            patch(
                "nemo_automodel.components.models.common.utils.safe_import_from",
                return_value=(False, object()),
            ),
            pytest.raises(ImportError, match=match),
        ):
            initializer(*args)


class TestIsOptimStep:
    def teardown_method(self):
        set_is_optim_step(False)

    def test_default_is_false(self):
        set_is_optim_step(False)
        assert get_is_optim_step() is False

    def test_set_true(self):
        set_is_optim_step(True)
        assert get_is_optim_step() is True

    def test_set_false(self):
        set_is_optim_step(True)
        set_is_optim_step(False)
        assert get_is_optim_step() is False


class TestIsFirstMicrobatch:
    def teardown_method(self):
        set_is_first_microbatch(None)

    def test_default_is_none(self):
        set_is_first_microbatch(None)
        assert get_is_first_microbatch() is None

    def test_set_true(self):
        set_is_first_microbatch(True)
        assert get_is_first_microbatch() is True

    def test_set_false(self):
        set_is_first_microbatch(False)
        assert get_is_first_microbatch() is False

    def test_set_none(self):
        set_is_first_microbatch(True)
        set_is_first_microbatch(None)
        assert get_is_first_microbatch() is None

    def test_grad_accumulation_lifecycle(self):
        """Simulate the typical GA lifecycle: True -> False -> True -> False."""
        # Start of optimizer step: first microbatch
        set_is_first_microbatch(True)
        assert get_is_first_microbatch() is True

        # After first microbatch
        set_is_first_microbatch(False)
        assert get_is_first_microbatch() is False

        # Next optimizer step: first microbatch again
        set_is_first_microbatch(True)
        assert get_is_first_microbatch() is True

        # After first microbatch
        set_is_first_microbatch(False)
        assert get_is_first_microbatch() is False


@contextmanager
def _fake_te_autocast():
    """Stand in for ``transformer_engine.pytorch.quantization.autocast`` and record every ``enabled`` flag."""
    entered: list[bool] = []

    @contextmanager
    def autocast(enabled: bool = True, **kwargs):
        entered.append(enabled)
        yield

    quantization = types.ModuleType("transformer_engine.pytorch.quantization")
    quantization.autocast = autocast
    te_pytorch = types.ModuleType("transformer_engine.pytorch")
    te_pytorch.quantization = quantization
    te_root = types.ModuleType("transformer_engine")
    te_root.pytorch = te_pytorch
    modules = {
        "transformer_engine": te_root,
        "transformer_engine.pytorch": te_pytorch,
        "transformer_engine.pytorch.quantization": quantization,
    }
    with patch("nemo_automodel.components.models.common.utils.HAVE_TE", True), patch.dict("sys.modules", modules):
        yield entered


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


@contextmanager
def _fake_te_nvfp4_recipe(available: bool = True):
    """Stand in for ``transformer_engine.common.recipe``, with or without ``NVFP4BlockScaling``."""
    recipe_module = types.ModuleType("transformer_engine.common.recipe")
    if available:

        class NVFP4BlockScaling:
            def __init__(self, **kwargs: bool | str) -> None:
                self.kwargs = kwargs

        recipe_module.NVFP4BlockScaling = NVFP4BlockScaling
    te_common = types.ModuleType("transformer_engine.common")
    te_common.recipe = recipe_module
    te_root = types.ModuleType("transformer_engine")
    te_root.common = te_common
    modules = {
        "transformer_engine": te_root,
        "transformer_engine.common": te_common,
        "transformer_engine.common.recipe": recipe_module,
    }
    with patch("nemo_automodel.components.models.common.utils.HAVE_TE", True), patch.dict("sys.modules", modules):
        yield


class TestTEFp8Config:
    def test_default_recipe(self):
        cfg = TEFp8Config()
        assert cfg.recipe == "current"

    def test_block_recipe(self):
        cfg = TEFp8Config(recipe="block")
        assert cfg.recipe == "block"

    def test_passthrough_recipe_object(self):
        """Non-string recipe objects are passed through directly."""
        sentinel = object()
        cfg = TEFp8Config(recipe=sentinel)
        assert cfg.build_recipe() is sentinel

    def test_maybe_te_autocast_without_te(self):
        """Without TE installed, maybe_te_autocast returns nullcontext."""
        cfg = TEFp8Config()
        with patch("nemo_automodel.components.models.common.utils.HAVE_TE", False):
            ctx = cfg.maybe_te_autocast()
            assert isinstance(ctx, nullcontext)

    def test_build_recipe_without_te(self):
        """Without TE installed, build_recipe returns None."""
        cfg = TEFp8Config()
        with patch("nemo_automodel.components.models.common.utils.HAVE_TE", False):
            assert cfg.build_recipe() is None

    def test_exclude_fqns_default_and_normalization(self):
        assert TEFp8Config().exclude_fqns == ("lm_head",)
        assert TEFp8Config(exclude_fqns=["lm_head", "model.score"]).exclude_fqns == ("lm_head", "model.score")
        assert TEFp8Config(exclude_fqns="score").exclude_fqns == ("score",)
        assert TEFp8Config(exclude_fqns=()).exclude_fqns == ()
        with pytest.raises(ValueError, match="exclude_fqns"):
            TEFp8Config(exclude_fqns=[""])
        cfg = BackendConfig(te_fp8={"recipe": "block", "exclude_fqns": ["lm_head"]}, experts="te", dispatcher="deepep")
        assert cfg.te_fp8.exclude_fqns == ("lm_head",)

    def test_module_excluded_uses_substring_match_like_torchao_filter_fqns(self):
        cfg = TEFp8Config(exclude_fqns=["lm_head"])
        assert cfg.module_excluded("lm_head")
        assert cfg.module_excluded("model.language_model.lm_head")
        assert not cfg.module_excluded("model.layers.0.mlp.gate")

    def test_exclude_modules_wraps_matching_forwards_once(self):
        """Excluded modules run inside the disabled autocast, others do not; idempotent; survives deepcopy."""
        model = nn.Sequential(OrderedDict(body=nn.Linear(4, 4), lm_head=nn.Linear(4, 3)))
        x = torch.randn(2, 4)
        expected = model(x)
        cfg = TEFp8Config(recipe="block")
        with _fake_te_autocast() as entered:
            assert cfg.exclude_modules(model) == ["lm_head"]
            assert cfg.exclude_modules(model) == ["lm_head"]  # idempotent: one wrapper, not two
            torch.testing.assert_close(model(x), expected)
            assert entered == [False]  # lm_head once, with the autocast disabled; body never
            clone = copy.deepcopy(model)
            assert clone.lm_head.forward.__self__ is clone.lm_head
            torch.testing.assert_close(clone(x), expected)
            assert entered == [False, False]
        assert list(model.state_dict()) == ["body.weight", "body.bias", "lm_head.weight", "lm_head.bias"]
        assert "forward" not in vars(model.body)

    def test_exclude_modules_keeps_an_installed_instance_forward(self):
        """A forward installed before the exclusion (tensor-parallel style, pipeline wrapper) still runs, outside FP8."""
        calls: list[str] = []

        def parallel_forward(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
            calls.append("tp")
            return nn.Linear.forward(module, x) + 1.0

        model = nn.Sequential(OrderedDict(body=nn.Linear(4, 4), o_proj=nn.Linear(4, 4)))
        model.o_proj.forward = types.MethodType(parallel_forward, model.o_proj)
        x = torch.randn(2, 4)
        expected = model(x)
        assert calls == ["tp"]
        cfg = TEFp8Config(recipe="block", exclude_fqns=["o_proj"])
        with _fake_te_autocast() as entered:
            assert cfg.exclude_modules(model) == ["o_proj"]
            torch.testing.assert_close(model(x), expected)
            assert calls == ["tp", "tp"] and entered == [False]
            installed = model.o_proj._te_fp8_excluded_forward
            assert installed.__func__ is parallel_forward and installed.__self__ is model.o_proj
            assert cfg.exclude_modules(model) == ["o_proj"]  # re-applying keeps the installed forward
            assert model.o_proj._te_fp8_excluded_forward is installed
            clone = copy.deepcopy(model)
            assert clone.o_proj._te_fp8_excluded_forward.__self__ is clone.o_proj
            torch.testing.assert_close(clone(x), expected)
            assert calls == ["tp", "tp", "tp"] and entered == [False, False]

    def test_exclude_modules_is_a_no_op_without_te_or_exclusions(self):
        model = nn.Sequential(OrderedDict(lm_head=nn.Linear(2, 2)))
        module = "nemo_automodel.components.models.common.utils"
        with patch(f"{module}.HAVE_TE", False):
            assert TEFp8Config().exclude_modules(model) == []
        with patch(f"{module}.HAVE_TE", True):
            assert TEFp8Config(exclude_fqns=()).exclude_modules(model) == []
        assert "forward" not in vars(model.lm_head)


class TestBackendConfigTeFp8:
    def test_te_fp8_default_is_none(self):
        """BackendConfig.te_fp8 defaults to None."""
        cfg = BackendConfig()
        assert cfg.te_fp8 is None

    def test_te_fp8_dict_normalized(self):
        """A dict te_fp8 is converted to TEFp8Config."""
        cfg = BackendConfig(te_fp8={"recipe": "block"}, experts="te", dispatcher="deepep")
        assert isinstance(cfg.te_fp8, TEFp8Config)
        assert cfg.te_fp8.recipe == "block"

    def test_te_fp8_requires_te_backend(self):
        """te_fp8 requires linear='te' or experts='te'."""
        with pytest.raises(ValueError, match="te_fp8 requires at least one TE backend"):
            BackendConfig(te_fp8=TEFp8Config(), linear="torch", experts="torch")


class TestTENvfp4Config:
    def test_default_options_keep_the_recipe_defaults(self):
        cfg = TENvfp4Config()
        assert cfg.recipe_kwargs() == {}
        assert not cfg.resident_weights
        with _fake_te_nvfp4_recipe():
            assert cfg.build_recipe().kwargs == {}

    def test_options_reach_the_te_recipe(self):
        cfg = TENvfp4Config(
            random_hadamard_transform=False,
            stochastic_rounding=False,
            weight_blocks_2d=False,
            row_scaled_activation=True,
            four_over_six="activations",
            backward="high_precision",
            resident_weights=True,
        )
        with _fake_te_nvfp4_recipe():
            built = cfg.build_recipe()
        assert built.kwargs == {
            "disable_rht": True,
            "disable_stochastic_rounding": True,
            "disable_2d_quantization": True,
            "row_scaled_activation": True,
            "nvfp4_4over6": "activations",
            "backward_override": "high_precision",
        }

    def test_build_recipe_without_te(self):
        with patch("nemo_automodel.components.models.common.utils.HAVE_TE", False):
            assert TENvfp4Config().build_recipe() is None

    def test_build_recipe_needs_te_with_nvfp4(self):
        """A TE build without NVFP4BlockScaling is reported."""
        with _fake_te_nvfp4_recipe(available=False):
            with pytest.raises(ImportError, match="NVFP4BlockScaling"):
                TENvfp4Config().build_recipe()

    @pytest.mark.parametrize(
        "options",
        [
            {"row_scaled_activation": True},
            {"four_over_six": "activations"},
            {"four_over_six": "all"},
            {"four_over_six": "sometimes", "random_hadamard_transform": False},
            {"backward": "float"},
        ],
    )
    def test_unsupported_options_are_rejected(self, options):
        with pytest.raises(ValueError, match="te_nvfp4"):
            TENvfp4Config(**options)

    def test_precision_map_left_as_a_mapping_is_rejected(self):
        """A YAML block without ``_target_`` reaches the config as a mapping; it has no ``resolve``."""
        mapping = {"high_precision_tail_layers": 14, "mamba_in_proj": "nvfp4"}
        with pytest.raises(ValueError, match=r"te_nvfp4.precision_map must be a precision map object.*`_target_`"):
            TENvfp4Config(precision_map=mapping)
        with pytest.raises(ValueError, match=r"te_nvfp4.precision_map must be a precision map object"):
            BackendConfig(te_nvfp4={"precision_map": mapping})

    def test_precision_map_and_exclude_fqns_are_mutually_exclusive(self):
        """A map keeps modules in high precision itself, so a config with a map has an empty exclusion list."""
        precision_map = _LastNamePrecisionMap({})
        message = r"te_nvfp4.precision_map and te_nvfp4.exclude_fqns are mutually exclusive, got exclude_fqns="
        # The default list is not empty.
        with pytest.raises(ValueError, match=message + r"\['lm_head'\] .*exclusions in the map and set .* to \[\]"):
            TENvfp4Config(precision_map=precision_map)
        with pytest.raises(ValueError, match=message + r"\['model.score'\]"):
            TENvfp4Config(precision_map=precision_map, exclude_fqns="model.score")
        with pytest.raises(ValueError, match=message + r"\['lm_head', 'q_proj'\]"):
            BackendConfig(te_nvfp4={"precision_map": precision_map, "exclude_fqns": ["lm_head", "q_proj"]})
        cfg = TENvfp4Config(precision_map=precision_map, exclude_fqns=[])
        assert cfg.precision_map is precision_map
        assert cfg.exclude_fqns == ()
        # Without a map the list is the one of every Transformer Engine quantization config.
        assert TENvfp4Config(exclude_fqns=["model.score"]).exclude_fqns == ("model.score",)

    def test_exclude_fqns_default_and_normalization(self):
        assert TENvfp4Config().exclude_fqns == ("lm_head",)
        assert TENvfp4Config(exclude_fqns=["lm_head", "model.score"]).exclude_fqns == ("lm_head", "model.score")
        assert TENvfp4Config(exclude_fqns="score").module_excluded("model.score")
        with pytest.raises(ValueError, match="te_nvfp4.exclude_fqns"):
            TENvfp4Config(exclude_fqns=[""])


@dataclass(frozen=True)
class _LastNamePrecisionMap:
    """Test map: precision by the last part of a GEMM module name; ``inner`` uses its parent as scope."""

    precisions: dict[str, str]
    drop: str | None = None

    def resolve(self, module_names: list[str], gemm_module_names: list[str]) -> TEPrecisionPlan:
        modules = tuple(
            TEModulePrecision(
                name=name,
                module_class=name.rsplit(".", 1)[-1],
                layer=None,
                precision=self.precisions[name.rsplit(".", 1)[-1]],
                scope=name.rsplit(".", 1)[0] if name.endswith(".inner") else name,
            )
            for name in gemm_module_names
            if name != self.drop
        )
        return TEPrecisionPlan(modules=modules, num_layers=0, high_precision_tail_layers=0)


class TestTENvfp4PrecisionMap:
    @staticmethod
    def _model() -> nn.ModuleDict:
        from transformer_engine.pytorch import Linear

        return nn.ModuleDict(
            {
                "quantized": Linear(4, 4),
                "eight_bit": Linear(4, 4),
                "block": nn.Sequential(OrderedDict(inner=Linear(4, 4), norm=nn.LayerNorm(4))),
                "torch_linear": nn.Linear(4, 4),
            }
        )

    def test_scopes_get_the_forward_of_their_precision(self, caplog):
        precision_map = _LastNamePrecisionMap({"quantized": "nvfp4", "eight_bit": "mxfp8", "inner": "high_precision"})
        x = torch.randn(2, 4)
        with _fake_te_mxfp8_autocast() as entered:
            from transformer_engine.pytorch.quantization import autocast

            model = self._model()
            expected = model.block(model.eight_bit(model.quantized(x)))
            plan = precision_map.resolve(
                [name for name, _ in model.named_modules()], [name for name, _ in te_gemm_modules(model)]
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

    def test_mxfp8_scopes_quantize_only_inside_an_enabled_autocast(self):
        """A forward without the Transformer Engine autocast, or inside a disabled one, stays in high precision."""
        x = torch.randn(2, 4)
        with _fake_te_mxfp8_autocast() as entered:
            from transformer_engine.pytorch import Linear
            from transformer_engine.pytorch.quantization import autocast

            model = nn.ModuleDict({"mlp": Linear(4, 4)})
            expected = model.mlp(x)
            precision_map = _LastNamePrecisionMap({"mlp": "mxfp8"})
            apply_precision_plan(model, precision_map.resolve(["", "mlp"], ["mlp"]))
            assert "forward" in vars(model.mlp)
            torch.testing.assert_close(model.mlp(x), expected)
            with autocast(enabled=False):
                torch.testing.assert_close(model.mlp(x), expected)
            assert entered == [(False, None)]
            with autocast(enabled=True):
                torch.testing.assert_close(model.mlp(x), expected)
            assert entered == [(False, None), (True, None), (True, "MXFP8BlockScaling")]

    def test_exclude_modules_hands_the_model_to_the_map(self):
        prepared = []

        class _Recording(TEPrecisionMap):
            def prepare(self, model: nn.Module) -> None:
                prepared.append(model)

        model = nn.Linear(2, 2)
        assert TENvfp4Config(precision_map=_Recording(), exclude_fqns=()).exclude_modules(model) == []
        assert prepared == [model]

    def test_plan_rejects_inconsistent_scopes(self):
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


class TestBackendConfigTeNvfp4:
    def test_te_nvfp4_default_is_none(self):
        cfg = BackendConfig()
        assert cfg.te_nvfp4 is None
        assert cfg.te_quantization is None

    def test_te_nvfp4_dict_normalized(self):
        cfg = BackendConfig(
            te_nvfp4={"stochastic_rounding": False, "resident_weights": True}, experts="te", dispatcher="deepep"
        )
        assert cfg.te_nvfp4 == TENvfp4Config(stochastic_rounding=False, resident_weights=True)
        assert cfg.te_fp8 is None
        assert cfg.te_quantization is cfg.te_nvfp4

    def test_empty_te_nvfp4_keeps_the_recipe_defaults(self):
        cfg = BackendConfig(te_nvfp4={}, experts="te", dispatcher="deepep")
        assert cfg.te_nvfp4 == TENvfp4Config()

    def test_te_quantization_is_te_fp8_when_that_is_set(self):
        cfg = BackendConfig(te_fp8={"recipe": "block"}, experts="te", dispatcher="deepep")
        assert cfg.te_nvfp4 is None
        assert cfg.te_quantization is cfg.te_fp8

    def test_te_nvfp4_and_te_fp8_together_are_rejected(self):
        with pytest.raises(ValueError, match="te_fp8 and te_nvfp4 are mutually exclusive"):
            BackendConfig(te_nvfp4={}, te_fp8={"recipe": "mxfp8"}, experts="te", dispatcher="deepep")

    def test_te_nvfp4_requires_te_backend(self):
        with pytest.raises(ValueError, match="te_nvfp4 requires at least one TE backend"):
            BackendConfig(te_nvfp4={}, linear="torch", experts="torch")

    def test_te_nvfp4_rejects_unsupported_options(self):
        with pytest.raises(ValueError, match="te_nvfp4.backward"):
            BackendConfig(te_nvfp4={"backward": "float"}, experts="te", dispatcher="deepep")


class TestGetRopeConfig:
    def test_extracts_rope_theta(self):
        config = SimpleNamespace(
            rope_parameters={"rope_theta": 1_000_000, "rope_type": "default"},
        )
        rope_theta, _, _ = get_rope_config(config)
        assert rope_theta == 1_000_000

    def test_returns_rope_parameters_as_rope_scaling(self):
        params = {"rope_theta": 10_000, "rope_type": "default"}
        config = SimpleNamespace(rope_parameters=params)
        _, rope_parameters, _ = get_rope_config(config)
        assert rope_parameters is params

    def test_partial_rotary_factor(self):
        config = SimpleNamespace(
            rope_parameters={"rope_theta": 10_000, "partial_rotary_factor": 0.5},
        )
        _, _, partial_rotary_factor = get_rope_config(config)
        assert partial_rotary_factor == 0.5

    def test_partial_rotary_factor_defaults_to_one(self):
        config = SimpleNamespace(
            rope_parameters={"rope_theta": 10_000},
        )
        _, _, partial_rotary_factor = get_rope_config(config)
        assert partial_rotary_factor == 1.0

    def test_yarn_scaling_parameters(self):
        config = SimpleNamespace(
            rope_parameters={
                "rope_theta": 1_000_000,
                "rope_type": "yarn",
                "factor": 4.0,
                "beta_slow": 1.0,
                "beta_fast": 32.0,
                "original_max_position_embeddings": 8192,
            },
        )
        rope_theta, rope_parameters, _ = get_rope_config(config)
        assert rope_theta == 1_000_000
        assert rope_parameters["factor"] == 4.0
        assert rope_parameters["beta_slow"] == 1.0
        assert rope_parameters["beta_fast"] == 32.0
        assert rope_parameters["original_max_position_embeddings"] == 8192


class TestGenerationConfigHelpers:
    def test_from_model_config_takes_stop_tokens_from_a_pretrained_config(self):
        from transformers import PretrainedConfig

        config = PretrainedConfig(bos_token_id=3, eos_token_id=[2, 11], pad_token_id=0)

        generation_config = generation_config_from_model_config(config)

        assert generation_config.bos_token_id == 3
        assert generation_config.eos_token_id == [2, 11]
        assert generation_config.pad_token_id == 0

    def test_from_model_config_falls_back_to_defaults_for_plain_objects(self):
        generation_config = generation_config_from_model_config(SimpleNamespace(eos_token_id=2))

        assert generation_config.eos_token_id is None

    def test_load_pretrained_returns_none_without_any_config_file(self, tmp_path):
        assert load_pretrained_generation_config(tmp_path) is None

    def test_load_pretrained_falls_back_to_the_generation_fields_of_config_json(self, tmp_path):
        """Legacy checkpoints keep do_sample/temperature in config.json only; HF reads them from there."""
        import json

        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "llama", "eos_token_id": 2, "do_sample": True, "temperature": 0.5})
        )

        generation_config = load_pretrained_generation_config(tmp_path)

        assert generation_config.eos_token_id == 2
        assert generation_config.do_sample is True
        assert generation_config.temperature == 0.5

    def test_restore_replaces_a_real_generation_config_only(self, tmp_path):
        from transformers import GenerationConfig

        GenerationConfig(eos_token_id=[2, 11]).save_pretrained(tmp_path)
        can_generate = nn.Module()
        can_generate.generation_config = GenerationConfig()
        cannot_generate = nn.Module()

        restore_pretrained_generation_config(can_generate, tmp_path)
        restore_pretrained_generation_config(cannot_generate, tmp_path)

        assert can_generate.generation_config.eos_token_id == [2, 11]
        assert not hasattr(cannot_generate, "generation_config")

    def test_load_pretrained_fallback_keeps_an_override_that_equals_a_default(self, tmp_path):
        """``eos_token_id=None`` is a deliberate "do not stop"; it equals the default, so
        the diff-based merge cannot see it and the caller has to name it."""
        import json

        from transformers import PretrainedConfig

        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama", "eos_token_id": 0}))

        generation_config = load_pretrained_generation_config(
            tmp_path, config=PretrainedConfig(eos_token_id=None), config_overrides={"eos_token_id"}
        )

        assert generation_config.eos_token_id is None

    def test_load_pretrained_reads_the_subfolder_of_a_local_directory(self, tmp_path):
        """transformers resolves an absolute local path to the parent's generation_config.json
        even with ``subfolder`` set; the helper joins the directory itself so the child wins."""
        from transformers import GenerationConfig

        GenerationConfig(eos_token_id=7).save_pretrained(tmp_path)
        GenerationConfig(eos_token_id=[2, 11]).save_pretrained(tmp_path / "child")

        assert load_pretrained_generation_config(tmp_path, subfolder="child").eos_token_id == [2, 11]

    def test_load_pretrained_forwards_the_hub_loading_options(self, tmp_path):
        from transformers import GenerationConfig

        GenerationConfig(eos_token_id=3).save_pretrained(tmp_path)

        with patch.object(GenerationConfig, "from_pretrained", wraps=GenerationConfig.from_pretrained) as spy:
            load_pretrained_generation_config(tmp_path, revision="abc123", token="secret", local_files_only=True)

        _, forwarded = spy.call_args
        assert forwarded["revision"] == "abc123"
        assert forwarded["token"] == "secret"
        assert forwarded["local_files_only"] is True

    @pytest.mark.parametrize("disk_eos", [2, None], ids=["conflicting", "omitted"])
    def test_load_pretrained_fallback_keeps_explicit_config_overrides(self, tmp_path, disk_eos):
        """An eos override passed at load time already sits on the in-memory config; the
        config.json fallback must neither replace it with the file's value nor with None."""
        import json

        from transformers import PretrainedConfig

        raw = {"model_type": "llama", "do_sample": True}
        if disk_eos is not None:
            raw["eos_token_id"] = disk_eos
        (tmp_path / "config.json").write_text(json.dumps(raw))

        generation_config = load_pretrained_generation_config(tmp_path, config=PretrainedConfig(eos_token_id=9))

        assert generation_config.eos_token_id == 9
        assert generation_config.do_sample is True

    def test_load_pretrained_reads_the_checkpoint_generation_file(self, tmp_path):
        from transformers import GenerationConfig

        GenerationConfig(eos_token_id=[2, 11], do_sample=True, temperature=0.7).save_pretrained(tmp_path)

        generation_config = load_pretrained_generation_config(tmp_path)

        assert generation_config.eos_token_id == [2, 11]
        assert generation_config.do_sample is True
        assert generation_config.temperature == 0.7
