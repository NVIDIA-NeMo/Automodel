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
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

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
    """Stand in for the Transformer Engine autocast and ``MXFP8BlockScaling``; record ``(enabled, recipe name)``."""
    entered: list[tuple[bool, str | None]] = []

    class MXFP8BlockScaling:
        pass

    @contextmanager
    def autocast(enabled: bool = True, recipe: MXFP8BlockScaling | None = None):
        entered.append((enabled, None if recipe is None else type(recipe).__name__))
        yield

    quantization = types.ModuleType("transformer_engine.pytorch.quantization")
    quantization.autocast = autocast
    te_pytorch = types.ModuleType("transformer_engine.pytorch")
    te_pytorch.quantization = quantization
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
            {"backward": "mxfp8"},
            {"backward": "mxfp8", "random_hadamard_transform": False, "resident_weights": True},
            {"mxfp8_fqns": [""]},
        ],
    )
    def test_unsupported_options_are_rejected(self, options):
        with pytest.raises(ValueError, match="te_nvfp4"):
            TENvfp4Config(**options)

    def test_mxfp8_fqns_default_and_normalization(self):
        assert TENvfp4Config().mxfp8_fqns == ()
        assert TENvfp4Config(mxfp8_fqns=["layers.2.", "layers.3."]).mxfp8_fqns == ("layers.2.", "layers.3.")
        assert TENvfp4Config(mxfp8_fqns="layers.3.").mxfp8_fqns == ("layers.3.",)

    def test_route_mxfp8_modules_wraps_the_outermost_matches(self):
        """Matching modules run under the MXFP8 recipe, their children are not wrapped again; idempotent."""
        model = nn.ModuleDict(
            {
                "layers": nn.ModuleDict(
                    {
                        "0": nn.Sequential(OrderedDict(mlp=nn.Linear(4, 4))),
                        "1": nn.Sequential(OrderedDict(mlp=nn.Sequential(OrderedDict(experts=nn.Linear(4, 4))))),
                    }
                ),
                "lm_head": nn.Linear(4, 3),
            }
        )
        x = torch.randn(2, 4)
        expected = model.lm_head(model.layers["1"](model.layers["0"](x)))
        cfg = TENvfp4Config(mxfp8_fqns=["layers.1."])
        with _fake_te_mxfp8_autocast() as entered:
            assert cfg.route_mxfp8_modules(model) == ["layers.1.mlp"]
            assert cfg.route_mxfp8_modules(model) == ["layers.1.mlp"]  # idempotent: one wrapper, not two
            cfg.prepare_modules(model)  # also keeps lm_head out of the autocast
            torch.testing.assert_close(model.lm_head(model.layers["1"](model.layers["0"](x))), expected)
            assert entered == [(True, "MXFP8BlockScaling"), (False, None)]
            clone = copy.deepcopy(model)
            assert clone.layers["1"].mlp.forward.__self__ is clone.layers["1"].mlp
        assert "forward" not in vars(model.layers["0"].mlp)
        assert "forward" not in vars(model.layers["1"].mlp.experts)

    def test_route_mxfp8_modules_rejects_an_excluded_module(self):
        model = nn.ModuleDict({"lm_head": nn.Linear(4, 3)})
        with _fake_te_mxfp8_autocast():
            with pytest.raises(ValueError, match="exclude_fqns and te_nvfp4.mxfp8_fqns"):
                TENvfp4Config(mxfp8_fqns=["lm_head"]).route_mxfp8_modules(model)

    def test_route_mxfp8_modules_is_a_no_op_without_te_or_names(self):
        model = nn.ModuleDict({"mlp": nn.Linear(2, 2)})
        module = "nemo_automodel.components.models.common.utils"
        with patch(f"{module}.HAVE_TE", False):
            assert TENvfp4Config(mxfp8_fqns=["mlp"]).route_mxfp8_modules(model) == []
        with patch(f"{module}.HAVE_TE", True):
            assert TENvfp4Config().route_mxfp8_modules(model) == []
        assert "forward" not in vars(model.mlp)

    def test_exclude_fqns_default_and_normalization(self):
        assert TENvfp4Config().exclude_fqns == ("lm_head",)
        assert TENvfp4Config(exclude_fqns=["lm_head", "model.score"]).exclude_fqns == ("lm_head", "model.score")
        assert TENvfp4Config(exclude_fqns="score").module_excluded("model.score")
        with pytest.raises(ValueError, match="te_nvfp4.exclude_fqns"):
            TENvfp4Config(exclude_fqns=[""])


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
