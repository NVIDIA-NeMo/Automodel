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

"""Lightweight contract tests for the Qwen-Image DMD2 recipe."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import yaml

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.models.qwen_image.dmd2 import QwenImageDMD2Adapter
from nemo_automodel.recipes.diffusion import dmd2 as dmd2_module
from nemo_automodel.recipes.diffusion.dmd2 import (
    DMD2DiffusionRecipe,
    _expand_negative_conditioning,
    _load_negative_prompt_embedding,
    _PreparedDMD2Batch,
    _require_modelopt_dmd2,
)
from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE_CONFIG = _REPO_ROOT / "examples" / "diffusion" / "dmd2" / "qwen_image_dmd2.yaml"


class _FakeDMDConfig:
    """Minimal Pydantic-like DMD config used to test config ownership boundaries."""

    model_fields = {
        "ema": object(),
        "gan_loss_weight_gen": object(),
        "guidance_scale": object(),
        "sample_t_cfg": object(),
        "student_sample_steps": object(),
        "student_update_freq": object(),
    }
    validated_payload: dict | None = None

    @classmethod
    def model_validate(cls, payload):
        """Record and materialize the merged ModelOpt-owned configuration."""
        cls.validated_payload = deepcopy(payload)
        ema = payload.get("ema")
        return SimpleNamespace(
            ema=SimpleNamespace(**ema) if isinstance(ema, dict) else ema,
            gan_loss_weight_gen=payload["gan_loss_weight_gen"],
            guidance_scale=payload.get("guidance_scale"),
            sample_t_cfg=SimpleNamespace(**payload["sample_t_cfg"]),
            student_sample_steps=payload["student_sample_steps"],
            student_update_freq=payload["student_update_freq"],
        )


class _FakeBaseDMDConfig:
    """ModelOpt recipe result exposing the Pydantic ``model_dump`` contract."""

    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        """Return an isolated copy so the integration cannot mutate the base recipe."""
        return deepcopy(self.payload)


class _FixedStepScheduler:
    """Five outer steps whose final accumulation window is partial."""

    def __init__(self) -> None:
        self.step = 0
        self.epochs = range(1)
        self.dataloader = None

    def __iter__(self):
        """Yield one student step followed by four fake-score steps."""
        for step in range(5):
            self.step = step
            yield [{}] if step == 4 else [{}, {}]
            self.step = step + 1

    @property
    def is_ckpt_step(self) -> bool:
        """Disable checkpoint writes in the loop unit test."""
        return False


class _CountingLRScheduler:
    """Minimal LR scheduler exposing AutoModel's step counter contract."""

    def __init__(self) -> None:
        self.num_steps = 0

    def step(self, increment: int) -> None:
        """Advance by the number of completed student updates."""
        self.num_steps += increment


def _base_recipe_config(**dmd2_overrides):
    """Return the smallest valid AutoModel-side DMD2 configuration."""
    dmd2 = {
        "model_adapter": _model_adapter_config(),
        **dmd2_overrides,
    }
    return {
        "model": {"mode": "finetune"},
        "fsdp": {"tp_size": 1, "cp_size": 1, "pp_size": 1},
        "dmd2": dmd2,
        "data": {"dataloader": {"train_text_encoder": False}},
    }


def _model_adapter_config(**overrides):
    """Return the declarative Qwen-Image DMD2 adapter configuration."""
    return {
        "_target_": "nemo_automodel.components.models.qwen_image.dmd2.QwenImageDMD2Adapter",
        "guidance": None,
        "gan_feature_indices": [30],
        "gan_num_blocks": 60,
        "gan_inner_dim": 3072,
        **overrides,
    }


def _bare_recipe(config):
    """Construct a recipe without invoking model or distributed setup."""
    recipe = DMD2DiffusionRecipe.__new__(DMD2DiffusionRecipe)
    recipe.cfg = ConfigNode(config)
    adapter_config = recipe.cfg.get("dmd2.model_adapter", None)
    if adapter_config is not None and hasattr(adapter_config, "instantiate"):
        recipe._model_adapter = adapter_config.instantiate()
    return recipe


def _resolved_config(**overrides):
    """Return a valid resolved DMD configuration with selected overrides."""
    values = {
        "student_update_freq": 5,
        "student_sample_steps": 1,
        "sample_t_cfg": SimpleNamespace(t_list=None),
        "gan_loss_weight_gen": 0.0,
        "ema": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_require_modelopt_dmd2_fails_actionably_when_optional_dependency_is_missing(monkeypatch):
    monkeypatch.setattr(dmd2_module, "safe_import", lambda *args, **kwargs: (False, None))

    with pytest.raises(ImportError, match=r"uv sync --extra dmd2"):
        _require_modelopt_dmd2()


def test_require_modelopt_dmd2_resolves_only_the_supported_public_contract(monkeypatch):
    fastgen = SimpleNamespace(DMDConfig=_FakeDMDConfig, load_dmd_config=Mock())

    monkeypatch.setattr(dmd2_module, "safe_import", lambda *args, **kwargs: (True, fastgen))

    api = _require_modelopt_dmd2()

    assert api.dmd_config_cls is _FakeDMDConfig
    assert api.load_dmd_config is fastgen.load_dmd_config


def test_require_modelopt_dmd2_rejects_an_incompatible_modelopt_build(monkeypatch):
    fastgen = SimpleNamespace(DMDConfig=_FakeDMDConfig)

    monkeypatch.setattr(dmd2_module, "safe_import", lambda *args, **kwargs: (True, fastgen))

    with pytest.raises(ImportError, match="Missing symbols: load_dmd_config"):
        _require_modelopt_dmd2()


def test_load_negative_prompt_embedding_preserves_embedding_and_mask_safely(tmp_path, monkeypatch):
    path = tmp_path / "negative.pt"
    embedding = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.int32)
    torch.save({"embed": embedding, "prompt_embeds_mask": mask}, path)
    original_torch_load = torch.load
    load_kwargs = {}

    def recording_load(*args, **kwargs):
        load_kwargs.update(kwargs)
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(dmd2_module.torch, "load", recording_load)

    loaded_embedding, loaded_mask = _load_negative_prompt_embedding(path)

    torch.testing.assert_close(loaded_embedding, embedding.squeeze(0))
    torch.testing.assert_close(loaded_mask, mask.squeeze(0).long())
    assert loaded_embedding.device.type == "cpu"
    assert loaded_embedding.is_contiguous()
    assert loaded_mask.is_contiguous()
    assert load_kwargs == {"map_location": "cpu", "weights_only": True}


def test_load_and_expand_bare_negative_prompt_embedding_uses_an_all_valid_mask(tmp_path):
    path = tmp_path / "negative.pt"
    embedding = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    torch.save(embedding, path)

    loaded_embedding, loaded_mask = _load_negative_prompt_embedding(path)
    expanded_embedding, expanded_mask = _expand_negative_conditioning(
        loaded_embedding,
        loaded_mask,
        batch_size=3,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert expanded_embedding.shape == (3, 2, 4)
    assert expanded_embedding.dtype is torch.bfloat16
    assert expanded_embedding.is_contiguous()
    assert expanded_mask.shape == (3, 2)
    assert expanded_mask.dtype is torch.long
    assert torch.all(expanded_mask == 1)
    for batch_index in range(3):
        torch.testing.assert_close(expanded_embedding[batch_index].float(), embedding)


def test_load_negative_prompt_embedding_rejects_non_binary_mask(tmp_path):
    path = tmp_path / "negative.pt"
    torch.save(
        {
            "embed": torch.ones(1, 2, 4),
            "mask": torch.tensor([[1, 2]]),
        },
        path,
    )

    with pytest.raises(ValueError, match="zeros and ones"):
        _load_negative_prompt_embedding(path)


def test_load_cfg_conditioning_suppresses_an_all_valid_mask_for_flash_attention(tmp_path):
    path = tmp_path / "negative.pt"
    torch.save(
        {
            "embed": torch.ones(1, 3, 4),
            "mask": torch.ones(1, 3, dtype=torch.long),
        },
        path,
    )
    recipe = _bare_recipe(_base_recipe_config(negative_prompt_embedding_path=str(path)))
    recipe._dmd_config = SimpleNamespace(guidance_scale=4.0)
    recipe.attention_backend = "flash"

    embedding, mask = recipe._load_cfg_conditioning()

    assert embedding.shape == (3, 4)
    assert mask is None


def test_load_cfg_conditioning_rejects_a_padded_mask_for_flash_attention(tmp_path):
    path = tmp_path / "negative.pt"
    torch.save(
        {
            "embed": torch.ones(1, 3, 4),
            "mask": torch.tensor([[1, 1, 0]]),
        },
        path,
    )
    recipe = _bare_recipe(_base_recipe_config(negative_prompt_embedding_path=str(path)))
    recipe._dmd_config = SimpleNamespace(guidance_scale=4.0)
    recipe.attention_backend = "flash"

    with pytest.raises(ValueError, match="padded negative-prompt mask"):
        recipe._load_cfg_conditioning()


def test_resolve_dmd_config_deep_merges_modelopt_fields_and_ignores_trainer_fields():
    recipe = _bare_recipe(
        _base_recipe_config(
            recipe_path="general/distillation/dmd2_qwen_image",
            fake_score_lr=2.0e-5,
            sample_t_cfg={"shift": 5.0},
        )
    )
    base_payload = {
        "student_update_freq": 5,
        "student_sample_steps": 2,
        "sample_t_cfg": {
            "min_t": 0.001,
            "max_t": 0.999,
            "shift": 3.0,
            "t_list": [1.0, 0.5, 0.0],
        },
        "gan_loss_weight_gen": 0.0,
        "guidance_scale": 4.0,
        "ema": None,
    }
    load_dmd_config = Mock(return_value=_FakeBaseDMDConfig(base_payload))
    recipe._modelopt = SimpleNamespace(
        dmd_config_cls=_FakeDMDConfig,
        load_dmd_config=load_dmd_config,
    )

    resolved = recipe._resolve_dmd_config()

    load_dmd_config.assert_called_once_with("general/distillation/dmd2_qwen_image")
    assert resolved.sample_t_cfg.shift == 5.0
    assert resolved.sample_t_cfg.min_t == 0.001
    assert resolved.sample_t_cfg.max_t == 0.999
    assert _FakeDMDConfig.validated_payload is not None
    assert "fake_score_lr" not in _FakeDMDConfig.validated_payload
    assert "model_adapter" not in _FakeDMDConfig.validated_payload
    assert base_payload["sample_t_cfg"]["shift"] == 3.0


def test_resolve_dmd_config_rejects_unknown_fields_before_modelopt_validation():
    recipe = _bare_recipe(
        _base_recipe_config(
            recipe_path="general/distillation/dmd2_qwen_image",
            misspelled_update_frequency=5,
        )
    )
    recipe._modelopt = SimpleNamespace(
        dmd_config_cls=_FakeDMDConfig,
        load_dmd_config=Mock(
            return_value=_FakeBaseDMDConfig(
                {
                    "student_update_freq": 5,
                    "student_sample_steps": 1,
                    "sample_t_cfg": {"t_list": None},
                    "gan_loss_weight_gen": 0.0,
                    "guidance_scale": None,
                    "ema": None,
                }
            )
        ),
    )

    with pytest.raises(ValueError, match="misspelled_update_frequency"):
        recipe._resolve_dmd_config()


@pytest.mark.parametrize(
    ("config_overrides", "error"),
    [
        ({"model": {"mode": "inference"}}, "model.mode=finetune"),
        ({"peft": {"dim": 8}}, "full-parameter training"),
        ({"ddp": {"enabled": True}}, "FSDP2"),
        ({"fsdp": None}, "fsdp.*configuration block"),
        ({"fsdp": {"tp_size": 2, "cp_size": 1, "pp_size": 1}}, "tp_size=2"),
        ({"dmd2": {}}, "model_adapter"),
        ({"data": {"dataloader": {"train_text_encoder": True}}}, "precomputed text embeddings"),
    ],
)
def test_validate_recipe_scope_rejects_unsupported_automodel_modes(config_overrides, error):
    config = _base_recipe_config()
    for key, value in config_overrides.items():
        config[key] = value
    recipe = _bare_recipe(config)

    with pytest.raises(ValueError, match=error):
        recipe._validate_recipe_scope()


def test_validate_recipe_scope_accepts_native_qwen_image_fsdp_data_parallelism():
    recipe = _bare_recipe(_base_recipe_config())

    recipe._validate_recipe_scope()


@pytest.mark.parametrize(
    ("resolved", "recipe_dmd2", "error"),
    [
        (_resolved_config(student_update_freq=0), {}, "student_update_freq"),
        (_resolved_config(student_sample_steps=0), {}, "student_sample_steps"),
        (
            _resolved_config(
                student_sample_steps=2,
                sample_t_cfg=SimpleNamespace(t_list=[1.0, 0.0]),
            ),
            {},
            r"student_sample_steps \+ 1",
        ),
        (
            _resolved_config(
                student_sample_steps=2,
                sample_t_cfg=SimpleNamespace(t_list=[1.0, 0.5, 0.5]),
            ),
            {},
            "strictly decreasing",
        ),
        (_resolved_config(gan_loss_weight_gen=-0.1), {}, "non-negative"),
        (
            _resolved_config(gan_loss_weight_gen=0.1),
            {
                "model_adapter": _model_adapter_config(
                    gan_feature_indices=[60],
                    gan_num_blocks=60,
                )
            },
            "outside",
        ),
        (
            _resolved_config(
                ema=SimpleNamespace(fsdp2=False, mode="full_tensor"),
            ),
            {},
            "exact checkpoint resume",
        ),
    ],
)
def test_validate_resolved_dmd_config_enforces_schedule_gan_and_ema_invariants(
    resolved,
    recipe_dmd2,
    error,
):
    recipe = _bare_recipe(_base_recipe_config(**recipe_dmd2))

    with pytest.raises(ValueError, match=error):
        recipe._validate_resolved_dmd_config(resolved)


def test_prepare_micro_batch_preserves_positive_mask_and_expands_negative_conditioning():
    recipe = _bare_recipe(_base_recipe_config())
    recipe.device = torch.device("cpu")
    recipe.compute_dtype = torch.float32
    recipe.discriminator = None
    recipe._gan_capture_shape = None
    recipe._negative_prompt_embedding = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    recipe._negative_prompt_mask = torch.tensor([1, 1, 0], dtype=torch.long)
    positive_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )

    prepared = recipe._prepare_micro_batch(
        {
            "image_latents": torch.zeros(2, 4, 8, 12),
            "text_embeddings": torch.ones(2, 5, 4),
            "text_embeddings_mask": positive_mask,
        }
    )

    torch.testing.assert_close(prepared.text_mask, positive_mask)
    assert prepared.negative_text_embeddings.shape == (2, 3, 4)
    assert prepared.negative_text_mask.shape == (2, 3)
    torch.testing.assert_close(prepared.negative_text_mask[0], recipe._negative_prompt_mask)
    torch.testing.assert_close(prepared.negative_text_mask[1], recipe._negative_prompt_mask)
    assert prepared.noise.shape == prepared.latents.shape
    assert prepared.noise.dtype == prepared.latents.dtype


def test_prepare_micro_batch_rejects_a_mask_that_does_not_match_text_embeddings():
    recipe = _bare_recipe(_base_recipe_config())
    recipe.device = torch.device("cpu")
    recipe.compute_dtype = torch.float32
    recipe.discriminator = None
    recipe._gan_capture_shape = None
    recipe._negative_prompt_embedding = None
    recipe._negative_prompt_mask = None

    with pytest.raises(ValueError, match=r"text mask must have shape \[B, S\]"):
        recipe._prepare_micro_batch(
            {
                "image_latents": torch.zeros(2, 4, 8, 8),
                "text_embeddings": torch.ones(2, 5, 4),
                "text_embeddings_mask": torch.ones(2, 4, dtype=torch.long),
            }
        )


def test_prepare_micro_batch_suppresses_only_an_all_valid_positive_mask_for_flash_attention():
    recipe = _bare_recipe(_base_recipe_config())
    recipe.device = torch.device("cpu")
    recipe.compute_dtype = torch.float32
    recipe.attention_backend = "flash"
    recipe.discriminator = None
    recipe._gan_capture_shape = None
    recipe._negative_prompt_embedding = None
    recipe._negative_prompt_mask = None
    micro_batch = {
        "image_latents": torch.zeros(2, 4, 8, 8),
        "text_embeddings": torch.ones(2, 5, 4),
        "text_embeddings_mask": torch.ones(2, 5, dtype=torch.long),
    }

    prepared = recipe._prepare_micro_batch(micro_batch)

    assert prepared.text_mask is None
    micro_batch["text_embeddings_mask"][1, -1] = 0
    with pytest.raises(ValueError, match="padded positive-prompt mask"):
        recipe._prepare_micro_batch(micro_batch)


@pytest.mark.parametrize("restore_from", [None, "/tmp/dmd2-checkpoint"])
def test_load_checkpoint_defers_until_all_dmd2_state_is_registered(restore_from):
    recipe = DMD2DiffusionRecipe.__new__(DMD2DiffusionRecipe)
    recipe._dmd2_state_ready = False
    recipe._restore_was_deferred = False
    recipe._deferred_restore_from = None

    with (
        patch.object(TrainDiffusionRecipe, "load_checkpoint", autospec=True) as parent_load,
        patch.object(DMD2DiffusionRecipe, "_validate_restored_dmd2_state", autospec=True) as validate_restore,
    ):
        recipe.load_checkpoint(restore_from)

        assert recipe._restore_was_deferred is True
        assert recipe._deferred_restore_from == restore_from
        parent_load.assert_not_called()

        recipe._dmd2_state_ready = True
        recipe.load_checkpoint(restore_from)

    parent_load.assert_called_once_with(recipe, restore_from)
    validate_restore.assert_called_once_with(recipe)


def test_validate_restored_dmd2_state_accepts_a_consistent_training_phase():
    recipe = _bare_recipe(_base_recipe_config())
    recipe.step_scheduler = SimpleNamespace(step=11)
    recipe._dmd_config = SimpleNamespace(student_update_freq=5)
    recipe.dmd2_state = SimpleNamespace(student_update_count=3)
    recipe.lr_scheduler = [SimpleNamespace(num_steps=3)]

    recipe._validate_restored_dmd2_state()


@pytest.mark.parametrize(
    ("student_update_count", "scheduler_steps", "error"),
    [
        (2, 2, "checkpoint phase is inconsistent"),
        (3, 2, "LR scheduler is inconsistent"),
    ],
)
def test_validate_restored_dmd2_state_rejects_partial_state(
    student_update_count,
    scheduler_steps,
    error,
):
    recipe = _bare_recipe(_base_recipe_config())
    recipe.step_scheduler = SimpleNamespace(step=11)
    recipe._dmd_config = SimpleNamespace(student_update_freq=5)
    recipe.dmd2_state = SimpleNamespace(student_update_count=student_update_count)
    recipe.lr_scheduler = [SimpleNamespace(num_steps=scheduler_steps)]

    with pytest.raises(ValueError, match=error):
        recipe._validate_restored_dmd2_state()


def test_training_loop_enforces_one_student_to_four_fake_updates_with_accumulation():
    recipe = _bare_recipe(_base_recipe_config())
    recipe.model = torch.nn.Linear(1, 1, bias=False)
    recipe.fake_score_model = torch.nn.Linear(1, 1, bias=False)
    recipe.teacher_model = torch.nn.Identity()
    recipe.discriminator = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        recipe.model.weight.fill_(1.0)
        recipe.fake_score_model.weight.fill_(1.0)
        recipe.discriminator.weight.fill_(1.0)
    recipe.optimizer = torch.optim.SGD(recipe.model.parameters(), lr=0.1)
    recipe.fake_score_optimizer = torch.optim.SGD(recipe.fake_score_model.parameters(), lr=0.1)
    recipe.discriminator_optimizer = torch.optim.SGD(recipe.discriminator.parameters(), lr=0.1)
    student_step = Mock(wraps=recipe.optimizer.step)
    fake_score_step = Mock(wraps=recipe.fake_score_optimizer.step)
    discriminator_step = Mock(wraps=recipe.discriminator_optimizer.step)
    recipe.optimizer.step = student_step
    recipe.fake_score_optimizer.step = fake_score_step
    recipe.discriminator_optimizer.step = discriminator_step

    recipe._dmd_config = SimpleNamespace(student_update_freq=5)
    update_ema = Mock()
    recipe._dmd_pipeline = SimpleNamespace(
        compute_student_loss=Mock(side_effect=lambda *args, **kwargs: {"total": recipe.model.weight.square().sum()}),
        compute_fake_score_loss=Mock(
            side_effect=lambda *args, **kwargs: {"total": recipe.fake_score_model.weight.square().sum()}
        ),
        compute_discriminator_loss=Mock(
            side_effect=lambda *args, **kwargs: {"total": recipe.discriminator.weight.square().sum()}
        ),
        update_ema=update_ema,
    )
    recipe.dmd2_state = SimpleNamespace(student_update_count=0)
    lr_scheduler = _CountingLRScheduler()
    recipe.lr_scheduler = [lr_scheduler]
    recipe.step_scheduler = _FixedStepScheduler()
    recipe.dataloader = [{}]
    recipe.sampler = None
    recipe.num_epochs = 1
    recipe.global_batch_size = 1
    recipe.local_batch_size = 1
    recipe.dp_size = 1
    recipe.device = torch.device("cpu")
    recipe.clip_grad_max_norm = 100.0
    recipe.grad_clip_foreach = False
    recipe.check_loss = True
    recipe.log_every = 0
    recipe._sync_device = Mock()
    recipe._finalize_and_close_checkpointer = Mock()
    recipe._transformer_engine_fp8_context = Mock(side_effect=lambda: nullcontext())
    recipe._prepare_micro_batch = Mock(
        return_value=_PreparedDMD2Batch(
            latents=torch.zeros(1, 1, 1, 1),
            noise=torch.ones(1, 1, 1, 1),
            text_embeddings=torch.zeros(1, 1, 1),
            text_mask=None,
            negative_text_embeddings=torch.zeros(1, 1, 1),
            negative_text_mask=None,
        )
    )

    with (
        patch.object(dmd2_module, "is_main_process", return_value=False),
        patch.object(dmd2_module, "_count_local_batch_group_samples", return_value=1),
        patch.object(dmd2_module, "prepare_for_grad_accumulation") as prepare_accumulation,
        patch.object(dmd2_module, "prepare_for_final_backward") as prepare_final_backward,
        patch.object(dmd2_module, "prepare_after_first_microbatch") as prepare_after_first,
        patch.object(dmd2_module, "clip_grad_norm", return_value=torch.tensor(0.0)),
        patch.object(dmd2_module.torch.cuda, "is_available", return_value=False),
    ):
        recipe.run_train_validation_loop()

    assert recipe._dmd_pipeline.compute_student_loss.call_count == 2
    assert recipe._dmd_pipeline.compute_fake_score_loss.call_count == 7
    assert recipe._dmd_pipeline.compute_discriminator_loss.call_count == 7
    assert student_step.call_count == 1
    assert fake_score_step.call_count == 4
    assert discriminator_step.call_count == 4
    torch.testing.assert_close(recipe.model.weight, torch.tensor([[0.8]]))
    torch.testing.assert_close(recipe.fake_score_model.weight, torch.tensor([[0.8**4]]))
    torch.testing.assert_close(recipe.discriminator.weight, torch.tensor([[0.8**4]]))
    update_ema.assert_called_once_with(iteration=1)
    assert recipe.dmd2_state.student_update_count == 1
    assert lr_scheduler.num_steps == 1
    assert recipe.step_scheduler.step == 5
    assert prepare_accumulation.call_count == 5
    assert prepare_final_backward.call_count == 5
    assert prepare_after_first.call_count == 5
    recipe._finalize_and_close_checkpointer.assert_called_once_with()


def test_gan_feature_capture_is_attached_once_per_latent_resolution():
    recipe = _bare_recipe(_base_recipe_config())
    recipe.discriminator = object()
    recipe.teacher_model = object()
    recipe._gan_capture_shape = None
    attach_feature_capture = Mock()
    recipe._model_adapter = SimpleNamespace(attach_feature_capture=attach_feature_capture)

    recipe._ensure_gan_feature_capture(32, 48)
    recipe._ensure_gan_feature_capture(32, 48)
    recipe._ensure_gan_feature_capture(64, 48)

    assert attach_feature_capture.call_count == 2
    assert attach_feature_capture.call_args_list[0].args == (recipe.teacher_model,)
    assert attach_feature_capture.call_args_list[0].kwargs == {
        "height": 32,
        "width": 48,
    }
    assert attach_feature_capture.call_args_list[1].kwargs["height"] == 64
    assert attach_feature_capture.call_args_list[1].kwargs["width"] == 48
    assert recipe._gan_capture_shape == (64, 48)


def test_gan_feature_capture_is_a_noop_when_gan_is_disabled():
    recipe = _bare_recipe(_base_recipe_config())
    recipe.discriminator = None
    recipe.teacher_model = object()
    recipe._gan_capture_shape = None
    recipe._model_adapter = SimpleNamespace(attach_feature_capture=Mock())

    recipe._ensure_gan_feature_capture(32, 48)

    recipe._model_adapter.attach_feature_capture.assert_not_called()
    assert recipe._gan_capture_shape is None


def test_qwen_image_dmd2_example_yaml_preserves_the_production_contract():
    config = yaml.safe_load(_EXAMPLE_CONFIG.read_text(encoding="utf-8"))

    assert config["recipe"] == "DMD2DiffusionRecipe"
    assert config["model"]["pretrained_model_name_or_path"] == "Qwen/Qwen-Image"
    assert config["model"]["attention_backend"] == "flash"

    dmd2 = config["dmd2"]
    assert dmd2["recipe_path"] == "general/distillation/dmd2_qwen_image"
    adapter = dmd2["model_adapter"]
    assert adapter["_target_"].endswith("qwen_image.dmd2.QwenImageDMD2Adapter")
    assert adapter["gan_feature_indices"] == [30]
    assert adapter["gan_num_blocks"] == 60
    assert adapter["gan_inner_dim"] == 3072
    assert isinstance(ConfigNode(config).dmd2.model_adapter.instantiate(), QwenImageDMD2Adapter)
    assert dmd2["guidance_scale"] == 4.0
    assert dmd2["negative_prompt_embedding_path"] == "PATH_TO_NEGATIVE_PROMPT_EMBEDDING"
    assert dmd2["student_update_freq"] == 5
    assert dmd2["student_sample_steps"] == 4
    assert len(dmd2["sample_t_cfg"]["t_list"]) == dmd2["student_sample_steps"] + 1
    assert dmd2["gan_loss_weight_gen"] > 0
    assert dmd2["gan_r1_reg_weight"] >= 0

    dataloader = config["data"]["dataloader"]
    assert dataloader["_target_"].endswith("build_text_to_image_multiresolution_dataloader")
    assert dataloader["train_text_encoder"] is False
    assert "negative_prompt_embedding_path" not in dataloader

    assert config["fsdp"]["tp_size"] == 1
    assert config["fsdp"]["cp_size"] == 1
    assert config["fsdp"]["pp_size"] == 1
    assert config["checkpoint"]["enabled"] is True
