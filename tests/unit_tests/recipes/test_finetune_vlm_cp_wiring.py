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

"""Tests for VLM context-parallel wiring in ``recipes/vlm/finetune.py``.

Training forward/backward and CP sharding are owned by :class:`Engine`. These
tests cover the VLM recipe responsibilities that remain around that core:
pipeline media staging setup, vision-frame context publication, and the
validation handoff plus epoch-level DP aggregation.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import nemo_automodel.recipes.vlm.finetune as vlm_finetune
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed.cp_vision_frame_shard import CpVisionFrameShardingConfig
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM


class _UnsupportedVisionModel:
    supports_cp_vision_frame_sharding = False


class _SupportedVisionModel:
    supports_cp_vision_frame_sharding = True


class _PackedCPModel:
    def __init__(self, *, supported):
        self.supports_cp_with_sequence_packing = supported


def test_cp_vision_frame_sharding_rejects_model_without_capability():
    policy = CpVisionFrameShardingConfig(enabled=True)

    with pytest.raises(
        ValueError,
        match=r"_UnsupportedVisionModel declares supports_cp_vision_frame_sharding=False",
    ):
        vlm_finetune._validate_cp_vision_frame_sharding_support(_UnsupportedVisionModel(), policy)


def test_cp_vision_frame_sharding_accepts_model_with_capability():
    policy = CpVisionFrameShardingConfig(enabled=True)

    vlm_finetune._validate_cp_vision_frame_sharding_support(_SupportedVisionModel(), policy)


def test_disabled_cp_vision_frame_sharding_accepts_model_without_capability():
    policy = CpVisionFrameShardingConfig(enabled=False)

    vlm_finetune._validate_cp_vision_frame_sharding_support(_UnsupportedVisionModel(), policy)


def test_packed_cp_rejects_unsupported_qwen_backend_before_dataloader_build():
    with pytest.raises(ValueError, match="Disable sequence packing, use cp_size=1"):
        vlm_finetune._validate_cp_packing_support(
            _PackedCPModel(supported=False),
            packing_enabled=True,
            cp_size=8,
        )


@pytest.mark.parametrize(("packing_enabled", "cp_size"), [(False, 8), (True, 1)])
def test_packed_cp_validation_is_inactive_without_composed_path(packing_enabled, cp_size):
    vlm_finetune._validate_cp_packing_support(
        _PackedCPModel(supported=False),
        packing_enabled=packing_enabled,
        cp_size=cp_size,
    )


def test_packed_cp_accepts_model_owned_backend():
    vlm_finetune._validate_cp_packing_support(
        _PackedCPModel(supported=True),
        packing_enabled=True,
        cp_size=8,
    )


def test_cp_vision_frame_sharding_context_resets_published_group_after_failure(monkeypatch):
    """The recipe must restore vision frame-sharding state when the model forward raises."""
    recipe = object.__new__(FinetuneRecipeForVLM)
    group = object()
    token = object()
    policy = CpVisionFrameShardingConfig(enabled=True)

    class _Mesh(dict):
        mesh_dim_names = ("cp",)

    recipe.device_mesh = _Mesh(cp=SimpleNamespace(size=lambda: 2, get_group=lambda: group))
    recipe.cp_vision_frame_sharding = policy
    events = []

    def _set(actual_group, *, config):
        events.append(("set", actual_group, config))
        return token

    def _reset(actual_token):
        events.append(("reset", actual_token))

    monkeypatch.setattr(vlm_finetune, "set_cp_vision_group", _set)
    monkeypatch.setattr(vlm_finetune, "reset_cp_vision_group", _reset)

    with pytest.raises(RuntimeError, match="forward failed"):
        with recipe._cp_vision_frame_sharding_context():
            events.append(("forward",))
            raise RuntimeError("forward failed")

    assert events[0] == ("set", group, policy)
    assert events[1] == ("forward",)
    assert events[2] == ("reset", token)


class _FakePPModel:
    def __init__(self, stage0):
        self.parts = [stage0]
        self.pp_batch_size = 4
        self.pp_microbatch_size = 2
        self.scale_grads_in_schedule = False
        self.info = SimpleNamespace(has_first_stage=True, has_last_stage=False, stages=[], schedule=None)


class _StageWithCPPreembedInForward:
    # Sunk VLM (minimax/qwen3_5/qwen3_5_moe/step3p7): embeds + shards per
    # microbatch inside forward and pulls media from the PP side channel, so
    # media MUST still be staged for PP under CP.
    def prepare_model_inputs_for_cp(self):
        return {}


class _StageWithoutCPPrepare:
    pass


def _patch_pp_setup_minimals(
    monkeypatch,
    *,
    cp_size,
    stage0,
    dataloader_calls,
    validation_loader_config=None,
):
    monkeypatch.setattr(vlm_finetune, "AutoPipeline", _FakePPModel)
    monkeypatch.setattr("nemo_automodel.engine._engine.AutoPipeline", _FakePPModel)
    monkeypatch.setattr(
        vlm_finetune,
        "initialize_distributed",
        lambda *a, **k: SimpleNamespace(world_size=1, is_main=True, device=torch.device("cpu"), rank=0),
    )
    monkeypatch.setattr(vlm_finetune, "setup_logging", lambda: None)
    monkeypatch.setattr(vlm_finetune, "apply_cache_compatibility_patches", lambda: None)
    monkeypatch.setattr(vlm_finetune, "StatefulRNG", lambda *args, **kwargs: "rng")
    monkeypatch.setattr(
        "nemo_automodel.recipes._typed_config.RecipeConfig.loss_fn",
        property(lambda self: SimpleNamespace(build=lambda: MaskedCrossEntropy(reduction="sum"))),
    )
    monkeypatch.setattr(vlm_finetune, "_supports_logits_to_keep", lambda model: True)
    monkeypatch.setattr(
        vlm_finetune,
        "create_distributed_setup_from_config",
        lambda cfg, world_size: SimpleNamespace(
            mesh_context=SimpleNamespace(
                pp_enabled=True,
                device_mesh=None,
                moe_mesh=None,
                cp_size=cp_size,
                pp_size=2,
            ),
            strategy_config=SimpleNamespace(),
            pipeline_config=SimpleNamespace(scale_grads_in_schedule=False),
            moe_parallel_config=None,
            activation_checkpointing=False,
        ),
    )

    def _stub_build_checkpoint_config(*args, **kwargs):
        cfg = SimpleNamespace(checkpoint_dir="ckpts", model_state_dict_keys=None)
        cfg.build = lambda **kw: SimpleNamespace(
            config=cfg,
            load_base_model=lambda *args, **kwargs: None,
            maybe_wait_for_staging=lambda: None,
            close=lambda: None,
        )
        return cfg

    monkeypatch.setattr(
        "nemo_automodel.recipes._typed_config.RecipeConfig.checkpoint",
        property(lambda self: _stub_build_checkpoint_config()),
    )
    monkeypatch.setattr(vlm_finetune, "build_model", lambda *args, **kwargs: _FakePPModel(stage0))
    monkeypatch.setattr(
        "nemo_automodel.recipes._typed_config.RecipeConfig.optimizer",
        property(
            lambda self: SimpleNamespace(
                build=lambda *args, **kwargs: [SimpleNamespace(param_groups=[{"lr": 0.01}], step=lambda: None)]
            )
        ),
    )

    def _build_dataloader(**kwargs):
        dataloader_calls.append(kwargs)
        return SimpleNamespace(dataloader="dl", processor="processor")

    loader_config = SimpleNamespace(
        packing=None,
        resolve_packing_attn_implementation=lambda **kwargs: None,
        build=_build_dataloader,
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes._typed_config.RecipeConfig.vlm_dataloader",
        property(lambda self: loader_config),
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes._typed_config.RecipeConfig.vlm_validation_dataloader",
        property(lambda self: validation_loader_config),
    )
    monkeypatch.setattr(vlm_finetune, "ScopedRNG", lambda **kwargs: nullcontext())
    monkeypatch.setattr(
        "nemo_automodel.components.training.step_scheduler.StepSchedulerConfig.build",
        lambda self, *args, **kwargs: SimpleNamespace(step=0, epoch=0, epochs=[]),
    )
    monkeypatch.setattr(
        "nemo_automodel.components.optim.optimizer.LRSchedulerConfig.build", lambda self, *args, **kwargs: []
    )
    monkeypatch.setattr(
        vlm_finetune,
        "build_metric_logger",
        lambda *args, **kwargs: SimpleNamespace(log=lambda *args, **kwargs: None, close=lambda: None),
    )
    monkeypatch.setattr(vlm_finetune.torch.cuda, "reset_peak_memory_stats", lambda: None, raising=False)
    monkeypatch.setattr(FinetuneRecipeForVLM, "_log_experiment_details", lambda self: None)
    monkeypatch.setattr(FinetuneRecipeForVLM, "_log_library_versions", lambda self: None)
    monkeypatch.setattr(FinetuneRecipeForVLM, "_log_model_and_optimizer_details", lambda *args, **kwargs: None)
    monkeypatch.setattr(FinetuneRecipeForVLM, "_setup_garbage_collection", lambda *args, **kwargs: None)
    monkeypatch.setattr(FinetuneRecipeForVLM, "load_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(FinetuneRecipeForVLM, "_log_step_scheduler_details", lambda *args, **kwargs: None)
    monkeypatch.setattr(FinetuneRecipeForVLM, "_get_dp_rank", lambda self, include_cp=False: 0)
    monkeypatch.setattr(FinetuneRecipeForVLM, "_get_tp_rank", lambda self: 0)
    monkeypatch.setattr(FinetuneRecipeForVLM, "_get_pp_rank", lambda self: 0)
    monkeypatch.setattr(FinetuneRecipeForVLM, "_get_dp_group_size", lambda self, include_cp=False: 1)


def _minimal_pp_setup_cfg():
    return ConfigNode(
        {
            "model": {
                "pretrained_model_name_or_path": "dummy/model",
                "backend": {},
            },
            "dataset": {"path_or_dataset": "dummy"},
            "dataloader": {},
            "step_scheduler": {"local_batch_size": 4, "global_batch_size": 4},
            "optimizer": {},
            "loss_fn": {},
            "checkpoint": {"best_metric_key": "default"},
            "distributed": {"pipeline": {"pp_microbatch_size": 2}},
        }
    )


@pytest.mark.parametrize(
    ("cp_size", "stage0", "expected_pp_n_microbatches"),
    [
        # Sunk VLM under CP: pulls media from the PP side channel per microbatch,
        # so media MUST be staged (regression guard for the 156-vs-160 vision
        # RoPE mismatch when raw media was left for torch pipelining to row-chunk).
        (2, _StageWithCPPreembedInForward(), 2),
        # Sunk VLM without CP: PP still stages media.
        (1, _StageWithCPPreembedInForward(), 2),
        # No CP hook at all: standard PP staging.
        (2, _StageWithoutCPPrepare(), 2),
    ],
)
def test_setup_always_stages_pp_media_under_pp(
    monkeypatch,
    cp_size,
    stage0,
    expected_pp_n_microbatches,
):
    """Under PP, media is always staged per microbatch (pp_n_microbatches set) — CP
    and the model's pre-embed flavor no longer skip it. Every PP-capable VLM is sunk
    and pulls media from the PP side channel; leaving raw media on schedule.step
    desyncs the vision RoPE (156-vs-160)."""
    dataloader_calls = []
    _patch_pp_setup_minimals(monkeypatch, cp_size=cp_size, stage0=stage0, dataloader_calls=dataloader_calls)
    trainer = FinetuneRecipeForVLM(_minimal_pp_setup_cfg())

    trainer.setup()

    assert dataloader_calls[0]["pp_n_microbatches"] == expected_pp_n_microbatches
    assert dataloader_calls[0]["cp_size"] == cp_size
    assert trainer.engine.pipeline is trainer.pp
    assert trainer.engine.microbatch_size == 1


def test_setup_stages_pp_validation_media_and_preserves_packing_wiring(monkeypatch):
    dataloader_calls = []
    packing_resolutions = []
    configure_packing_calls = []

    def _resolve_validation_packing(**kwargs):
        packing_resolutions.append(kwargs)
        return "sdpa"

    validation_loader_config = SimpleNamespace(
        drop_last=True,
        packing=SimpleNamespace(packing_format="neat"),
        resolve_packing_attn_implementation=_resolve_validation_packing,
        build=lambda **kwargs: (
            dataloader_calls.append(kwargs) or SimpleNamespace(dataloader="val_dl", processor="processor")
        ),
    )
    _patch_pp_setup_minimals(
        monkeypatch,
        cp_size=1,
        stage0=_StageWithoutCPPrepare(),
        dataloader_calls=dataloader_calls,
        validation_loader_config=validation_loader_config,
    )
    monkeypatch.setattr(
        "nemo_automodel.components.models.common.packing.configure_packing",
        lambda **kwargs: configure_packing_calls.append(kwargs),
    )
    trainer = FinetuneRecipeForVLM(_minimal_pp_setup_cfg())

    trainer.setup()

    assert len(dataloader_calls) == 2
    validation_call = dataloader_calls[1]
    assert validation_call["pp_n_microbatches"] == 2
    assert validation_call["packing_attn_implementation"] == "sdpa"
    assert validation_call["cp_size"] == 1
    assert packing_resolutions == [{"model_attn_implementation": "sdpa", "cp_size": 1}]
    assert configure_packing_calls == [{"attn_implementation": "sdpa"}]
    assert trainer.val_dataloader == "val_dl"


def test_setup_rejects_incomplete_pp_validation_batches(monkeypatch):
    dataloader_calls = []
    validation_loader_config = SimpleNamespace(
        drop_last=False,
        packing=None,
        resolve_packing_attn_implementation=lambda **kwargs: None,
        build=lambda **kwargs: pytest.fail("validation loader must not build before drop_last validation"),
    )
    _patch_pp_setup_minimals(
        monkeypatch,
        cp_size=1,
        stage0=_StageWithoutCPPrepare(),
        dataloader_calls=dataloader_calls,
        validation_loader_config=validation_loader_config,
    )
    trainer = FinetuneRecipeForVLM(_minimal_pp_setup_cfg())

    with pytest.raises(ValueError, match=r"validation_dataloader\.drop_last=true"):
        trainer.setup()

    assert len(dataloader_calls) == 1


def test_train_loop_runs_validation_when_pipeline_is_enabled():
    class _SingleStepScheduler:
        epochs = (0,)
        step = 1
        epoch = 0
        is_val_step = True
        is_ckpt_step = False
        sigterm_flag = False

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __iter__(self):
            yield [object()]

    recipe = object.__new__(FinetuneRecipeForVLM)
    model_part = SimpleNamespace(train=MagicMock())
    recipe.model_parts = [model_part]
    recipe.step_scheduler = _SingleStepScheduler()
    recipe.val_dataloader = object()
    recipe.pp_enabled = True
    recipe._make_progress_bar = MagicMock(return_value=None)
    recipe._run_train_optim_step = MagicMock(return_value=SimpleNamespace(metrics={"loss": 1.0}))
    recipe.log_train_metrics = MagicMock()
    recipe._update_progress_bar = MagicMock()
    validation_metrics = SimpleNamespace(metrics={"val_loss": 0.25})
    recipe._run_validation_epoch = MagicMock(return_value=validation_metrics)
    recipe.log_val_metrics = MagicMock()
    recipe.save_checkpoint = MagicMock()
    recipe._maybe_collect_garbage = MagicMock()
    recipe.metric_logger_train = SimpleNamespace(close=MagicMock())
    recipe.metric_logger_valid = SimpleNamespace(close=MagicMock())
    recipe._finalize_and_close_checkpointer = MagicMock()

    recipe.run_train_validation_loop()

    recipe._run_validation_epoch.assert_called_once_with(recipe.val_dataloader)
    recipe.log_val_metrics.assert_called_once_with(validation_metrics)
    assert model_part.train.call_count == 2
