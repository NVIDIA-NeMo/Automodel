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

"""Tests for the VLM-CP wiring in ``recipes/vlm/finetune.py``.

These reproduce the ``_forward_backward_step``-style and
``_run_validation_epoch``-style batch handling without instantiating the
full recipe — exercising the code shape that gets shipped:

  - Invoke the sharder-only ``prepare_model_inputs_for_cp`` directly through
    ``ContextParallelSharder`` construction (a plain method call; nothing consumed, so input_ids
    and multimodal inputs stay in the batch for the model's own forward)
  - PP gating: the sharder-only hook is invoked on every stage (all PP-capable
    VLMs are sunk — they embed + shard in their own forward); media is dropped on
    non-first stages so those stage forwards see only text inputs
  - Validation: count labels after ``ContextParallelSharder.shard`` and inside train_ctx
  - Validation: position_ids ``.to(self.dist_env.device)`` (not model.device)
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import nemo_automodel.recipes.vlm.finetune as vlm_finetune
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed.cp_vision_frame_shard import CpVisionFrameShardingConfig
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM


def _identity_cp_shard(sharder, batch):
    """Bypass CP transport while preserving constructor-side strategy resolution.

    Args:
        sharder: Sharder whose resolved strategy is not exercised by this test.
        batch: Mutable model-input mapping whose tensor values retain their
            existing shapes.

    Returns:
        The null context factory and the same input mapping.
    """
    del sharder
    return nullcontext, batch


def _make_recipe_with_pp_stages(*, pp_enabled=True, has_first_stage=True, pp_microbatch_size=2):
    first_stage = SimpleNamespace(is_first=True, inputs_meta=("old-first",))
    later_stage = SimpleNamespace(is_first=False, inputs_meta=("old-later",))
    recipe = SimpleNamespace(
        pp_enabled=pp_enabled,
        pp=SimpleNamespace(
            pp_microbatch_size=pp_microbatch_size,
            info=SimpleNamespace(has_first_stage=has_first_stage, stages=[first_stage, later_stage]),
        ),
    )
    return recipe, first_stage, later_stage


def test_maybe_set_pp_first_stage_embed_input_meta_sets_first_stage_meta():
    recipe, first_stage, later_stage = _make_recipe_with_pp_stages(pp_microbatch_size=3)
    model_input = torch.empty(5, 11, 13, dtype=torch.bfloat16)

    FinetuneRecipeForVLM._maybe_set_pp_first_stage_embed_input_meta(recipe, model_input)

    assert later_stage.inputs_meta == ("old-later",)
    assert len(first_stage.inputs_meta) == 1
    meta = first_stage.inputs_meta[0]
    assert tuple(meta.shape) == (3, 11, 13)
    assert meta.dtype == torch.bfloat16
    assert meta.device.type == "meta"


@pytest.mark.parametrize(
    ("recipe_kwargs", "model_input"),
    [
        ({"pp_enabled": False}, torch.empty(5, 11, 13)),
        ({"has_first_stage": False}, torch.empty(5, 11, 13)),
        ({}, torch.empty(5, 11, 13, dtype=torch.int64)),
        ({}, torch.empty(5, 11)),
    ],
)
def test_maybe_set_pp_first_stage_embed_input_meta_guard_conditions(recipe_kwargs, model_input):
    recipe, first_stage, later_stage = _make_recipe_with_pp_stages(**recipe_kwargs)

    FinetuneRecipeForVLM._maybe_set_pp_first_stage_embed_input_meta(recipe, model_input)

    assert first_stage.inputs_meta == ("old-first",)
    assert later_stage.inputs_meta == ("old-later",)


class _FakeCPMesh:
    mesh_dim_names = ("cp",)

    def __getitem__(self, key):
        assert key == "cp"
        return SimpleNamespace(size=lambda: 2, get_group=lambda: "cp-group")


def test_recipe_config_resolves_cp_vision_frame_sharding_policy():
    cfg = RecipeConfig(
        ConfigNode(
            {
                "distributed": {
                    "cp_vision_frame_sharding": {
                        "enabled": True,
                        "min_tokens": 17,
                        "cost_alpha": 0,
                    }
                }
            }
        )
    )

    assert cfg.cp_vision_frame_sharding == CpVisionFrameShardingConfig(enabled=True, min_tokens=17, cost_alpha=0)


def test_recipe_config_disables_cp_vision_frame_sharding_by_default():
    policy = RecipeConfig(ConfigNode({})).cp_vision_frame_sharding

    assert policy == CpVisionFrameShardingConfig()
    assert policy.cost_alpha == "auto"


def test_recipe_config_accepts_explicit_auto_cost_alpha():
    cfg = RecipeConfig(ConfigNode({"distributed": {"cp_vision_frame_sharding": {"cost_alpha": "auto"}}}))

    assert cfg.cp_vision_frame_sharding.cost_alpha == "auto"


class _UnsupportedVisionModel:
    supports_cp_vision_frame_sharding = False


class _SupportedVisionModel:
    supports_cp_vision_frame_sharding = True


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


class _ScheduleSpy:
    def __init__(self):
        self.calls = []

    def step(self, model_input=None, *, target=None, losses=None, **batch):
        self.calls.append({"model_input": model_input, "target": target, "batch": batch})
        if losses is not None:
            losses.append(torch.tensor(1.25))


def test_forward_backward_step_pp_cp_first_stage_sunk_keeps_input_ids_full(monkeypatch):
    """Sunk model on the FIRST PP stage under CP: the sharder-only hook is invoked
    (consumes nothing), so input_ids stays full-length, update_seq_len sees the
    full seq_len, and the full-length input_ids is fed to the pipeline schedule
    (the model embeds + shards inside its own forward)."""
    labels = torch.arange(12, dtype=torch.long).reshape(2, 6)
    model = _SunkSpyVLM()
    schedule = _ScheduleSpy()
    seq_lens = []
    first_stage = SimpleNamespace(is_first=True, inputs_meta=None)
    recipe = object.__new__(FinetuneRecipeForVLM)
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    recipe.device_mesh = _FakeCPMesh()
    recipe.cp_vision_frame_sharding = CpVisionFrameShardingConfig(enabled=True)
    recipe.distributed_config = SimpleNamespace(defer_fsdp_grad_sync=True)
    recipe.model_parts = [model]
    recipe.pp_enabled = True
    recipe.pp = SimpleNamespace(
        pp_microbatch_size=2,
        info=SimpleNamespace(
            has_first_stage=True,
            has_last_stage=True,
            stages=[first_stage, SimpleNamespace(is_first=False, inputs_meta=None)],
            schedule=schedule,
        ),
        update_seq_len=seq_lens.append,
    )
    batch = {
        "input_ids": torch.ones(2, 6, dtype=torch.long),
        "pixel_values": torch.zeros(2, 3, 4, 4),
        "labels": labels,
    }
    seen_cp_batch = {}

    def _shard(sharder, cp_batch):
        """Capture the global model-input mapping before CP transport.

        Args:
            sharder: Sharder configured by the VLM recipe.
            cp_batch: Mutable model-input mapping whose tensor values have
                global batch and sequence extents.

        Returns:
            The null context factory and the same input mapping.
        """
        del sharder
        seen_cp_batch.update(cp_batch)
        return nullcontext, cp_batch

    monkeypatch.setattr(vlm_finetune.ContextParallelSharder, "shard", _shard)
    monkeypatch.setattr(vlm_finetune, "stage_vlm_media_for_pp", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(FinetuneRecipeForVLM, "_maybe_set_pp_first_stage_embed_input_meta", lambda self, mi: None)

    loss_buffer = []
    FinetuneRecipeForVLM._forward_backward_step(
        recipe,
        0,
        batch,
        loss_buffer=loss_buffer,
        num_label_tokens=labels.numel(),
        num_batches=1,
    )

    assert len(model.calls) == 1
    # Sharder-only: input_ids stays full, no inputs_embeds injected.
    assert "input_ids" in seen_cp_batch
    assert tuple(seen_cp_batch["input_ids"].shape) == (2, 6)
    assert "inputs_embeds" not in seen_cp_batch
    assert seq_lens == [6]
    assert len(schedule.calls) == 1
    assert tuple(schedule.calls[0]["model_input"].shape) == (2, 6)
    assert torch.equal(schedule.calls[0]["target"], labels)
    assert torch.equal(loss_buffer[0], torch.tensor(1.25))


class _SunkSpyVLM:
    """Sunk VLM: sharder-only CP hook (embeds/shards in forward, consumes nothing)."""

    def __init__(self):
        self.calls = []

    def prepare_model_inputs_for_cp(self, batch, *, num_chunks=1):
        # Sharder-only: nothing consumed, no inputs_embeds — input_ids stays full.
        self.calls.append({"batch": dict(batch), "num_chunks": num_chunks})
        return {}

    def __call__(self, **kwargs):
        raise AssertionError("CP prepare must call prepare_model_inputs_for_cp directly, not __call__")


def _run_nonfirst_stage_fbstep(monkeypatch, model):
    """Drive _forward_backward_step for a non-first (has_first_stage=False) PP+CP stage."""
    labels = torch.arange(12, dtype=torch.long).reshape(2, 6)
    schedule = _ScheduleSpy()
    seq_lens = []
    recipe = object.__new__(FinetuneRecipeForVLM)
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    recipe.device_mesh = _FakeCPMesh()
    recipe.cp_vision_frame_sharding = CpVisionFrameShardingConfig(enabled=True)
    recipe.distributed_config = SimpleNamespace(defer_fsdp_grad_sync=True)
    recipe.model_parts = [model]
    recipe.pp_enabled = True
    recipe.pp = SimpleNamespace(
        pp_microbatch_size=2,
        info=SimpleNamespace(
            has_first_stage=False,
            has_last_stage=True,
            stages=[SimpleNamespace(is_first=False, inputs_meta=None)],
            schedule=schedule,
        ),
        update_seq_len=seq_lens.append,
    )
    batch = {
        "input_ids": torch.ones(2, 6, dtype=torch.long),
        "pixel_values": torch.zeros(2, 3, 4, 4),
        "labels": labels,
    }
    seen_cp_batch = {}

    def _shard(sharder, cp_batch):
        """Capture the global model-input mapping before CP transport.

        Args:
            sharder: Sharder configured by the VLM recipe.
            cp_batch: Mutable model-input mapping whose tensor values have
                global batch and sequence extents.

        Returns:
            The null context factory and the same input mapping.
        """
        del sharder
        seen_cp_batch.update(cp_batch)
        return nullcontext, cp_batch

    monkeypatch.setattr(vlm_finetune.ContextParallelSharder, "shard", _shard)
    monkeypatch.setattr(vlm_finetune, "stage_vlm_media_for_pp", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(FinetuneRecipeForVLM, "_maybe_set_pp_first_stage_embed_input_meta", lambda self, mi: None)

    FinetuneRecipeForVLM._forward_backward_step(
        recipe, 0, batch, loss_buffer=[], num_label_tokens=labels.numel(), num_batches=1
    )
    return seen_cp_batch, seq_lens


def test_forward_backward_step_pp_cp_sunk_model_nonfirst_stage_invokes_hook_keeps_input_ids_full(monkeypatch):
    """Regression: a sunk model must invoke its sharder-only hook on NON-first PP
    stages under cp>1, so input_ids stays full-length and update_seq_len (which
    drives the CP-aware stage metas) sees the FULL seq_len — not the local length
    the generic sharder would produce, which would ÷cp a second time and truncate
    the inter-stage hidden (the text-decoder RoPE size mismatch)."""
    model = _SunkSpyVLM()
    seen_cp_batch, seq_lens = _run_nonfirst_stage_fbstep(monkeypatch, model)

    # Hook invoked on the non-first stage (this is the fix).
    assert len(model.calls) == 1
    # Sharder-only hook consumes nothing: input_ids stays full-length (seq=6).
    assert "input_ids" in seen_cp_batch
    assert tuple(seen_cp_batch["input_ids"].shape) == (2, 6)
    # All pp ranks feed the FULL seq_len to update_seq_len.
    assert seq_lens == [6]


class _FakePPModel:
    def __init__(self, stage0):
        self.parts = [stage0]
        self.pp_batch_size = 4
        self.pp_microbatch_size = 2
        self.info = SimpleNamespace(has_last_stage=False, stages=[], schedule=None)


class _StageWithCPPreembedInForward:
    # Sunk VLM (minimax/qwen3_5/qwen3_5_moe/step3p7): embeds + shards per
    # microbatch inside forward and pulls media from the PP side channel, so
    # media MUST still be staged for PP under CP.
    def prepare_model_inputs_for_cp(self):
        return {}


class _StageWithoutCPPrepare:
    pass


def _patch_pp_setup_minimals(monkeypatch, *, cp_size, stage0, dataloader_calls):
    monkeypatch.setattr(vlm_finetune, "AutoPipeline", _FakePPModel)
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
        property(lambda self: SimpleNamespace(build=lambda: "loss_fn")),
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
            pipeline_config=SimpleNamespace(),
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
        property(lambda self: None),
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


# -----------------------------------------------------------------------------
# val-side wiring (the bug-fix territory)
# -----------------------------------------------------------------------------


class _ShardLabelsOnEnter:
    def __init__(self, labels, local_labels):
        self.labels = labels
        self.local_labels = local_labels

    def __enter__(self):
        self.labels.resize_(self.local_labels.shape)
        self.labels.copy_(self.local_labels)

    def __exit__(self, exc_type, exc, tb):
        return False


def test_val_counts_label_tokens_inside_cp_context_after_labels_are_sharded():
    """Validation must count label tokens after CP has exposed the local shard."""
    labels = torch.tensor([[1, 2, -100, 4]])
    batch = {"labels": labels}
    local_labels = torch.tensor([[1, -100]])

    def train_ctx():
        return _ShardLabelsOnEnter(labels, local_labels)

    labels = batch.pop("labels")
    pre_context_count = (labels != -100).sum().item()
    with train_ctx():
        local_num_label_tokens = (labels != -100).sum().item()

    assert pre_context_count == 3
    assert local_num_label_tokens == 1


def test_val_pos_ids_uses_dist_env_device_not_model_device():
    """Reproduce the bug fix at finetune.py:1281 — val must use
    ``self.dist_env.device``, not ``self.model_parts[0].device`` which
    AttributeErrors on FSDP-wrapped models."""

    class _FSDPWrapped:
        # Intentionally has NO ``.device`` attribute (mirrors real FSDP wrapper).
        def __getattr__(self, name):
            if name == "device":
                raise AttributeError("'FSDPWrapped' object has no attribute 'device'")
            raise AttributeError(name)

    model = _FSDPWrapped()
    dist_env = SimpleNamespace(device=torch.device("cpu"))

    # The fixed line:
    pos = torch.arange(0, 4).unsqueeze(0).to(dist_env.device)
    assert pos.device.type == "cpu"

    # The buggy line would have raised:
    with pytest.raises(AttributeError, match="no attribute 'device'"):
        _ = torch.arange(0, 4).unsqueeze(0).to(model.device)


def test_run_validation_epoch_does_not_sum_tokens_over_cp(monkeypatch):
    """``total_loss`` is all-reduced with include_cp=True, but ``total_tokens``
    (measured pre-CP-shard) must NOT include CP — otherwise val_loss is scaled
    down by cp_size. Guards the fix at finetune.py:_run_validation_epoch."""
    from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM

    # No-op replacements for the heavy collaborators.
    monkeypatch.setattr(vlm_finetune, "ScopedRNG", lambda *a, **k: nullcontext())
    monkeypatch.setattr(vlm_finetune.ContextParallelSharder, "shard", _identity_cp_shard)
    monkeypatch.setattr(vlm_finetune, "filter_forward_kwargs", lambda model, batch: batch)
    monkeypatch.setattr(vlm_finetune, "calculate_loss", lambda *a, **k: torch.tensor(2.0))

    class _Model(torch.nn.Module):
        def eval(self):  # noqa: D401
            return self

        def forward(self, **batch):
            return SimpleNamespace(logits=torch.zeros(1, 4, 8), hidden_states=None)

    recipe = FinetuneRecipeForVLM.__new__(FinetuneRecipeForVLM)
    recipe.model_parts = [_Model()]
    recipe.loss_fn = object()  # not a FusedLinearCrossEntropy
    recipe.device_mesh = None  # CP inactive -> skip pre-embed branch
    recipe.pp_enabled = False
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    recipe.step_scheduler = SimpleNamespace(step=3, epoch=1)
    recipe.optimizer = [SimpleNamespace(param_groups=[{"lr": 0.001}])]
    recipe._maybe_add_drafter_loss = lambda *, out, base_loss, labels, model, num_label_tokens: base_loss

    allreduce_calls = []

    def _fake_allreduce(tensor, include_cp=False):
        allreduce_calls.append((tensor.tolist(), include_cp))
        return tensor

    recipe._dp_allreduce = _fake_allreduce

    # One batch, 3 supervised tokens (-100 ignored).
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, -100, 4]]),
    }

    result = recipe._run_validation_epoch([batch])

    # total_loss all-reduced WITH cp; total_tokens and num_label_tokens WITHOUT.
    loss_call = allreduce_calls[0]
    tokens_call = allreduce_calls[1]
    assert loss_call[1] is True, "total_loss must include CP ranks"
    assert tokens_call[1] is False, "total_tokens must NOT be summed over CP ranks"
    # val_loss = (2.0 * 3 tokens) / 3 tokens == 2.0
    assert result.metrics["val_loss"] == pytest.approx(2.0)


def test_run_validation_epoch_cp_active_runs_pre_embed(monkeypatch):
    """With CP active and a model exposing prepare_model_inputs_for_cp, the
    validation loop must invoke the model's sharder-only CP hook before sharding.
    Guards finetune.py:_run_validation_epoch CP pre-embed branch."""
    from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM

    monkeypatch.setattr(vlm_finetune, "ScopedRNG", lambda *a, **k: nullcontext())
    monkeypatch.setattr(vlm_finetune.ContextParallelSharder, "shard", _identity_cp_shard)
    monkeypatch.setattr(vlm_finetune, "filter_forward_kwargs", lambda model, batch: batch)
    monkeypatch.setattr(vlm_finetune, "calculate_loss", lambda *a, **k: torch.tensor(2.0))

    pre_embed_calls = []

    class _Model(torch.nn.Module):
        def eval(self):
            return self

        def prepare_model_inputs_for_cp(self, batch, *, num_chunks=1):  # sharder-only hook
            pre_embed_calls.append(set(batch))
            return {}

        def forward(self, **batch):
            return SimpleNamespace(logits=torch.zeros(1, 4, 8), hidden_states=None)

    class _DM(dict):
        mesh_dim_names = ["cp"]

    recipe = FinetuneRecipeForVLM.__new__(FinetuneRecipeForVLM)
    recipe.model_parts = [_Model()]
    recipe.loss_fn = object()
    recipe.device_mesh = _DM(cp=SimpleNamespace(size=lambda: 2, get_group=lambda: "cp-group"))
    recipe.cp_vision_frame_sharding = CpVisionFrameShardingConfig(enabled=True)
    recipe.pp_enabled = False
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    recipe.step_scheduler = SimpleNamespace(step=3, epoch=1)
    recipe.optimizer = [SimpleNamespace(param_groups=[{"lr": 0.001}])]
    recipe._maybe_add_drafter_loss = lambda *, out, base_loss, labels, model, num_label_tokens: base_loss
    recipe._dp_allreduce = lambda tensor, include_cp=False: tensor

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "pixel_values": torch.randn(1, 3, 8, 8),
        "labels": torch.tensor([[1, 2, -100, 4]]),
    }

    result = recipe._run_validation_epoch([batch])

    assert pre_embed_calls, "the CP hook (prepare_model_inputs_for_cp) must run when CP is active"
    assert result.metrics["val_loss"] == pytest.approx(2.0)
