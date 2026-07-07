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

  - Iterate the umbrella ``VLM_INPUT_KEYS`` to filter the batch
  - Call model via ``__call__`` with ``_pre_embed_only=True`` (so FSDP2
    forward pre-hooks fire)
  - Pop *all* umbrella keys from batch after prepare step
  - Update batch with the prepared dict (which carries ``inputs_embeds``)
  - Validation: count labels after make_cp_batch_and_ctx and inside train_ctx
  - Validation: position_ids ``.to(self.dist_env.device)`` (not model.device)
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import nemo_automodel.recipes.vlm.finetune as vlm_finetune
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed import cp_utils as cp_utils_mod
from nemo_automodel.components.utils.model_utils import VLM_INPUT_KEYS
from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM

# -----------------------------------------------------------------------------
# Helpers reproducing the recipe's CP-prepare block (train + val flavors).
# -----------------------------------------------------------------------------


def _train_cp_prepare(_model, batch, *, pp_enabled=False, has_first_stage=True):
    """Replicates the train-side CP prepare block in
    ``recipes/vlm/finetune.py::_forward_backward_step`` (lines 960-967)."""
    if not hasattr(_model, "prepare_model_inputs_for_cp"):
        return batch
    if not pp_enabled or has_first_stage:
        prepared = _model(_pre_embed_only=True, _cp_batch=batch, num_chunks=1)
        # dispatcher contract: None values mark consumed keys -> removed
        for k, v in prepared.items():
            if v is None:
                batch.pop(k, None)
            else:
                batch[k] = v
    else:
        for k in VLM_INPUT_KEYS:
            if k != "input_ids":
                batch.pop(k, None)
    return batch


# -----------------------------------------------------------------------------
# Spy model: records what it was called with and returns a controllable dict.
# -----------------------------------------------------------------------------


class _SpyVLM:
    def __init__(self, prepared=None):
        self.calls = []
        self.prepared = prepared or {"inputs_embeds": torch.zeros(1, 4, 8)}

    def prepare_model_inputs_for_cp(self, **kwargs):
        # Existence required so recipe's hasattr() check fires; never called by
        # the recipe (the recipe routes through __call__).
        return self.prepared

    def __call__(self, *, _pre_embed_only=False, **kwargs):
        record = {"_pre_embed_only": _pre_embed_only, **kwargs}
        if isinstance(record.get("_cp_batch"), dict):
            # snapshot: the dispatcher mutates the live batch after this call
            record["_cp_batch"] = dict(record["_cp_batch"])
        self.calls.append(record)
        if _pre_embed_only:
            # Mirror the production contract: consumed raw inputs are returned
            # as None markers so the dispatcher removes them from the batch.
            # This spy "consumes" every input key it received except loss/mask
            # bookkeeping (a real model lists exactly what it embedded).
            cp_batch = kwargs.get("_cp_batch") or {}
            consumed = {k: None for k in cp_batch if k not in ("labels", "attention_mask", "position_ids")}
            return {**consumed, **self.prepared}
        raise AssertionError("recipe must use _pre_embed_only=True for the CP prepare step")


# -----------------------------------------------------------------------------
# train-side wiring
# -----------------------------------------------------------------------------


def test_train_cp_prepare_routes_through_call_with_pre_embed_only_flag():
    """The recipe must invoke model(...) — NOT the bound prepare_model_inputs_for_cp —
    so FSDP2's forward pre-hook fires."""
    inputs_embeds = torch.randn(1, 4, 8)
    model = _SpyVLM(prepared={"inputs_embeds": inputs_embeds})
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "pixel_values": torch.zeros(1, 3, 4, 4),
        "labels": torch.tensor([[1, 2, 3, 4]]),
    }
    out_batch = _train_cp_prepare(model, batch)

    assert len(model.calls) == 1
    assert model.calls[0]["_pre_embed_only"] is True
    # the whole batch rides through the opaque _cp_batch kwarg
    assert "input_ids" in model.calls[0]["_cp_batch"]
    assert "pixel_values" in model.calls[0]["_cp_batch"]
    # The returned batch contains inputs_embeds (not input_ids)
    assert "inputs_embeds" in out_batch
    assert torch.equal(out_batch["inputs_embeds"], inputs_embeds)
    assert "input_ids" not in out_batch


def test_train_cp_prepare_pops_all_vlm_input_keys_from_batch():
    """Keys the model declares consumed (returned as None) must be removed
    after the prepare step. Other keys (labels, attention_mask, etc.) remain."""
    model = _SpyVLM(prepared={"inputs_embeds": torch.zeros(1, 4, 8)})
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "pixel_values": torch.zeros(1, 3, 4, 4),
        "image_flags": torch.tensor([[1]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4),
    }
    out_batch = _train_cp_prepare(model, batch)

    # Multimodal keys gone
    assert "input_ids" not in out_batch
    assert "pixel_values" not in out_batch
    assert "image_flags" not in out_batch
    # Non-multimodal keys preserved
    assert "labels" in out_batch
    assert "attention_mask" in out_batch


def test_train_cp_prepare_passes_the_whole_batch_opaquely():
    """The hook receives the batch dict itself (no framework-side key filter):
    the model — not a central registry — decides which keys it reads."""
    model = _SpyVLM()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        # No pixel_values; no sound_features; etc.
        "labels": torch.tensor([[1, 2, 3, 4]]),
    }
    _train_cp_prepare(model, batch)

    cp_batch = model.calls[0]["_cp_batch"]
    assert set(cp_batch) == {"input_ids", "labels"}


def test_train_cp_prepare_skipped_when_model_has_no_prepare_model_inputs_for_cp():
    """If the model lacks the method, the prepare step is skipped — batch stays
    intact for the standard LLM/SDPA path."""

    class _NoPrepareLLM:
        def __call__(self, **kw):
            raise AssertionError("should not be called when model lacks prepare_model_inputs_for_cp")

    model = _NoPrepareLLM()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
    }
    out = _train_cp_prepare(model, batch)
    assert "input_ids" in out  # untouched
    assert "inputs_embeds" not in out


def test_train_cp_prepare_allows_grad_through_pre_embed():
    """Pre-embed must keep grad enabled for trainable multimodal towers."""

    class _GradSensitive:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def prepare_model_inputs_for_cp(self, **kw):
            return {"inputs_embeds": torch.zeros(1, 4, 8)}

        def __call__(self, **kw):
            assert torch.is_grad_enabled(), "prepare step must keep gradients enabled"
            return {"inputs_embeds": self.weight * torch.ones(1, 4, 8)}

    model = _GradSensitive()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
    }
    with torch.enable_grad():
        out = _train_cp_prepare(model, batch)

    assert out["inputs_embeds"].requires_grad


def test_train_cp_prepare_keeps_only_model_returned_cp_metadata():
    """The recipe should not preserve VLM metadata itself after pre-embed.

    Model-specific CP metadata, such as Gemma4 ``mm_token_type_ids``, must be
    returned from the model's pre-embed call when later attention needs it.
    """
    mm_token_type_ids = torch.tensor([[1, 1, 0, 0]])
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "mm_token_type_ids": mm_token_type_ids,
        "labels": torch.tensor([[1, 2, 3, 4]]),
    }
    out = _train_cp_prepare(_SpyVLM(prepared={"inputs_embeds": torch.zeros(1, 4, 8)}), dict(batch))
    assert "mm_token_type_ids" not in out

    prepared = {"inputs_embeds": torch.zeros(1, 4, 8), "mm_token_type_ids": mm_token_type_ids}
    out = _train_cp_prepare(_SpyVLM(prepared=prepared), dict(batch))
    assert torch.equal(out["mm_token_type_ids"], mm_token_type_ids)


def test_train_cp_prepare_pp_first_stage_preembeds_inputs():
    """When CP and PP are both enabled, only the first stage should materialize
    multimodal inputs before sequence sharding."""
    inputs_embeds = torch.randn(1, 4, 8)
    model = _SpyVLM(prepared={"inputs_embeds": inputs_embeds})
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "pixel_values": torch.zeros(1, 3, 4, 4),
        "patch_pixel_values": torch.zeros(1, 2, 3, 4, 4),
        "num_patches": torch.tensor([2]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
    }

    out_batch = _train_cp_prepare(model, batch, pp_enabled=True, has_first_stage=True)

    assert len(model.calls) == 1
    assert model.calls[0]["_pre_embed_only"] is True
    assert "patch_pixel_values" in model.calls[0]["_cp_batch"]
    assert "num_patches" in model.calls[0]["_cp_batch"]
    assert "inputs_embeds" in out_batch
    assert "input_ids" not in out_batch
    assert "pixel_values" not in out_batch


def test_train_cp_prepare_pp_later_stage_drops_media_without_preembedding():
    """Later PP stages should not run the media encoder, but should remove
    unneeded media tensors before CP batch processing."""
    model = _SpyVLM()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "pixel_values": torch.zeros(1, 3, 4, 4),
        "patch_pixel_values": torch.zeros(1, 2, 3, 4, 4),
        "num_patches": torch.tensor([2]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
    }

    out_batch = _train_cp_prepare(model, batch, pp_enabled=True, has_first_stage=False)

    assert model.calls == []
    assert "input_ids" in out_batch
    assert "inputs_embeds" not in out_batch
    assert "pixel_values" not in out_batch
    assert "patch_pixel_values" not in out_batch
    assert "num_patches" not in out_batch
    assert "labels" in out_batch


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
        return SimpleNamespace(size=lambda: 2)


class _ScheduleSpy:
    def __init__(self):
        self.calls = []

    def step(self, model_input=None, *, target=None, losses=None, **batch):
        self.calls.append({"model_input": model_input, "target": target, "batch": batch})
        if losses is not None:
            losses.append(torch.tensor(1.25))


def test_forward_backward_step_pp_cp_first_stage_uses_inputs_embeds(monkeypatch):
    inputs_embeds = torch.randn(2, 6, 8)
    labels = torch.arange(12, dtype=torch.long).reshape(2, 6)
    model = _SpyVLM(prepared={"inputs_embeds": inputs_embeds})
    schedule = _ScheduleSpy()
    seq_lens = []
    first_stage = SimpleNamespace(is_first=True, inputs_meta=None)
    recipe = object.__new__(FinetuneRecipeForVLM)
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    recipe.device_mesh = _FakeCPMesh()
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

    def _make_cp_batch_and_ctx(device_mesh, cp_batch, *args, **kwargs):
        seen_cp_batch.update(cp_batch)
        return nullcontext, cp_batch

    monkeypatch.setattr(cp_utils_mod, "make_cp_batch_and_ctx", _make_cp_batch_and_ctx)
    monkeypatch.setattr(vlm_finetune, "stage_vlm_media_for_pp", lambda *args, **kwargs: nullcontext())

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
    assert model.calls[0]["_pre_embed_only"] is True
    assert "inputs_embeds" in seen_cp_batch
    assert "input_ids" not in seen_cp_batch
    assert "pixel_values" not in seen_cp_batch
    assert seq_lens == [inputs_embeds.shape[1]]
    assert len(schedule.calls) == 1
    assert schedule.calls[0]["model_input"] is inputs_embeds
    assert torch.equal(schedule.calls[0]["target"], labels)
    assert schedule.calls[0]["batch"] == {}
    assert tuple(first_stage.inputs_meta[0].shape) == (2, 6, 8)
    assert first_stage.inputs_meta[0].dtype == inputs_embeds.dtype
    assert torch.equal(loss_buffer[0], torch.tensor(1.25))


class _FakePPModel:
    def __init__(self, stage0):
        self.parts = [stage0]
        self.pp_batch_size = 4
        self.pp_microbatch_size = 2
        self.info = SimpleNamespace(has_last_stage=False, stages=[], schedule=None)


class _StageWithCPPrepare:
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

    def _build_dataloader(*args, **kwargs):
        dataloader_calls.append(kwargs)
        return "dl", "processor"

    monkeypatch.setattr(vlm_finetune, "build_dataloader", _build_dataloader)
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
        (2, _StageWithCPPrepare(), None),
        (1, _StageWithCPPrepare(), 2),
        (2, _StageWithoutCPPrepare(), 2),
    ],
)
def test_setup_skips_pp_media_prechunk_when_cp_preembeds_vlm_inputs(
    monkeypatch,
    cp_size,
    stage0,
    expected_pp_n_microbatches,
):
    dataloader_calls = []
    _patch_pp_setup_minimals(monkeypatch, cp_size=cp_size, stage0=stage0, dataloader_calls=dataloader_calls)
    trainer = FinetuneRecipeForVLM(_minimal_pp_setup_cfg())

    trainer.setup()

    assert dataloader_calls[0]["pp_n_microbatches"] == expected_pp_n_microbatches


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
    monkeypatch.setattr(cp_utils_mod, "make_cp_batch_and_ctx", lambda mesh, batch, *a, **k: (nullcontext, batch))
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
    validation loop must run the model's _pre_embed_only pass before sharding.
    Guards finetune.py:_run_validation_epoch CP pre-embed branch."""
    from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM

    monkeypatch.setattr(vlm_finetune, "ScopedRNG", lambda *a, **k: nullcontext())
    monkeypatch.setattr(cp_utils_mod, "make_cp_batch_and_ctx", lambda mesh, batch, *a, **k: (nullcontext, batch))
    monkeypatch.setattr(vlm_finetune, "filter_forward_kwargs", lambda model, batch: batch)
    monkeypatch.setattr(vlm_finetune, "calculate_loss", lambda *a, **k: torch.tensor(2.0))

    pre_embed_calls = []

    class _Model(torch.nn.Module):
        def eval(self):
            return self

        def prepare_model_inputs_for_cp(self, **kwargs):  # marker presence matters
            return {"inputs_embeds": torch.zeros(1, 4, 8)}

        def forward(self, _pre_embed_only=False, **batch):
            if _pre_embed_only:
                pre_embed_calls.append(set(batch))
                return self.prepare_model_inputs_for_cp(**batch)
            return SimpleNamespace(logits=torch.zeros(1, 4, 8), hidden_states=None)

    class _DM(dict):
        mesh_dim_names = ["cp"]

    recipe = FinetuneRecipeForVLM.__new__(FinetuneRecipeForVLM)
    recipe.model_parts = [_Model()]
    recipe.loss_fn = object()
    recipe.device_mesh = _DM(cp=SimpleNamespace(size=lambda: 2))
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

    assert pre_embed_calls, "the _pre_embed_only pass must run when CP is active"
    assert result.metrics["val_loss"] == pytest.approx(2.0)
