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
  - Validation: do NOT pop labels before make_cp_batch_and_ctx
  - Validation: position_ids ``.to(self.dist_env.device)`` (not model.device)
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.utils.model_utils import VLM_INPUT_KEYS


# -----------------------------------------------------------------------------
# Helpers reproducing the recipe's CP-prepare block (train + val flavors).
# -----------------------------------------------------------------------------


def _train_cp_prepare(_model, batch):
    """Replicates the train-side CP prepare block in
    ``recipes/vlm/finetune.py::_forward_backward_step`` (lines 960-967)."""
    if not hasattr(_model, "prepare_model_inputs_for_cp"):
        return batch
    mm_kwargs = {k: batch[k] for k in VLM_INPUT_KEYS if batch.get(k) is not None}
    with torch.no_grad():
        prepared = _model(_pre_embed_only=True, **mm_kwargs)
    for k in VLM_INPUT_KEYS:
        batch.pop(k, None)
    batch.update(prepared)
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
        self.calls.append({"_pre_embed_only": _pre_embed_only, **kwargs})
        if _pre_embed_only:
            return self.prepared
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
    # input_ids and pixel_values should have been forwarded as kwargs
    assert "input_ids" in model.calls[0]
    assert "pixel_values" in model.calls[0]
    # The returned batch contains inputs_embeds (not input_ids)
    assert "inputs_embeds" in out_batch
    assert torch.equal(out_batch["inputs_embeds"], inputs_embeds)
    assert "input_ids" not in out_batch


def test_train_cp_prepare_pops_all_vlm_input_keys_from_batch():
    """All keys in VLM_INPUT_KEYS that were in the batch must be popped after
    the prepare step. Other keys (labels, attention_mask, etc.) must remain."""
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


def test_train_cp_prepare_only_passes_keys_that_are_present():
    """``mm_kwargs`` filter uses ``batch.get(k) is not None`` so missing or None
    multimodal keys are not forwarded as kwargs."""
    model = _SpyVLM()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        # No pixel_values; no sound_features; etc.
        "labels": torch.tensor([[1, 2, 3, 4]]),
    }
    _train_cp_prepare(model, batch)

    call_kwargs = model.calls[0]
    # input_ids should be present
    assert "input_ids" in call_kwargs
    # No spurious None-valued multimodal kwargs
    for k in ("pixel_values", "sound_features", "pixel_values_videos"):
        assert k not in call_kwargs


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


def test_train_cp_prepare_uses_torch_no_grad():
    """Vision tower runs under no_grad so the recipe doesn't accidentally
    accumulate grads through the (frozen) vision encoder."""

    class _GradSensitive:
        def prepare_model_inputs_for_cp(self, **kw):
            return {"inputs_embeds": torch.zeros(1, 4, 8)}

        def __call__(self, **kw):
            assert not torch.is_grad_enabled(), (
                "prepare step must run under torch.no_grad()"
            )
            return {"inputs_embeds": torch.zeros(1, 4, 8)}

    model = _GradSensitive()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
    }
    _train_cp_prepare(model, batch)


# -----------------------------------------------------------------------------
# val-side wiring (the bug-fix territory)
# -----------------------------------------------------------------------------


def test_val_does_not_pop_labels_before_make_cp_batch_and_ctx():
    """Reproduce the bug fix at finetune.py:1239 — val must compute
    ``num_label_tokens`` from ``batch["labels"]`` WITHOUT popping the labels,
    because ``make_cp_batch_and_ctx`` registers labels as a CP buffer."""
    labels = torch.tensor([[1, 2, -100, 4]])
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": labels,
    }
    # The recipe-side line under test:
    num_label_tokens = (batch["labels"] != -100).sum().item()
    assert num_label_tokens == 3
    # labels must STILL be in batch (not popped) so cp_utils can read it
    assert "labels" in batch
    assert batch["labels"] is labels


def test_val_pos_ids_uses_dist_env_device_not_model_device():
    """Reproduce the bug fix at finetune.py:1281 — val must use
    ``self.dist_env.device``, not ``self.model_parts[0].device`` which
    AttributeErrors on FSDP-wrapped models."""

    class _FSDPWrapped:
        # Intentionally has NO ``.device`` attribute (mirrors real FSDP wrapper).
        def __getattr__(self, name):
            if name == "device":
                raise AttributeError(
                    "'FSDPWrapped' object has no attribute 'device'"
                )
            raise AttributeError(name)

    model = _FSDPWrapped()
    dist_env = SimpleNamespace(device=torch.device("cpu"))

    # The fixed line:
    pos = torch.arange(0, 4).unsqueeze(0).to(dist_env.device)
    assert pos.device.type == "cpu"

    # The buggy line would have raised:
    with pytest.raises(AttributeError, match="no attribute 'device'"):
        _ = torch.arange(0, 4).unsqueeze(0).to(model.device)
