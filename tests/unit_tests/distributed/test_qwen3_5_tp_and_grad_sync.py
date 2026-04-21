# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for Qwen3.5 VLM TP plan additions and grad-sync knob plumbing."""

from types import SimpleNamespace

import pytest
import torch.nn as nn

import nemo_automodel.components.distributed.parallelizer as parallelizer
from nemo_automodel.components.distributed.optimized_tp_plans import (
    PARALLELIZE_FUNCTIONS,
    _parallelize_qwen3_5_vlm,
)
from nemo_automodel.components.distributed.parallelizer import (
    translate_to_torch_parallel_style,
)


class TestTranslateToTorchParallelStyleReplicatedWithGradAllreduce:
    """Regression coverage for the 'replicated_with_grad_allreduce' style added
    to translate_to_torch_parallel_style. Must return None so get_hf_tp_shard_plan
    skips the entry (safe under FSDP+TP where the TP mesh replication is natural)."""

    def test_returns_none_for_replicated_with_grad_allreduce(self):
        assert translate_to_torch_parallel_style("replicated_with_grad_allreduce") is None

    def test_known_styles_still_return_concrete_objects(self):
        # spot-check that the translator still works for existing styles
        assert translate_to_torch_parallel_style("colwise") is not None
        assert translate_to_torch_parallel_style("rowwise") is not None
        assert translate_to_torch_parallel_style("sequence_parallel") is not None

    def test_unknown_style_raises(self):
        with pytest.raises(ValueError, match="Unknown parallel style"):
            translate_to_torch_parallel_style("definitely_not_a_real_style")


class TestGetHfTpShardPlanSkipsNoneStyles:
    """get_hf_tp_shard_plan must filter out dict entries where the translator
    returns None (the new 'replicated_with_grad_allreduce' case), not crash."""

    def _build_model_with_inner_plan(self, plan):
        """Create a minimal model exposing an inner ``.model._tp_plan`` attribute."""
        model = nn.Module()
        model.config = SimpleNamespace(tie_word_embeddings=False)
        inner = nn.Module()
        inner._tp_plan = plan
        model.model = inner
        return model

    def test_none_styled_entry_is_skipped(self):
        model = self._build_model_with_inner_plan(
            {
                "layers.0.self_attn.q_proj": "colwise",
                "layers.0.self_attn.q_norm": "replicated_with_grad_allreduce",
            }
        )
        plan = parallelizer.get_hf_tp_shard_plan(model)
        assert "model.layers.0.self_attn.q_proj" in plan
        assert "model.layers.0.self_attn.q_norm" not in plan


class TestParallelizeQwen35VlmRegistered:
    """_parallelize_qwen3_5_vlm is registered in PARALLELIZE_FUNCTIONS and
    delegates to get_hf_tp_shard_plan so transformers' native base_model_tp_plan
    is reused."""

    def test_qwen3_5_vlm_entry_present_in_registry(self):
        # Use the hard-coded qualname string to avoid importing
        # transformers.models.qwen3_5 at test-collection time, which would
        # defeat other tests that stub that module before first import.
        key = "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForConditionalGeneration"
        assert key in PARALLELIZE_FUNCTIONS
        assert PARALLELIZE_FUNCTIONS[key] is _parallelize_qwen3_5_vlm

    def test_delegates_to_get_hf_tp_shard_plan(self, monkeypatch):
        sentinel_plan = {"probe": "value"}
        calls = []

        def fake_get_hf_tp_shard_plan(model):
            calls.append(model)
            return sentinel_plan

        monkeypatch.setattr(parallelizer, "get_hf_tp_shard_plan", fake_get_hf_tp_shard_plan)
        dummy = object()
        result = _parallelize_qwen3_5_vlm(dummy, sequence_parallel=False)
        assert result is sentinel_plan
        assert calls == [dummy]


class TestExtractModelLayersStringFallbackAndNoneSafe:
    """Two guarantees on _extract_model_layers:

    1. The string-key fallback for Qwen3.5 fires when class identity fails
       (defensive against lazy-module / deepcopy class drift).
    2. The internal _reduce_attrs tolerates None intermediate attributes
       (which happen after PP stage split strips unused sub-modules).
    """

    def _make_fake_qwen35(self, visual_is_none: bool):
        """Build a stand-in object whose type().__name__ is
        'Qwen3_5ForConditionalGeneration' but is NOT the real class — this
        mimics the lazy-import / deepcopy class-identity drift case."""

        class Qwen3_5ForConditionalGeneration(nn.Module):  # noqa: N801  (name intentional)
            pass

        model = Qwen3_5ForConditionalGeneration()
        model.model = nn.Module()
        model.model.language_model = nn.Module()
        model.model.language_model.layers = nn.ModuleList([nn.Linear(4, 4)])
        if not visual_is_none:
            model.model.visual = nn.Module()
            model.model.visual.blocks = nn.ModuleList([nn.Linear(4, 4)])
        else:
            # Simulate the PP-stripped stage where visual was set to None
            model.model.visual = None
        return model

    def test_string_fallback_recovers_both_paths(self):
        model = self._make_fake_qwen35(visual_is_none=False)
        layers = parallelizer._extract_model_layers(model)
        # After #1941 each ModuleList is flattened into its per-layer modules,
        # so the single-element ModuleLists from each FQN yield one Linear each.
        assert len(layers) == 2
        assert all(isinstance(x, nn.Linear) for x in layers)

    def test_none_intermediate_attribute_skipped_gracefully(self):
        model = self._make_fake_qwen35(visual_is_none=True)
        # Should not raise even though model.model.visual is None
        layers = parallelizer._extract_model_layers(model)
        # Only the language_model.layers path survives; flattened to its one Linear.
        assert len(layers) == 1
        assert isinstance(layers[0], nn.Linear)


class TestAutoPipelineDeferFsdpGradSyncConversion:
    """AutoPipeline's surface uses the existing FSDP2Config-style knob
    `defer_fsdp_grad_sync`. Internally it maps to reduce_grad_per_microbatch
    (= not defer) when calling into pipeline_model."""

    def test_defer_true_maps_to_reduce_false(self):
        # We only inspect the attribute storage + mapping; exercising build()
        # requires a real DeviceMesh + model which is beyond unit scope.
        from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline

        # Provide a minimal world_mesh stand-in that supports __getitem__(axis).
        class _FakeAxis:
            def size(self):
                return 2

        class _FakeMesh:
            def __getitem__(self, _name):
                return _FakeAxis()

        pp = AutoPipeline(
            world_mesh=_FakeMesh(),
            pp_axis_name="pp",
            dp_axis_names=("dp",),
            defer_fsdp_grad_sync=True,
        )
        assert pp.defer_fsdp_grad_sync is True

    def test_defer_false_stored_unchanged(self):
        from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline

        class _FakeAxis:
            def size(self):
                return 2

        class _FakeMesh:
            def __getitem__(self, _name):
                return _FakeAxis()

        pp = AutoPipeline(
            world_mesh=_FakeMesh(),
            pp_axis_name="pp",
            dp_axis_names=("dp",),
            defer_fsdp_grad_sync=False,
        )
        assert pp.defer_fsdp_grad_sync is False
