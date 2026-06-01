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

"""Smoke tests for Qwen2.5-Omni Thinker model module.

These do NOT load real HF checkpoints (parity tests under
``tests/functional_tests/qwen2_5_omni/`` cover that). The point here is to
catch wiring regressions cheaply: imports, MRO, registry resolution, and the
state-dict adapter attachment.
"""

import pytest

torch = pytest.importorskip("torch")


def test_imports():
    from nemo_automodel.components.models.qwen2_5_omni.model import (
        ModelClass,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    assert ModelClass is Qwen2_5OmniThinkerForConditionalGeneration


def test_mro_includes_hf_checkpointing_mixin():
    from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
    from nemo_automodel.components.models.qwen2_5_omni.model import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    mro_names = {c.__name__ for c in Qwen2_5OmniThinkerForConditionalGeneration.__mro__}
    assert "HFCheckpointingMixin" in mro_names
    # And no MoE FSDP sync mixin should be present (dense model).
    assert "MoEFSDPSyncMixin" not in mro_names
    # Sanity: must still inherit the HF Thinker class.
    assert "Qwen2_5OmniThinkerForConditionalGeneration" in mro_names
    assert any("Qwen2_5OmniPreTrainedModel" in n for n in mro_names)
    assert HFCheckpointingMixin in Qwen2_5OmniThinkerForConditionalGeneration.__mro__


def test_registry_resolves_to_thinker_class():
    from nemo_automodel._transformers.registry import ModelRegistry
    from nemo_automodel.components.models.qwen2_5_omni.model import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    for arch in (
        "Qwen2_5OmniModel",
        "Qwen2_5OmniForConditionalGeneration",
        "Qwen2_5OmniThinkerForConditionalGeneration",
    ):
        cls = ModelRegistry.get_model_cls_from_model_arch(arch)
        assert cls is Qwen2_5OmniThinkerForConditionalGeneration, (
            f"Registry maps {arch!r} to {cls.__name__}, expected the NeMo Thinker class"
        )


def test_resolve_thinker_config_unwraps_full_omni_config():
    """``_resolve_thinker_config`` returns the thinker sub-config when given the full Omni config."""
    from nemo_automodel.components.models.qwen2_5_omni.model import _resolve_thinker_config

    class FakeThinker:
        torch_dtype = None

    class FakeOmni:
        thinker_config = FakeThinker()

    assert _resolve_thinker_config(FakeOmni()) is FakeOmni().thinker_config.__class__ or isinstance(
        _resolve_thinker_config(FakeOmni()), FakeThinker
    )

    # If a thinker config is passed directly, it should be returned untouched.
    thinker_only = FakeThinker()
    assert _resolve_thinker_config(thinker_only) is thinker_only


@pytest.mark.parametrize(
    "processor_name",
    ["Qwen2_5OmniProcessor"],
)
def test_marker_processor_set_includes_qwen2_5_omni(processor_name):
    from nemo_automodel.components.datasets.vlm.collate_fns import _IMSTART_TEMPLATE_PROCESSORS

    assert processor_name in _IMSTART_TEMPLATE_PROCESSORS


def test_collate_fn_dispatch_includes_qwen2_5_omni():
    from nemo_automodel.components.datasets.vlm.collate_fns import (
        COLLATE_FNS,
        qwen2_5_omni_asr_collate_fn,
    )

    assert COLLATE_FNS.get("Qwen2_5OmniProcessor") is qwen2_5_omni_asr_collate_fn


def test_collate_fn_is_alias_of_qwen3_omni_asr():
    """``qwen2_5_omni_asr_collate_fn`` must delegate to ``qwen3_omni_asr_collate_fn`` for processor-agnostic logic."""
    from unittest.mock import MagicMock

    from nemo_automodel.components.datasets.vlm import collate_fns as cf

    sentinel = object()
    captured = {}

    def fake_qwen3_asr(examples, processor):
        captured["examples"] = examples
        captured["processor"] = processor
        return sentinel

    real = cf.qwen3_omni_asr_collate_fn
    cf.qwen3_omni_asr_collate_fn = fake_qwen3_asr
    try:
        out = cf.qwen2_5_omni_asr_collate_fn([{"x": 1}], processor=MagicMock())
    finally:
        cf.qwen3_omni_asr_collate_fn = real

    assert out is sentinel
    assert captured["examples"] == [{"x": 1}]
