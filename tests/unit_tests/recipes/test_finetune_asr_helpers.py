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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from nemo_automodel._transformers import NeMoAutoModelForCTC, NeMoAutoModelForSpeechSeq2Seq
from nemo_automodel.recipes.asr.finetune import build_model


@pytest.fixture(autouse=True)
def _mock_missing_cuda(monkeypatch):
    """Patch CUDA APIs that fail on CPU-only builds."""
    if torch.cuda.is_available():
        yield
        return
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [], raising=False)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _: None, raising=False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _: None, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0, raising=False)
    yield


class DummyASRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.config = SimpleNamespace()

    def forward(self, x):  # pragma: no cover
        return self.linear(x)


class _NeMoModelConfig:
    """Mimics a NeMoAutoModel config (is_nemo_auto_model=True path)."""

    def __init__(self, target):
        self._target_ = target

    def instantiate(self, **kwargs):
        return DummyASRModel()

    def get(self, key, default=None):
        return getattr(self, key, default)


class _BYOMConfig:
    """Mimics a custom (non-NeMoAutoModel) model config (BYOM path)."""

    def __init__(self):
        self._target_ = lambda: None  # any callable not in the NeMoAutoModel set

    def instantiate(self, **kwargs):
        # BYOM path must call instantiate() with NO kwargs
        assert kwargs == {}, f"BYOM instantiate() should receive no kwargs, got {kwargs}"
        return DummyASRModel()

    def get(self, key, default=None):
        return getattr(self, key, default)


# ---------------------------------------------------------------------------
# NeMoAutoModel path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        NeMoAutoModelForSpeechSeq2Seq.from_pretrained,
        NeMoAutoModelForSpeechSeq2Seq.from_config,
        NeMoAutoModelForCTC.from_pretrained,
        NeMoAutoModelForCTC.from_config,
    ],
)
def test_build_model_nemo_auto_model_path(target):
    """NeMoAutoModel targets call instantiate(**kwargs) and return the model directly."""
    captured = {}

    class CapturingConfig(_NeMoModelConfig):
        def instantiate(self, **kwargs):
            captured.update(kwargs)
            return DummyASRModel()

    model = build_model(cfg_model=CapturingConfig(target=target), cfg_freeze=None, cfg_peft=None, seed=42)

    assert isinstance(model, DummyASRModel)
    # Infrastructure kwargs must have been forwarded
    assert "peft_config" in captured
    assert "device_mesh" in captured


# ---------------------------------------------------------------------------
# BYOM path
# ---------------------------------------------------------------------------


def test_build_model_byom_calls_infrastructure():
    """Non-NeMoAutoModel configs take the BYOM path: instantiate() + apply_model_infrastructure()."""
    cfg_model = _BYOMConfig()
    fake_mesh = MagicMock(name="MeshContext")
    fake_infra = (MagicMock(), MagicMock(), MagicMock(), MagicMock())  # wrapper, pp, parallel, qat
    applied_model = DummyASRModel()

    with (
        patch("nemo_automodel.recipes.asr.finetune.MeshContext") as mock_mesh_cls,
        patch("nemo_automodel.recipes.asr.finetune.instantiate_infrastructure") as mock_infra,
        patch("nemo_automodel.recipes.asr.finetune.apply_model_infrastructure") as mock_apply,
    ):
        mock_mesh_cls.from_meshes.return_value = fake_mesh
        mock_infra.return_value = fake_infra
        mock_apply.return_value = applied_model

        result = build_model(cfg_model=cfg_model, cfg_freeze=None, cfg_peft=None, seed=42)

    assert result is applied_model

    mock_mesh_cls.from_meshes.assert_called_once_with(None, None)  # device_mesh=None, moe_mesh=None
    mock_infra.assert_called_once()
    mock_apply.assert_called_once()

    apply_kwargs = mock_apply.call_args.kwargs
    assert apply_kwargs["is_meta_device"] is False
    assert apply_kwargs["load_base_model"] is False
    assert apply_kwargs["mesh"] is fake_mesh
    assert apply_kwargs["model_wrapper"] is fake_infra[0]
    assert apply_kwargs["autopipeline"] is fake_infra[1]
    assert apply_kwargs["parallelize_fn"] is fake_infra[2]
    assert apply_kwargs["qat_quantizer"] is fake_infra[3]


def test_build_model_byom_loss_fn_from_pipeline_config():
    """When a pipeline_config is provided, its loss_fn is passed to apply_model_infrastructure."""
    fake_loss_fn = MagicMock(name="loss_fn")
    pipeline_config = SimpleNamespace(loss_fn=fake_loss_fn)

    with (
        patch("nemo_automodel.recipes.asr.finetune.MeshContext"),
        patch("nemo_automodel.recipes.asr.finetune.instantiate_infrastructure") as mock_infra,
        patch("nemo_automodel.recipes.asr.finetune.apply_model_infrastructure") as mock_apply,
    ):
        mock_infra.return_value = (None, None, None, None)
        mock_apply.return_value = DummyASRModel()

        build_model(
            cfg_model=_BYOMConfig(),
            cfg_freeze=None,
            cfg_peft=None,
            seed=42,
            pipeline_config=pipeline_config,
        )

    assert mock_apply.call_args.kwargs["loss_fn"] is fake_loss_fn


def test_build_model_byom_no_pipeline_config_loss_fn_is_none():
    """Without a pipeline_config, loss_fn passed to apply_model_infrastructure is None."""
    with (
        patch("nemo_automodel.recipes.asr.finetune.MeshContext"),
        patch("nemo_automodel.recipes.asr.finetune.instantiate_infrastructure") as mock_infra,
        patch("nemo_automodel.recipes.asr.finetune.apply_model_infrastructure") as mock_apply,
    ):
        mock_infra.return_value = (None, None, None, None)
        mock_apply.return_value = DummyASRModel()

        build_model(cfg_model=_BYOMConfig(), cfg_freeze=None, cfg_peft=None, seed=42)

    assert mock_apply.call_args.kwargs["loss_fn"] is None
