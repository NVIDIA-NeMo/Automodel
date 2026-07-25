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

"""Qwen-Image-owned DMD2 adapter and parallelization tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from nemo_automodel.components.distributed import parallelizer as parallelizer_module
from nemo_automodel.components.distributed.parallelizer import register_full_block_checkpointing_strategy
from nemo_automodel.components.models.qwen_image import dmd2 as qwen_dmd2_module
from nemo_automodel.components.models.qwen_image.dmd2 import QwenImageDMD2Adapter


class _FakeDiscriminator(nn.Module):
    """Tiny stand-in that records Qwen-Image discriminator geometry."""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        super().__init__()
        type(self).last_kwargs = kwargs
        self.anchor = nn.Parameter(torch.ones(1))


def _resolve_adapter(monkeypatch, **adapter_kwargs) -> tuple[QwenImageDMD2Adapter, Mock, Mock]:
    """Create an adapter backed by fake Model Optimizer Qwen symbols."""
    pipeline_cls = Mock(return_value=object())
    feature_capture = Mock()
    symbols = {
        "QwenImageDMDPipeline": pipeline_cls,
        "attach_feature_capture": feature_capture,
        "update_feature_capture_shape": Mock(),
        "Discriminator_ImageDiT": _FakeDiscriminator,
    }
    monkeypatch.setattr(
        qwen_dmd2_module,
        "safe_import_from",
        lambda module, name, **kwargs: (True, symbols[name]),
    )
    adapter = QwenImageDMD2Adapter(**adapter_kwargs)
    adapter.require_modelopt_dependencies()
    return adapter, pipeline_cls, feature_capture


def test_qwen_image_adapter_owns_modelopt_pipeline_discriminator_and_feature_hook(monkeypatch):
    adapter, pipeline_cls, feature_capture = _resolve_adapter(
        monkeypatch,
        guidance=3.5,
        gan_feature_indices=[7, 11],
        gan_num_blocks=60,
        gan_inner_dim=128,
    )
    config = SimpleNamespace(gan_loss_weight_gen=0.03)
    discriminator = adapter.build_discriminator(
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    teacher = nn.Identity()
    student = nn.Identity()
    fake_score = nn.Identity()

    adapter.attach_feature_capture(teacher, height=32, width=48)
    pipeline = adapter.build_pipeline(
        student=student,
        teacher=teacher,
        fake_score=fake_score,
        config=config,
        discriminator=discriminator,
    )

    assert isinstance(discriminator, _FakeDiscriminator)
    assert discriminator.training
    assert _FakeDiscriminator.last_kwargs == {
        "feature_indices": {7, 11},
        "num_blocks": 60,
        "inner_dim": 128,
    }
    feature_capture.assert_called_once_with(
        teacher,
        feature_indices=[7, 11],
        h_lat=32,
        w_lat=48,
    )
    assert pipeline is pipeline_cls.return_value
    pipeline_cls.assert_called_once_with(
        student=student,
        teacher=teacher,
        fake_score=fake_score,
        config=config,
        discriminator=discriminator,
        guidance=3.5,
    )


def test_qwen_image_adapter_rejects_incomplete_modelopt_plugin(monkeypatch):
    monkeypatch.setattr(
        qwen_dmd2_module,
        "safe_import_from",
        lambda module, name, **kwargs: (name != "update_feature_capture_shape", Mock()),
    )
    adapter = QwenImageDMD2Adapter()

    with pytest.raises(ImportError, match="Qwen-Image plugin fixes"):
        adapter.require_modelopt_dependencies()


def test_qwen_image_adapter_validates_transformer_and_gan_geometry():
    class QwenImageTransformer2DModel(nn.Module):
        pass

    adapter = QwenImageDMD2Adapter(
        gan_feature_indices=[60],
        gan_num_blocks=60,
    )

    adapter.validate_transformer(QwenImageTransformer2DModel(), name="student")
    with pytest.raises(TypeError, match="QwenImageTransformer2DModel"):
        adapter.validate_transformer(nn.Identity(), name="student")
    with pytest.raises(ValueError, match="outside"):
        adapter.validate_dmd_config(SimpleNamespace(gan_loss_weight_gen=0.03))


def test_qwen_image_adapter_normalizes_flash_masks():
    all_valid = torch.ones(2, 4, dtype=torch.long)
    padded = all_valid.clone()
    padded[1, -1] = 0

    assert (
        QwenImageDMD2Adapter.normalize_text_mask(
            all_valid,
            attention_backend="flash",
            prompt_kind="positive",
        )
        is None
    )
    torch.testing.assert_close(
        QwenImageDMD2Adapter.normalize_text_mask(
            padded,
            attention_backend=None,
            prompt_kind="positive",
        ),
        padded,
    )
    with pytest.raises(ValueError, match="padded negative-prompt mask"):
        QwenImageDMD2Adapter.normalize_text_mask(
            padded[1],
            attention_backend="flash",
            prompt_kind="negative",
        )


def test_qwen_image_parallelization_checkpoints_full_blocks_before_native_fsdp():
    class QwenImageTransformer2DModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(4, 4), nn.SiLU()),
                    nn.Sequential(nn.Linear(4, 4), nn.SiLU()),
                ]
            )

    model = QwenImageTransformer2DModel()
    device_mesh = object()
    model_class_name = "QwenImageTransformer2DModel"
    missing = object()
    previous_strategy = parallelizer_module.PARALLELIZATION_STRATEGIES.pop(model_class_name, missing)
    try:
        adapter = QwenImageDMD2Adapter()
        register_full_block_checkpointing_strategy(
            model_class_name=adapter.parallel_model_class_name,
            checkpoint_blocks=adapter.checkpoint_transformer_blocks,
        )
        strategy = parallelizer_module.PARALLELIZATION_STRATEGIES[model_class_name]
        register_full_block_checkpointing_strategy(
            model_class_name=adapter.parallel_model_class_name,
            checkpoint_blocks=adapter.checkpoint_transformer_blocks,
        )
        assert parallelizer_module.PARALLELIZATION_STRATEGIES[model_class_name] is strategy

        with patch.object(
            parallelizer_module.DefaultParallelizationStrategy,
            "parallelize",
            autospec=True,
            return_value=model,
        ) as native_parallelize:
            result = strategy.parallelize(
                model,
                device_mesh=device_mesh,
                activation_checkpointing=True,
            )

        assert result is model
        assert all(hasattr(block, "_checkpoint_wrapped_module") for block in model.transformer_blocks)
        native_parallelize.assert_called_once_with(
            strategy,
            model,
            device_mesh,
            activation_checkpointing=False,
        )
    finally:
        parallelizer_module.PARALLELIZATION_STRATEGIES.pop(model_class_name, None)
        if previous_strategy is not missing:
            parallelizer_module.PARALLELIZATION_STRATEGIES[model_class_name] = previous_strategy
