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

"""BAGEL import smoke test.

Only asserts that the top-level symbols are importable and are classes. It
does not instantiate the full model because BAGEL construction needs the
checkpoint-sized nested configs and optional GPU dependencies.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace


def test_bagel_imports() -> None:
    from nemo_automodel.components.models.bagel import (
        BagelConfig,
        BagelForUnifiedMultimodal,
    )
    from nemo_automodel.recipes.multimodal.finetune import FinetuneRecipeForMultimodal

    assert inspect.isclass(BagelConfig)
    assert inspect.isclass(BagelForUnifiedMultimodal)
    assert inspect.isclass(FinetuneRecipeForMultimodal)


def test_bagel_stage2_config_selects_mot_decoder() -> None:
    from nemo_automodel.components.models.bagel.configuration import BagelConfig
    from nemo_automodel.components.models.bagel.model import _prepare_config_for_stage

    cfg = BagelConfig(visual_gen=False, stage=2)

    _prepare_config_for_stage(cfg)

    assert cfg.visual_gen is True
    assert cfg.text_config.layer_module == "Qwen2MoTDecoderLayer"


def test_bagel_stage1_config_drops_generation_path() -> None:
    from nemo_automodel.components.models.bagel.configuration import BagelConfig
    from nemo_automodel.components.models.bagel.model import _prepare_config_for_stage

    cfg = BagelConfig(visual_gen=True, stage=1)
    cfg.text_config.layer_module = None

    _prepare_config_for_stage(cfg)

    assert cfg.visual_gen is False
    assert cfg.text_config.layer_module == "Qwen2DecoderLayer"


def test_bagel_from_pretrained_passes_backend_to_model(monkeypatch, tmp_path) -> None:
    import nemo_automodel.components.models.bagel.model as bagel_model

    config = SimpleNamespace(stage=None)
    captured = {}

    monkeypatch.setattr(
        bagel_model.BagelConfig,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: config),
    )

    def _fake_init(self, cfg, backend=None):
        self.config = cfg
        captured["backend"] = backend

    monkeypatch.setattr(bagel_model.BagelForUnifiedMultimodal, "__init__", _fake_init)
    monkeypatch.setattr(
        bagel_model.BagelForUnifiedMultimodal,
        "load_state_dict",
        lambda self, state_dict, strict: ([], []),
    )

    def _fake_load_checkpoint(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(bagel_model, "load_bagel_checkpoint_state_dict", _fake_load_checkpoint)

    backend = {"linear": "te"}
    bagel_model.BagelForUnifiedMultimodal.from_pretrained(tmp_path, stage=2, backend=backend)

    assert captured["backend"] is backend
    assert captured["stage"] == 2
