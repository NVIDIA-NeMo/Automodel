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

"""CPU contracts for the optional model-owned DeepSelect backend."""

import dataclasses

import pytest
import torch

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next import qsa
from nemo_automodel.components.models.qwen3_8_flash_next.backend import Qwen3_8_FlashNextBackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.config import Qwen3_8_FlashNextTextConfig
from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration


def test_model_owned_backend_resolves_yaml_and_mapping() -> None:
    settings = {"attn": "flex", "linear": "torch", "qsa_topk": "deepselect", "rope_fusion": False}
    yaml_backend = ConfigNode(
        {
            "_target_": "nemo_automodel.components.models.qwen3_8_flash_next.backend.Qwen3_8_FlashNextBackendConfig",
            **settings,
        }
    ).instantiate()
    model_type = Qwen3_8_FlashNextForConditionalGeneration
    mapping_backend = model_type.backend_config_resolver(settings)
    assert isinstance(mapping_backend, Qwen3_8_FlashNextBackendConfig)
    assert dataclasses.asdict(yaml_backend) == dataclasses.asdict(mapping_backend)
    assert mapping_backend.qsa_topk == "deepselect"
    assert "qsa_topk" not in {f.name for f in dataclasses.fields(BackendConfig)}
    assert settings["qsa_topk"] == "deepselect"


@pytest.mark.parametrize("model_owned", [False, True])
def test_default_selector_does_not_load_deepselect(monkeypatch: pytest.MonkeyPatch, model_owned: bool) -> None:
    monkeypatch.setattr(qsa, "safe_import", lambda name: pytest.fail("default selector must not load DeepSelect"))
    config = Qwen3_8_FlashNextTextConfig(hidden_size=16, indexer_n_heads=2, indexer_head_dim=8)
    backend_cls = Qwen3_8_FlashNextBackendConfig if model_owned else BackendConfig
    backend = backend_cls(linear="torch", attn="flex", rope_fusion=False)
    indexer = qsa.Qwen3_8_FlashNextQSAIndexer(config, backend)
    assert indexer.topk_backend == "torch"
    assert all(not p.requires_grad for p in indexer.parameters())


def test_invalid_selector_fails_at_configuration() -> None:
    with pytest.raises(ValueError, match="QSA top-k backend"):
        Qwen3_8_FlashNextBackendConfig(qsa_topk="invalid")


def test_deepselect_rejects_cpu_inputs() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        qsa.select_qsa_token_ids(
            torch.ones(1, 4, 1, 1),
            torch.ones(1, 1, 1, 1),
            torch.tensor([4]),
            token_budget=2048,
            compress_ratio=4,
            topk_backend="deepselect",
        )


def test_deepselect_validates_budget_before_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qsa, "safe_import", lambda name: pytest.fail("validate budget before optional import"))
    config = Qwen3_8_FlashNextTextConfig(indexer_budget=1024)
    with pytest.raises(ValueError, match="512 selected blocks"):
        qsa.Qwen3_8_FlashNextQSAIndexer(config, Qwen3_8_FlashNextBackendConfig(qsa_topk="deepselect"))


def test_deepselect_missing_dependency_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    qsa._get_deepselect.cache_clear()
    monkeypatch.setattr(qsa, "safe_import", lambda name: (False, None))
    try:
        with pytest.raises(ImportError, match="requires DeepSelect"):
            qsa.Qwen3_8_FlashNextQSAIndexer(
                Qwen3_8_FlashNextTextConfig(),
                Qwen3_8_FlashNextBackendConfig(qsa_topk="deepselect"),
            )
    finally:
        qsa._get_deepselect.cache_clear()
