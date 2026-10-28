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

"""Real VLM decoder metadata with a CPU attention oracle, not GPU Flash kernels."""

from contextlib import closing
from dataclasses import replace

import pytest
import torch
from transformers import Qwen2_5_VLForConditionalGeneration

from nemo_automodel._transformers.registry import resolve_image_position_state
from nemo_automodel.components.datasets.zephon.vlm_config import ZephonVlmSourceConfig
from nemo_automodel.components.models.common.packing import PackingCapabilities
from tests.integration.zephon.test_mistral3_images import _loader as mistral_loader
from tests.integration.zephon.test_mistral3_images import _model as mistral_model
from tests.integration.zephon.test_mistral3_images import source as source
from tests.integration.zephon.test_zephon_vlm_loader import config as config
from tests.utils.vlm_fixtures import inputs as inputs
from tests.utils.vlm_fixtures import processor as processor
from tests.utils.vlm_fixtures import tiny_vlm_config
from tests.utils.zephon_attention import compare_decoder_attention


@pytest.mark.parametrize("backend", ["sdpa", "flash_attention_2"])
def test_vlm_configured_backend_reaches_packed_loader(config, inputs, backend, monkeypatch):
    from types import SimpleNamespace

    from nemo_automodel._transformers import NeMoAutoModelForImageTextToText
    from nemo_automodel._transformers.kernel_patches import _apply_preload_overrides
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.components.models.common.packing import configure_packing_for_models
    from nemo_automodel.recipes.vlm.finetune import build_model

    class Consumer(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            return kwargs

    def construct(*, attn_implementation, has_packed_sequence, **kwargs):
        assert has_packed_sequence is True
        selected, _ = _apply_preload_overrides(1, 1, has_packed_sequence, attn_implementation, False)
        model = Consumer()
        model.config = SimpleNamespace(_attn_implementation=selected)
        return model

    monkeypatch.setattr(NeMoAutoModelForImageTextToText, "from_pretrained", construct)
    model = build_model(
        cfg_model=ConfigNode({"_target_": construct, "attn_implementation": backend}),
        cfg_peft=None,
        cfg_freeze=None,
        seed=42,
        has_packed_sequence=config.is_packed,
    )
    assert model.config._attn_implementation == backend
    position_model = Qwen2_5_VLForConditionalGeneration(tiny_vlm_config())
    with closing(
        config.build(
            pretrained_model_name_or_path=inputs[0],
            dp_rank=0,
            dp_world_size=1,
            batch_size=2,
            position_state_factory=lambda: resolve_image_position_state(position_model),
            packing_contract=configure_packing_for_models([model]),
        ).dataloader
    ) as loader:
        batch = next(loader)
        assert batch["position_ids"].shape == (3, 2, 63)
        assert batch["_packed_seq_ids"].gt(1).any()
        received = model(**batch)
        for name in ("position_ids", "pixel_values", "image_grid_thw"):
            torch.testing.assert_close(received[name], batch[name], atol=0, rtol=0)
        if backend == "sdpa":
            assert received["attention_mask"].shape == (2, 1, 63, 63)
        else:
            assert "attention_mask" not in received
            assert received["cu_seq_lens_q"][-1] == batch["input_ids"].numel()
            assert received["cu_seq_lens_q"].numel() > 3


def test_qwen_mrope_and_media_reach_flash_decoder_unchanged(config, inputs, monkeypatch):
    torch.manual_seed(84)
    model = Qwen2_5_VLForConditionalGeneration(tiny_vlm_config()).train()
    for xdoc in (False, True):
        selected = replace(
            config,
            dataset_config=ZephonVlmSourceConfig(inject_fake_images=False),
            zephon=replace(
                config.zephon, training=replace(config.zephon.training, allow_cross_document_attention=xdoc)
            ),
        )
        loaders = [
            selected.build(
                pretrained_model_name_or_path=inputs[0],
                dp_rank=0,
                dp_world_size=1,
                batch_size=2,
                position_state_factory=lambda: resolve_image_position_state(model),
                packing_contract=PackingCapabilities(kind),
            ).dataloader
            for kind in ("block_causal", "flash_varlen")
        ]
        try:
            dense, flash = [next(loader) for loader in loaders]
            assert dense["position_ids"].shape == (3, 2, 63)
            assert dense["pixel_values"].numel()
            assert bool(dense["_packed_seq_ids"].gt(1).any()) is not xdoc
            compare_decoder_attention(model, dense, flash, monkeypatch)
        finally:
            for loader in loaders:
                loader.close()


def test_mistral_images_and_scalar_positions_reach_flash_decoder_unchanged(source, monkeypatch):
    torch.manual_seed(85)
    model = mistral_model()
    for xdoc in (False, True):
        loaders = [
            mistral_loader(source, model, xdoc=xdoc, packing_contract=PackingCapabilities(kind))
            for kind in ("block_causal", "flash_varlen")
        ]
        try:
            dense, flash = [next(loader) for loader in loaders]
            assert dense["position_ids"].shape == (2, 63)
            assert dense["image_sizes"].numel()
            assert bool(dense["_packed_seq_ids"].gt(1).any()) is not xdoc
            compare_decoder_attention(model, dense, flash, monkeypatch)
        finally:
            for loader in loaders:
                loader.close()


@pytest.mark.parametrize("xdoc", [False, True])
def test_flash_accepts_default_fake_policy_for_mixed_documents(config, inputs, xdoc):
    config = replace(
        config,
        zephon=replace(config.zephon, training=replace(config.zephon.training, allow_cross_document_attention=xdoc)),
    )
    model = Qwen2_5_VLForConditionalGeneration(tiny_vlm_config())
    with closing(
        config.build(
            pretrained_model_name_or_path=inputs[0],
            dp_rank=0,
            dp_world_size=1,
            batch_size=2,
            position_state_factory=lambda: resolve_image_position_state(model),
            packing_contract=PackingCapabilities("flash_varlen"),
        ).dataloader
    ) as loader:
        batches = list(loader)
        assert batches
        for batch in batches:
            assert batch["labels"].ne(-100).any()
            assert (
                batch["input_ids"].eq(model.config.image_token_id).sum()
                == (batch["image_grid_thw"].prod(-1) // 4).sum()
            )
