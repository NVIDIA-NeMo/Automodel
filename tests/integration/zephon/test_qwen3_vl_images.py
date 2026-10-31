# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Real native Qwen3-VL DeepStack consumption through the shared Zephon loader."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from nemo_automodel._transformers.registry import (
    ModelRegistry,
    resolve_image_position_state,
)
from nemo_automodel.components.datasets.zephon.config import (
    ZephonOptions,
    ZephonPackingOptions,
    ZephonRuntimeOptions,
    ZephonSource,
    ZephonSourceOptions,
    ZephonWorkerOptions,
)
from nemo_automodel.components.datasets.zephon.vlm_config import (
    ZephonVlmDataloaderConfig,
    ZephonVlmSourceConfig,
)
from nemo_automodel.components.models.common.packing import (
    PackingCapabilities,
    configure_packing,
)
from nemo_automodel.components.models.qwen3_vl.model import (
    Qwen3VLForConditionalGeneration,
)
from tests.utils.qwen3_vlm_fixtures import processor as processor
from tests.utils.qwen3_vlm_fixtures import qwen2_processor as qwen2_processor
from tests.utils.qwen3_vlm_fixtures import tiny_qwen3_vl_config
from tests.utils.vlm_fixtures import inputs as inputs
from tests.utils.zephon_attention import compare_decoder_attention
from tests.utils.zephon_positions import assert_joined_position_ranges


@pytest.fixture
def config(tmp_path, inputs):
    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / "images.jsonl").write_text("".join(json.dumps(document) + "\n" for document in inputs[1] * 3))
    return ZephonVlmDataloaderConfig(
        dataset_config=ZephonVlmSourceConfig(inject_fake_images=False),
        zephon=ZephonOptions(
            sources=(ZephonSource(name="images", path=str(directory), fmt="jsonl"),),
            packing=ZephonPackingOptions(
                seq_len=63,
                pack_algorithm="first_fit",
                pack_num_bins=1,
                max_num_documents=2,
                max_num_documents_scope="bin",
                shuffle_after_pack=False,
            ),
            source=ZephonSourceOptions(
                chunk_size=3, shuffle_shards=False, shuffle_within_shard=False, shuffle_block_size=None
            ),
            workers=ZephonWorkerOptions(tokenize_parallelism=1),
            runtime=ZephonRuntimeOptions(runner="inline", mtp_mode=False),
        ),
    )


def _model(backend="sdpa"):
    config = tiny_qwen3_vl_config()
    config._attn_implementation = backend
    cls = ModelRegistry.resolve_custom_model_cls(config.architectures[0], config)
    assert cls is Qwen3VLForConditionalGeneration
    return cls(config).float().train()


def _loader(config, inputs, model, *, packing_contract=PackingCapabilities("block_causal")):
    return config.build(
        pretrained_model_name_or_path=inputs[0],
        dp_rank=0,
        dp_world_size=1,
        batch_size=2,
        position_state_factory=lambda: resolve_image_position_state(model),
        packing_contract=packing_contract,
    ).dataloader


def _forward(model, batch):
    configure_packing(model.config._attn_implementation, model=model)
    return model(**{name: value for name, value in batch.items() if name != "labels"}, use_cache=False).logits


def _loss(logits, labels):
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum")


def test_native_deepstack_packed_matches_separate_logits_loss_and_gradients(config, inputs):
    torch.manual_seed(84)
    packed_model = _model()
    separate_model = deepcopy(packed_model)
    packed_loader = _loader(config, inputs, packed_model)
    separate_loader = _loader(
        replace(config, zephon=replace(config.zephon, packing=replace(config.zephon.packing, max_num_documents=1))),
        inputs,
        separate_model,
    )
    try:
        packed = next(packed_loader)
        individual = [next(separate_loader), next(separate_loader)]
        actual = _forward(packed_model, packed)
        expected = [_forward(separate_model, batch) for batch in individual]
        for document_index in range(4):
            batch_index, item_index = divmod(document_index, 2)
            packed_valid = packed["_packed_seq_ids"][batch_index].eq(item_index + 1)
            separate_valid = individual[batch_index]["_packed_seq_ids"][item_index].gt(0)
            torch.testing.assert_close(
                actual[batch_index, packed_valid],
                expected[batch_index][item_index, separate_valid],
                atol=1e-6,
                rtol=1e-5,
            )
            torch.testing.assert_close(
                packed["position_ids"][:, batch_index, packed_valid],
                individual[batch_index]["position_ids"][:, item_index, separate_valid],
            )
        actual_loss = _loss(actual, packed["labels"])
        expected_loss = sum(_loss(output, batch["labels"]) for output, batch in zip(expected, individual, strict=True))
        torch.testing.assert_close(actual_loss, expected_loss)
        actual_loss.backward()
        expected_loss.backward()
        for name, parameter in packed_model.named_parameters():
            torch.testing.assert_close(
                parameter.grad, separate_model.get_parameter(name).grad, atol=1e-5, rtol=1e-4, msg=name
            )
        deepstack_parameters = [p for name, p in packed_model.named_parameters() if "deepstack_merger_list" in name]
        assert deepstack_parameters
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in deepstack_parameters)
        assert any(p.grad.abs().sum() > 0 for p in deepstack_parameters)
        assert packed["input_ids"].eq(9).sum() == (packed["image_grid_thw"].prod(-1) // 4).sum()
        assert packed["position_ids"].shape == (3, 2, 63)
    finally:
        packed_loader.close()
        separate_loader.close()


@pytest.mark.parametrize("backend", ["sdpa", "eager"])
def test_xdoc_joins_coordinates_and_preserves_native_deepstack(config, inputs, backend):
    torch.manual_seed(85)
    model = _model(backend)
    config = replace(config, dataset_config=ZephonVlmSourceConfig(inject_fake_images=True))
    off_loader = _loader(config, inputs, model)
    on_loader = _loader(
        replace(
            config,
            zephon=replace(
                config.zephon, training=replace(config.zephon.training, allow_cross_document_attention=True)
            ),
        ),
        inputs,
        model,
    )
    try:
        off, on = next(off_loader), next(on_loader)
        assert_joined_position_ranges(off["position_ids"], on["position_ids"], off["_packed_seq_ids"])
        for name in off.keys() - {"_packed_seq_ids", "position_ids"}:
            torch.testing.assert_close(off[name], on[name], atol=0, rtol=0)
        before_off, before_on = _forward(model, off), _forward(model, on)
        changed = off["input_ids"].clone()
        changed[off["_packed_seq_ids"].eq(1) & changed.eq(5)] = 7
        later = off["_packed_seq_ids"].gt(1)
        assert later.any()
        # Both packs contain a real image, so text-only subdocuments add no fake.
        assert not (off["_packed_seq_ids"].eq(0) & ~off["padding_mask"]).any()
        torch.testing.assert_close(
            _forward(model, {**off, "input_ids": changed})[later], before_off[later], atol=0, rtol=0
        )
        assert not torch.allclose(_forward(model, {**on, "input_ids": changed})[later], before_on[later])
        loss = _loss(before_on, on["labels"])
        loss.backward()
        gradient = model.model.visual.patch_embed.proj.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    finally:
        off_loader.close()
        on_loader.close()


def test_process_worker_checkpoint_replays_expanded_qwen3_documents(config, inputs):
    model = _model()
    config = replace(
        config,
        zephon=replace(
            config.zephon,
            runtime=replace(config.zephon.runtime, runner="process"),
            packing=replace(config.zephon.packing, pack_algorithm="best_fit"),
        ),
    )
    original, restored = _loader(config, inputs, model), _loader(config, inputs, model)
    try:
        next(original)
        state = original.state_dict()
        expected = list(original)
        assert expected
        restored.load_state_dict(state)
        actual = list(restored)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        original.close()
        restored.close()


def test_native_deepstack_flash_metadata_matches_dense_attention(config, inputs, monkeypatch):
    model = _model()
    vision_calls = []

    def assert_language_metadata_does_not_reach_vision(_module, _args, kwargs):
        assert not {"_packed_seq_ids", "cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"} & kwargs.keys()
        vision_calls.append(True)

    model.model.visual.register_forward_pre_hook(assert_language_metadata_does_not_reach_vision, with_kwargs=True)
    for xdoc in (False, True):
        selected = replace(
            config,
            zephon=replace(
                config.zephon, training=replace(config.zephon.training, allow_cross_document_attention=xdoc)
            ),
        )
        loaders = [
            _loader(selected, inputs, model, packing_contract=PackingCapabilities(kind))
            for kind in ("block_causal", "flash_varlen")
        ]
        try:
            dense, flash = [next(loader) for loader in loaders]
            compare_decoder_attention(model, dense, flash, monkeypatch)
        finally:
            for loader in loaders:
                loader.close()

    assert len(vision_calls) == 4
