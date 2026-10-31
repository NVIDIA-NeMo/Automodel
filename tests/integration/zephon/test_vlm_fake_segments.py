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

"""Leading fake segments preserve whole documents and unblock mixed-data Flash."""

import json
from contextlib import closing
from dataclasses import replace

import pytest
import torch
from PIL import Image
from transformers import Mistral3ForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, Qwen3VLProcessor

from nemo_automodel._transformers.registry import resolve_image_position_state
from nemo_automodel.components.datasets.vlm.datasets import encode_vlm_document
from nemo_automodel.components.datasets.zephon.config import (
    ZephonOptions,
    ZephonPackingOptions,
    ZephonRuntimeOptions,
    ZephonSource,
    ZephonSourceOptions,
    ZephonWorkerOptions,
)
from nemo_automodel.components.datasets.zephon.vlm import prepare_fake_image_document
from nemo_automodel.components.datasets.zephon.vlm_config import ZephonVlmDataloaderConfig, ZephonVlmSourceConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.packing import configure_packing
from nemo_automodel.components.models.gemma4_moe.model import Gemma4ForConditionalGeneration
from nemo_automodel.components.models.qwen3_vl.model import Qwen3VLForConditionalGeneration
from tests.utils.gemma4_vlm_fixtures import gemma4_processor, tiny_gemma4_config
from tests.utils.mistral3_vlm_fixtures import build_model_config, build_processor
from tests.utils.qwen3_vlm_fixtures import tiny_qwen3_vl_config
from tests.utils.vlm_fixtures import processor as processor
from tests.utils.vlm_fixtures import tiny_vlm_config
from tests.utils.zephon_attention import compare_decoder_attention


@pytest.fixture(params=["qwen2", "qwen3", "mistral", "gemma"])
def family(request, processor):
    if request.param == "qwen2":
        return request.param, processor, Qwen2_5_VLForConditionalGeneration(tiny_vlm_config()).float()
    if request.param == "qwen3":
        selected = Qwen3VLProcessor(
            tokenizer=processor.tokenizer,
            image_processor=processor.image_processor,
            video_processor=processor.video_processor,
            chat_template=processor.chat_template,
        )
        return request.param, selected, Qwen3VLForConditionalGeneration(tiny_qwen3_vl_config()).float()
    if request.param == "mistral":
        config = build_model_config()
        config._attn_implementation = "sdpa"
        return request.param, build_processor(), Mistral3ForConditionalGeneration(config).float()
    config = tiny_gemma4_config()
    config._attn_implementation = "sdpa"
    return (
        request.param,
        gemma4_processor(),
        Gemma4ForConditionalGeneration(
            config,
            backend=BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch"),
        ).float(),
    )


def _source(tmp_path, processor):
    processor.save_pretrained(tmp_path / "processor")
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "blue").save(image)

    def document(*, with_image):
        return {
            "conversation": [
                {
                    "role": "user",
                    "content": ([{"type": "image", "image": str(image)}] if with_image else [])
                    + [{"type": "text", "text": "Describe image"}],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "the answer"}]},
            ]
        }

    documents = [
        document(with_image=False),
        document(with_image=False),
        document(with_image=True),
        document(with_image=False),
    ]
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "images.jsonl").write_text("".join(json.dumps(item) + "\n" for item in documents * 2))
    return documents, ZephonVlmDataloaderConfig(
        zephon=ZephonOptions(
            sources=(ZephonSource(name="images", path=str(corpus), fmt="jsonl"),),
            packing=ZephonPackingOptions(
                seq_len=255,
                pack_algorithm="first_fit",
                pack_num_bins=1,
                max_num_documents=2,
                max_num_documents_scope="bin",
                shuffle_after_pack=False,
            ),
            source=ZephonSourceOptions(
                chunk_size=4, shuffle_shards=False, shuffle_within_shard=False, shuffle_block_size=None
            ),
            workers=ZephonWorkerOptions(fetch_parallelism=1, tokenize_parallelism=1),
            runtime=ZephonRuntimeOptions(runner="inline", mtp_mode=False),
        )
    )


def _loader(config, tmp_path, model, *, kind="block_causal"):
    return config.build(
        pretrained_model_name_or_path=str(tmp_path / "processor"),
        dp_rank=0,
        dp_world_size=1,
        batch_size=2,
        position_state_factory=lambda: resolve_image_position_state(model),
        packing_contract=configure_packing("flash_attention_2" if kind == "flash_varlen" else "sdpa", model=model),
    ).dataloader


def _forward(model, batch):
    """Consume token fields [B,S], scalar/multi-axis positions and image media.

    Returns:
        Model logits [batch, sequence, vocab].
    """
    configure_packing("sdpa", model=model)
    return model(**{name: value for name, value in batch.items() if name != "labels"}, use_cache=False).logits


def test_leading_fake_only_for_all_text_packs_preserves_source_accounting_and_outputs(family, tmp_path, monkeypatch):
    _, processor, model = family
    documents, config = _source(tmp_path, processor)
    position_state = resolve_image_position_state(model)
    fake = prepare_fake_image_document(processor, position_state)
    fake_length = len(fake["input_ids"])
    raw_documents = [encode_vlm_document(document, processor, inject_fake_images=False) for document in documents]
    from nemo_automodel.components.datasets.zephon import vlm as vlm_module

    collated_inputs = []
    convert = vlm_module.convert_image_batch

    def capture_batch(batch, **kwargs):
        """Observe unshifted packed documents without changing payloads or tensors."""
        collated_inputs.append(batch)
        return convert(batch, **kwargs)

    monkeypatch.setattr(vlm_module, "convert_image_batch", capture_batch)
    loader = _loader(config, tmp_path, model)
    baseline_loader = _loader(
        replace(config, dataset_config=ZephonVlmSourceConfig(inject_fake_images=False)), tmp_path, model
    )
    try:
        actual = next(loader)
        raw = collated_inputs[0]
        first, second = raw.records
        assert first.meta.padding_length is None and second.meta.padding_length is None
        for record, indices, prefix in ((first, (0, 1), fake_length), (second, (2, 3), 0)):
            assert set(record.payload) == {"packed_samples"}
            payload = record.payload["packed_samples"]
            source_tokens = sum(len(raw_documents[index]["input_ids"]) for index in indices)
            assert record.meta.tags["_packing_metadata"]["total_length"] == source_tokens
            assert record.meta.tags["_packing_metadata"]["num_sequences"] == 2
            assert record.meta.component_sample_counts == {0: 2}
            assert sum(record.meta.component_token_counts.values()) == source_tokens
            assert len(payload["input_ids"]) == source_tokens + prefix
            assert source_tokens <= config.zephon.packing.seq_len + 1 - fake_length
            assert payload["n_images_per_sample"] == 1
        first_payload = first.payload["packed_samples"]
        assert not first_payload["processor_attention_mask"][:fake_length].any()
        assert not first_payload["loss_mask"][:fake_length].any()
        expected = next(baseline_loader)
        assert actual["labels"].ne(-100).sum() == expected["labels"].ne(-100).sum() == 12
        # The input shift removes the final text token; no required image slot is lost.
        image_token = (
            model.config.image_token_id if hasattr(model.config, "image_token_id") else model.config.image_token_index
        )
        assert (
            actual["input_ids"].eq(image_token).sum()
            == fake["input_ids"].eq(image_token).sum() + raw_documents[2]["input_ids"].eq(image_token).sum()
        )
        assert actual["_packed_seq_ids"][0, :fake_length].eq(0).all()
        valid, reference_valid = actual["_packed_seq_ids"].gt(0), expected["_packed_seq_ids"].gt(0)
        torch.testing.assert_close(actual["input_ids"][valid], expected["input_ids"][reference_valid])
        actual_logits, expected_logits = _forward(model, actual), _forward(model, expected)
        torch.testing.assert_close(actual_logits[valid], expected_logits[reference_valid], rtol=1e-5, atol=3e-6)
        loss = torch.nn.functional.cross_entropy(
            actual_logits.flatten(0, 1), actual["labels"].flatten(), ignore_index=-100
        )
        loss.backward()
        assert torch.isfinite(loss)
    finally:
        loader.close()
        baseline_loader.close()


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "mistral"], indirect=True)
def test_default_fake_segments_match_flash_metadata_for_both_xdoc_modes(family, tmp_path, monkeypatch):
    _, processor, model = family
    _, config = _source(tmp_path, processor)
    for xdoc in (False, True):
        selected = replace(
            config,
            zephon=replace(
                config.zephon, training=replace(config.zephon.training, allow_cross_document_attention=xdoc)
            ),
        )
        dense_loader = _loader(selected, tmp_path, model)
        flash_loader = _loader(selected, tmp_path, model, kind="flash_varlen")
        try:
            dense, flash = next(dense_loader), next(flash_loader)
            assert dense["_packed_seq_ids"][0, 0] == 0
            assert dense["_packed_seq_ids"][1, 0] > 0
            compare_decoder_attention(model, dense, flash, monkeypatch)
        finally:
            dense_loader.close()
            flash_loader.close()


@pytest.mark.parametrize("family", ["qwen2"], indirect=True)
def test_reserved_fake_capacity_validation_oversized_policy_and_process_resume(family, tmp_path):
    _, processor, model = family
    _, config = _source(tmp_path, processor)
    fake_length = len(prepare_fake_image_document(processor, resolve_image_position_state(model))["input_ids"])
    too_small = replace(
        config, zephon=replace(config.zephon, packing=replace(config.zephon.packing, seq_len=fake_length - 1))
    )
    with pytest.raises(ValueError, match="reserved fake-image segment length"):
        _loader(too_small, tmp_path, model)

    capacity = 32
    full_capacity = capacity + fake_length
    oversized = {
        "conversation": [
            {"role": "user", "content": [{"type": "text", "text": "Describe " * (capacity - 4)}]},
            {"role": "assistant", "content": [{"type": "text", "text": "the answer"}]},
        ]
    }
    length = len(encode_vlm_document(oversized, processor, inject_fake_images=False)["input_ids"])
    assert capacity < length <= full_capacity  # Fits full capacity, but not after the explicit reserve.
    corpus = tmp_path / "oversized"
    corpus.mkdir()
    (corpus / "images.jsonl").write_text("".join(json.dumps(oversized) + "\n" for _ in range(4)))
    selected = replace(
        config,
        zephon=replace(
            config.zephon,
            sources=(ZephonSource(name="images", path=str(corpus), fmt="jsonl"),),
            packing=replace(config.zephon.packing, seq_len=full_capacity - 1, drop_oversized=False),
        ),
    )
    with closing(_loader(selected, tmp_path, model)) as loader:
        with pytest.raises(ValueError, match="exceeds max_length"):
            next(loader)
    with closing(
        _loader(
            replace(
                selected, zephon=replace(selected.zephon, packing=replace(selected.zephon.packing, drop_oversized=True))
            ),
            tmp_path,
            model,
        )
    ) as loader:
        assert list(loader) == []

    process_config = replace(
        config, zephon=replace(config.zephon, runtime=replace(config.zephon.runtime, runner="process"))
    )
    original, restored = _loader(process_config, tmp_path, model), _loader(process_config, tmp_path, model)
    try:
        first = next(original)
        assert first["_packed_seq_ids"][0, 0] == 0
        checkpoint = original.state_dict()
        expected = list(original)
        assert expected
        restored.load_state_dict(checkpoint)
        actual = list(restored)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        original.close()
        restored.close()
