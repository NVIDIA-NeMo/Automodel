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

"""Worker state, processor lineage and model-owned position parity."""

import pytest
import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from zephon.ops import OpContext
from zephon.types import SampleBatch, SampleMeta, SampleRecord

from nemo_automodel.components.datasets.vlm.datasets import VlmMediaTokenMismatchError, encode_vlm_document
from nemo_automodel.components.datasets.zephon.vlm import (
    VlmEncoderConfig,
    _validate_flash_image_eligibility,
    image_grids,
    image_pixels,
    merge_image_documents,
)
from nemo_automodel.components.models.qwen2_5_vl.preprocessing import Qwen2_5_VLPositionState
from tests.utils.vlm_fixtures import inputs as inputs
from tests.utils.vlm_fixtures import processor as processor
from tests.utils.vlm_fixtures import tiny_vlm_config


def test_position_state_matches_actual_model_and_operator_preserves_lineage(inputs, processor):
    processor_path, documents = inputs
    config = tiny_vlm_config()
    state = Qwen2_5_VLPositionState(config)
    model = Qwen2_5_VLForConditionalGeneration(config)
    encoded = encode_vlm_document(documents[1], processor)
    expected, _ = model.model.get_rope_index(
        encoded["input_ids"].unsqueeze(0),
        mm_token_type_ids=encoded["mm_token_type_ids"].unsqueeze(0),
        image_grid_thw=encoded["image_grid_thw"],
        attention_mask=encoded["attention_mask"].unsqueeze(0),
    )
    actual = state.positions(
        encoded["input_ids"],
        mm_token_type_ids=encoded["mm_token_type_ids"].unsqueeze(0),
        image_grid_thw=encoded["image_grid_thw"],
        attention_mask=encoded["attention_mask"],
    )
    torch.testing.assert_close(actual, expected[:, 0], rtol=0, atol=0)
    operator = VlmEncoderConfig().build(pretrained_model_name_or_path=processor_path, positions=state)
    operator.setup(OpContext({}))
    metadata = SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0, lineage=(2, 4))
    source = SampleRecord(meta=metadata, payload=documents[1])
    sentinel = SampleRecord(meta=metadata.with_tombstone(), payload={})
    try:
        result, control = operator.process_many([source, sentinel])
        assert result.meta is metadata and control is sentinel
        assert source.payload is documents[1] and "input_ids" not in source.payload
        torch.testing.assert_close(result.payload["position_ids"], actual)
        # ZUG's static harness does not call setup. Exercise the initialized worker
        # contract explicitly: singleton/group execution and placeholder metadata
        # must produce identical payloads without depending on previous calls.
        other = SampleRecord(meta=SampleMeta(sample_id=(0, 0, 2), lane_id=7, chunk_id=8), payload=documents[0])
        singleton = operator.process_one(source)[0]
        grouped = operator.process_many([other, source])[1]
        placeholder = SampleRecord(
            meta=SampleMeta(sample_id=metadata.sample_id, lane_id=9, chunk_id=4), payload=documents[1]
        )
        replayed = operator.process_one(placeholder)[0]
        for value in (singleton, grouped, replayed):
            for key in result.payload:
                torch.testing.assert_close(value.payload[key], result.payload[key], rtol=0, atol=0)
    finally:
        operator.teardown()
        operator.teardown()


def test_flash_eligibility_accepts_document_edges_but_rejects_internal_holes():
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    # Excluded document edges stay separate from the contiguous eligible content.
    _validate_flash_image_eligibility(positions, torch.tensor([[False, True, True, False, False, True, True, False]]))
    with pytest.raises(ValueError, match="processor-excluded tokens inside a document"):
        _validate_flash_image_eligibility(
            positions, torch.tensor([[False, True, False, True, True, True, False, False]])
        )


def test_flash_eligibility_checks_each_document_of_each_batch():
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 0], [0, 1, 2, 3, 4, 5, 0, 0]])
    eligible = torch.tensor(
        [[False, True, False, False, True, False, False, False], [False, True, True, True, False, False, False, False]]
    )
    _validate_flash_image_eligibility(positions, eligible)
    eligible[1, 2] = False
    with pytest.raises(ValueError, match="processor-excluded tokens inside a document"):
        _validate_flash_image_eligibility(positions, eligible)


def test_media_field_callbacks_use_none_for_absence_and_preserve_real_shapes():
    for shape in ((0, 3), (0, 2), (0, 17, 2)):
        assert image_grids([None, torch.empty(shape)], length=8, shifted=True) is None
    assert image_pixels([None, torch.empty(0, 3, 8, 8)], length=8, shifted=True) is None
    for shape in ((2, 3), (2, 2), (2, 17, 2)):
        value = torch.arange(torch.tensor(shape).prod()).reshape(shape)
        result = image_grids([None, value], length=8, shifted=True)
        assert result.shape == shape
        torch.testing.assert_close(result, value)


def test_fake_prefix_preserves_shifted_targets_without_masking_other_document_starts():
    documents = [
        {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(ids),
            "position_ids": torch.arange(len(ids)),
            "attention_mask": torch.ones(len(ids), dtype=torch.bool),
        }
        for ids in ([10, 11, 12], [20, 21, 22])
    ]
    fake = {
        "input_ids": torch.tensor([90, 91]),
        "labels": torch.tensor([-100, -100]),
        "position_ids": torch.arange(2),
        "attention_mask": torch.zeros(2, dtype=torch.bool),
        "pixel_values": torch.zeros(1, 3, 2, 2),
        "image_sizes": torch.tensor([[2, 2]]),
    }
    ordinary = merge_image_documents(documents)
    injected = merge_image_documents(documents, fake_document=fake)
    shifted = []
    for payload in (ordinary, injected):
        record = SampleRecord(
            payload={name: payload[name] for name in ("input_ids", "loss_mask", "positions")},
            meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0),
        )
        batch = SampleBatch((record,)).to_training(
            tokens_field="input_ids",
            dtype=torch.long,
            return_labels=True,
            pad_to_length=10,
            pad_token_id=0,
            mask_segment_starts=False,
        )
        shifted.append(batch["labels"][batch["labels"].ne(-100)])
    torch.testing.assert_close(shifted[0], torch.tensor([11, 12, 20, 21, 22]))
    torch.testing.assert_close(shifted[1], shifted[0])
    assert not injected["loss_mask"][2]  # The first label still has no real preceding input.
    assert injected["loss_mask"][5]  # The second real document's start keeps Zephon's policy.
    assert documents[0]["labels"].tolist() == [10, 11, 12]
    assert fake["labels"].tolist() == [-100, -100]


def test_encoder_caps_only_independent_worker_threads(inputs, monkeypatch):
    from nemo_automodel.components.datasets.zephon import vlm

    processor_path, _ = inputs
    operator = VlmEncoderConfig().build(
        pretrained_model_name_or_path=processor_path, positions=Qwen2_5_VLPositionState(tiny_vlm_config())
    )
    requested = []
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    monkeypatch.setenv("MKL_NUM_THREADS", "7")
    monkeypatch.setattr(torch, "set_num_threads", requested.append)
    operator.setup(OpContext({}))
    operator.teardown()
    assert requested == []  # Inline/thread runners must not reconfigure the trainer.
    assert vlm.os.environ["OMP_NUM_THREADS"] == "6" and vlm.os.environ["MKL_NUM_THREADS"] == "7"
    monkeypatch.setattr(vlm.os, "getpid", lambda: operator._owner_pid + 1)
    operator.setup(OpContext({}))
    operator.teardown()
    assert requested == [1]
    assert vlm.os.environ["OMP_NUM_THREADS"] == "1" and vlm.os.environ["MKL_NUM_THREADS"] == "1"


def test_encoder_fails_on_corrupt_media_without_dropping_or_replacing(inputs, tmp_path):
    processor_path, documents = inputs
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    document = {
        "conversation": [
            {"role": "user", "content": [{"type": "image", "image": str(corrupt)}]},
            {"role": "assistant", "content": [{"type": "text", "text": "the answer"}]},
        ]
    }
    operator = VlmEncoderConfig().build(
        pretrained_model_name_or_path=processor_path, positions=Qwen2_5_VLPositionState(tiny_vlm_config())
    )
    operator.setup(OpContext({}))
    try:
        record = SampleRecord(meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0), payload=document)
        with pytest.raises((OSError, ValueError)):
            operator.process_many([record])
        assert record.payload is document and "input_ids" not in document
        # A failed input does not add hidden retry/drop counters to the encoder.
        valid = SampleRecord(meta=record.meta, payload=documents[0])
        first, second = operator.process_one(valid)[0], operator.process_one(valid)[0]
        for key in first.payload:
            torch.testing.assert_close(first.payload[key], second.payload[key], rtol=0, atol=0)
    finally:
        operator.teardown()


def test_encoder_media_drop_is_record_pure_and_does_not_hide_processor_failures(inputs, tmp_path, monkeypatch, caplog):
    processor_path, documents = inputs
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    document = {"conversation": [{"role": "user", "content": [{"type": "image", "image": str(corrupt)}]}]}
    operator = VlmEncoderConfig(on_media_error="drop").build(
        pretrained_model_name_or_path=processor_path, positions=Qwen2_5_VLPositionState(tiny_vlm_config())
    )
    assert not operator.traits().indexable and operator.traits().calibration_replayable
    operator.setup(OpContext({}))
    metadata = SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0)
    corrupt_record = SampleRecord(meta=metadata, payload=document)
    valid = SampleRecord(meta=SampleMeta(sample_id=(0, 0, 1), lane_id=0, chunk_id=0), payload=documents[0])
    sentinel = SampleRecord(meta=metadata.with_tombstone(), payload={})
    try:
        closed = operator.process_one(corrupt_record)
        assert len(closed) == 1 and closed[0].meta.is_sentinel
        assert closed[0].meta.contribution_cursors() == corrupt_record.meta.contribution_cursors()
        grouped = operator.process_many([corrupt_record, valid, corrupt_record, sentinel])
        data = [record for record in grouped if not record.meta.is_sentinel]
        assert len(data) == 1 and data[0].meta is valid.meta and grouped[-1] is sentinel
        replayed = operator.process_one(valid)[0]
        for key in replayed.payload:
            torch.testing.assert_close(data[0].payload[key], replayed.payload[key], rtol=0, atol=0)
        assert "sample IDs/reasons" in caplog.text and str(metadata.sample_id) in caplog.text
        assert corrupt_record.payload is document and "input_ids" not in document
        original_apply = operator._processor.apply_chat_template

        def broken_processor(*args, **kwargs):
            raise OSError("processor-side configuration failed")

        monkeypatch.setattr(operator._processor, "apply_chat_template", broken_processor)
        with pytest.raises(OSError, match="processor-side configuration failed"):
            operator.process_one(valid)
        monkeypatch.setattr(operator._processor, "apply_chat_template", original_apply)
    finally:
        operator.teardown()


def test_media_token_mismatch_has_a_narrow_droppable_type(processor):
    document = {
        "conversation": [
            {"role": "user", "content": [{"type": "text", "text": "Describe image"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "the answer"}]},
        ]
    }
    # Truncating an expanded fake image loses required slots but keeps its grid.
    with pytest.raises(VlmMediaTokenMismatchError):
        encode_vlm_document(document, processor, max_length=4, truncate=True)


@pytest.mark.parametrize("media_type", ["video", "audio"])
def test_unsupported_media_is_rejected_before_encoding_even_with_drop_policy(inputs, monkeypatch, media_type):
    from nemo_automodel.components.datasets.zephon import vlm

    processor_path, _ = inputs
    operator = VlmEncoderConfig(on_media_error="drop").build(
        pretrained_model_name_or_path=processor_path, positions=Qwen2_5_VLPositionState(tiny_vlm_config())
    )
    operator.setup(OpContext({}))

    def must_not_encode(*args, **kwargs):
        raise AssertionError("unsupported media reached decoding/processor execution")

    monkeypatch.setattr(vlm, "encode_vlm_document", must_not_encode)
    document = {
        "conversation": [
            {"role": "user", "content": [{"type": "image", "image": "unreadable.png"}]},
            {"role": "user", "content": [{"type": media_type, media_type: "unsupported"}]},
        ]
    }
    record = SampleRecord(meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0), payload=document)
    try:
        with pytest.raises(ValueError, match=f"only image and text.*{media_type}"):
            operator.process_one(record)
        assert record.payload is document
    finally:
        operator.teardown()


@pytest.mark.parametrize("output_format", ["batched", "thd"])
def test_image_batch_uses_gathered_metadata_request_only_for_batched_output(output_format):
    from nemo_automodel.components.datasets.zephon.output import convert_batch_output
    from nemo_automodel.components.datasets.zephon.vlm import convert_image_batch
    from nemo_automodel.components.models.common.packing import PackingCapabilities

    documents = [
        {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(ids),
            "position_ids": torch.arange(len(ids)),
            "attention_mask": torch.ones(len(ids), dtype=torch.bool),
        }
        for ids in ([10, 11, 12], [20, 21, 22])
    ]
    payload = merge_image_documents(documents)
    batch = SampleBatch((SampleRecord(payload=payload, meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0)),))
    converted = convert_image_batch(
        batch,
        sequence_length=8,
        pad_token_id=0,
        packing_contract=PackingCapabilities("block_causal", return_packed_sequence_metadata=True),
        output_format=output_format,
    )
    assert ("packed_token_indices" in converted) is (output_format == "batched")
    assert ("cu_seqlens" in converted) is (output_format == "batched")
    result = convert_batch_output(converted, output_format=output_format)
    assert ("seq_lens" in result) is (output_format == "thd")
    assert ("seq_lens_padded" in result) is (output_format == "thd")
