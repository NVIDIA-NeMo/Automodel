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

"""Local checkpoint validation and recipe routing without a running pipeline."""

from unittest.mock import Mock

import pytest
import torch

from nemo_automodel.components.datasets.packing import DEFAULT_PACKED_SEQUENCE_CONTRACT
from nemo_automodel.components.datasets.zephon.loader import ZephonDataLoader
from nemo_automodel.recipes.base_recipe import is_dataloader, is_distributed_stateful


@pytest.mark.parametrize("state", [{}, {"zephon": {}}, {"zephon": b""}])
def test_checkpoint_rejects_missing_or_empty_bytes(state):
    pipeline = Mock()
    loader = ZephonDataLoader(
        pipeline=pipeline, packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT, convert_batch=Mock()
    )
    with pytest.raises(ValueError):
        loader.load_state_dict(state)
    pipeline.restore.assert_not_called()


def test_checkpoint_leaves_byte_validation_to_zephon():
    from zephon import Pipeline

    source = Mock()
    loader = ZephonDataLoader(
        pipeline=Pipeline(source), packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT, convert_batch=Mock()
    )
    with pytest.raises(ValueError, match="checkpoint byte format"):
        loader.load_state_dict({"zephon": b"garbage"})
    assert source.mock_calls == []


def test_checkpoint_passes_opaque_bytes_without_reserialization():
    checkpoint = b"opaque-zephon-checkpoint"
    pipeline = Mock()
    pipeline.checkpoint.return_value = checkpoint
    loader = ZephonDataLoader(
        pipeline=pipeline, packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT, convert_batch=Mock()
    )
    state = loader.state_dict()
    assert state["zephon"] is checkpoint
    pipeline.checkpoint.assert_called_once_with(as_bytes=True)
    loader.load_state_dict(state)
    pipeline.restore.assert_called_once_with(checkpoint)


def test_actual_base_recipe_checkpoint_route():
    loader = ZephonDataLoader(pipeline=Mock(), packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT, convert_batch=Mock())
    assert not is_dataloader(loader)
    assert is_distributed_stateful(loader)


def test_load_template_does_not_initialize_or_checkpoint_pipeline():
    pipeline = Mock()
    loader = ZephonDataLoader(
        pipeline=pipeline, packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT, convert_batch=Mock()
    )
    assert loader.state_dict_template() == {"zephon": b""}
    assert pipeline.mock_calls == []


def test_exhaustion_checkpoints_before_explicit_close():
    from unittest.mock import MagicMock

    pipeline = MagicMock()
    pipeline.__iter__.return_value = (item for item in ())
    pipeline.checkpoint.return_value = b"terminal-checkpoint"
    loader = ZephonDataLoader(
        pipeline=pipeline, packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT, convert_batch=Mock()
    )
    assert list(loader) == []
    assert list(loader) == []
    pipeline.close.assert_not_called()
    assert loader.state_dict()["zephon"]
    pipeline.checkpoint.assert_called_once_with(as_bytes=True)
    loader.close()
    pipeline.close.assert_called_once()


def test_pipeline_startup_error_closes_runtime():
    from unittest.mock import MagicMock

    pipeline = MagicMock()
    pipeline.__iter__.side_effect = RuntimeError("source failed")
    loader = ZephonDataLoader(
        pipeline=pipeline, packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT, convert_batch=Mock()
    )
    with pytest.raises(RuntimeError, match="source failed"):
        next(loader)
    pipeline.close.assert_called_once()


def test_closed_loader_reports_error_instead_of_source_exhaustion():
    pipeline = Mock()
    loader = ZephonDataLoader(
        pipeline=pipeline, packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT, convert_batch=Mock()
    )
    loader.close()
    with pytest.raises(RuntimeError, match="ZephonDataLoader is closed"):
        next(loader)
    pipeline.checkpoint.assert_not_called()


def test_iterator_cleanup_without_public_pipeline_close():
    pipeline = Mock(spec=[])
    iterator = Mock(spec=["close"])
    loader = ZephonDataLoader(
        pipeline=pipeline, packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT, convert_batch=Mock()
    )
    loader._iterator = iterator
    loader.close()
    loader.close()
    iterator.close.assert_called_once()
    with pytest.raises(RuntimeError, match="ZephonDataLoader is closed"):
        next(loader)


def test_public_pipeline_cleanup_errors_are_not_swallowed():
    pipeline = Mock(spec=["close"])
    pipeline.close.side_effect = RuntimeError("pipeline cleanup failed")
    loader = ZephonDataLoader(
        pipeline=pipeline, packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT, convert_batch=Mock()
    )
    with pytest.raises(RuntimeError, match="pipeline cleanup failed"):
        loader.close()


@pytest.mark.parametrize("attention_kind", ["block_causal", "flash_varlen"])
@pytest.mark.parametrize("output_format", ["batched", "thd"])
def test_text_xdoc_position_policy_preserves_boundaries_padding_and_supervision(attention_kind, output_format):
    from zephon import SampleBatch, SampleMeta, SampleRecord

    from nemo_automodel.components.datasets.zephon.loader import convert_text_batch
    from nemo_automodel.components.models.common.packing import PackingCapabilities

    payload = {
        "input_ids": [2, 3, 4, 5, 6, 1, 1],
        "positions": [0, 1, 2, 0, 1, 0, 1],
        "loss_mask": [0, 1, 1, 0, 1, 0, 0],
    }
    sample_batch = SampleBatch(
        records=tuple(
            SampleRecord(
                payload=payload,
                meta=SampleMeta(
                    sample_id=(0, 0, index),
                    lane_id=0,
                    chunk_id=0,
                    tags={"_packing_metadata": {"padding_length": 2}},
                ),
            )
            for index in range(2)
        )
    )
    off, on = [
        convert_text_batch(
            sample_batch,
            packing_contract=PackingCapabilities(attention_kind, return_packed_sequence_metadata=True),
            output_format=output_format,
            ignore_index=-7,
            mask_segment_starts=True,
            allow_cross_document_attention=xdoc,
        )
        for xdoc in (False, True)
    ]
    assert off["position_ids"].tolist() == [[0, 1, 2, 0, 1, 0]] * 2
    assert on["position_ids"].tolist() == [[0, 1, 2, 3, 4, 0]] * 2
    assert on["document_ids"].tolist() == [[1, 1, 1, 2, 2, 0]] * 2
    assert on["attention_document_ids"].tolist() == [[1, 1, 1, 1, 1, 0]] * 2
    assert "attention_mask" not in on
    for batch in (off, on):
        assert ("packed_token_indices" in batch) is (output_format == "batched")
        assert ("cu_seqlens" in batch) is (output_format == "batched")
    if output_format == "batched":
        assert off["cu_seqlens"].tolist() == [[0, 3, 5]] * 2
        assert on["cu_seqlens"].tolist() == [[0, 5]] * 2
    assert off["labels"].tolist() == [[3, 4, -7, 6, -7, -7]] * 2
    for name in ("input_ids", "labels", "padding_mask", "document_ids"):
        torch.testing.assert_close(on[name], off[name])
    assert payload["positions"] == [0, 1, 2, 0, 1, 0, 1]


def test_loader_handoff_renames_owned_mapping_without_copying():
    from unittest.mock import MagicMock

    from zephon import SampleBatch, SampleMeta, SampleRecord

    from nemo_automodel.components.models.common.packing import PackingCapabilities

    ids = torch.tensor([[1, 1, 2, 2]])
    for kind in ("block_causal", "flash_varlen", "document_ids"):
        payload = {"document_ids": ids, "attention_document_ids": ids}
        item = SampleBatch(
            (SampleRecord(meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0), payload=payload),)
        )
        pipeline = MagicMock()
        pipeline.__iter__.return_value = (batch for batch in (item,))
        loader = ZephonDataLoader(
            pipeline=pipeline,
            packing_contract=PackingCapabilities(kind),
            convert_batch=lambda batch: batch.records[0].payload,
        )
        output = next(loader)
        assert output is payload
        assert output["_packed_seq_ids"] is ids
        key = "attention_mask" if kind == "document_ids" else "_packed_seq_ids"
        assert output[key] is ids
        assert "document_ids" not in output and "attention_document_ids" not in output
        loader.close()


def test_original_membership_is_opt_in_and_does_not_change_effective_segmentation():
    from unittest.mock import MagicMock

    from zephon import SampleBatch, SampleMeta, SampleRecord

    from nemo_automodel.components.models.common.packing import PackingCapabilities

    original = torch.tensor([[1, 1, 2, 2, 0]])
    selected = torch.tensor([[1, 1, 1, 1, 0]])
    for retain_original in (False, True):
        payload = {"document_ids": original, "attention_document_ids": selected}
        item = SampleBatch(
            (SampleRecord(meta=SampleMeta(sample_id=(0, 0, 0), lane_id=0, chunk_id=0), payload=payload),)
        )
        pipeline = MagicMock()
        pipeline.__iter__.return_value = (batch for batch in (item,))
        loader = ZephonDataLoader(
            pipeline=pipeline,
            packing_contract=PackingCapabilities("block_causal", return_document_ids=retain_original),
            convert_batch=lambda batch: batch.records[0].payload,
        )
        try:
            output = next(loader)
            assert output["_packed_seq_ids"] is selected
            assert ("document_ids" in output) is retain_original
            if retain_original:
                assert output["document_ids"] is original
        finally:
            loader.close()


def test_consumer_collates_original_batch_and_accounts_for_every_document():
    from functools import partial
    from unittest.mock import MagicMock

    from zephon import SampleBatch, SampleMeta, SampleRecord

    from nemo_automodel.components.datasets.zephon.loader import convert_text_batch

    records = tuple(
        SampleRecord(
            meta=SampleMeta(
                sample_id=(0, 0, index),
                lane_id=0,
                chunk_id=index,
                component_sample_counts={index: 1},
                component_token_counts={index: 4},
            ),
            payload={"input_ids": [1, 2, 3, 4], "positions": [0, 1, 0, 1]},
        )
        for index in range(2)
    )
    batch = SampleBatch(records)
    pipeline = MagicMock()
    pipeline.__iter__.return_value = (item for item in (batch,))
    converter = Mock(
        wraps=partial(
            convert_text_batch,
            packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT,
            ignore_index=-100,
            mask_segment_starts=False,
            allow_cross_document_attention=False,
        )
    )
    loader = ZephonDataLoader(
        pipeline=pipeline, packing_contract=DEFAULT_PACKED_SEQUENCE_CONTRACT, convert_batch=converter
    )
    try:
        converted = next(loader)
        converter.assert_called_once_with(batch)
        assert converted["input_ids"].tolist() == [[1, 2, 3]] * 2
        assert converted["_packed_seq_ids"].tolist() == [[1, 1, 2]] * 2
        assert loader._component_tokens == {0: 4, 1: 4}
        assert batch.records is records
        assert records[0].meta.component_token_counts == {0: 4}
        assert records[1].meta.component_sample_counts == {1: 1}
        assert records[0].payload["input_ids"] == [1, 2, 3, 4]
    finally:
        loader.close()
