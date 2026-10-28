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

"""CPU integration spike: real image processor -> Zephon bins -> to_training."""

import json
from contextlib import closing
from operator import itemgetter

import pytest
import torch
from zephon import Dataset, InMemoryShard, Pipeline, StaticMixtureWorkSource

from nemo_automodel.components.datasets.zephon.vlm import (
    VlmEncoderConfig,
    image_grids,
    image_pixels,
    merge_image_documents,
    model_positions,
)
from nemo_automodel.components.models.qwen2_5_vl.preprocessing import Qwen2_5_VLPositionState
from tests.utils.vlm_fixtures import inputs as inputs
from tests.utils.vlm_fixtures import processor as processor
from tests.utils.vlm_fixtures import tiny_vlm_config


def _pipeline(inputs, algorithm, runner, *, on_media_error="raise"):
    processor_path, documents = inputs
    source = StaticMixtureWorkSource(
        [Dataset.from_dict("images", {0: InMemoryShard(documents)})],
        {"images": 1.0},
        chunk_size=3,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )
    operator = VlmEncoderConfig(parallelism=2, on_media_error=on_media_error).build(
        pretrained_model_name_or_path=processor_path, positions=Qwen2_5_VLPositionState(tiny_vlm_config())
    )
    pipeline = Pipeline(source).add_op(operator)
    pipeline.pack_sequences(max_length=41, algorithm=algorithm, num_bins=2, pack_payloads=merge_image_documents)
    pipeline.map_transform(itemgetter("packed_samples"), parallelism=1)
    pipeline.batch(2, drop_last=False)
    pipeline.options(deterministic=True, max_workers=2, runner=runner, mtp_mode=False, shm_enabled=False)
    return pipeline


def _convert(batch):
    result = batch.to_training(
        dtype=torch.long,
        return_labels=True,
        return_loss_mask=True,
        return_padding_mask=True,
        return_cu_seqlens=True,
        pad_to_length=41,
        pad_token_id=1,
        mask_segment_starts=True,
        extra_fields=("processor_attention_mask",),
        field_collators={
            "pixel_values": image_pixels,
            "image_grid_thw": image_grids,
            "model_position_ids": model_positions,
        },
        exclude_fields=("ids", "texts"),
    )
    result["processor_attention_mask"] = result["processor_attention_mask"].bool()
    return result


@pytest.mark.parametrize("algorithm,runner", [("first_fit", "inline"), ("best_fit", "process")])
def test_real_processor_packing_fields_and_replay(inputs, algorithm, runner):
    pipeline = _pipeline(inputs, algorithm, runner)
    pipeline.validate(strict=True)
    baseline = [_convert(batch) for batch in pipeline]
    assert len(baseline) >= 2  # Resume must have a nonempty suffix.
    assert sum(b["image_grid_thw"].shape[0] for b in baseline) == 7  # Six real images plus one fake.
    assert sum(b["labels"].ne(-100).sum().item() for b in baseline) == 18  # Six complete answers/EOS.
    assert any((~b["processor_attention_mask"] & ~b["padding_mask"]).any() for b in baseline)
    for batch in baseline:
        assert batch["input_ids"].shape[1] == batch["labels"].shape[1] == 40
        assert batch["model_position_ids"].shape == (3, *batch["input_ids"].shape)
        slots = batch["input_ids"].eq(9).sum().item()
        assert slots == (batch["image_grid_thw"].prod(dim=-1) // 4).sum().item()
        assert batch["pixel_values"].shape[0] == slots * 4
        assert batch["pixel_values"].dtype == torch.float32
        assert not torch.equal(batch["pixel_values"], batch["pixel_values"].bfloat16().float())
        assert batch["labels"][batch["input_ids"].eq(9)].eq(-100).all()
        assert not batch["processor_attention_mask"][batch["padding_mask"]].any()
    first = _pipeline(inputs, algorithm, runner)
    with closing(iter(first)) as iterator:
        prefix = [_convert(next(iterator))]
        checkpoint = json.loads(json.dumps(first.checkpoint()))
    resumed = _pipeline(inputs, algorithm, runner)
    resumed.restore(checkpoint)
    actual = prefix + [_convert(batch) for batch in resumed]
    torch.testing.assert_close(actual, baseline, atol=0, rtol=0)


def test_corrupt_media_drop_survives_packing_and_checkpoint_replay(inputs, tmp_path):
    processor_path, documents = inputs
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"unreadable image")
    invalid = {"conversation": [{"role": "user", "content": [{"type": "image", "image": str(corrupt)}]}]}
    # Nine source records form complete three-record source chunks.
    corpus = processor_path, [invalid, *documents[:3], invalid, *documents[3:], invalid]
    baseline = [_convert(batch) for batch in _pipeline(corpus, "first_fit", "process", on_media_error="drop")]
    assert sum(batch["labels"].ne(-100).sum().item() for batch in baseline) == 18
    pipeline = _pipeline(corpus, "first_fit", "process", on_media_error="drop")
    with closing(iter(pipeline)) as iterator:
        prefix = _convert(next(iterator))
        checkpoint = json.loads(json.dumps(pipeline.checkpoint()))
    restored = _pipeline(corpus, "first_fit", "process", on_media_error="drop")
    restored.restore(checkpoint)
    replayed = [prefix, *(_convert(batch) for batch in restored)]
    torch.testing.assert_close(replayed, baseline, atol=0, rtol=0)
