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

import json
from pathlib import Path

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_automodel.components.datasets.diffusion.image_edit_dataset import ImageEditDataloaderConfig, ImageEditDataset
from nemo_automodel.components.datasets.diffusion.sampler import SequentialBucketSampler

MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"


def _signature(
    target_shape: tuple[int, int, int],
    context_shapes: list[tuple[int, int, int]],
) -> dict[str, object]:
    return {
        "target": list(target_shape),
        "contexts": [list(shape) for shape in context_shapes],
    }


def _make_payload(
    sample_index: int,
    *,
    target_shape: tuple[int, int, int] = (4, 8, 8),
    context_shapes: list[tuple[int, int, int]] | None = None,
    prompt_length: int = 4,
    include_token_lengths: bool = True,
    conditioning_shape: tuple[int, ...] | None = (3,),
) -> dict[str, object]:
    """Build one cached image-edit payload.

    Returns:
        Payload containing a target latent of shape ``[channels, height,
        width]``, context latents of shape ``[channels, height, width]``, prompt
        embeddings of shape ``[sequence, hidden]``, a prompt mask of shape
        ``[sequence]``, and optional conditioning tensors with shapes declared
        by ``conditioning_shape``.
    """
    if context_shapes is None:
        context_shapes = [(4, 8, 8)]
    signature = _signature(target_shape, context_shapes)
    conditioning_tensors = {}
    conditioning_shapes = {}
    if conditioning_shape is not None:
        conditioning_tensors["image_grid_thw"] = torch.arange(
            max(torch.tensor(conditioning_shape).prod().item(), 1),
            dtype=torch.int64,
        ).reshape(conditioning_shape)
        conditioning_shapes["image_grid_thw"] = list(conditioning_shape)

    metadata = {
        "original_ids": {"row_index": sample_index, "image_id": f"image-{sample_index}"},
        "target_spatial_shape": list(target_shape[-2:]),
        "context_spatial_shapes": [list(shape[-2:]) for shape in context_shapes],
        "compound_bucket_signature": signature,
        "conditioning_shapes": conditioning_shapes,
        "mask_column": f"mask-{sample_index}.png",
    }
    if include_token_lengths:
        metadata.update(
            {
                "target_token_length": 10 + sample_index,
                "context_token_lengths": [20 + sample_index + index for index in range(len(context_shapes))],
                "text_token_length": prompt_length - sample_index,
            }
        )
    prompt_attention_mask = torch.ones(prompt_length, dtype=torch.bool)
    if include_token_lengths:
        prompt_attention_mask[prompt_length - sample_index :] = False

    return {
        "target_latent": torch.full(target_shape, sample_index, dtype=torch.bfloat16),
        "context_latents": [
            torch.full(shape, sample_index + context_index + 1, dtype=torch.bfloat16)
            for context_index, shape in enumerate(context_shapes)
        ],
        "prompt_embeddings": torch.arange(prompt_length * 6, dtype=torch.float32).reshape(prompt_length, 6),
        "prompt_attention_mask": prompt_attention_mask,
        "conditioning_tensors": conditioning_tensors,
        "metadata": metadata,
    }


def _write_cache(
    cache_dir: Path,
    payloads: list[dict[str, object]],
    *,
    shard_signatures: list[dict[str, object]] | None = None,
) -> None:
    """Write cached payloads and their manifest.

    Args:
        cache_dir: Directory receiving tensor payloads and metadata files.
        payloads: Payload mappings whose target and context latents have shape
            ``[channels, height, width]``, prompt embeddings have shape
            ``[sequence, hidden]``, and prompt masks have shape ``[sequence]``.
        shard_signatures: Optional metadata signatures to write instead of the
            signatures embedded in ``payloads``.
    """
    records = []
    for index, payload in enumerate(payloads):
        cache_file = cache_dir / f"sample_{index:04d}.pt"
        torch.save(payload, cache_file)
        signature = payload["metadata"]["compound_bucket_signature"]
        if shard_signatures is not None:
            signature = shard_signatures[index]
        records.append(
            {
                "cache_file": cache_file.name,
                "compound_bucket_signature": signature,
            }
        )

    shard_file = cache_dir / "metadata_shard_0000.json"
    shard_file.write_text(json.dumps(records), encoding="utf-8")
    manifest = {
        "dataset_name": "osunlp/MagicBrush",
        "split": "dev",
        "row_limit": len(payloads),
        "preprocessing_config": {
            "max_pixels": 1024 * 1024,
            "model_name": MODEL_NAME,
        },
        "shards": [shard_file.name],
    }
    (cache_dir / "metadata.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_dataloader_collates_cache_contract_and_cpu_token_counts(tmp_path: Path) -> None:
    first = _make_payload(0, prompt_length=3)
    second = _make_payload(1, prompt_length=5)
    _write_cache(tmp_path, [first, second])

    result = ImageEditDataloaderConfig(
        cache_dir=str(tmp_path),
        drop_last=False,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    ).build(dp_rank=0, dp_world_size=1, batch_size=2)
    batch = next(iter(result.dataloader))

    assert isinstance(result.dataloader, StatefulDataLoader)
    assert isinstance(result.sampler, SequentialBucketSampler)
    assert batch["data_type"] == "image"
    assert batch["image_latents"].shape == (2, 4, 8, 8)
    assert len(batch["context_latents"]) == 1
    assert batch["context_latents"][0].shape == (2, 4, 8, 8)
    assert batch["text_embeddings"].shape == (2, 5, 6)
    assert batch["text_attention_mask"].shape == (2, 5)
    assert torch.count_nonzero(batch["text_embeddings"][0, 3:]) == 0
    assert torch.count_nonzero(batch["text_attention_mask"][0, 3:]) == 0
    assert batch["conditioning_tensors"]["image_grid_thw"].shape == (2, 3)

    metadata = batch["metadata"]
    assert metadata["batch_size"] == 2
    assert metadata["target_token_counts"] == [10, 11]
    assert metadata["context_token_counts"] == [20, 21]
    assert metadata["text_token_counts"] == [3, 4]
    assert metadata["total_token_counts"] == [33, 36]
    assert metadata["target_token_count"] == 21
    assert metadata["context_token_count"] == 41
    assert metadata["text_token_count"] == 7
    assert metadata["total_token_count"] == 69
    assert all(isinstance(value, int) for value in metadata["total_token_counts"])
    assert metadata["samples"][0]["mask_column"] == "mask-0.png"
    assert result.dataloader.state_dict()


def test_dataloader_rng_is_global_state_independent_and_resume_is_exact(tmp_path: Path) -> None:
    """Loader iterator bookkeeping uses rank-local state and resumes exactly."""
    _write_cache(tmp_path, [_make_payload(index) for index in range(4)])
    config = ImageEditDataloaderConfig(
        cache_dir=str(tmp_path),
        drop_last=False,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        seed=123,
    )
    result = config.build(dp_rank=0, dp_world_size=1, batch_size=1)

    torch.manual_seed(2026)
    global_state = torch.get_rng_state().clone()
    iterator = iter(result.dataloader)
    next(iterator)
    checkpoint = result.dataloader.state_dict()
    expected_remaining_ids = [batch["metadata"]["samples"][0]["original_ids"]["row_index"] for batch in iterator]

    restored = config.build(dp_rank=0, dp_world_size=1, batch_size=1)
    restored.dataloader.load_state_dict(checkpoint)
    actual_remaining_ids = [
        batch["metadata"]["samples"][0]["original_ids"]["row_index"] for batch in restored.dataloader
    ]

    assert actual_remaining_ids == expected_remaining_ids
    assert torch.equal(torch.get_rng_state(), global_state)


def test_dataset_falls_back_to_spatial_and_sequence_token_lengths(tmp_path: Path) -> None:
    payload = _make_payload(
        0,
        target_shape=(4, 6, 8),
        context_shapes=[(4, 3, 5), (4, 2, 7)],
        prompt_length=9,
        include_token_lengths=False,
        conditioning_shape=None,
    )
    _write_cache(tmp_path, [payload])

    sample = ImageEditDataset(str(tmp_path))[0]

    assert sample["metadata"]["target_token_length"] == 48
    assert sample["metadata"]["context_token_lengths"] == [15, 14]
    assert sample["metadata"]["text_token_length"] == 9
    assert sample["conditioning_tensors"] == {}
    assert len(sample["context_latents"]) == 2
    assert torch.all(sample["context_latents"][0] == 1)
    assert torch.all(sample["context_latents"][1] == 2)


def test_compound_bucketing_separates_context_shapes(tmp_path: Path) -> None:
    first = _make_payload(0, context_shapes=[(4, 8, 8)])
    second = _make_payload(0, context_shapes=[(4, 4, 16)])
    _write_cache(tmp_path, [first, second])

    dataset = ImageEditDataset(str(tmp_path))
    result = ImageEditDataloaderConfig(
        cache_dir=str(tmp_path),
        drop_last=False,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    ).build(dp_rank=0, dp_world_size=1, batch_size=2)
    batches = list(result.dataloader)

    assert dataset.get_bucket_info()["total_buckets"] == 2
    assert len(batches) == 2
    assert {tuple(batch["context_latents"][0].shape[-2:]) for batch in batches} == {(8, 8), (4, 16)}


def test_dataloader_rejects_unsafe_dynamic_batch_sizing(tmp_path: Path) -> None:
    """Dynamic sizing cannot ignore ordered context latent cost."""
    _write_cache(tmp_path, [_make_payload(0)])

    with pytest.raises(ValueError, match="compound target/context latent signature"):
        ImageEditDataloaderConfig(cache_dir=str(tmp_path), dynamic_batch_size=True).build(
            dp_rank=0,
            dp_world_size=1,
            batch_size=1,
        )


def test_dataset_rejects_tensor_shape_that_disagrees_with_shard_signature(tmp_path: Path) -> None:
    payload = _make_payload(0)
    wrong_signature = _signature((4, 16, 4), [(4, 8, 8)])
    _write_cache(tmp_path, [payload], shard_signatures=[wrong_signature])

    dataset = ImageEditDataset(str(tmp_path))
    with pytest.raises(ValueError, match="do not match metadata shard signature"):
        dataset[0]


def test_dataset_rejects_invalid_cache_tensor_contract(tmp_path: Path) -> None:
    payload = _make_payload(0)
    payload["prompt_attention_mask"] = torch.ones(3, dtype=torch.bool)
    _write_cache(tmp_path, [payload])

    dataset = ImageEditDataset(str(tmp_path))
    with pytest.raises(ValueError, match="prompt_attention_mask must have shape"):
        dataset[0]


def test_dataset_rejects_empty_context_list(tmp_path: Path) -> None:
    payload = _make_payload(0, context_shapes=[])
    _write_cache(tmp_path, [payload])

    with pytest.raises(ValueError, match="non-empty ordered list"):
        ImageEditDataset(str(tmp_path))


def test_dataset_rejects_conditioning_shape_mismatch(tmp_path: Path) -> None:
    payload = _make_payload(0)
    payload["metadata"]["conditioning_shapes"]["image_grid_thw"] = [1, 3]
    _write_cache(tmp_path, [payload])

    dataset = ImageEditDataset(str(tmp_path))
    with pytest.raises(ValueError, match="must contain exactly 1 dimensions"):
        dataset[0]


def test_dataloader_rejects_incompatible_conditioning_tensors(tmp_path: Path) -> None:
    first = _make_payload(0, conditioning_shape=(3,))
    second = _make_payload(0, conditioning_shape=(1, 3))
    _write_cache(tmp_path, [first, second])
    result = ImageEditDataloaderConfig(
        cache_dir=str(tmp_path),
        drop_last=False,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    ).build(dp_rank=0, dp_world_size=1, batch_size=2)

    with pytest.raises(ValueError, match="same shape and dtype"):
        next(iter(result.dataloader))


def test_dataset_rejects_cache_path_outside_cache_directory(tmp_path: Path) -> None:
    payload = _make_payload(0)
    outside_file = tmp_path.parent / f"{tmp_path.name}_outside.pt"
    torch.save(payload, outside_file)
    signature = payload["metadata"]["compound_bucket_signature"]
    shard_file = tmp_path / "metadata_shard_0000.json"
    shard_file.write_text(
        json.dumps([{"cache_file": str(outside_file), "compound_bucket_signature": signature}]),
        encoding="utf-8",
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "shards": [shard_file.name],
            }
        ),
        encoding="utf-8",
    )

    dataset = ImageEditDataset(str(tmp_path))
    with pytest.raises(ValueError, match="outside cache directory"):
        dataset[0]
