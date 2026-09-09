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
#
# Includes Apache-2.0 code adapted from ByteDance-Seed/Bagel. Upstream references:
#   https://github.com/bytedance-seed/BAGEL
#   data/dataset_base.py
# Upstream copyright: Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Class names, attribute names, and per-sample RNG-consumption ordering match
# the ``data_status`` / buffer state contract.
#
# DIVERGENCE: ``PackedDataset.__iter__`` reseeds Python, NumPy, and Torch RNGs
# at worker start. BAGEL's packer consumes global RNG state while selecting
# sample groups, applying conditioning dropout, and sampling flow-matching
# timesteps; reseeding here keeps the packed-data stream reproducible even if
# model construction consumed RNG before the DataLoader starts iterating.

"""Packed-sequence iterable for BAGEL training."""

from __future__ import annotations

import copy
import json
import logging
import random
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch

from .datasets import BagelDatasetConfig
from .transforms import ImageTransform
from .utils import (
    get_flattened_position_ids_extrapolate,
    get_flattened_position_ids_interpolate,
    len2weight,
    patchify,
    prepare_attention_mask_per_sample,
)
from .video import FrameSampler

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Compatibility alias for the former packing-only config. The canonical
# config now owns complete packed-dataset construction.
DataConfig = BagelDatasetConfig


class PackedDataset(torch.utils.data.IterableDataset):
    """Greedy pack of samples drawn from weighted groups into token-budgeted batches.

    The dataset reseeds at iterator start so AM sees a deterministic
    BAGEL-compatible packed-data stream regardless of earlier RNG consumption
    during model construction or checkpoint loading.
    """

    def __init__(
        self,
        data_config: "BagelDatasetConfig",
        tokenizer: "PreTrainedTokenizerBase",
        special_tokens: Mapping[str, object],
        local_rank: int,
        world_size: int,
        num_workers: int,
        expected_num_tokens: int = 32768,
        max_num_tokens_per_sample: int = 16384,
        max_num_tokens: int = 36864,
        prefer_buffer_before: int = 16384,
        max_buffer_size: int = 50,
        interpolate_pos: bool = False,
        use_flex: bool = False,
        data_status: object | None = None,
        dataset_info: Mapping[str, object] | None = None,
        global_seed: int | None = None,
    ) -> None:
        """Initialize the packed BAGEL data stream.

        Args:
            data_config: Declarative grouped-dataset configuration.
            tokenizer: Tokenizer used to encode text samples.
            special_tokens: BAGEL special tokens exposed to dataset transforms.
            local_rank: Rank used to shard source datasets.
            world_size: Number of ranks consuming the dataset.
            num_workers: Number of dataloader workers per rank.
            expected_num_tokens: Preferred packed token count.
            max_num_tokens_per_sample: Maximum tokens accepted from one sample.
            max_num_tokens: Hard token limit for one packed result.
            prefer_buffer_before: Token threshold at which buffered samples are preferred.
            max_buffer_size: Maximum number of samples retained for greedy packing.
            interpolate_pos: Whether to interpolate image position identifiers.
            use_flex: Whether to emit metadata for flex attention.
            data_status: Optional checkpointed source-dataset status.
            dataset_info: Resolved dataset metadata keyed by source name.
            global_seed: Optional seed used to reset the iterator deterministically.
        """
        super().__init__()
        self.expected_num_tokens = expected_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.prefer_buffer_before = prefer_buffer_before
        self.max_num_tokens = max_num_tokens
        self.max_buffer_size = max_buffer_size
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.use_flex = use_flex
        self._global_seed = global_seed
        for k, v in special_tokens.items():
            setattr(self, k, v)

        if dataset_info is None:
            raise ValueError(
                "PackedDataset requires explicit dataset_info paths. Set "
                "'dataset.dataset_info_path' or 'dataset.dataset_info' in the training config."
            )
        self._dataset_info = dataset_info

        grouped_datasets, is_mandatory, grouped_weights = self.build_datasets(data_config.grouped_datasets, data_status)
        self.grouped_datasets = grouped_datasets
        self.is_mandatory = is_mandatory
        self.grouped_weights = grouped_weights
        self.data_config = data_config
        self.interpolate_pos = interpolate_pos
        self._loaded_state = None
        self._resume_buffer = []
        self._resume_sequence_status = self.set_sequence_status()
        self._yielded_batches = 0
        self._drop_counters = {}
        if self.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

    def _log_drop(self, reason, message, *args, every=100):
        count = self._drop_counters.get(reason, 0) + 1
        self._drop_counters[reason] = count
        if count <= 3 or count % every == 0:
            logger.warning("PackedDataset drop[%s]=%d: " + message, reason, count, *args)

    def _rng_state_dict(self):
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }

    def _load_rng_state_dict(self, state):
        if not state:
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])

    def _grouped_dataset_state_dicts(self):
        states = []
        for dataset in self.grouped_datasets:
            if hasattr(dataset, "state_dict"):
                states.append(dataset.state_dict())
            else:
                states.append(None)
        return states

    def _load_grouped_dataset_state_dicts(self, states):
        if states is None:
            return
        for dataset, state in zip(self.grouped_datasets, states):
            if hasattr(dataset, "load_state_dict"):
                dataset.load_state_dict(state)

    def _set_resume_point(self, buffer, yielded_batches):
        self._resume_buffer = copy.deepcopy(buffer)
        self._resume_sequence_status = self.set_sequence_status()
        self._yielded_batches = yielded_batches

    def state_dict(self):
        return {
            "version": 1,
            "yielded_batches": int(self._yielded_batches),
            "buffer": copy.deepcopy(self._resume_buffer),
            "sequence_status": copy.deepcopy(self._resume_sequence_status),
            "rng": self._rng_state_dict(),
            "grouped_dataset_states": self._grouped_dataset_state_dicts(),
        }

    def load_state_dict(self, state_dict):
        self._loaded_state = copy.deepcopy(state_dict)

    def build_datasets(self, datasets_metainfo, data_status):
        from .datasets import DATASET_REGISTRY

        datasets = []
        is_mandatory = []
        grouped_weights = []
        for grouped_dataset_name, dataset_args in datasets_metainfo.items():
            is_mandatory.append(dataset_args.pop("is_mandatory", False))
            grouped_weights.append(dataset_args.pop("weight", 0.0))

            if "frame_sampler_args" in dataset_args.keys():
                frame_sampler = FrameSampler(**dataset_args.pop("frame_sampler_args"))
                dataset_args["frame_sampler"] = frame_sampler
            if "image_transform_args" in dataset_args.keys():
                transform = ImageTransform(**dataset_args.pop("image_transform_args"))
                dataset_args["transform"] = transform
            if "vit_image_transform_args" in dataset_args.keys():
                vit_transform = ImageTransform(**dataset_args.pop("vit_image_transform_args"))
                dataset_args["vit_transform"] = vit_transform

            assert "dataset_names" in dataset_args.keys()
            dataset_names = dataset_args.pop("dataset_names")
            dataset_args["data_dir_list"] = []
            for item in dataset_names:
                if self.local_rank == 0:
                    logger.info("Preparing Dataset %s/%s", grouped_dataset_name, item)
                meta_info = self._dataset_info[grouped_dataset_name][item]
                dataset_args["data_dir_list"].append(meta_info["data_dir"])

                if "parquet_info_path" in meta_info.keys():
                    if "parquet_info" not in dataset_args.keys():
                        dataset_args["parquet_info"] = {}
                    with open(meta_info["parquet_info_path"], "r") as f:
                        parquet_info = json.load(f)
                    dataset_args["parquet_info"].update(parquet_info)

                if "json_dir" in meta_info.keys():
                    if "json_dir_list" not in dataset_args.keys():
                        dataset_args["json_dir_list"] = [meta_info["json_dir"]]
                    else:
                        dataset_args["json_dir_list"].append(meta_info["json_dir"])

                if "jsonl_path" in meta_info.keys():
                    if "jsonl_path_list" not in dataset_args.keys():
                        dataset_args["jsonl_path_list"] = [meta_info["jsonl_path"]]
                    else:
                        dataset_args["jsonl_path_list"].append(meta_info["jsonl_path"])

            resume_data_status = dataset_args.pop("resume_data_status", True)
            if data_status is not None and grouped_dataset_name in data_status.keys() and resume_data_status:
                data_status_per_group = data_status[grouped_dataset_name]
            else:
                data_status_per_group = None
            dataset = DATASET_REGISTRY[grouped_dataset_name](
                dataset_name=grouped_dataset_name,
                tokenizer=self.tokenizer,
                local_rank=self.local_rank,
                world_size=self.world_size,
                num_workers=self.num_workers,
                data_status=data_status_per_group,
                **dataset_args,
            )
            datasets.append(dataset)

        return datasets, is_mandatory, grouped_weights

    def set_epoch(self, seed):
        # Stash the seed so ``__iter__`` can derive the per-worker reseed
        # even when the caller did not set ``data_config.data_seed``.
        self._data_seed = int(seed)
        for dataset in self.grouped_datasets:
            dataset.set_epoch(seed)

    def set_sequence_status(self):
        sequence_status = dict(
            curr=0,
            sample_lens=list(),
            packed_position_ids=list(),
            nested_attention_masks=list(),
            split_lens=list(),
            attn_modes=list(),
            packed_text_ids=list(),
            packed_text_indexes=list(),
            packed_label_ids=list(),
            ce_loss_indexes=list(),
            ce_loss_weights=list(),
            vae_image_tensors=list(),
            packed_latent_position_ids=list(),
            vae_latent_shapes=list(),
            packed_vae_token_indexes=list(),
            packed_timesteps=list(),
            mse_loss_indexes=list(),
            packed_vit_tokens=list(),
            vit_token_seqlens=list(),
            packed_vit_position_ids=list(),
            packed_vit_token_indexes=list(),
        )
        return sequence_status

    def to_tensor(self, sequence_status):
        data = dict(
            sequence_length=sum(sequence_status["sample_lens"]),
            sample_lens=sequence_status["sample_lens"],
            packed_text_ids=torch.tensor(sequence_status["packed_text_ids"]),
            packed_text_indexes=torch.tensor(sequence_status["packed_text_indexes"]),
            packed_position_ids=torch.tensor(sequence_status["packed_position_ids"]),
        )
        if not self.use_flex:
            data["nested_attention_masks"] = sequence_status["nested_attention_masks"]
        else:
            sequence_len = data["sequence_length"]
            pad_len = self.max_num_tokens - sequence_len
            data["split_lens"] = sequence_status["split_lens"] + [pad_len]
            data["attn_modes"] = sequence_status["attn_modes"] + ["causal"]
            data["sample_lens"] += [pad_len]

        if len(sequence_status["vae_image_tensors"]) > 0:
            image_tensors = sequence_status.pop("vae_image_tensors")
            image_sizes = [item.shape for item in image_tensors]
            max_image_size = [max(item) for item in list(zip(*image_sizes))]
            padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
            for i, image_tensor in enumerate(image_tensors):
                padded_images[i, :, : image_tensor.shape[1], : image_tensor.shape[2]] = image_tensor

            data["padded_images"] = padded_images
            data["patchified_vae_latent_shapes"] = sequence_status["vae_latent_shapes"]
            data["packed_latent_position_ids"] = torch.cat(sequence_status["packed_latent_position_ids"], dim=0)
            data["packed_vae_token_indexes"] = torch.tensor(sequence_status["packed_vae_token_indexes"])

        if len(sequence_status["packed_vit_tokens"]) > 0:
            data["packed_vit_tokens"] = torch.cat(sequence_status["packed_vit_tokens"], dim=0)
            data["packed_vit_position_ids"] = torch.cat(sequence_status["packed_vit_position_ids"], dim=0)
            data["packed_vit_token_indexes"] = torch.tensor(sequence_status["packed_vit_token_indexes"])
            data["vit_token_seqlens"] = torch.tensor(sequence_status["vit_token_seqlens"])

        if len(sequence_status["packed_timesteps"]) > 0:
            data["packed_timesteps"] = torch.tensor(sequence_status["packed_timesteps"])
            data["mse_loss_indexes"] = torch.tensor(sequence_status["mse_loss_indexes"])

        if len(sequence_status["packed_label_ids"]) > 0:
            data["packed_label_ids"] = torch.tensor(sequence_status["packed_label_ids"])
            data["ce_loss_indexes"] = torch.tensor(sequence_status["ce_loss_indexes"])
            data["ce_loss_weights"] = torch.tensor(sequence_status["ce_loss_weights"])

        return data

    def __iter__(self):
        # Upstream ``PackedDataset.__iter__`` consumes the global ``random``
        # state per-sample (lines 269/323/355/399 of upstream
        # ``data/dataset_base.py``) for dropout decisions. ``set_epoch``
        # only reseeds the local file-shuffle rng — it does NOT touch the
        # global RNG. Any code the caller runs between ``set_seed(...)`` and
        # the first ``next(loader_iter)`` that consumes the global ``random``
        # / ``numpy`` / ``torch`` state (e.g. model construction with a
        # different RNG footprint than upstream's explicit chain) will shift
        # the pack sequence. Baking the reseed into ``__iter__`` at the very
        # last moment — after dataloader workers have forked and per-worker
        # ``torch.initial_seed()`` has been set — makes the pack stream
        # reproducible across callers.
        #
        # Derivation matches the intent of upstream's ``global_seed *
        # world_size + rank`` convention, extended with the worker id so
        # different DataLoader workers see disjoint streams.
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        # Prefer _global_seed (BAGEL's "global_seed" semantics) for the
        # in-__iter__ reseed; fall back to _data_seed if not set. The data-
        # file shuffle inside each DistributedIterableDataset.set_epoch keeps
        # using _data_seed (whatever was passed to PackedDataset.set_epoch),
        # matching BAGEL's pretrain_unified_navit.py:641 which calls
        # set_epoch(data_seed) but seeds rank_seed with global_seed.
        loaded_state = self._loaded_state
        if loaded_state is not None:
            self._load_grouped_dataset_state_dicts(loaded_state.get("grouped_dataset_states"))
            buffer = loaded_state.get("buffer") or []
            sequence_status = loaded_state.get("sequence_status") or self.set_sequence_status()
            batch_data_indexes = []
            self._yielded_batches = int(loaded_state.get("yielded_batches", 0))
            self._load_rng_state_dict(loaded_state.get("rng"))
            self._loaded_state = None
        else:
            base_seed = int(getattr(self, "_global_seed", None) or getattr(self, "_data_seed", 0))
            rank_seed = base_seed * max(1, self.world_size) + int(self.local_rank)
            rank_seed = rank_seed * max(1, self.num_workers) + int(worker_id)
            rank_seed = rank_seed & 0xFFFFFFFF
            random.seed(rank_seed)
            np.random.seed(rank_seed)
            torch.manual_seed(rank_seed)
            sequence_status = self.set_sequence_status()
            batch_data_indexes = []
            buffer = []

        total_weights = sum(self.grouped_weights)
        assert total_weights > 0.0
        group_cumprobs = [sum(self.grouped_weights[: i + 1]) / total_weights for i in range(len(self.grouped_weights))]
        dataset_iters = [iter(dataset) for dataset in self.grouped_datasets]
        while True:
            if sequence_status["curr"] == 0:
                for group_index, group_iter in enumerate(dataset_iters):
                    if self.is_mandatory[group_index]:
                        while True:
                            sample = next(group_iter)
                            num_tokens = sample["num_tokens"] + 2 * len(sample["sequence_plan"])
                            if num_tokens < self.max_num_tokens_per_sample:
                                sequence_status = self.pack_sequence(sample, sequence_status)
                                batch_data_indexes.append(sample["data_indexes"])
                                break
                            else:
                                self._log_drop(
                                    "overlength_mandatory_sample", "skipping sample with length %s", num_tokens
                                )
                                continue

            if sequence_status["curr"] < self.prefer_buffer_before and len(buffer) > 0:
                sample = buffer.pop(0)
                sample_from_buffer = True
            else:
                n = random.random()
                group_index = 0
                for i, cumprob in enumerate(group_cumprobs):
                    if n < cumprob:
                        group_index = i
                        break
                sample = next(dataset_iters[group_index])
                sample_from_buffer = False

            num_tokens = sample["num_tokens"] + 2 * len(sample["sequence_plan"])
            if num_tokens > self.max_num_tokens_per_sample:
                self._log_drop("overlength_sample", "skipping sample with length %s", num_tokens)
                continue

            if sequence_status["curr"] + num_tokens > self.max_num_tokens:
                if len(buffer) < self.max_buffer_size and not sample_from_buffer:
                    buffer.append(sample)
                else:
                    logger.info("Yielding data with length %s", sum(sequence_status["sample_lens"]))
                    data = self.to_tensor(sequence_status)
                    data["batch_data_indexes"] = batch_data_indexes
                    self._set_resume_point(buffer, self._yielded_batches + 1)
                    yield data
                    sequence_status = self.set_sequence_status()
                    batch_data_indexes = []
                continue

            sequence_status = self.pack_sequence(sample, sequence_status)
            batch_data_indexes.append(sample["data_indexes"])

            if sequence_status["curr"] >= self.expected_num_tokens:
                data = self.to_tensor(sequence_status)
                data["batch_data_indexes"] = batch_data_indexes
                self._set_resume_point(buffer, self._yielded_batches + 1)
                yield data
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []

    def pack_sequence(self, sample, sequence_status):
        image_tensor_list = sample["image_tensor_list"]
        text_ids_list = sample["text_ids_list"]
        sequence_plan = sample["sequence_plan"]

        split_lens, attn_modes = list(), list()
        curr = sequence_status["curr"]
        curr_rope_id = 0
        sample_lens = 0

        for item in sequence_plan:
            split_start = item.get("split_start", True)
            if split_start:
                curr_split_len = 0

            if item["type"] == "text":
                text_ids = text_ids_list.pop(0)
                if item["enable_cfg"] == 1 and random.random() < self.data_config.text_cond_dropout_prob:
                    continue

                shifted_text_ids = [self.bos_token_id] + text_ids
                sequence_status["packed_text_ids"].extend(shifted_text_ids)
                sequence_status["packed_text_indexes"].extend(range(curr, curr + len(shifted_text_ids)))
                if item["loss"] == 1:
                    sequence_status["ce_loss_indexes"].extend(range(curr, curr + len(shifted_text_ids)))
                    sequence_status["ce_loss_weights"].extend(
                        [len2weight(len(shifted_text_ids))] * len(shifted_text_ids)
                    )
                    sequence_status["packed_label_ids"].extend(text_ids + [self.eos_token_id])
                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                # add a <|im_end|> token
                sequence_status["packed_text_ids"].append(self.eos_token_id)
                sequence_status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:
                    sequence_status["ce_loss_indexes"].append(curr)
                    sequence_status["ce_loss_weights"].append(1.0)
                    sequence_status["packed_label_ids"].append(item["special_token_label"])
                curr += 1
                curr_split_len += 1

                attn_modes.append("causal")
                sequence_status["packed_position_ids"].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len

            elif item["type"] == "vit_image":
                image_tensor = image_tensor_list.pop(0)
                if item["enable_cfg"] == 1 and random.random() < self.data_config.vit_cond_dropout_prob:
                    curr_rope_id += 1
                    continue

                sequence_status["packed_text_ids"].append(self.start_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                curr += 1
                curr_split_len += 1

                vit_tokens = patchify(image_tensor, self.data_config.vit_patch_size)
                num_img_tokens = vit_tokens.shape[0]
                sequence_status["packed_vit_token_indexes"].extend(range(curr, curr + num_img_tokens))
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status["packed_vit_tokens"].append(vit_tokens)
                sequence_status["vit_token_seqlens"].append(num_img_tokens)
                sequence_status["packed_vit_position_ids"].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1),
                        image_tensor.size(2),
                        self.data_config.vit_patch_size,
                        max_num_patches_per_side=self.data_config.max_num_patch_per_side,
                    )
                )

                sequence_status["packed_text_ids"].append(self.end_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:
                    sequence_status["ce_loss_indexes"].append(curr)
                    sequence_status["ce_loss_weights"].append(1.0)
                    sequence_status["packed_label_ids"].append(item["special_token_label"])
                curr += 1
                curr_split_len += 1

                attn_modes.append("full")
                sequence_status["packed_position_ids"].extend([curr_rope_id] * curr_split_len)
                curr_rope_id += 1

            elif item["type"] == "vae_image":
                image_tensor = image_tensor_list.pop(0)
                if item["enable_cfg"] == 1 and random.random() < self.data_config.vae_cond_dropout_prob:
                    curr_rope_id += 1
                    continue

                sequence_status["packed_text_ids"].append(self.start_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                curr += 1
                curr_split_len += 1

                sequence_status["vae_image_tensors"].append(image_tensor)
                sequence_status["packed_latent_position_ids"].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1),
                        image_tensor.size(2),
                        self.data_config.vae_image_downsample,
                        max_num_patches_per_side=self.data_config.max_latent_size,
                    )
                )
                H, W = image_tensor.shape[1:]
                h = H // self.data_config.vae_image_downsample
                w = W // self.data_config.vae_image_downsample
                sequence_status["vae_latent_shapes"].append((h, w))

                num_img_tokens = w * h
                sequence_status["packed_vae_token_indexes"].extend(range(curr, curr + num_img_tokens))
                if item["loss"] == 1:
                    sequence_status["mse_loss_indexes"].extend(range(curr, curr + num_img_tokens))
                    if split_start:
                        timestep = np.random.randn()
                else:
                    timestep = float("-inf")

                sequence_status["packed_timesteps"].extend([timestep] * num_img_tokens)
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status["packed_text_ids"].append(self.end_of_image)
                sequence_status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:
                    sequence_status["ce_loss_indexes"].append(curr)
                    sequence_status["ce_loss_weights"].append(1.0)
                    sequence_status["packed_label_ids"].append(item["special_token_label"])
                curr += 1
                curr_split_len += 1

                if split_start:
                    if item["loss"] == 1 and "frame_delta" not in item.keys():
                        attn_modes.append("noise")
                    else:
                        attn_modes.append("full")
                sequence_status["packed_position_ids"].extend([curr_rope_id] * (num_img_tokens + 2))
                if "frame_delta" in item.keys():
                    curr_rope_id += item["frame_delta"]
                elif item["loss"] == 0:
                    curr_rope_id += 1

            if item.get("split_end", True):
                split_lens.append(curr_split_len)
                sample_lens += curr_split_len

        sequence_status["curr"] = curr
        sequence_status["sample_lens"].append(sample_lens)
        if not self.use_flex:
            sequence_status["nested_attention_masks"].append(prepare_attention_mask_per_sample(split_lens, attn_modes))
        else:
            sequence_status["split_lens"].extend(split_lens)
            sequence_status["attn_modes"].extend(attn_modes)

        return sequence_status
