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
#   data/vlm_dataset.py
#   data/t2i_dataset.py
#   data/interleave_datasets/edit_dataset.py
# Upstream copyright: Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Class names and yielded-dict schema match the packing layer contract.

"""BAGEL multimodal datasets: VLM-SFT, T2I pretrain, unified image editing.

The three dataset families emit the data-yield surface needed for the
3-group ``example.yaml`` mixture to be iterated by AM's
:class:`.packing.PackedDataset`. Stage 1 consumes only the understanding-side
loss-bearing pieces; Stage 2 additionally consumes VAE-image plan entries for
flow-matching loss. VAE encode, MSE computation, noise sampling, and timestep
embedding are intentionally training/model concerns, not dataset concerns.
"""

from __future__ import annotations

import io
import json
import logging
import os
import random

from PIL import Image, ImageFile, PngImagePlugin

from .distributed_iterable import DistributedIterableDataset
from .interleave import UnifiedEditIterableDataset
from .parquet_utils import get_parquet_data_paths, init_arrow_pf_fs
from .utils import pil_img2rgb

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
_MaximumDecompressedSize = 1024
_MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = _MaximumDecompressedSize * _MegaByte


class SftJSONLIterableDataset(DistributedIterableDataset):
    """Iterable over conversation JSONL rows where each row may reference images."""

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        frame_sampler,
        jsonl_path_list,
        data_dir_list,
        num_used_data,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        shuffle_lines=False,
        shuffle_seed=0,
    ):
        """
        Args:
            jsonl_path_list: list of jsonl file paths
            data_dir_list: list of image dirs aligned with ``jsonl_path_list``
            num_used_data: list of row counts to draw from each jsonl
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.frame_sampler = frame_sampler
        self.set_data_status(data_status)
        self.data_paths = self.get_data_paths(
            jsonl_path_list,
            data_dir_list,
            num_used_data,
            shuffle_lines,
            shuffle_seed,
        )
        self.set_epoch()

    def get_data_paths(
        self,
        jsonl_path_list,
        data_dir_list,
        num_used_data,
        shuffle_lines,
        shuffle_seed,
    ):
        data_paths = []
        for jsonl_path, image_dir, num_data_point in zip(jsonl_path_list, data_dir_list, num_used_data):
            with open(jsonl_path, "r") as f:
                raw_data = f.readlines()
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            raw_data = raw_data[:num_data_point]
            data_paths.extend([(json_data, image_dir) for json_data in raw_data])
        return data_paths

    def change_format(self, data, num_images):
        elements = []
        for conversation in data["conversations"]:
            if conversation["from"] == "human":
                if "<image>" not in conversation["value"]:
                    elements.append({"type": "text", "has_loss": 0, "text": conversation["value"]})
                else:
                    text_list = conversation["value"].split("<image>")
                    for idx, text in enumerate(text_list):
                        if text.strip() != "":
                            elements.append({"type": "text", "has_loss": 0, "text": text.strip()})
                        if (idx != len(text_list) - 1) and (idx < num_images):
                            elements.append({"type": "image"})
            elif conversation["from"] == "gpt":
                elements.append({"type": "text", "has_loss": 1, "text": conversation["value"]})
        return elements

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        worker_data_status = self._get_worker_data_status(worker_id)
        if worker_data_status is not None:
            row_start_id = worker_data_status + 1
        else:
            row_start_id = 0
        transform_stride = self.transform.stride

        logger.info(
            "rank-%s worker-%s dataset-%s: resuming data at row#%s",
            self.local_rank,
            worker_id,
            self.dataset_name,
            row_start_id,
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (data, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                num_tokens = 0
                image_tensor_list = []
                text_ids_list = []
                sequence_plan = []

                try:
                    data_item = json.loads(data)
                    raw_images = None
                    if "image" in data_item:
                        if type(data_item["image"]) == list:
                            raw_images = [
                                pil_img2rgb(Image.open(os.path.join(image_dir, image))) for image in data_item["image"]
                            ]
                        else:
                            raw_images = [pil_img2rgb(Image.open(os.path.join(image_dir, data_item["image"])))]
                    elif "video" in data_item:
                        raw_images = self.frame_sampler(os.path.join(image_dir, data_item["video"]))
                        special_tokens = "<image>" * len(raw_images)
                        for item in data_item["conversations"]:
                            if "<video>" in item["value"]:
                                item["value"] = item["value"].replace("<video>", special_tokens)
                                break
                            else:
                                raise ValueError("Cannot find <video> in the conversation!")
                except Exception:
                    self._log_drop("jsonl_parse", "failed to parse multimodal JSONL row", exc_info=True)
                    continue

                if raw_images:
                    for raw_image in raw_images:
                        image_tensor = self.transform(raw_image, img_num=len(raw_images))
                        image_tensor_list.append(image_tensor)
                        height, width = image_tensor.shape[1:]
                        num_tokens += width * height // transform_stride**2

                elements = self.change_format(data_item, len(image_tensor_list))

                for item in elements:
                    if item["type"] == "text":
                        text_data = item["text"]
                        text_ids = self.tokenizer.encode(text_data)
                        if len(text_ids) > 0:
                            text_ids_list.append(text_ids)
                            num_tokens += len(text_ids)
                            current_plan = {
                                "type": "text",
                                "enable_cfg": 0,
                                "loss": item["has_loss"],
                                "special_token_loss": 0,
                                "special_token_label": None,
                            }
                            sequence_plan.append(current_plan)
                    elif item["type"] == "image":
                        current_plan = {
                            "type": "vit_image",
                            "enable_cfg": 0,
                            "loss": 0,
                            "special_token_loss": 0,
                            "special_token_label": None,
                        }
                        sequence_plan.append(current_plan)

                has_loss = [item["loss"] for item in sequence_plan]
                if sum(has_loss) == 0:
                    self._log_drop("no_loss_labels", "skipping sample without loss labels")
                    continue

                self._set_worker_resume_data_status(worker_id, row_idx)
                yield dict(
                    image_tensor_list=image_tensor_list,
                    text_ids_list=text_ids_list,
                    sequence_plan=sequence_plan,
                    num_tokens=num_tokens,
                    data_indexes={
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    },
                )

            row_start_id = 0
            logger.info("%s repeat in rank-%s worker-%s", self.dataset_name, self.local_rank, worker_id)


class T2IIterableDataset(DistributedIterableDataset):
    """Iterable over a parquet-sharded (image, captions) text-to-image dataset.

    Each parquet row is expected to carry:
      * ``image``: raw image bytes.
      * ``captions``: JSON-encoded ``{key: caption}`` dict; one caption is
        sampled uniformly per row.

    The yielded dict carries only one image tensor (the VAE input). Stage 1
    models ignore the VAE branch; Stage 2 consumes it via the PackedDataset
    ``vae_image`` branch + flow-matching loss. No VAE encoding happens in
    the data pipeline — just tensor preparation.
    """

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        data_dir_list,
        num_used_data,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
    ):
        """
        Args:
            data_dir_list: list of directories containing parquet shards
            num_used_data: list of path counts to draw per directory (>= num shards)
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.set_data_status(data_status)
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data):
        return get_parquet_data_paths(data_dir_list, num_used_data)

    def __iter__(self):
        # Lazy-import pyarrow so AM installs without it still import this module.
        import pyarrow.parquet as pq

        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        worker_data_status = self._get_worker_data_status(worker_id)
        if worker_data_status is not None:
            parquet_start_id = worker_data_status[0]
            row_group_start_id = worker_data_status[1]
            row_start_id = worker_data_status[2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        logger.info(
            "rank-%s worker-%s dataset-%s: resuming data at parquet#%s, rg#%s, row#%s",
            self.local_rank,
            worker_id,
            self.dataset_name,
            parquet_start_id,
            row_group_start_id,
            row_start_id,
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for parquet_idx, parquet_file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                fs = init_arrow_pf_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    fr = pq.ParquetFile(f)
                    row_group_ids = list(range(fr.num_row_groups))
                    row_group_ids_ = row_group_ids[row_group_start_id:]

                    for row_group_id in row_group_ids_:
                        df = fr.read_row_group(row_group_id).to_pandas()
                        df = df.iloc[row_start_id:]

                        for row_idx, row in df.iterrows():
                            num_tokens = 0
                            try:
                                image_byte = row["image"]
                                image = pil_img2rgb(Image.open(io.BytesIO(image_byte)))
                            except Exception as e:
                                self._log_drop(
                                    "t2i_image_parse",
                                    "error %s in rg#%s, %s",
                                    e,
                                    row_group_id,
                                    parquet_file_path,
                                )
                                continue
                            image_tensor = self.transform(image)
                            height, width = image_tensor.shape[1:]
                            num_tokens += width * height // transform_stride**2

                            try:
                                caption_dict = row["captions"]
                                caption_dict = json.loads(caption_dict)
                            except Exception as e:
                                self._log_drop(
                                    "t2i_caption_parse",
                                    "error %s in rg#%s, %s",
                                    e,
                                    row_group_id,
                                    parquet_file_path,
                                )
                                continue

                            caps_token = [self.tokenizer.encode(v) for _, v in caption_dict.items()]
                            if len(caps_token) == 0:
                                self._log_drop(
                                    "t2i_no_caption", "no caption in rg#%s, %s", row_group_id, parquet_file_path
                                )
                                caption_token = self.tokenizer.encode(" ")
                            else:
                                caption_token = random.choice(caps_token)

                            sequence_plan, text_ids_list = [], []
                            text_ids = caption_token
                            num_tokens += len(caption_token)
                            text_ids_list.append(text_ids)
                            sequence_plan.append(
                                {
                                    "type": "text",
                                    "enable_cfg": 1,
                                    "loss": 0,
                                    "special_token_loss": 0,
                                    "special_token_label": None,
                                }
                            )

                            sequence_plan.append(
                                {
                                    "type": "vae_image",
                                    "enable_cfg": 0,
                                    "loss": 1,
                                    "special_token_loss": 0,
                                    "special_token_label": None,
                                }
                            )

                            sample = dict(
                                image_tensor_list=[image_tensor],
                                text_ids_list=text_ids_list,
                                num_tokens=num_tokens,
                                sequence_plan=sequence_plan,
                                data_indexes={
                                    "data_indexes": [parquet_idx, row_group_id, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                },
                            )
                            self._set_worker_resume_data_status(worker_id, [parquet_idx, row_group_id, row_idx])
                            yield sample

                        row_start_id = 0
                    row_group_start_id = 0
            parquet_start_id = 0
            logger.info("%s repeat in rank-%s worker-%s", self.dataset_name, self.local_rank, worker_id)


# Registry mirroring upstream ``data/dataset_info.py::DATASET_REGISTRY``.
# The packing layer consumes this when building grouped datasets.
DATASET_REGISTRY = {
    "t2i_pretrain": T2IIterableDataset,
    "vlm_sft": SftJSONLIterableDataset,
    "unified_edit": UnifiedEditIterableDataset,
}


# Kept for import compatibility only. Public training configs must provide
# dataset paths explicitly through ``dataset_info`` or ``dataset_info_path``.
DEFAULT_DATASET_INFO = {}


def make_bagel_multimodal_dataset(
    *,
    tokenizer,
    special_tokens,
    grouped_datasets,
    local_rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    expected_num_tokens: int = 32768,
    max_num_tokens_per_sample: int = 16384,
    max_num_tokens: int = 36864,
    prefer_buffer_before: int = 16384,
    max_buffer_size: int = 50,
    interpolate_pos: bool = False,
    use_flex: bool = False,
    data_status=None,
    data_seed: int = 42,
    text_cond_dropout_prob: float = 0.1,
    vit_cond_dropout_prob: float = 0.4,
    vae_cond_dropout_prob: float = 0.1,
    vae_image_downsample: int = 16,
    max_latent_size: int = 32,
    vit_patch_size: int = 14,
    max_num_patch_per_side: int = 70,
    dataset_info=None,
):
    """Build a BAGEL packed dataset for Stage 1 or Stage 2 training.

    ``grouped_datasets`` is the YAML dict produced by ``data/configs/*.yaml``.
    ``dataset_info`` is required and contains the concrete local paths for each
    dataset named by ``grouped_datasets``.

    Returns the constructed :class:`PackedDataset` instance. The caller is
    still responsible for invoking ``set_epoch`` immediately before
    DataLoader iteration (and for the pre-iter RNG reseed — see
    :meth:`PackedDataset.set_epoch` docstring).
    """
    # Lazy import to avoid a circular: packing imports DATASET_REGISTRY from
    # here, so we defer the packing import until factory call time.
    from .packing import DataConfig, PackedDataset

    data_config = DataConfig(
        grouped_datasets=grouped_datasets,
        text_cond_dropout_prob=text_cond_dropout_prob,
        vit_cond_dropout_prob=vit_cond_dropout_prob,
        vae_cond_dropout_prob=vae_cond_dropout_prob,
        vae_image_downsample=vae_image_downsample,
        max_latent_size=max_latent_size,
        vit_patch_size=vit_patch_size,
        max_num_patch_per_side=max_num_patch_per_side,
    )
    data_config.data_seed = data_seed

    return PackedDataset(
        data_config=data_config,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        local_rank=local_rank,
        world_size=world_size,
        num_workers=num_workers,
        expected_num_tokens=expected_num_tokens,
        max_num_tokens_per_sample=max_num_tokens_per_sample,
        max_num_tokens=max_num_tokens,
        prefer_buffer_before=prefer_buffer_before,
        max_buffer_size=max_buffer_size,
        interpolate_pos=interpolate_pos,
        use_flex=use_flex,
        data_status=data_status,
        dataset_info=dataset_info,
    )
