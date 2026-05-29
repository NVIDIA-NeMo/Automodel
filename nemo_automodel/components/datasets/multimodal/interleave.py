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
#   data/interleave_datasets/edit_dataset.py
#   data/interleave_datasets/interleave_t2i_dataset.py
# Upstream copyright: Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Class names, per-sample RNG ordering, and yielded-dict keys match the packing
# layer and ``data_status`` resume contract.

"""Interleaved-image parquet datasets for BAGEL editing + joint recipes.

Provides:

* :class:`InterleavedBaseIterableDataset` – mixin that exposes
  ``_init_data`` / ``_add_text`` / ``_add_image`` / ``_add_video``
  builders for per-row assembly of the packed-sequence plan.
* :class:`ParquetStandardIterableDataset` – base class that iterates
  per-row-group over a list of parquet files; subclasses override
  ``parse_row`` to turn a pandas row into a ``dict`` compatible with
  :class:`.packing.PackedDataset`.
* :class:`UnifiedEditIterableDataset` – concrete parse_row that emits
  interleaved (input-image, instruction, output-image) samples from an
  image-editing parquet schema (``image_list`` + ``instruction_list``).

When ``visual_gen=False`` these samples can still flow through packing while
the model ignores VAE / flow-matching tensors. Stage 2 consumes the same
yielded sample dicts for edit-generation loss.
"""

from __future__ import annotations

import io
import logging
import random

from PIL import Image, ImageFile, PngImagePlugin

from .distributed_iterable import DistributedIterableDataset
from .parquet_utils import get_parquet_data_paths, init_arrow_pf_fs
from .utils import pil_img2rgb

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
_MaximumDecompressedSize = 1024
_MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = _MaximumDecompressedSize * _MegaByte


class InterleavedBaseIterableDataset(DistributedIterableDataset):
    """Builder mixin for interleaved image/text/video sequence plans.

    Subclasses still provide ``__init__`` + ``__iter__`` + ``parse_row``
    (via :class:`ParquetStandardIterableDataset`); this class only holds
    the per-item append helpers used inside ``parse_row``.
    """

    def _init_data(self):
        data = {
            "sequence_plan": [],
            "text_ids_list": [],
            "image_tensor_list": [],
            "num_tokens": 0,
        }
        return data

    def _add_text(self, data, text, need_loss, enable_cfg=True):
        text_ids = self.tokenizer.encode(text)
        data["num_tokens"] += len(text_ids)
        data["text_ids_list"].append(text_ids)
        data["sequence_plan"].append(
            {
                "type": "text",
                "enable_cfg": int(enable_cfg),
                "loss": int(need_loss),
                "special_token_loss": 0,
                "special_token_label": None,
            }
        )
        return data

    def _add_image(self, data, image, need_loss, need_vae, need_vit, enable_cfg=True):
        assert need_loss or need_vae or need_vit

        if need_loss:
            data["sequence_plan"].append(
                {
                    "type": "vae_image",
                    "enable_cfg": 0,
                    "loss": 1,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data["num_tokens"] += width * height // self.transform.stride**2
            data["image_tensor_list"].append(image_tensor)

        if need_vae:
            data["sequence_plan"].append(
                {
                    "type": "vae_image",
                    "enable_cfg": int(enable_cfg),
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data["num_tokens"] += width * height // self.transform.stride**2
            data["image_tensor_list"].append(image_tensor.clone())

        if need_vit:
            data["sequence_plan"].append(
                {
                    "type": "vit_image",
                    "enable_cfg": int(enable_cfg),
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                },
            )
            vit_image_tensor = self.vit_transform(image)
            height, width = vit_image_tensor.shape[1:]
            data["num_tokens"] += width * height // self.vit_transform.stride**2
            data["image_tensor_list"].append(vit_image_tensor)

        return data

    def _add_video(self, data, frames, frame_indexes, need_loss, need_vae, enable_cfg=True):
        assert int(need_loss) + int(need_vae) == 1

        if need_loss:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
                current_sequence_plan = {
                    "type": "vae_image",
                    "enable_cfg": 0,
                    "loss": 1,
                    "special_token_loss": 0,
                    "special_token_label": None,
                    "split_start": idx == 0,
                    "split_end": idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan["frame_delta"] = frame_indexes[idx + 1] - frame_idx
                data["sequence_plan"].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data["image_tensor_list"].append(image_tensor)
                data["num_tokens"] += width * height // self.transform.stride**2

        elif need_vae:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
                current_sequence_plan = {
                    "type": "vae_image",
                    "enable_cfg": int(enable_cfg),
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                    "split_start": idx == 0,
                    "split_end": idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan["frame_delta"] = frame_indexes[idx + 1] - frame_idx
                data["sequence_plan"].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data["image_tensor_list"].append(image_tensor)
                data["num_tokens"] += width * height // self.transform.stride**2

        return data


class ParquetStandardIterableDataset(DistributedIterableDataset):
    """Base class: iterate per-(file, row_group) across a list of parquet shards.

    Subclasses override :meth:`parse_row` to turn one pandas row into the
    dict schema consumed by :class:`.packing.PackedDataset`.
    """

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        vit_transform,
        data_dir_list,
        num_used_data,
        parquet_info,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
    ):
        """
        Args:
            data_dir_list: list of data directories containing parquet files
            num_used_data: list of row-count caps per directory
            vit_transform: input transform for the ViT tower
            parquet_info: dict mapping ``parquet_path -> {"num_row_groups": int}``
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.set_data_status(data_status)
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data, parquet_info)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data, parquet_info):
        row_groups = []
        for data_dir, num_data_path in zip(data_dir_list, num_used_data):
            data_paths = get_parquet_data_paths([data_dir], [num_data_path])
            for data_path in data_paths:
                if data_path in parquet_info.keys():
                    num_row_groups = parquet_info[data_path]["num_row_groups"]
                    for rg_idx in range(num_row_groups):
                        row_groups.append((data_path, rg_idx))
        return row_groups

    def parse_row(self, row):
        raise NotImplementedError

    def __iter__(self):
        # Lazy import: pyarrow is optional unless you actually iterate a
        # parquet-backed dataset.
        import pyarrow.parquet as pq

        file_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        worker_data_status = self._get_worker_data_status(worker_id)
        if worker_data_status is not None:
            global_row_group_start_id = worker_data_status[0]
            row_start_id = worker_data_status[1] + 1
        else:
            global_row_group_start_id = 0
            row_start_id = 0

        logger.info(
            "rank-%s worker-%s dataset-%s: resuming data at global_rg#%s, row#%s",
            self.local_rank,
            worker_id,
            self.dataset_name,
            global_row_group_start_id,
            row_start_id,
        )

        while True:
            file_paths_per_worker_ = file_paths_per_worker[global_row_group_start_id:]
            for global_row_group_idx, (parquet_file_path, row_group_id) in enumerate(
                file_paths_per_worker_, start=global_row_group_start_id
            ):
                fs = init_arrow_pf_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    try:
                        fr = pq.ParquetFile(f)
                        df = fr.read_row_group(row_group_id).to_pandas()
                        df = df.iloc[row_start_id:]
                    except Exception as e:
                        logger.warning("Error %s in rg#%s, %s", e, row_group_id, parquet_file_path)
                        continue

                    for row_idx, row in df.iterrows():
                        try:
                            data = self.parse_row(row)
                            if len(data) == 0:
                                continue
                            data["data_indexes"] = {
                                "data_indexes": [global_row_group_idx, row_idx],
                                "worker_id": worker_id,
                                "dataset_name": self.dataset_name,
                            }
                        except Exception as e:
                            logger.warning("Error %s in rg#%s, %s", e, row_group_id, parquet_file_path)
                            continue
                        self._set_worker_resume_data_status(worker_id, [global_row_group_idx, row_idx])
                        yield data

                    row_start_id = 0
            global_row_group_start_id = 0
            logger.info("%s repeat in rank-%s worker-%s", self.dataset_name, self.local_rank, worker_id)


class UnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    """Image-editing dataset: ``(input, instruction, output)`` chains over parquet.

    Row schema (upstream BAGEL ``seedxedit_multi`` + compatibles):
      ``image_list``: list of raw image bytes (at least 2).
      ``instruction_list``: list of lists; ``instruction_list[i]`` is a
        set of equivalent phrasings for the edit that turns
        ``image_list[i]`` into ``image_list[i+1]``.
    """

    def parse_row(self, row):
        image_num = len(row["image_list"])
        # Randomly choose start and end frames, up to a 2-step chain.
        start_idx = random.choice(range(image_num - 1))
        max_end = min(start_idx + 3, image_num)
        end_idx = random.choice(range(start_idx + 1, max_end))

        data = self._init_data()
        data = self._add_image(
            data,
            pil_img2rgb(Image.open(io.BytesIO(row["image_list"][start_idx]))),
            need_loss=False,
            need_vae=True,
            need_vit=True,
        )

        # With p=0.5, concatenate multiple instructions into one edit pass.
        if end_idx - start_idx > 1 and random.random() < 0.5:
            if end_idx == image_num - 1:
                end_idx -= 1

            instruction = ""
            for idx in range(start_idx + 1, end_idx + 1):
                instruction += random.choice(row["instruction_list"][idx - 1]) + ". "
            data = self._add_text(data, instruction.rstrip(), need_loss=False)
            data = self._add_image(
                data,
                pil_img2rgb(Image.open(io.BytesIO(row["image_list"][end_idx]))),
                need_loss=True,
                need_vae=False,
                need_vit=False,
            )
        else:
            for idx in range(start_idx + 1, end_idx + 1):
                instruction = random.choice(row["instruction_list"][idx - 1])
                data = self._add_text(data, instruction, need_loss=False)
                if idx != end_idx:
                    data = self._add_image(
                        data,
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True,
                        need_vae=True,
                        need_vit=True,
                    )
                else:
                    data = self._add_image(
                        data,
                        pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True,
                        need_vae=False,
                        need_vit=False,
                    )
        return data
