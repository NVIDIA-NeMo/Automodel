# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Helpers for the zero-YAML LoRA CLI path."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _path_or_dataset_id(value: str) -> str:
    """Return an absolute path for local data, otherwise keep the dataset id."""
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    if path.suffix.lower() in {".json", ".jsonl", ".ndjson", ".parquet"}:
        raise FileNotFoundError(f"Training data file was not found: {value}")
    return value


def _default_global_batch_size(local_batch_size: int, nproc_per_node: int | None) -> int:
    """Choose a small effective batch size that is divisible by local workers."""
    if nproc_per_node is None or nproc_per_node <= 0:
        return 8
    per_step_batch = local_batch_size * nproc_per_node
    return max(per_step_batch, math.ceil(8 / per_step_batch) * per_step_batch)


def _iter_local_json_rows(path: Path, *, limit: int = 100):
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as fp:
            for idx, line in enumerate(fp):
                if idx >= limit:
                    return
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            yield from data[:limit]
        else:
            yield data


def _content_part_has_media(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    part_type = str(part.get("type", "")).lower()
    return part_type in {"image", "image_url", "input_image", "video", "video_url", "input_video"} or any(
        key in part for key in ("image", "image_url", "video", "video_url")
    )


def _content_has_media(content: Any) -> bool:
    if isinstance(content, list):
        return any(_content_part_has_media(part) for part in content)
    if isinstance(content, dict):
        return _content_part_has_media(content)
    if isinstance(content, str):
        return "<image>" in content.lower() or "<video>" in content.lower()
    return False


def _row_has_vlm_content(row: dict[str, Any]) -> bool:
    if row.get("images") or row.get("videos"):
        return True

    for key in ("messages", "conversation"):
        messages = row.get(key)
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and _content_has_media(message.get("content")):
                    return True
    return False


def detect_openai_chat_data_kind(data_path_or_dataset_id: str) -> str:
    """Return ``"vlm"`` when local OpenAI chat data contains image/video parts."""
    path = Path(data_path_or_dataset_id).expanduser()
    if not path.exists() or not path.is_file():
        return "llm"
    for row in _iter_local_json_rows(path):
        if isinstance(row, dict) and _row_has_vlm_content(row):
            return "vlm"
    return "llm"


def build_lora_chat_config(
    model_id: str,
    data_path_or_dataset_id: str,
    *,
    nproc_per_node: int | None = None,
) -> dict[str, Any]:
    """Build the default LoRA SFT config for ``automodel <model> <data>``.

    The generated config intentionally uses existing recipe and dataset
    components. It is just a YAML-free front door for the standard local
    fine-tuning path.
    """
    local_batch_size = 1
    global_batch_size = _default_global_batch_size(local_batch_size, nproc_per_node)
    data_source = _path_or_dataset_id(data_path_or_dataset_id)
    safe_model_name = model_id.strip("/").replace("/", "__") or "model"

    return {
        "recipe": "TrainFinetuneRecipeForNextTokenPrediction",
        "dist_env": {
            "backend": "nccl",
            "timeout_minutes": 30,
        },
        "rng": {
            "_target_": "nemo_automodel.components.training.rng.StatefulRNG",
            "seed": 1111,
            "ranked": True,
        },
        "model": {
            "_target_": "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained",
            "pretrained_model_name_or_path": model_id,
            "torch_dtype": "auto",
            "trust_remote_code": False,
            "attn_implementation": "sdpa",
        },
        "distributed": {
            "strategy": "fsdp2",
        },
        "step_scheduler": {
            "global_batch_size": global_batch_size,
            "local_batch_size": local_batch_size,
            "num_epochs": 1,
            "ckpt_every_steps": 100,
            "val_every_steps": None,
        },
        "optimizer": {
            "_target_": "torch.optim.Adam",
            "lr": 5.0e-6,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
        },
        "lr_scheduler": {
            "lr_decay_style": "cosine",
            "lr_warmup_steps": 0,
        },
        "checkpoint": {
            "enabled": True,
            "model_save_format": "safetensors",
            "checkpoint_dir": f"./checkpoints/{safe_model_name}-lora",
            "save_consolidated": True,
            "dequantize_base_checkpoint": True,
        },
        "dataset": {
            "_target_": "nemo_automodel.components.datasets.llm.chat_dataset.OpenAIChatDataset",
            "path_or_dataset_id": data_source,
            "split": "train",
            "seq_length": 2048,
            "padding": "do_not_pad",
            "truncation": "longest_first",
        },
        "dataloader": {
            "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
            "collate_fn": "nemo_automodel.components.datasets.utils.default_collater",
            "shuffle": True,
        },
        "peft": {
            "_target_": "nemo_automodel.components._peft.lora.PeftConfig",
            "dim": 16,
            "alpha": 32,
            "dropout": 0.0,
            "target_modules": "*_proj",
            "use_triton": True,
        },
        "loss_fn": {
            "_target_": "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy",
        },
    }


def build_lora_vlm_config(
    model_id: str,
    data_path_or_dataset_id: str,
    *,
    nproc_per_node: int | None = None,
) -> dict[str, Any]:
    """Build the default LoRA VLM SFT config for multimodal OpenAI chat JSONL."""
    local_batch_size = 1
    global_batch_size = _default_global_batch_size(local_batch_size, nproc_per_node)
    data_source = _path_or_dataset_id(data_path_or_dataset_id)
    safe_model_name = model_id.strip("/").replace("/", "__") or "model"

    return {
        "recipe": "FinetuneRecipeForVLM",
        "step_scheduler": {
            "global_batch_size": global_batch_size,
            "local_batch_size": local_batch_size,
            "num_epochs": 1,
            "ckpt_every_steps": 100,
            "val_every_steps": None,
        },
        "dist_env": {
            "backend": "nccl",
            "timeout_minutes": 30,
        },
        "rng": {
            "_target_": "nemo_automodel.components.training.rng.StatefulRNG",
            "seed": 1111,
            "ranked": True,
        },
        "model": {
            "_target_": "nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained",
            "pretrained_model_name_or_path": model_id,
            "torch_dtype": "auto",
            "trust_remote_code": False,
            "attn_implementation": "sdpa",
        },
        "processor": {
            "_target_": "transformers.AutoProcessor.from_pretrained",
            "pretrained_model_name_or_path": model_id,
            "trust_remote_code": False,
        },
        "checkpoint": {
            "enabled": True,
            "model_save_format": "safetensors",
            "checkpoint_dir": f"./checkpoints/{safe_model_name}-vlm-lora",
            "save_consolidated": True,
            "dequantize_base_checkpoint": True,
        },
        "distributed": {
            "strategy": "fsdp2",
        },
        "dataset": {
            "_target_": "nemo_automodel.components.datasets.vlm.datasets.make_openai_vlm_chat_dataset",
            "path_or_dataset": data_source,
            "split": "train",
        },
        "dataloader": {
            "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
            "num_workers": 0,
            "pin_memory": True,
        },
        "peft": {
            "_target_": "nemo_automodel.components._peft.lora.PeftConfig",
            "exclude_modules": ["*vision_tower*", "*vision*", "*visual*", "*image_encoder*", "*lm_head*"],
            "dim": 8,
            "alpha": 32,
            "dropout": 0.0,
            "use_triton": True,
        },
        "loss_fn": {
            "_target_": "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy",
        },
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 1.0e-5,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
        },
        "freeze_config": {
            "freeze_embeddings": True,
            "freeze_vision_tower": True,
            "freeze_language_model": False,
        },
    }


def build_lora_config(
    model_id: str,
    data_path_or_dataset_id: str,
    *,
    nproc_per_node: int | None = None,
    data_kind: str = "auto",
) -> dict[str, Any]:
    """Build the default LoRA config, choosing LLM or VLM from the data shape."""
    if data_kind == "auto":
        data_kind = detect_openai_chat_data_kind(data_path_or_dataset_id)
    if data_kind == "vlm":
        return build_lora_vlm_config(model_id, data_path_or_dataset_id, nproc_per_node=nproc_per_node)
    if data_kind == "llm":
        return build_lora_chat_config(model_id, data_path_or_dataset_id, nproc_per_node=nproc_per_node)
    raise ValueError(f"Unsupported data kind for quick LoRA config: {data_kind}")
