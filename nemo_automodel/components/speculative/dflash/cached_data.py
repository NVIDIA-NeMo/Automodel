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

"""Read the unshifted tensor rows produced by SpecForge's DFlash visual scoring capture."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import NotRequired, TypedDict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

VISUAL_SCORE_SCALE = 10_000
_TEXT_ONLY_SCORE = -1
_FORMAT_VERSION = 1
_HIDDEN_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class CachedDFlashRow(TypedDict):
    """One unpadded record with token ids, assistant mask and optional scores of shape
    [sequence], and target hidden states of shape [sequence, layers * hidden].
    """

    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    hidden_states: torch.Tensor
    visual_score: NotRequired[torch.Tensor]


class CachedDFlashBatch(CachedDFlashRow):
    """A right-padded batch of cache rows, with each row treated as one document."""

    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    seq_lens: torch.Tensor
    doc_remaining: torch.Tensor


class CachedDFlashDataset(Dataset[CachedDFlashRow]):
    """Read one SpecForge ``.ckpt`` per item, in lexicographic filename order.

    A ``manifest.json`` identifies the capture and its settings. Setup reads the
    manifest and enumerates the files. Tensors are read and validated on demand.
    The trainer determines full-block anchor eligibility after batching.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        target_model_name_or_path: str,
        target_layer_ids: Sequence[int],
        target_hidden_size: int,
        target_vocab_size: int,
        block_size: int,
        mask_token_id: int,
        max_seq_len: int,
        require_visual_score: bool = False,
    ) -> None:
        """Validate the cache manifest against the model and recipe settings.

        Args:
            cache_dir: Directory containing the manifest and checkpoint rows.
            target_model_name_or_path: Exact teacher model ID or path recorded at capture.
            target_layer_ids: Decoder layer indices in hidden-state concatenation order.
            target_hidden_size: Width of one teacher layer.
            target_vocab_size: Size of the teacher's output vocabulary.
            block_size: DFlash block size, including the unmasked anchor token.
            mask_token_id: Token used for the masked draft positions.
            max_seq_len: Maximum allowed length of an unpadded cached row.
            require_visual_score: Require a score channel for MMFlash.
        """
        if block_size < 2 or max_seq_len < block_size:
            raise ValueError("Cached DFlash requires 2 <= block_size <= max_seq_len")
        if target_hidden_size < 1 or not 0 <= mask_token_id < target_vocab_size:
            raise ValueError("Cached DFlash needs a positive target hidden size and an in-vocabulary mask_token_id")
        layer_ids = list(target_layer_ids)
        if not layer_ids or any(type(layer) is not int or layer < 0 for layer in layer_ids):
            raise ValueError("Cached DFlash requires nonnegative integer target_layer_ids")
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("Cached DFlash requires distinct target_layer_ids")

        cache_dir = Path(cache_dir).resolve()
        manifest_path = cache_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing DFlash cache manifest at {manifest_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{manifest_path}: invalid manifest JSON") from exc
        if not isinstance(manifest, dict):
            raise ValueError(f"{manifest_path}: expected a JSON object")

        expected = {
            "format_version": _FORMAT_VERSION,
            "target_model_name_or_path": target_model_name_or_path,
            "target_layer_ids": layer_ids,
            "target_hidden_size": target_hidden_size,
            "target_vocab_size": target_vocab_size,
            "block_size": block_size,
            "mask_token_id": mask_token_id,
        }
        for name, value in expected.items():
            recorded = manifest.get(name)
            if type(recorded) is not type(value) or recorded != value:
                raise ValueError(f"{manifest_path}: {name} is {recorded!r}, expected {value!r}")
        if any(type(layer) is not int for layer in manifest["target_layer_ids"]):
            raise ValueError(f"{manifest_path}: target_layer_ids must contain integers")

        hidden_dtype = manifest.get("hidden_states_dtype")
        if not isinstance(hidden_dtype, str) or hidden_dtype not in _HIDDEN_DTYPES:
            raise ValueError(f"{manifest_path}: hidden_states_dtype must be one of {', '.join(_HIDDEN_DTYPES)}")
        visual_scale = manifest.get("visual_score_scale")
        if "visual_score_scale" not in manifest or (
            visual_scale is not None and (type(visual_scale) is not int or visual_scale != VISUAL_SCORE_SCALE)
        ):
            raise ValueError(f"{manifest_path}: visual_score_scale must be {VISUAL_SCORE_SCALE} or null")
        if require_visual_score and visual_scale is None:
            raise ValueError(f"{manifest_path}: MMFlash requires visual_score_scale={VISUAL_SCORE_SCALE}")

        num_samples = manifest.get("num_samples")
        if type(num_samples) is not int or num_samples < 1:
            raise ValueError(f"{manifest_path}: num_samples must be a positive integer")
        self._paths = sorted(path for path in cache_dir.glob("*.ckpt") if path.is_file())
        if len(self._paths) != num_samples:
            raise ValueError(f"{manifest_path}: num_samples is {num_samples}, found {len(self._paths)} .ckpt files")

        self._fields = {"input_ids", "loss_mask", "hidden_states"}
        if visual_scale is not None:
            self._fields.add("visual_score")
        self._hidden_dtype = _HIDDEN_DTYPES[hidden_dtype]
        self._feature_width = len(layer_ids) * target_hidden_size
        self._target_vocab_size = target_vocab_size
        self._max_seq_len = max_seq_len

    def __len__(self) -> int:
        """Return the number of cached rows."""
        return len(self._paths)

    def __getitem__(self, index: int) -> CachedDFlashRow:
        """Load an unshifted record on CPU.

        Args:
            index: Index into the sorted checkpoint files.

        Returns:
            Int64 ``input_ids`` and binary ``loss_mask`` of shape [sequence].
            ``hidden_states`` has shape [sequence, layers * hidden] and the
            dtype recorded in the manifest. Declared ``visual_score`` is an
            int64 tensor of shape [sequence] aligned with the same positions.
        """
        path = self._paths[index]
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a dictionary of cache tensors")
        missing = self._fields.difference(raw)
        unexpected = raw.keys() - self._fields
        if missing or unexpected:
            raise ValueError(
                f"{path}: invalid cache row fields (missing {sorted(missing)}, unexpected {list(unexpected)})"
            )
        for field in self._fields:
            tensor = raw[field]
            if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or tensor.device.type != "cpu":
                raise ValueError(f"{path}: {field} must be a dense CPU tensor")

        input_ids, loss_mask, hidden_states = raw["input_ids"], raw["loss_mask"], raw["hidden_states"]
        if input_ids.ndim != 1 or input_ids.dtype != torch.long:
            raise ValueError(f"{path}: input_ids must be int64 with shape [sequence]")
        length = input_ids.size(0)
        if not 1 <= length <= self._max_seq_len:
            raise ValueError(f"{path}: sequence length {length} is outside [1, {self._max_seq_len}]")
        # input_ids also supplies the unshifted hard labels.
        if (input_ids < 0).any() or (input_ids >= self._target_vocab_size).any():
            raise ValueError(f"{path}: input_ids contains a token outside [0, {self._target_vocab_size})")
        if (
            loss_mask.shape != input_ids.shape
            or loss_mask.dtype not in (torch.bool, torch.long)
            or ((loss_mask != 0) & (loss_mask != 1)).any()
        ):
            raise ValueError(f"{path}: loss_mask must be binary, bool or int64, and match input_ids")
        if hidden_states.shape != (length, self._feature_width) or hidden_states.dtype != self._hidden_dtype:
            raise ValueError(
                f"{path}: hidden_states must have shape ({length}, {self._feature_width}) "
                f"and dtype {self._hidden_dtype}, got {tuple(hidden_states.shape)} and {hidden_states.dtype}"
            )

        row: CachedDFlashRow = {
            "input_ids": input_ids,
            "loss_mask": loss_mask.long(),
            "hidden_states": hidden_states.detach(),
        }
        if "visual_score" in raw:
            score = raw["visual_score"]
            if score.shape != input_ids.shape or score.dtype != torch.long:
                raise ValueError(f"{path}: visual_score must be int64 and match input_ids")
            if (score < _TEXT_ONLY_SCORE).any() or (score > VISUAL_SCORE_SCALE).any():
                raise ValueError(
                    f"{path}: visual_score values must be {_TEXT_ONLY_SCORE} or in [0, {VISUAL_SCORE_SCALE}]"
                )
            text_only = score == _TEXT_ONLY_SCORE
            if text_only.any():
                if not text_only.all():
                    raise ValueError(f"{path}: visual_score must use -1 on every token of a text-only row")
            elif (score[loss_mask == 0] != 0).any():
                raise ValueError(f"{path}: visual_score must be zero outside assistant supervision on an image row")
            row["visual_score"] = score
        return row


def collate_cached_dflash(examples: list[CachedDFlashRow], *, pad_token_id: int) -> CachedDFlashBatch:
    """Batch right-padded rows, keeping their original sequence boundaries.

    Args:
        examples: Cache records with ``input_ids``, ``loss_mask``, and optional
            ``visual_score`` of shape [sequence] and ``hidden_states`` of shape
            [sequence, layers * hidden]. Sequences can have different lengths.
        pad_token_id: Valid vocabulary ID used for padding input ids.

    Returns:
        ``input_ids``, ``loss_mask``, ``attention_mask``, ``position_ids``,
        ``doc_remaining``, and optional ``visual_score`` have shape
        [batch, sequence]. ``hidden_states`` has shape
        [batch, sequence, layers * hidden]. ``attention_mask`` marks real tokens
        and ``doc_remaining`` counts subsequent real tokens; padding has zero
        loss mask and hidden states, and score -1. ``seq_lens`` has shape
        [batch, 1] and records each row's unpadded length.
    """
    if not examples:
        raise ValueError("Cannot collate an empty DFlash cache batch")
    has_scores = ["visual_score" in row for row in examples]
    if any(has_scores) and not all(has_scores):
        raise ValueError("All DFlash cache rows in a batch must share the same visual_score layout")

    input_ids = pad_sequence([row["input_ids"] for row in examples], batch_first=True, padding_value=pad_token_id)
    # Materialize the batch dimension so DataLoader can pin the positions.
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids).clone()
    seq_lens = torch.tensor([row["input_ids"].numel() for row in examples], device=input_ids.device).unsqueeze(1)
    # The trainer's document-boundary contract lets every row keep its own
    # full-block anchors even when its neighbors need a wider padded tensor.
    doc_remaining = seq_lens - positions - 1
    batch: CachedDFlashBatch = {
        "input_ids": input_ids,
        "loss_mask": pad_sequence([row["loss_mask"] for row in examples], batch_first=True),
        "attention_mask": doc_remaining >= 0,
        "hidden_states": pad_sequence([row["hidden_states"] for row in examples], batch_first=True),
        "position_ids": positions,
        "seq_lens": seq_lens,
        "doc_remaining": doc_remaining.clamp_min_(0),
    }
    if all(has_scores):
        batch["visual_score"] = pad_sequence(
            [row["visual_score"] for row in examples], batch_first=True, padding_value=_TEXT_ONLY_SCORE
        )
    return batch


@dataclass(frozen=True, kw_only=True)
class CachedDFlashDataConfig:
    """Settings for a cache-backed DFlash dataloader."""

    cache_dir: str | Path
    max_seq_len: int
    batch_size: int
    require_visual_score: bool = False
    num_workers: int = 0
    shuffle: bool = True
    shuffle_seed: int = 42

    def build(
        self,
        *,
        target_model_name_or_path: str,
        target_layer_ids: Sequence[int],
        target_hidden_size: int,
        target_vocab_size: int,
        block_size: int,
        mask_token_id: int,
        pad_token_id: int,
        distributed: bool = False,
    ) -> DataLoader[CachedDFlashRow]:
        """Build a cache reader and its CPU dataloader.

        Args:
            target_model_name_or_path: Exact teacher model ID or path recorded at capture.
            target_layer_ids: Decoder layer indices in hidden-state concatenation order.
            target_hidden_size: Width of one teacher layer.
            target_vocab_size: Size of the teacher's output vocabulary.
            block_size: DFlash block size, including the unmasked anchor token.
            mask_token_id: Token used for the masked draft positions.
            pad_token_id: In-vocabulary token used to pad uneven batches.
            distributed: Shard samples across the default process group.

        Returns:
            Dataloader yielding right-padded CPU batches as defined by
            ``collate_cached_dflash``.
        """
        if not 0 <= pad_token_id < target_vocab_size:
            raise ValueError(f"Cached DFlash pad_token_id must be in [0, {target_vocab_size})")
        dataset = CachedDFlashDataset(
            self.cache_dir,
            target_model_name_or_path=target_model_name_or_path,
            target_layer_ids=target_layer_ids,
            target_hidden_size=target_hidden_size,
            target_vocab_size=target_vocab_size,
            block_size=block_size,
            mask_token_id=mask_token_id,
            max_seq_len=self.max_seq_len,
            require_visual_score=self.require_visual_score,
        )
        sampler = None
        if distributed or self.shuffle:
            sampler = DistributedSampler(
                dataset,
                num_replicas=None if distributed else 1,
                rank=None if distributed else 0,
                shuffle=self.shuffle,
                seed=self.shuffle_seed,
            )
        use_cuda = torch.cuda.is_available()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=partial(collate_cached_dflash, pad_token_id=pad_token_id),
            sampler=sampler,
            generator=torch.Generator().manual_seed(self.shuffle_seed),
            num_workers=self.num_workers,
            pin_memory=use_cuda,
            persistent_workers=self.num_workers > 0,
            multiprocessing_context="forkserver" if self.num_workers > 0 and use_cuda else None,
        )
