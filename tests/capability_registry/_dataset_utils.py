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

"""Build a small list of real-recipe batches for capability validation.

The default dataset for LLMs is HellaSwag (matches the Llama-3.1-8B recipe at
``examples/llm_finetune/llama3_1/llama3_1_8b_hellaswag_pp.yaml``). A per-model
override map is provided for future models whose recipes use a different dataset.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from nemo_automodel.components.datasets.llm.hellaswag import HellaSwag
from nemo_automodel.components.datasets.utils import default_collater


# Per-model dataset overrides. Empty by default; extend as new models onboard.
# Each entry maps an HF model id prefix -> a builder callable taking
# (tokenizer, num_samples) and returning an iterable dataset.
_MODEL_DATASET_OVERRIDES: dict[str, Any] = {}


def build_training_batches(
    *,
    model_id: str,
    num_steps: int,
    local_batch_size: int = 1,
    pad_seq_len_divisible: int = 16,
    split: str = "train",
    trust_remote_code: bool = False,
) -> list[dict[str, torch.Tensor]]:
    """Build ``num_steps`` real batches for the validation procedure.

    Defaults to HellaSwag (``rowan/hellaswag``), which is what the Llama-3.1-8B
    PP recipe uses and which works for any causal LM with a tokenizer. The same
    list of batches is consumed by both the reference and variant runs.

    Args:
        model_id: HF model id (drives the tokenizer choice).
        num_steps: Number of batches (= number of training steps including the
            final capture step).
        local_batch_size: Per-rank batch size. Default 1 keeps memory tight.
        pad_seq_len_divisible: Pad sequence lengths to a multiple of this
            value. The default of 16 covers ``cp_size <= 8`` (CP requires
            ``seq_len % (2 * cp_size) == 0``).
        split: Dataset split.
        trust_remote_code: Forwarded to ``AutoTokenizer``.

    Returns:
        A list of dicts with keys ``input_ids``, ``labels``, ``attention_mask``,
        ... — one dict per batch, all on CPU; move to device per-rank.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    builder = _MODEL_DATASET_OVERRIDES.get(model_id)
    if builder is not None:
        dataset = builder(tokenizer=tokenizer, num_samples=num_steps * local_batch_size)
    else:
        dataset = HellaSwag(
            path_or_dataset="rowan/hellaswag",
            tokenizer=tokenizer,
            split=split,
            num_samples_limit=num_steps * local_batch_size,
            pad_to_max_length=False,
        )

    # Use a non-shuffling loader so reference and variant runs see batches in
    # the same order without coordinating an RNG.
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        collate_fn=lambda b: default_collater(b, pad_seq_len_divisible=pad_seq_len_divisible),
        drop_last=False,
    )

    batches: list[dict[str, torch.Tensor]] = []
    for i, batch in enumerate(loader):
        if i >= num_steps:
            break
        batches.append(batch)
    if len(batches) < num_steps:
        raise RuntimeError(
            f"Dataset produced only {len(batches)} batches but {num_steps} were requested. "
            f"Increase num_samples_limit or use a different dataset."
        )
    return batches
