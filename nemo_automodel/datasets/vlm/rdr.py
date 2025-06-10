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

import torch
from nemo_automodel.datasets.llm.hf_dataset import HFDatasetBuilder
from nemo_automodel.datasets.vlm.utils import extract_skipped_token_ids


def make_rdr_dataset(
    path_or_dataset="quintend/rdr-items", processor=None, split="train", **kwargs
):
    """
    Load and preprocess the RDR dataset for image-to-text fine-tuning.

    Args:
        path_or_dataset (str): Path or identifier for the RDR dataset.
        processor: The model processor (tokenizer + image processor).
        split (str): Dataset split to load.
        limit_dataset_samples (int, optional): Limit number of samples.
        instruction (str): Instruction text for prompting.
        **kwargs: Additional arguments.

    Returns:
        HFDatasetBuilder: Dataset builder compatible with VLM finetune.py.
    """

    def fmt(sample):
        instruction = "Describe accurately the given image."
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["text"]}],
            },
        ]
        return {"conversation": conversation, "images": [sample["image"]]}

    collate_fn = None
    if processor is not None:
        skipped_tokens = extract_skipped_token_ids(processor)

        def collate_fn(examples):
            text = []
            images = []

            for example in map(fmt, examples):
                text.append(
                    processor.apply_chat_template(
                        example["conversation"],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
                images += example["images"]

            batch = processor(
                text=text, images=images, padding=True, return_tensors="pt"
            )
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
            labels = batch["input_ids"].clone()[:, 1:]
            labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
            labels[torch.isin(labels, skipped_tokens)] = -100
            batch["labels"] = labels

            return batch

    return HFDatasetBuilder(
        path_or_dataset=path_or_dataset, split=split, collate_fn=collate_fn, **kwargs
    )
