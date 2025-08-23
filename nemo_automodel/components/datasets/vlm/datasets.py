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

import json
import random

from datasets import load_dataset

from nemo_automodel.components.datasets.vlm.utils import json2token


def make_rdr_dataset(path_or_dataset="quintend/rdr-items", split="train", **kwargs):
    """Load and preprocess the RDR dataset for image-to-text fine-tuning.

    Args:
        path_or_dataset (str): Path or identifier for the RDR dataset.
        split (str): Dataset split to load.
        **kwargs: Additional arguments.

    Returns:
        Dataset: The processed dataset.
    """
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example):
        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": "Describe this image."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example["text"]}],
                },
            ],
        }

    return [format(example) for example in dataset]
    # return dataset.map(format, batched=False)


def make_cord_v2_dataset(
    path_or_dataset="naver-clova-ix/cord-v2",
    split="train",
    **kwargs,
):
    """Load and preprocess the CORD-V2 dataset for image-to-text fine-tuning."""
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example):
        ground_truth = json.loads(example["ground_truth"])
        if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
            assert isinstance(ground_truth["gt_parses"], list)
            gt_jsons = ground_truth["gt_parses"]
        else:
            assert "gt_parse" in ground_truth and isinstance(
                ground_truth["gt_parse"],
                dict,
            )
            gt_jsons = [ground_truth["gt_parse"]]

        text = random.choice(
            [json2token(gt_json, sort_json_key=True) for gt_json in gt_jsons],
        )

        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": "Describe this image."},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": text}]},
            ],
        }

    return [format(example) for example in dataset]
    # return dataset.map(format, batched=False, num_proc=8,remove_columns=["ground_truth"])


def make_medpix_dataset(path_or_dataset="medpix-dataset/medpix-dataset", split="train", **kwargs):
    """Load and preprocess the MedPix dataset for image-to-text fine-tuning."""
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example):
        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image_id"]},
                        {"type": "text", "text": example["question"]},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]},
            ],
        }

    return [format(example) for example in dataset]


def make_cv17_dataset(path_or_dataset="ysdede/commonvoice_17_tr_fixed", split="train", **kwargs):
    """Load and preprocess the CommonVoice 17 dataset for audio-to-text fine-tuning."""
    dataset = load_dataset(path_or_dataset, split=split)
    all_columns = dataset.column_names
    columns_to_remove = [col for col in all_columns if col not in ["audio", "transcription"]]
    dataset = dataset.remove_columns(columns_to_remove)

    def format(example):
        return {
            "conversation": [
                {"role": "user", "content": "<|audio_1|>Transcribe the Turkish audio clip."},
                {"role": "assistant", "content": example["transcription"]},
            ],
            "audio": (example["audio"]["array"], example["audio"]["sampling_rate"]),
        }

    ret = [format(example) for example in dataset]
    return ret
