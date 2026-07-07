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

"""Offline fixtures for the separate-mesh KD functional test."""

from __future__ import annotations

import pathlib

from torch.utils.data import Dataset


class TinyKDDataset(Dataset):
    """Small deterministic next-token dataset."""

    def __init__(self, num_samples: int = 16, seq_length: int = 8):
        self.samples = []
        for sample_index in range(num_samples):
            input_ids = [3 + (sample_index + position) % 24 for position in range(seq_length)]
            labels = input_ids[1:] + [2]
            labels[0] = -100
            self.samples.append({"input_ids": input_ids, "labels": labels})

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


def make_tiny_kd_dataset(tokenizer=None, num_samples: int = 16, seq_length: int = 8) -> TinyKDDataset:
    """Build the deterministic test dataset; ``tokenizer`` verifies recipe injection."""
    del tokenizer
    return TinyKDDataset(num_samples=num_samples, seq_length=seq_length)


def create_tiny_llama_assets(output_dir: str | pathlib.Path) -> None:
    """Create a tiny local Llama checkpoint and compatible tokenizer."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2}
    vocab.update({f"token_{index}": index for index in range(3, 32)})
    tokenizer_impl = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer_impl.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(output_dir)

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            bos_token_id=2,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    model.save_pretrained(output_dir)
