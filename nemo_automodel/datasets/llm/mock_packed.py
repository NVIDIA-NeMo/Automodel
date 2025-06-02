
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

from datasets import Dataset, Features, Sequence, Value
import random

def make_vocab(vocab_size:int=100):
    """
    Build a trivial vocab; index 0=<pad>, 1=<eos>, rest = word_i
    """
    vocab = {"<pad>": 0, "<eos>": 1}
    for i in range(2, vocab_size):
        vocab[f"tok_{i}"] = i
    return vocab

def gen_sentence_ids(vocab, mean_len:float, std_len:float, max_len:int):
    """ Sentence generator with Gaussian length control """

    words = list(vocab.values())[2:]     # exclude <pad>, <eos>
    # truncated Gaussian
    L = max(1, min(max_len, int(random.gauss(mean_len, std_len))))
    return random.choices(words, k=L) + [vocab["<eos>"]]

def flush_block(block, block_size):
    """ Flush helper (build position_ids that reset after <eos>) """
    pos, pos_ids = 0, []
    for tid in block:
        pos_ids.append(pos)
        pos = 0 if tid == 1 else pos + 1          # 1 == <eos>
    return {
        "input_ids": block,
        "attention_mask": [1]*block_size,
        "labels": block.copy(),
        "position_ids": pos_ids,
    }

def build_packed_dataset(
        *,
        num_blocks:int          = 10,
        block_size:int          = 128,
        mean_len:float          = 20.0,
        std_len:float           = 6.0,
        vocab_size:int          = 100,
        max_sentence_len:int    = 64,
        seed:int                = 0,
):
    """ dataset builder """
    random.seed(seed)
    vocab = make_vocab(vocab_size)
    current, examples = [], []

    while len(examples) < num_blocks:
        sent = gen_sentence_ids(vocab, mean_len, std_len, max_sentence_len)

        # overflow? -> save current block if full
        if len(current) + len(sent) > block_size:
            if len(current) == block_size:
                examples.append(flush_block(current, block_size))
            current = []

        current.extend(sent)

    # Optional: emit last block if exactly full
    if len(current) == block_size and len(examples) < num_blocks:
        examples.append(flush_block(current, block_size))

    features = Features({
        "input_ids":     Sequence(Value("int64")),
        "attention_mask":Sequence(Value("int8")),
        "labels":        Sequence(Value("int64")),
        "position_ids":  Sequence(Value("int64")),
    })
    return Dataset.from_list(examples[:num_blocks], features=features), vocab

if __name__ == "__main__":
    ds, vocab = build_packed_dataset(
        num_blocks=3,
        block_size=32,
        mean_len=10,
        std_len=3,
        vocab_size=50,
    )
    print(ds)
    print("Row-0 lengths:", {k: len(v) for k,v in ds[0].items()})
    print("Row-0 position_ids:", ds[0]["position_ids"])