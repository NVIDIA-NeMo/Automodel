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

"""Generate a local tokenizer and finite JSONL corpus for the Zephon smoke recipe."""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


def main() -> None:
    """Write the deterministic tiny fixture without downloading model assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("work/zephon-tiny"))
    args = parser.parse_args()
    corpus = args.output / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, **{f"w{i}": i + 2 for i in range(62)}}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]"
    ).save_pretrained(args.output / "tokenizer")
    instructions = args.output / "instructions"
    instructions.mkdir(parents=True, exist_ok=True)
    with (instructions / "train.jsonl").open("w") as stream:
        for i in range(256):
            stream.write(
                json.dumps(
                    {"question": f"w{i % 62} w{(i + 1) % 62}", "answer": f"w{(i + 2) % 62} w{(i + 3) % 62} [EOS]"}
                )
                + "\n"
            )
    with (corpus / "train.jsonl").open("w") as stream:
        for i in range(256):
            stream.write(json.dumps({"text": " ".join(f"w{(i + j) % 62}" for j in range(31))}) + "\n")


if __name__ == "__main__":
    main()
