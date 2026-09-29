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

"""Prepare the Tulu3 Arrow data (DS41_DATA_ROOT) for the DeepSeek-V4.1-Flash Tulu3 recipes.

Two layouts are produced from allenai/tulu-3-sft-mixture with tulu3_chat_template.jinja:
  - tulu3_4k_padded/{train,validation}: ChatDataset (chat mode), seq_length=4096,
    padding=max_length; input for deepseek_v41_flash_tulu3_cp*.yaml.
  - tulu3_32k_packed/{train,validation}: ChatDataset with seq_length=32768, no padding,
    then pack_dataset(packed_sequence_size=32768, cp_size=1, pad_to_multiple_of=2);
    input for deepseek_v41_flash_tulu3_packed_cp8_32k*.yaml (prepacked THD).
Both carry globally shifted, assistant-only labels ({% generation %} blocks of the
template), positional attention masks, and EOS == pad (DeepSeek's pad token is
<|end_of_sentence|>), as the recipe headers require.

Example:
  python examples/llm_finetune/deepseek_v41/prepare_tulu3_data.py \
      --model-path "$DS41_CHECKPOINT" --out-root /data/ds41_tulu3 --num-proc 32
  export DS41_DATA_ROOT=/data/ds41_tulu3/tulu3_4k_padded   # or tulu3_32k_packed
"""

import argparse
import os
import time

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset
from nemo_automodel.components.datasets.llm.packed_sequence import pack_dataset, tokenize_dataset_parallel

DATASET_ID = "allenai/tulu-3-sft-mixture"
KEEP_COLUMNS = ("input_ids", "labels", "attention_mask", "loss_mask", "position_ids", "seq_lens", "seq_lens_padded")


def build_chat_dataset(
    tokenizer: PreTrainedTokenizerBase,
    template: str,
    split: str,
    seq_length: int,
    padding: str | bool,
    shuffle_seed: int,
) -> ChatDataset:
    """Build a Tulu3 split with assistant-only labels and the requested padding.

    Args:
        tokenizer: DeepSeek tokenizer used to encode each conversation.
        template: Jinja chat template containing generation blocks.
        split: Hugging Face split or slice expression.
        seq_length: Maximum token count per conversation.
        padding: Tokenizer padding mode.
        shuffle_seed: Seed applied before slicing the dataset.

    Returns:
        Lazily tokenized chat dataset.
    """
    return ChatDataset(
        DATASET_ID,
        tokenizer,
        split=split,
        shuffle_seed=shuffle_seed,
        seq_length=seq_length,
        padding=padding,
        truncation=True,
        chat_template=template,
    )


def materialize(chat_ds: ChatDataset, num_proc: int) -> Dataset:
    """Tokenize chat rows in parallel and retain the training columns.

    Args:
        chat_ds: Lazily tokenized chat dataset.
        num_proc: Number of tokenization worker processes.

    Returns:
        Arrow-backed dataset containing tokenized training rows.
    """
    ds = tokenize_dataset_parallel(chat_ds, num_proc=num_proc)
    drop = [c for c in ds.column_names if c not in KEEP_COLUMNS]
    return ds.remove_columns(drop) if drop else ds


def summarize(name: str, ds: Dataset, pad_token_id: int | None) -> None:
    """Print length, supervision, and padding statistics for prepared rows.

    Args:
        name: Label identifying the layout and split.
        ds: Nonempty dataset containing input_ids and labels columns.
        pad_token_id: Token ID counted as padding in the first row.
    """
    n = len(ds)
    row = ds[0]
    lens = [len(ds[i]["input_ids"]) for i in range(min(n, 256))]
    sup = [sum(1 for x in ds[i]["labels"] if x != -100) for i in range(min(n, 256))]
    print(
        f"[{name}] rows={n} columns={ds.column_names} "
        f"input_ids_len(min/mean/max over first 256)={min(lens)}/{sum(lens) / len(lens):.0f}/{max(lens)} "
        f"supervised_tokens(mean over first 256)={sum(sup) / len(sup):.0f} "
        f"first_row_pad_count={sum(1 for x in row['input_ids'] if x == pad_token_id)}",
        flush=True,
    )


def main() -> None:
    """Prepare and save the padded and packed Tulu3 datasets requested by the CLI."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path", required=True, help="local HF snapshot of deepseek-ai/DeepSeek-V4.1-Flash (DS41_CHECKPOINT)"
    )
    p.add_argument(
        "--template",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tulu3_chat_template.jinja"),
        help="chat template; defaults to tulu3_chat_template.jinja next to this script",
    )
    p.add_argument(
        "--out-root",
        required=True,
        help="output root; DS41_DATA_ROOT is <out-root>/tulu3_4k_padded or <out-root>/tulu3_32k_packed",
    )
    p.add_argument("--shuffle-seed", type=int, default=42)
    p.add_argument("--num-proc", type=int, default=32)
    # 4k right-padded rows (CP recipe): 100 updates x GBS 64/128 consume 6400/12800 rows.
    p.add_argument("--padded-train", type=int, default=16384)
    p.add_argument("--padded-val", type=int, default=256)
    p.add_argument("--padded-seq-length", type=int, default=4096)
    # 32k packed: 100 updates x GBS 64 = 6400 packs ~ 2.1e8 tokens; Tulu3 averages ~700 tokens/row -> ~400k rows.
    p.add_argument("--packed-train", type=int, default=400000)
    p.add_argument("--packed-val", type=int, default=4096)
    p.add_argument("--packed-size", type=int, default=32768)
    p.add_argument("--skip-padded", action="store_true")
    p.add_argument("--skip-packed", action="store_true")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    template = open(args.template).read()
    print(
        f"tokenizer: eos={tokenizer.eos_token!r}({tokenizer.eos_token_id}) pad={tokenizer.pad_token!r}({tokenizer.pad_token_id}) "
        f"bos={tokenizer.bos_token!r}",
        flush=True,
    )
    if tokenizer.pad_token_id != tokenizer.eos_token_id:
        print("WARNING: upstream expects EOS and pad to share a token id", flush=True)

    # ChatDataset shuffles with shuffle_seed before slicing, so train/validation slices do not overlap.
    if not args.skip_padded:
        out = os.path.join(args.out_root, "tulu3_4k_padded")
        t0 = time.time()
        for name, sl in (
            ("train", f"train[:{args.padded_train}]"),
            ("validation", f"train[{args.padded_train}:{args.padded_train + args.padded_val}]"),
        ):
            ds = materialize(
                build_chat_dataset(tokenizer, template, sl, args.padded_seq_length, "max_length", args.shuffle_seed),
                args.num_proc,
            )
            summarize(f"4k_padded/{name}", ds, tokenizer.pad_token_id)
            ds.save_to_disk(os.path.join(out, name))
        print(f"4k_padded done in {time.time() - t0:.0f}s -> {out}", flush=True)

    if not args.skip_packed:
        out = os.path.join(args.out_root, "tulu3_32k_packed")
        t0 = time.time()
        for name, sl in (
            ("train", f"train[:{args.packed_train}]"),
            ("validation", f"train[{args.packed_train}:{args.packed_train + args.packed_val}]"),
        ):
            tok = materialize(
                build_chat_dataset(tokenizer, template, sl, args.packed_size, False, args.shuffle_seed),
                args.num_proc,
            )
            print(f"[32k_packed/{name}] tokenized rows={len(tok)}", flush=True)
            packed = pack_dataset(
                tok,
                split=None,
                packed_sequence_size=args.packed_size,
                padding_idx=tokenizer.pad_token_id,
                cp_size=1,
                pad_to_multiple_of=2,
            )
            summarize(f"32k_packed/{name}", packed, tokenizer.pad_token_id)
            packed.save_to_disk(os.path.join(out, name))
        print(f"32k_packed done in {time.time() - t0:.0f}s -> {out}", flush=True)

    print("PREPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
