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

"""Masking-sanity check for the CoderForge SFT training data.

Loads the EXACT training ChatDataset (same tokenizer / seq_length / JSONL) used by the
gemma4_31b_coderforge recipe and inspects the per-sample labels to confirm the SFT actually
supervised the right tokens. "Loss dropped" only means loss dropped on whatever was supervised;
if masking is broken (all-masked, everything-supervised, or wrong spans) then loss-down /
capability-down is a BUG, not a finding.

For a few samples it reports:
  - supervised fraction (labels != -100); flags ~0% (nothing learned) or ~100% (user/tool text
    also supervised -> the model imitates non-assistant text).
  - whether the gemma4 stop token <turn|> (106) falls inside supervised spans.
  - a decoded supervised span vs a decoded masked span, so a human can eyeball that ASSISTANT
    content (incl. tool_calls) is supervised and system/user/tool-result content is masked.
"""

import argparse
import json

from transformers import AutoTokenizer

from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset

GEMMA4 = "/path/to/checkpoints/hf_gemma4_31b_it"
JSONL = ("/path/to/coderforge_cache/"
         "togethercomputer_CoderForge-Preview_filtered_reward1_seq65536/data.jsonl")


def contiguous_spans(labels, keep):
    """Yield (start, end) index ranges where (labels[i] != -100) == keep."""
    spans, s = [], None
    for i, l in enumerate(labels):
        cond = (l != -100)
        if cond == keep and s is None:
            s = i
        elif cond != keep and s is not None:
            spans.append((s, i)); s = None
    if s is not None:
        spans.append((s, len(labels)))
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seq-length", type=int, default=65536)
    ap.add_argument("--jsonl", default=JSONL)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(GEMMA4)
    ds = ChatDataset(path_or_dataset_id=args.jsonl, tokenizer=tok, split="train",
                     seq_length=args.seq_length, padding="do_not_pad")
    print(f"Dataset size: {len(ds)}")

    for i in range(min(args.n, len(ds))):
        ex = ds[i]
        keys = list(ex.keys())
        ids = list(ex["input_ids"])
        labels = list(ex.get("labels", ex.get("loss_mask", [])))
        # If a loss_mask (0/1) was returned instead of -100 labels, normalize to label view.
        if "labels" not in ex and "loss_mask" in ex:
            labels = [ids[j] if labels[j] else -100 for j in range(len(ids))]
        n = len(ids)
        sup = [j for j in range(n) if labels[j] != -100]
        frac = 100 * len(sup) / max(1, n)
        sup_ids = [ids[j] for j in sup]
        print(f"\n===== sample {i} (keys={keys}) =====")
        print(f"  tokens={n}  supervised={len(sup)} ({frac:.1f}%)  106_in_supervised={106 in sup_ids}")
        if not sup:
            print("  !!! ALL-MASKED — nothing supervised (BUG).")
            continue
        if frac > 95:
            print("  !!! ~ALL supervised — user/tool text likely supervised too (BUG).")
        # Show the first supervised span (should be assistant content incl. tool_calls) and the
        # first masked span (should be system/user/tool-result).
        sup_spans = contiguous_spans(labels, keep=True)
        msk_spans = contiguous_spans(labels, keep=False)
        if sup_spans:
            a, b = sup_spans[0]
            print(f"  first SUPERVISED span [{a}:{b}] ->\n    {tok.decode(ids[a:min(b, a+80)])!r}")
        if msk_spans:
            a, b = msk_spans[0]
            print(f"  first MASKED span [{a}:{b}] ->\n    {tok.decode(ids[a:min(b, a+80)])!r}")

    print("\nVERDICT: supervised fraction should be moderate (roughly 10-60% for agent"
          " transcripts), 106 present in supervised spans, assistant text supervised &"
          " system/user/tool text masked. All-masked or ~all-supervised => masking bug.")


if __name__ == "__main__":
    main()
