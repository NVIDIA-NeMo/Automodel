"""Dump one training-ready v4_88k sample to a human-readable text file.

Takes a random row from the pre-filtered corpus, pushes it through exactly the
dataset adapter and collator the recipe uses (`make_v4_88k_dataset` +
`last_turn_collate_fn`), and writes what the model actually sees: the decoded
input_ids, the supervised span marked inline, the decoded labels, and a
token-level view of the mask boundary.

CPU only. Batch of one, so there is no padding to confuse the reading.

    python examples/vlm_finetune/qwen3_5_moe/dump_sample_v4_88k.py \
        --dataset data/v4_88k_filtered/train.parquet --out sample.txt
"""

import argparse
import importlib.util
import pathlib
import random

import torch
from transformers import AutoProcessor

IGNORE_INDEX = -100
_MODULE_PATH = pathlib.Path(__file__).resolve().parent / "v4_88k.py"


def _load_adapter():
    spec = importlib.util.spec_from_file_location("v4_88k_adapter", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runs(labels_row: torch.Tensor) -> list[tuple[int, int]]:
    """[start, end) bounds of each contiguous supervised span."""
    flags = labels_row.ne(IGNORE_INDEX).tolist()
    spans, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(flags)))
    return spans


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="data/v4_88k_filtered/train.parquet")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--out", default="sample_v4_88k.txt")
    p.add_argument("--seed", type=int, default=None, help="Default: OS entropy.")
    p.add_argument("--index", type=int, default=None, help="Force a row index.")
    p.add_argument("--context", type=int, default=24, help="Tokens of context at the mask boundary.")
    p.add_argument("--reasoning-revision", action="store_true",
                   help="Rebuild the final turn the way the pending corpus revision will ship it: "
                        "content=y_raw (the action only) and reasoning_content=z_raw (the THOUGHT "
                        "prose), instead of the two concatenated into content. SYNTHESIZED from the "
                        "existing columns -- no row in the current corpus is stored this way.")
    p.add_argument("--repr", dest="as_repr", action="store_true",
                   help="Render the decoded text as exact Python repr() literals (escapes visible, "
                        "one line per section, round-trips through ast.literal_eval).")
    args = p.parse_args()

    adapter = _load_adapter()
    processor = AutoProcessor.from_pretrained(args.model, padding_side="right")
    tokenizer = getattr(processor, "tokenizer", processor)

    dataset = adapter.make_v4_88k_dataset(path_or_dataset=args.dataset)

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    index = args.index if args.index is not None else random.Random(seed).randrange(len(dataset))
    row = dataset[index]

    if args.reasoning_revision:
        # The adapter forwards `reasoning_content` when a message carries it; the current
        # corpus never does (it concatenates "THOUGHT: " + z_raw into content). Rebuild
        # the final turn from the raw columns so the thinking-enabled render can be
        # inspected before the revision lands.
        import pyarrow.parquet as pq

        # pyarrow cannot convert a nested column out of a chunked array, so walk
        # record batches (which are contiguous) and stop at the wanted row.
        cols = ["messages", "z_raw", "y_raw"]
        raw, seen = None, 0
        for rb in pq.ParquetFile(args.dataset).iter_batches(batch_size=512, columns=cols):
            if seen + rb.num_rows > index:
                raw = rb.slice(index - seen, 1).to_pylist()[0]
                break
            seen += rb.num_rows
        if raw is None:
            raise SystemExit(f"row {index} not found in {args.dataset}")
        if not (raw["z_raw"] or "").strip():
            raise SystemExit(f"row {index} has an empty z_raw; pick a row that has reasoning")
        conversation = []
        for i, m in enumerate(raw["messages"]):
            turn = {"role": m["role"], "content": [{"type": "text", "text": m.get("content") or ""}]}
            if i == len(raw["messages"]) - 1:
                turn["content"] = [{"type": "text", "text": raw["y_raw"] or ""}]
                turn["reasoning_content"] = raw["z_raw"]
            conversation.append(turn)
        row = {"conversation": conversation, "n_tokens": row.get("n_tokens")}

    # Exactly the collator the recipe configures, on a batch of one.
    batch = adapter.last_turn_collate_fn([{"conversation": row["conversation"]}], processor)
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    spans = _runs(labels)
    n_sup = int(labels.ne(IGNORE_INDEX).sum())
    conv = row["conversation"]

    # default_collate_fn shifts: labels[p] supervises input_ids[p + 1].
    if spans:
        lo, hi = spans[-1]
        tok_lo, tok_hi = lo + 1, hi + 1
    else:
        tok_lo = tok_hi = len(input_ids)

    dec = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
    # In repr mode every decoded body is one exact Python string literal: whitespace,
    # newlines and special tokens are all visible and nothing is reflowed.
    show = repr if args.as_repr else (lambda s: s)
    out = []
    w = out.append

    w("=" * 100)
    w("v4_88k training-ready sample — decoded for manual review")
    w("=" * 100)
    w(f"dataset            : {args.dataset}")
    w(f"rows in dataset    : {len(dataset)}")
    w(f"row index          : {index}   (seed {seed})")
    w(f"model / tokenizer  : {args.model}")
    w(f"collator           : v4_88k.py:last_turn_collate_fn (batch of 1, no padding)")
    w(f"text rendering     : {'repr() literals' if args.as_repr else 'plain decoded text'}")
    if args.reasoning_revision:
        w("final turn         : *** SYNTHESIZED reasoning_content revision (content=y_raw, "
          "reasoning_content=z_raw) -- NOT how the current corpus is stored ***")
    w("")
    w(f"turns in row       : {len(conv)}  ({', '.join(t['role'] for t in conv)})")
    w(f"n_tokens (prefilter): {row.get('n_tokens')}")
    w(f"input_ids length   : {len(input_ids)}")
    w(f"attention_mask sum : {int(batch['attention_mask'][0].sum())}")
    w(f"batch keys         : {sorted(batch.keys())}")
    w(f"supervised tokens  : {n_sup}  ({100.0 * n_sup / len(input_ids):.2f}% of the sequence)")
    w(f"supervised runs    : {len(spans)}  {spans}   <- must be exactly 1 (final assistant turn)")
    w(f"supervised span    : label[{tok_lo - 1}:{tok_hi - 1}] => predicts input_ids[{tok_lo}:{tok_hi}]")
    w("")

    w("=" * 100)
    w("SECTION 1 — FULL MODEL INPUT (decoded input_ids), supervised region bracketed")
    w("Everything outside the >>>SUPERVISED<<< brackets contributes no loss.")
    w("=" * 100)
    w(show(dec(input_ids[:tok_lo])))
    w("")
    w(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SUPERVISED FROM HERE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    w(show(dec(input_ids[tok_lo:tok_hi])))
    w("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< SUPERVISED ENDS HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    w("")
    tail = dec(input_ids[tok_hi:])
    w(show(tail) if tail else "(nothing after the supervised span)")
    w("")

    w("=" * 100)
    w("SECTION 2 — LOSS TARGET (decoded labels, ignoring -100). This is the ONLY text trained on.")
    w("=" * 100)
    kept = labels[labels.ne(IGNORE_INDEX)]
    w(show(dec(kept)))
    w("")

    w("=" * 100)
    w("SECTION 3 — RAW FINAL ASSISTANT MESSAGE from the dataset row (for comparison with section 2)")
    w("=" * 100)
    w(show(conv[-1]["content"][0]["text"]))
    w("")

    w("=" * 100)
    w(f"SECTION 4 — TOKEN-LEVEL MASK BOUNDARY (+/- {args.context} tokens)")
    w("'mask' is the label at that position: '-' = ignored, id = supervised.")
    w("Note the next-token shift: label[p] is the target for the token at position p+1.")
    w("=" * 100)
    w(f"{'pos':>8}  {'input_id':>9}  {'label':>9}  {'mask':>5}  token")
    w("-" * 100)

    def rows(a, b):
        for pos in range(max(a, 0), min(b, len(input_ids))):
            lab = int(labels[pos])
            yield (
                f"{pos:>8}  {int(input_ids[pos]):>9}  {lab if lab != IGNORE_INDEX else -100:>9}  "
                f"{'-' if lab == IGNORE_INDEX else 'LOSS':>5}  "
                f"{tokenizer.convert_ids_to_tokens([int(input_ids[pos])])[0]!r}"
            )

    for line in rows(tok_lo - args.context, tok_lo + args.context):
        w(line)
    w(f"{'...':>8}")
    for line in rows(tok_hi - args.context, tok_hi + args.context):
        w(line)
    w("")
    w("=" * 100)
    w("CHECKS")
    w("=" * 100)
    _, suffixes = adapter._resolve_markers(tokenizer)
    for s in suffixes:
        w(f"generation-prompt suffix (masked out of the loss): {dec(s)!r} ids={s}")
    w(f"[{'PASS' if len(spans) == 1 else 'FAIL'}] exactly one supervised run")
    last_txt = (conv[-1]["content"][0]["text"] or "").strip()
    decoded_target = dec(kept).replace("<|im_end|>", "").strip()
    w(f"[{'PASS' if decoded_target.endswith(last_txt[-120:]) else 'FAIL'}] target ends with the final assistant message")
    starts_clean = not any(dec(kept).startswith(dec(s)) for s in suffixes)
    w(f"[{'PASS' if starts_clean else 'FAIL'}] target does NOT start with a generation-prompt suffix")
    w(f"[{'PASS' if int(input_ids[tok_hi - 1]) == tokenizer.convert_tokens_to_ids('<|im_end|>') else 'WARN'}] supervised span ends at <|im_end|>")
    w(f"[{'PASS' if len(input_ids) <= 40960 else 'FAIL'}] within the 40,960-token prefilter cap (no truncation)")

    text = "\n".join(out) + "\n"
    pathlib.Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {args.out}  ({len(text)} chars)  row={index} seed={seed} "
          f"len={len(input_ids)} supervised={n_sup} runs={len(spans)}")


if __name__ == "__main__":
    main()
