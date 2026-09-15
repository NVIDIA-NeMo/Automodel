"""Scan the v4_88k corpus for rows that carry real reasoning.

Two shapes can put reasoning into the supervised span:

* a ``reasoning_content`` field on a message (the upcoming corpus revision; the
  Qwen3.6 chat template renders it as ``<think>\\n`` + reasoning on the final turn),
* literal ``<think> ... </think>`` tags *inside* ``content`` with a non-empty body.

Either one changes what ``last_turn_collate_fn`` masks: the thinking-enabled
suffix ``<think>\\n`` matches instead of the full empty block, so the reasoning
becomes supervised. Reports counts for both, over the whole file.

CPU only.
"""

import argparse
import json
import pathlib
import re

import pyarrow.parquet as pq

THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="data/v4_88k_filtered/train.parquet")
    p.add_argument("--out", default=None, help="Write matching row indices as JSON.")
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    pf = pq.ParquetFile(args.dataset)
    schema = pf.schema_arrow
    print(f"file      : {args.dataset}")
    print(f"rows      : {pf.metadata.num_rows}")
    print(f"columns   : {schema.names}")
    msg_field = schema.field("messages").type
    print(f"messages  : {msg_field}")
    print()

    # Does the message struct even declare reasoning_content?
    value = msg_field.value_type if hasattr(msg_field, "value_type") else msg_field
    keys = [value.field(i).name for i in range(value.num_fields)] if hasattr(value, "num_fields") else []
    print(f"message keys declared in the schema: {keys}")
    has_rc_column = "reasoning_content" in keys
    print(f"'reasoning_content' present in schema: {has_rc_column}")
    print()

    n = 0
    rc_any = rc_final = 0            # rows with a non-empty reasoning_content
    think_any = think_final = 0      # rows with a non-empty <think>..</think> in content
    empty_think_final = 0            # the shape the current corpus uses
    hits = {"reasoning_content": [], "think_tag": []}

    for batch in pf.iter_batches(batch_size=args.batch_size, columns=["messages"]):
        for messages in batch.column("messages").to_pylist():
            idx = n
            n += 1
            if not messages:
                continue
            final = messages[-1]

            def rc(m):
                v = m.get("reasoning_content") if isinstance(m, dict) else None
                return bool(v and str(v).strip())

            if any(rc(m) for m in messages):
                rc_any += 1
                if rc(final):
                    rc_final += 1
                    if len(hits["reasoning_content"]) < 20:
                        hits["reasoning_content"].append(idx)

            def think_bodies(m):
                text = (m.get("content") or "") if isinstance(m, dict) else ""
                return [b for b in THINK.findall(text) if b.strip()]

            if any(think_bodies(m) for m in messages):
                think_any += 1
                if think_bodies(final):
                    think_final += 1
                    if len(hits["think_tag"]) < 20:
                        hits["think_tag"].append(idx)

            ftext = (final.get("content") or "") if isinstance(final, dict) else ""
            if "<think>" in ftext and not think_bodies(final):
                empty_think_final += 1

    print(f"scanned rows                                        : {n}")
    print(f"rows with non-empty reasoning_content (any turn)    : {rc_any}")
    print(f"  ... on the FINAL turn (the supervised one)        : {rc_final}")
    print(f"rows with non-empty <think>..</think> in content    : {think_any}")
    print(f"  ... on the FINAL turn (the supervised one)        : {think_final}")
    print(f"rows with an EMPTY <think> block in final content   : {empty_think_final}")
    print()
    print(f"example row indices: {json.dumps(hits)}")

    if args.out:
        pathlib.Path(args.out).write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "rows": n,
                    "reasoning_content_any": rc_any,
                    "reasoning_content_final": rc_final,
                    "think_tag_any": think_any,
                    "think_tag_final": think_final,
                    "empty_think_final": empty_think_final,
                    "examples": hits,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
