"""Check what LengthGroupedSampler actually yields, epoch over epoch.

Answers two questions for the 2-node recipe (ep8, lbs 4, 16 ranks):
  1. does the order change between epoch 0 and epoch 1?
  2. is the per-epoch chunk shuffle the only randomness, i.e. do batches keep
     the same length-homogeneous membership?

CPU only. Uses the real `n_tokens` column, exactly as
`LengthGroupedSamplerConfig.build` does.
"""

import argparse
import pathlib

from datasets import load_dataset

from nemo_automodel.components.datasets.llm.length_grouped_sampler import LengthGroupedSampler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="data/v4_88k_filtered/train.parquet")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch-size", type=int, default=4, help="local_batch_size")
    p.add_argument("--world-size", type=int, default=16)
    p.add_argument("--ranks", type=int, default=2, help="How many ranks to instantiate.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ds = load_dataset("parquet", data_files=args.dataset, split="train")
    lengths = list(ds["n_tokens"])
    out = []
    w = lambda s="": (out.append(s), print(s))

    w(f"dataset        : {args.dataset}  ({len(ds)} rows)")
    w(f"sampler        : seed={args.seed} batch_size={args.batch_size} "
      f"num_replicas={args.world_size} drop_last=True")
    w("")

    orders = {}
    for rank in range(args.ranks):
        s = LengthGroupedSampler(
            dataset=ds, batch_size=args.batch_size, seed=args.seed,
            num_replicas=args.world_size, rank=rank, drop_last=True, lengths=lengths,
        )
        w(f"--- rank {rank}: {len(s)} indices, {len(s) // args.batch_size} chunks ---")
        for epoch in range(args.epochs):
            s.set_epoch(epoch)
            order = list(iter(s))
            orders[(rank, epoch)] = order
            head = order[: args.batch_size * 2]
            w(f"  epoch {epoch}: first {len(head)} indices {head}  "
              f"lengths {[lengths[i] for i in head]}")

        w("")
        base = orders[(rank, 0)]
        for epoch in range(1, args.epochs):
            other = orders[(rank, epoch)]
            same_pos = sum(a == b for a, b in zip(base, other))
            w(f"  epoch 0 vs {epoch}: identical order? {base == other}   "
              f"same multiset? {sorted(base) == sorted(other)}   "
              f"indices at an identical position: {same_pos}/{len(base)} "
              f"({100.0 * same_pos / len(base):.2f}%)")

        # Does the *batch membership* change, or only the batch order?
        def batches(order):
            bs = args.batch_size
            return {tuple(order[i : i + bs]) for i in range(0, len(order), bs)}

        b0, b1 = batches(orders[(rank, 0)]), batches(orders[(rank, 1)])
        w(f"  batch membership identical across epochs? {b0 == b1}  "
          f"({len(b0 & b1)}/{len(b0)} batches shared)")
        w("")

    if args.ranks >= 2:
        w("--- cross-rank ---")
        for epoch in range(args.epochs):
            a, b = orders[(0, epoch)], orders[(1, epoch)]
            w(f"  epoch {epoch}: rank0/rank1 index overlap {len(set(a) & set(b))} "
              f"(must be 0: disjoint shards)")
            la = [lengths[i] for i in a[: args.batch_size]]
            lb = [lengths[i] for i in b[: args.batch_size]]
            w(f"            first-batch lengths rank0={la} rank1={lb}  "
              f"(aligned => little cross-rank padding)")

    if args.out:
        pathlib.Path(args.out).write_text("\n".join(str(x) for x in out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
