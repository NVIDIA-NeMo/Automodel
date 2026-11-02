"""Plot GLM-5.2 full-model CP parity: CP1 x EP64 vs CP2 (head/tail) x EP64, per-step loss and gradient norm.

Usage: python plot_cp_parity.py cp1/training.jsonl cp2/training.jsonl out.png
Two single-axis panels (loss, gradient norm) over the same steps; the per-step differences are printed
and the maximum absolute loss difference is annotated. Fixed categorical colors (blue = baseline,
orange = CP2 head/tail), thin lines, direct labels, recessive grid.
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEXT, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e5e1"
SERIES = (  # (label, color, style, width)
    ("CP1 x EP64 (64 GB300)", "#2a78d6", "-", 3.0),
    ("CP2 head/tail x EP64 (64 GB300)", "#eb6834", "--", 1.8),
)


def read(path: str) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "step" in rec and "loss" in rec:
            out[int(rec["step"])] = {"loss": float(rec["loss"]), "grad_norm": float(rec.get("grad_norm", float("nan")))}
    return out


def main() -> None:
    cp1_path, cp2_path, out = sys.argv[1:4]
    cp1, cp2 = read(cp1_path), read(cp2_path)
    steps = sorted(set(cp1) & set(cp2))
    if not steps:
        raise SystemExit("no common steps")
    dloss = [cp2[s]["loss"] - cp1[s]["loss"] for s in steps]
    dgn = [(cp2[s]["grad_norm"] - cp1[s]["grad_norm"]) / cp1[s]["grad_norm"] for s in steps]
    worst_loss = max(range(len(steps)), key=lambda i: abs(dloss[i]))
    worst_gn = max(range(len(steps)), key=lambda i: abs(dgn[i]))
    print(f"steps compared: {len(steps)} ({steps[0]}..{steps[-1]})")
    print(f"max |loss diff| = {abs(dloss[worst_loss]):.4f} at step {steps[worst_loss]} "
          f"(cp1 {cp1[steps[worst_loss]]['loss']:.4f}, cp2 {cp2[steps[worst_loss]]['loss']:.4f})")
    print(f"max |grad-norm rel diff| = {abs(dgn[worst_gn]):.4f} at step {steps[worst_gn]}")
    print("step  loss_cp1  loss_cp2  diff      gn_cp1   gn_cp2   rel")
    for i, s in enumerate(steps):
        print(f"{s:4d}  {cp1[s]['loss']:.4f}   {cp2[s]['loss']:.4f}   {dloss[i]:+.4f}   "
              f"{cp1[s]['grad_norm']:.4f}  {cp2[s]['grad_norm']:.4f}  {dgn[i]:+.4f}")

    plt.rcParams.update({"font.size": 11, "text.color": TEXT, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})
    fig, (ax_loss, ax_gn) = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150)
    for ax, key, title in ((ax_loss, "loss", "Training loss"), (ax_gn, "grad_norm", "Gradient norm")):
        for run, (label, color, style, width) in zip((cp1, cp2), SERIES):
            ax.plot(steps, [run[s][key] for s in steps], style, color=color, linewidth=width, label=label,
                    solid_capstyle="round")
        ax.set_title(title, loc="left", color=TEXT, fontsize=12)
        ax.set_xlabel("step")
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(x=0.02)
    ax_loss.set_ylabel("loss (nats / token)")
    ax_gn.set_ylabel("global grad norm")
    ax_loss.legend(frameon=False, loc="upper right", fontsize=10)
    s = steps[worst_loss]
    ax_loss.annotate(f"max |diff| {abs(dloss[worst_loss]):.4f} @ step {s}", xy=(s, cp2[s]["loss"]),
                     xytext=(0.30, 0.52), textcoords="axes fraction", color=MUTED, fontsize=9,
                     arrowprops={"arrowstyle": "-", "color": GRID, "lw": 1})
    ax_gn.text(0.98, 0.9, f"max relative diff {abs(dgn[worst_gn]):.1%} @ step {steps[worst_gn]}",
               transform=ax_gn.transAxes, ha="right", va="top", color=MUTED, fontsize=9)
    fig.suptitle("GLM-5.2 (78 layers, 256 experts) on Tulu3 packed to 32k: CP1 vs CP2 head/tail, "
                 "same data order, init and seed", x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
