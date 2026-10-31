"""Plot GLM-5.3 full-model 100-step training loss: BF16 vs FP8 (and the bf16-moment BF16 control).

Usage: python plot_loss_curves.py compare_3runs.tsv out.png
Single panel, one y-axis: loss per step for all runs on identical data order.
"""

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = [  # fixed categorical order (validated palette: blue, orange, aqua); (key, label, color, style, width, label dy)
    ("bf16_fp32mom", "BF16 (fp32 Adam moments), job 21091663", "#2a78d6", "-", 3.4, 9),
    ("fp8_bf16mom", "FP8 block (bf16 Adam moments), job 21104687", "#eb6834", "-", 1.8, 0),
    ("bf16_bf16mom", "BF16 (bf16 Adam moments), job 21104688", "#1baf7a", "--", 1.8, -9),
]
YLIM = (0.35, 0.80)  # steps >= 1; step-0 values are quoted in the note
TEXT, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e5e1"

tsv, out = sys.argv[1:3]
rows = list(csv.DictReader(open(tsv), delimiter="\t"))
steps = [int(r["step"]) for r in rows]
epoch1 = next((int(r["step"]) for r in rows if int(r["epoch"]) == 1), None)

fig, ax = plt.subplots(figsize=(9, 5.4), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=9)
ax.grid(True, axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)

step0 = []
for key, label, color, ls, lw, dy in SERIES:
    if key not in rows[0]:
        continue
    ys = [float(r[key]) for r in rows]
    ax.plot(steps, ys, color=color, linewidth=lw, linestyle=ls, label=label, solid_capstyle="round")
    ax.annotate(f"{ys[-1]:.4f}", (steps[-1], ys[-1]), textcoords="offset points", xytext=(6, dy), fontsize=8.5, color=TEXT, va="center")
    step0.append(f"{ys[0]:.4f}")
ax.set_ylim(*YLIM)
ax.annotate("y-axis clipped above 0.8; step-0 losses (identical weights): " + " / ".join(step0) + " in legend order",
            (0.01, 0.02), xycoords="axes fraction", fontsize=8.5, color=MUTED)
ax.set_ylabel("training loss", color=TEXT)
ax.set_xlabel("step", color=TEXT)
ax.set_title("GLM-5.3 full-parameter SFT on 32x8 H100 (PP4 x EP64): BF16 vs FP8 training loss\n"
             "packed 4k Tulu3 train[:100000], identical data order, GBS 256, lr 1e-5",
             color=TEXT, fontsize=10.5, loc="left")
ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 0.07), labelcolor=TEXT)
if epoch1 is not None:
    ax.axvline(epoch1, color=MUTED, linewidth=1, linestyle=":")
    ax.annotate(f"epoch 1 from step {epoch1}\n(train[:100000] repeats)", (epoch1, YLIM[1]), textcoords="offset points",
                xytext=(-6, -26), fontsize=8.5, color=MUTED, ha="right")
fig.tight_layout()
fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
print("wrote", out, "steps", len(steps), "series", [k for k, *_ in SERIES if k in rows[0]])
