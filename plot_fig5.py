"""
Reproduce Fig 5 from Jayaraman & Grauman (CVPR 2018):
  Active categorization accuracy vs. time — two panels.

Left  : PanoContext (2-class: bedroom vs living_room)
Right : ModelNet-10 (10-class: zero-shot cross-domain transfer)

Run:
  uv run python plot_fig5.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt

LEFT_JSON  = "results/transfer_panocontext.json"
RIGHT_JSON = "results/transfer_modelnet10.json"

COLORS = {
    "ours":         ("tab:purple", "-",  "o"),
    "random":       ("tab:orange", "-",  "s"),
    "large-action": ("tab:green",  "-",  "^"),
    "1-view":       ("tab:gray",   "--", ""),
}


def load(path):
    if not os.path.exists(path):
        print(f"ERROR: results file not found: {path}", file=sys.stderr)
        print("Run eval_transfer.py first to generate results.", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


left_data  = load(LEFT_JSON)
right_data = load(RIGHT_JSON)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

for ax, data, title in [
    (axes[0], left_data,  left_data["dataset"]),
    (axes[1], right_data, right_data["dataset"]),
]:
    T = data["T"]
    ts = list(range(1, T + 1))

    for name, accs in data["results"].items():
        color, ls, marker = COLORS.get(name, ("tab:blue", "-", "o"))
        ax.plot(ts, accs, color=color, linestyle=ls,
                marker=marker if marker else None, markersize=5,
                linewidth=1.8, label=name)

    ax.set_xlabel("time $t$", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(ts)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    all_vals = [v for accs in data["results"].values() for v in accs]
    ymin = max(0,  min(all_vals) - 5)
    ymax = min(100, max(all_vals) + 5)
    ax.set_ylim(ymin, ymax)

fig.suptitle("Fig 5 · Active categorization accuracy vs. time (policy transfer)",
             fontsize=10, y=1.01)
plt.tight_layout()

os.makedirs("results", exist_ok=True)
out = "results/fig5_policy_transfer.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

for data in [left_data, right_data]:
    print(f"\n{data['dataset']}")
    T = data["T"]
    header = f"{'Method':<15}" + "".join(f"  t={t}" for t in range(1, T+1))
    print(header)
    print("-" * len(header))
    for name, accs in data["results"].items():
        row = f"{name:<15}" + "".join(f"  {a:4.1f}" for a in accs)
        print(row)
