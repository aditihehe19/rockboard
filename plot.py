"""Training curves for every run in logs/, saved to results/training_curves.png.

Each point is the average over the last 100 attempts, like ep_rew_mean.
"""

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from rewards import EXPERIMENTS

COLORS = dict(zip(EXPERIMENTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
LABELS = {
    "builtin": "Built-in reward",
    "time_fee": "+ time fee",
    "hover_bonus": "+1 per calm frame (broken)",
    "calm_progress": "Calm progress + fee (fix)",
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

PANELS = [
    ("r", "Score PPO was paid per attempt\n(built-in + your rules)", None),
    ("builtin_score", "Lunar Lander score per attempt\n(built-in only)", (200, "solved (200)")),
    ("l", "Frames per attempt", (1000, "time limit (1,000)")),
]


def load_run(folder):
    """Merge the 16 Monitor logs of one run into a single 100-attempt rolling average."""
    logs = [pd.read_csv(f, skiprows=1) for f in glob.glob(f"{folder}/*.monitor.csv")]
    df = pd.concat(logs).sort_values("t")
    steps = df["l"].cumsum()
    rolled = df[["r", "builtin_score", "l"]].rolling(100, min_periods=20).mean()
    rolled["steps"] = steps
    return rolled.dropna()


def main():
    runs = sorted(Path(p).name for p in glob.glob("logs/*_seed*") if Path(p).is_dir())
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(9, 10), sharex=True, facecolor=SURFACE)

    for ax, (column, title, reference) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        if reference:
            value, text = reference
            ax.axhline(value, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            ax.annotate(text, (0, value), xytext=(4, 4), textcoords="offset points",
                        xycoords=("axes fraction", "data"), color=MUTED, fontsize=8)

        for run in runs:
            experiment, seed = run.rsplit("_seed", 1)
            if experiment not in EXPERIMENTS:
                continue
            df = load_run(f"logs/{run}")
            ax.plot(df["steps"], df[column], color=COLORS[experiment], linewidth=1.6,
                    linestyle="-" if seed == "0" else (0, (5, 2)))

    # One legend entry per experiment, plus a key for the two seeds.
    shown = [e for e in EXPERIMENTS if any(r.startswith(e + "_seed") for r in runs)]
    handles = [plt.Line2D([], [], color=COLORS[e], linewidth=2) for e in shown]
    handles += [plt.Line2D([], [], color=MUTED, linewidth=1.6, linestyle=s) for s in ("-", (0, (5, 2)))]
    names = [LABELS[e] for e in shown] + ["seed 0", "seed 1"]
    fig.legend(handles, names, loc="upper center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.0))

    axes[-1].set_xlabel("Training steps (frames)", color=MUTED)
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:g}M"))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path("results").mkdir(exist_ok=True)
    fig.savefig("results/training_curves.png", dpi=150, facecolor=SURFACE)
    print("Wrote results/training_curves.png")


if __name__ == "__main__":
    main()
