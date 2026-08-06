#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Plot the fixed-fold IOI paired predictive improvements."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    ("count_plus_PB_count", r"$P\!\times\!B$"),
    ("count_plus_PE_count", r"$P\!\times\!E$"),
    ("count_plus_BE_count", r"$B\!\times\!E$"),
    ("count_plus_all_pairs", "all"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.summary).set_index("model")
    rows = source.loc[[model for model, _ in MODEL_ORDER]]
    labels = [label for _, label in MODEL_ORDER]
    center = rows["delta_mae_mean"].to_numpy(float)
    lower = rows["delta_mae_q025"].to_numpy(float)
    upper = rows["delta_mae_q975"].to_numpy(float)
    errors = np.vstack([center - lower, upper - center])

    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5})
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#7A5195"]
    x = np.arange(len(rows))
    ax.errorbar(
        x,
        center,
        yerr=errors,
        fmt="none",
        ecolor="0.20",
        elinewidth=1.0,
        capsize=2.5,
        zorder=2,
    )
    ax.scatter(x, center, c=colors, s=27, zorder=3)
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"paired $\Delta$MAE vs count-additive")
    ax.set_ylim(-0.008, 0.165)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.55)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=320)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
