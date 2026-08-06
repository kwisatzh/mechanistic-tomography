#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Regenerate the paper's Claim-3 budget figure from the archived summary."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METHOD_STYLES = {
    ("first_order_omp", 5.0): dict(color="#4C9F70", ls="--", marker="x"),
    ("first_order_omp", 8.0): dict(color="#D95F5F", ls="--", marker="x"),
    ("lifted_omp", 5.0): dict(color="#2878B5", ls="-", marker="o"),
    ("lifted_omp", 8.0): dict(color="#F28E2B", ls="-", marker="o"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def plot_curve(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    threshold: float,
) -> None:
    for method in ("first_order_omp", "lifted_omp"):
        for epsilon in (5.0, 8.0):
            rows = summary[
                (summary["method"] == method)
                & (summary["epsilon"].astype(float) == epsilon)
            ].sort_values("budget_measurements")
            style = METHOD_STYLES[(method, epsilon)]
            label = f"{'first-order' if method == 'first_order_omp' else 'lifted'} OMP, $\\epsilon={epsilon:g}$"
            ax.errorbar(
                rows["budget_measurements"],
                rows[f"{metric}_mean"],
                yerr=rows[f"{metric}_sem"],
                label=label,
                linewidth=2.0,
                markersize=5.5,
                markeredgewidth=1.2,
                capsize=2.5,
                **style,
            )

    ax.axhline(threshold, color="#777777", ls="--", lw=1.1)
    ax.text(0.985, threshold + 0.018, f"target {threshold:.2f}", color="#666666",
            ha="right", va="bottom", transform=ax.get_yaxis_transform(), fontsize=9.5)
    ax.set_xscale("log", base=2)
    ax.set_xlim(14, 285)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([16, 32, 64, 128, 256], labels=["16", "32", "64", "128", "256"])
    ax.set_xlabel("aggregate measurements")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=7)
    ax.grid(axis="both", color="#D9D9D9", lw=0.6, alpha=0.7)


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)

    n_components = 64
    support = 4
    lifted_dimension = n_components + math.comb(n_components, 2)
    orientation = support * math.log(lifted_dimension / support)

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.dpi": 220,
        "savefig.dpi": 220,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 4.6), sharex=True)
    plot_curve(axes[0], summary, "heldout_r2", "held-out $R^2$", "Prediction quality", 0.95)
    plot_curve(axes[1], summary, "pair_topk_recall", "top-$k$ pair recovery", "Pair support recovery", 0.99)

    for ax in axes:
        ax.axvline(orientation, color="#7F7F7F", ls=":", lw=1.2)
    axes[0].text(orientation * 1.035, 0.72, "$4\\log(N/4)\\approx25$\n(orientation only)",
                 color="#555555", rotation=90, ha="left", va="center", fontsize=9)

    axes[0].text(0.99, 0.04, "exhaustive pair probing: 2016 measurements (off scale)",
                 transform=axes[0].transAxes, ha="right", va="bottom", fontsize=9,
                 color="#555555")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.01), columnspacing=1.4, handlelength=2.3)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.14, top=0.82, wspace=0.24)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"N={lifted_dimension}, k={support}, 4 log(N/4)={orientation:.2f}")
    print(args.output)


if __name__ == "__main__":
    main()
