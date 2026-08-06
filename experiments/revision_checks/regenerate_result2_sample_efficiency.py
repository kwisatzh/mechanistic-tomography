#!/usr/bin/env python3
"""Regenerate the Result 2 sample-efficiency figure from the saved sparse-recovery sweep."""

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--plotted-csv", type=Path)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row["method"] in {"omp", "ridge"} and 12 <= int(row["n_train"]) <= 64
    ]


def write_plotted_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "n_train", "n_components", "holdout_r2", "pearson_vs_mi"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def series(rows: list[dict[str, str]], method: str, metric: str) -> tuple[list[int], list[float]]:
    selected = sorted((row for row in rows if row["method"] == method), key=lambda row: int(row["n_train"]))
    return [int(row["n_train"]) for row in selected], [float(row[metric]) for row in selected]


def plot(rows: list[dict[str, str]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
    styles = {
        ("omp", "holdout_r2"): ("OMP held-out $R^2$", "#1f77b4", "o", "-"),
        ("omp", "pearson_vs_mi"): ("OMP Pearson", "#1f77b4", "s", "--"),
        ("ridge", "holdout_r2"): ("ridge held-out $R^2$", "#ff7f0e", "o", "-"),
        ("ridge", "pearson_vs_mi"): ("ridge Pearson", "#ff7f0e", "s", "--"),
    }
    for key, (label, color, marker, linestyle) in styles.items():
        x, y = series(rows, key[0], key[1])
        ax.plot(x, y, label=label, color=color, marker=marker, linestyle=linestyle, linewidth=1.8, markersize=4.5)

    ax.axhline(0.9, color="0.45", linewidth=1.0, linestyle=":")
    ax.text(63.5, 0.905, "0.90", color="0.38", fontsize=8, ha="right", va="bottom")
    ax.axvline(32, color="0.55", linewidth=1.0, linestyle=":")
    ax.text(32.8, -0.19, "$n=32$ coordinates", color="0.35", fontsize=8, rotation=90, va="bottom")
    ax.set_xlabel("aggregate measurements")
    ax.set_ylabel("score")
    ax.set_xlim(11, 65)
    ax.set_ylim(-0.25, 1.04)
    ax.set_xticks([12, 16, 24, 32, 48, 64])
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", fontsize=7.5, ncol=2, frameon=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    plot(rows, args.output_png)
    if args.plotted_csv:
        write_plotted_rows(args.plotted_csv, rows)


if __name__ == "__main__":
    main()
