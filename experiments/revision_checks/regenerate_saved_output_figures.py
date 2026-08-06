#!/usr/bin/env python3
"""Regenerate manuscript figures from existing HMM, planted, and calibration outputs."""

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_observer(rows: list[dict[str, str]], output: Path) -> None:
    colors = {
        "oracle": "#1f77b4",
        "linear_probe": "#2ca02c",
        "last_obs_proxy": "#bcbd22",
        "entangled_bad": "#7f7f7f",
        "noisy_oracle": "#d6279f",
    }
    labels = {
        "oracle": "oracle",
        "linear_probe": "probe",
        "last_obs_proxy": "proxy",
        "entangled_bad": "entangled",
    }
    offsets = {
        "oracle": (5, 7),
        "linear_probe": (-10, 22),
        "last_obs_proxy": (5, -15),
        "entangled_bad": (5, 5),
        "0.25": (5, 13),
        "0.5": (5, -19),
        "1.0": (5, 6),
        "1.5": (5, 6),
        "2.0": (5, 6),
        "3.0": (-34, 6),
    }
    fig, ax = plt.subplots(figsize=(5.8, 3.5), constrained_layout=True)
    noisy = sorted((row for row in rows if row["observer"] == "noisy_oracle"), key=lambda row: float(row["sigma"]))
    ax.plot(
        [float(row["observer_rmse"]) for row in noisy],
        [float(row["control_target_mse"]) for row in noisy],
        color=colors["noisy_oracle"],
        linewidth=1.1,
        alpha=0.65,
        zorder=1,
    )
    for row in rows:
        observer = row["observer"]
        x = float(row["observer_rmse"])
        y = float(row["control_target_mse"])
        ax.scatter(x, y, s=34, color=colors[observer], edgecolor="white", linewidth=0.5, zorder=2)
        if observer == "noisy_oracle":
            sigma = row["sigma"]
            text = rf"$\sigma={float(sigma):g}$"
            offset = offsets[sigma]
        else:
            text = labels[observer]
            offset = offsets[observer]
        ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", fontsize=7.5)
    ax.set_title("Observer error and closed-loop target error", fontsize=10)
    ax.set_xlabel("observer RMSE against reference state")
    ax.set_ylabel("closed-loop target MSE")
    ax.set_ylim(2.8, 16.35)
    ax.grid(alpha=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def plot_reach(rows: list[dict[str, str]], output: Path) -> None:
    order = [
        "raw_atp",
        "scalar_cal_atp",
        "multigain_atp",
        "finite_single",
        "subset_ridge",
        "first_order_omp",
        "lifted_omp",
    ]
    labels = {
        "raw_atp": "raw AtP",
        "scalar_cal_atp": "scalar-calibrated AtP",
        "multigain_atp": "multi-gain AtP",
        "finite_single": "finite singleton",
        "subset_ridge": "subset ridge",
        "first_order_omp": "first-order OMP",
        "lifted_omp": "lifted OMP",
    }
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if float(row["epsilon"]) == 8.0:
            grouped[row["method"]].append(float(row["heldout_r2"]))
    means = [statistics.mean(grouped[method]) for method in order]
    errors = [statistics.stdev(grouped[method]) for method in order]

    fig, ax = plt.subplots(figsize=(5.8, 3.5), constrained_layout=True)
    y = list(range(len(order)))
    colors = ["#7f8c8d"] * (len(order) - 1) + ["#2b6cb0"]
    ax.barh(y, means, xerr=errors, color=colors, alpha=0.92, capsize=3, error_kw={"linewidth": 1.0})
    ax.set_yticks(y, [labels[method] for method in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("held-out $R^2$")
    ax.set_title(r"Representational reach at $\epsilon=8$", fontsize=10)
    ax.grid(axis="x", alpha=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def plot_gain(rows: list[dict[str, str]], output: Path) -> None:
    grouped: dict[tuple[int, int], dict[float, float]] = defaultdict(dict)
    for row in rows:
        grouped[(int(row["context_seed"]), int(row["design_seed"]))][float(row["epsilon"])] = float(row["fitted_gain"])
    fig, ax = plt.subplots(figsize=(3.35, 2.35), constrained_layout=True)
    for values in grouped.values():
        ax.plot([5, 8], [values[5.0], values[8.0]], color="0.68", linewidth=0.8, marker="o", markersize=2.8)
    means = [statistics.mean(values[epsilon] for values in grouped.values()) for epsilon in (5.0, 8.0)]
    ax.plot([5, 8], means, color="#2b6cb0", linewidth=2.0, marker="o", markersize=4.5, label="mean")
    ax.set_xticks([5, 8], [r"$\epsilon=5$", r"$\epsilon=8$"])
    ax.set_ylabel("fitted scalar gain")
    ax.set_ylim(0.65, 1.05)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=7.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=320)
    plt.close(fig)


def plot_noise(rows: list[dict[str, str]], output: Path) -> None:
    colors = {5.0: "#1f77b4", 8.0: "#ff7f0e"}
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), constrained_layout=True)
    for epsilon in (5.0, 8.0):
        selected = sorted(
            (
                row
                for row in rows
                if row["method"] == "lifted_omp" and float(row["epsilon"]) == epsilon
            ),
            key=lambda row: float(row["noise_std"]),
        )
        noise = [float(row["noise_std"]) for row in selected]
        label = rf"$\epsilon={epsilon:g}$"
        axes[0].errorbar(
            noise,
            [float(row["heldout_r2_mean"]) for row in selected],
            yerr=[float(row["heldout_r2_sem"]) for row in selected],
            color=colors[epsilon],
            marker="o",
            linewidth=1.4,
            capsize=3,
            label=label,
        )
        axes[1].errorbar(
            noise,
            [float(row["pair_topk_recall_mean"]) for row in selected],
            yerr=[float(row["pair_topk_recall_sem"]) for row in selected],
            color=colors[epsilon],
            marker="o",
            linewidth=1.4,
            capsize=3,
            label=label,
        )

    axes[0].axhline(0.95, color="0.45", linestyle="--", linewidth=1, label="target 0.95")
    axes[0].set_title("Finite-effect prediction", fontsize=10)
    axes[0].set_ylabel("held-out $R^2$")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")

    axes[1].axhline(0.99, color="0.45", linestyle="--", linewidth=1, label="target 0.99")
    axes[1].set_title("Pair support recovery", fontsize=10)
    axes[1].set_ylabel("top-$k$ pair recovery")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower left")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("measurement noise std")
        ax.grid(alpha=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observer-csv", type=Path, required=True)
    parser.add_argument("--reach-csv", type=Path, required=True)
    parser.add_argument("--calibration-csv", type=Path, required=True)
    parser.add_argument("--noise-summary-csv", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_observer(read_csv(args.observer_csv), args.figure_dir / "observer_control.png")
    plot_reach(read_csv(args.reach_csv), args.figure_dir / "claim3_bar_bad.png")
    plot_gain(read_csv(args.calibration_csv), args.figure_dir / "calibration_factorial_gain_pairs.png")
    plot_noise(read_csv(args.noise_summary_csv), args.figure_dir / "claim3_noise_clean.png")


if __name__ == "__main__":
    main()
