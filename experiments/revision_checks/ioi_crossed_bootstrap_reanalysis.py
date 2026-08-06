#!/usr/bin/env python3
"""Crossed bootstrap for the saved IOI direct-effect robustness measurements."""

# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "D_P",
    "D_B",
    "D_E",
    "D_PB",
    "D_PE",
    "D_BE",
    "D_PBE",
    "I_PB",
    "I_PE",
    "I_BE",
    "I_PB_per_pair",
    "I_PE_per_pair",
    "I_BE_per_pair",
    "I_PE_minus_PB",
    "I_PB_minus_BE",
    "I_PE_minus_BE",
    "I_PBE_third_order",
)


def build_arrays(metrics: pd.DataFrame) -> tuple[list[str], list[str], list[str], np.ndarray]:
    frames = sorted(metrics["frame_id"].unique())
    orders = ["ABBA", "BABA"]
    ablations = ["mean", "zero"]
    prompt_ids = sorted(metrics["prompt_idx"].unique())
    if prompt_ids != list(range(len(prompt_ids))):
        raise ValueError("prompt_idx must be a shared contiguous name-pair index")
    arrays = np.empty((len(frames), len(orders), len(ablations), len(prompt_ids), len(METRICS)))
    for frame_index, frame_id in enumerate(frames):
        for order_index, order in enumerate(orders):
            for ablation_index, ablation in enumerate(ablations):
                selected = metrics[
                    (metrics["frame_id"] == frame_id)
                    & (metrics["order"] == order)
                    & (metrics["ablation"] == ablation)
                ].sort_values("prompt_idx")
                if len(selected) != len(prompt_ids):
                    raise ValueError(f"incomplete cell: {frame_id}, {order}, {ablation}")
                arrays[frame_index, order_index, ablation_index] = selected[list(METRICS)].to_numpy(float)
    return frames, orders, ablations, arrays


def crossed_bootstrap(metrics: pd.DataFrame, repeats: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, _orders, ablations, arrays = build_arrays(metrics)
    rng = np.random.default_rng(seed)
    samples = np.empty((repeats, len(ablations), len(METRICS)))
    n_frames, _n_orders, _n_ablations, n_pairs, _n_metrics = arrays.shape
    for repeat in range(repeats):
        frame_indices = rng.integers(0, n_frames, n_frames)
        pair_indices = rng.integers(0, n_pairs, n_pairs)
        draw = arrays[frame_indices, :, :, :, :][:, :, :, pair_indices, :]
        samples[repeat] = draw.mean(axis=(0, 1, 3))

    sample_rows = []
    summary_rows = []
    for ablation_index, ablation in enumerate(ablations):
        for metric_index, metric in enumerate(METRICS):
            values = samples[:, ablation_index, metric_index]
            sample_rows.extend(
                {
                    "bootstrap": repeat,
                    "ablation": ablation,
                    "metric": metric,
                    "value": value,
                }
                for repeat, value in enumerate(values)
            )
            summary_rows.append(
                {
                    "ablation": ablation,
                    "metric": metric,
                    "n_bootstrap": repeats,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "q025": float(np.quantile(values, 0.025)),
                    "q975": float(np.quantile(values, 0.975)),
                    "p_gt_zero": float(np.mean(values > 0.0)),
                    "resampled_lexical_frames": len(frames),
                    "retained_orders_per_frame": 2,
                    "shared_name_pairs": n_pairs,
                }
            )
    return pd.DataFrame(sample_rows), pd.DataFrame(summary_rows)


def leave_one_frame_out(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for heldout_frame in sorted(metrics["frame_id"].unique()):
        for ablation in ("mean", "zero"):
            remaining = metrics[
                (metrics["frame_id"] != heldout_frame) & (metrics["ablation"] == ablation)
            ]
            for metric in ("I_PB", "I_PE", "I_BE", "I_PE_minus_PB"):
                rows.append(
                    {
                        "heldout_frame": heldout_frame,
                        "ablation": ablation,
                        "metric": metric,
                        "remaining_frames_mean": float(remaining[metric].mean()),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    metrics = pd.read_csv(args.input_csv)
    samples, summary = crossed_bootstrap(metrics, args.repeats, args.seed)
    frame_loto = leave_one_frame_out(metrics)
    args.outdir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.outdir / "ioi_robustness_crossed_bootstrap.csv", index=False)
    summary.to_csv(args.outdir / "ioi_robustness_crossed_bootstrap_summary.csv", index=False)
    frame_loto.to_csv(args.outdir / "ioi_robustness_leave_one_frame_out.csv", index=False)
    selected = summary[
        summary["metric"].isin(["I_PB", "I_PE", "I_BE", "I_PE_minus_PB", "I_PBE_third_order"])
    ]
    print(selected[["ablation", "metric", "mean", "q025", "q975", "p_gt_zero"]].to_string(index=False))


if __name__ == "__main__":
    main()
