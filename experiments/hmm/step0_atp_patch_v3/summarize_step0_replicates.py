#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Summarize multiple Step-0 replicate directories.

Usage:
  python summarize_step0_replicates.py \
    --inputs runs/m4_baseline_seed7/step0_v2_seed*/step0_results.csv \
    --outdir runs/m4_baseline_seed7/step0_v2_replicate_summary
"""
from __future__ import annotations

import argparse
from pathlib import Path
import glob
import json

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="CSV paths or glob patterns for step0_results.csv files")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    paths = []
    for pat in args.inputs:
        matches = glob.glob(pat)
        if matches:
            paths.extend(matches)
        else:
            paths.append(pat)
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No input CSVs found")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["source"] = str(path)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(outdir / "step0_replicates_all.csv", index=False)

    numeric = [c for c in all_df.columns if c not in {"source", "mask_kind"} and pd.api.types.is_numeric_dtype(all_df[c])]
    summary = all_df.groupby("epsilon")[numeric].agg(["mean", "std", "count"])
    summary.to_csv(outdir / "step0_replicates_summary.csv")

    # Flatten for simple plotting.
    flat = summary.copy()
    flat.columns = ["_".join(col).strip("_") for col in flat.columns.values]
    flat = flat.reset_index()

    def err_plot(metric: str, ylabel: str, filename: str) -> None:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in flat:
            return
        plt.figure(figsize=(7, 4.5))
        plt.errorbar(flat["epsilon"], flat[mean_col], yerr=flat.get(std_col, None), marker="o", capsize=3)
        plt.xlabel("finite intervention scale epsilon")
        plt.ylabel(ylabel)
        plt.title(ylabel + " across Step-0 replicates")
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=180)
        plt.close()

    err_plot("finite_on_atp_slope", "slope: finite ≈ g · AtP", "replicate_atp_scale_slope.png")
    err_plot("finite_to_atp_norm_ratio", "norm ratio ||finite|| / ||AtP||", "replicate_atp_norm_ratio.png")
    err_plot("atp_vs_finite_pearson", "AtP vs finite Pearson", "replicate_atp_pearson.png")
    err_plot("atp_finite_calibration_gap", "finite-single R² - AtP R²", "replicate_atp_gap.png")
    err_plot("aggregate_additivity_gap", "1 - finite-single R²", "replicate_additivity_gap.png")

    with open(outdir / "inputs.json", "w") as f:
        json.dump({"inputs": paths}, f, indent=2)

    print(f"Read {len(paths)} CSVs")
    print(f"Wrote {outdir}")
    print(flat[[c for c in ["epsilon", "finite_on_atp_slope_mean", "finite_to_atp_norm_ratio_mean", "atp_vs_finite_pearson_mean", "atp_finite_calibration_gap_mean", "aggregate_additivity_gap_mean"] if c in flat]].to_string(index=False))


if __name__ == "__main__":
    main()
