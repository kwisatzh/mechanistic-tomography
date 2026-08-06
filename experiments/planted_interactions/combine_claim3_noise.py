#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
Combine Claim-3 noise-sweep runs whose per-run outputs have the same file names.

Expected layout, for example:
  runs/claim3_noise_0.01/claim3_results.csv
  runs/claim3_noise_0.01/claim3_metadata.json
  runs/claim3_noise_0.05/claim3_results.csv
  ...

Usage:
  python combine_claim3_noise.py \
    --inputs 'runs/claim3_noise_*/claim3_results.csv' \
    --outdir runs/claim3_noise_combined \
    --focus-epsilons 5 8
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_METHOD_ORDER = [
    "lifted_omp",
    "first_order_omp",
    "scalar_cal_atp",
    "multigain_atp",
    "raw_atp",
    "finite_single",
    "subset_ridge",
]


def expand_inputs(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            p = Path(pat)
            if p.exists():
                paths.append(p)
    # deterministic, unique
    return sorted(set(paths), key=lambda p: str(p))


def parse_noise_from_name(path: Path) -> Optional[float]:
    """Fallback parser for names like claim3_noise_0.05 or noise-0.1."""
    text = str(path.parent)
    m = re.search(r"noise[_=-]?([0-9]+(?:\.[0-9]+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def load_metadata(results_path: Path) -> dict:
    meta_path = results_path.parent / "claim3_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def infer_noise(results_path: Path, meta: dict) -> float:
    # Preferred: top-level args.noise_std in metadata.
    try:
        val = meta.get("args", {}).get("noise_std", None)
        if val is not None:
            return float(val)
    except Exception:
        pass
    # Sometimes in per-epsilon metas.
    try:
        metas = meta.get("metas", [])
        if metas and "noise_std" in metas[0]:
            return float(metas[0]["noise_std"])
    except Exception:
        pass
    # Fallback to directory name.
    val = parse_noise_from_name(results_path)
    if val is not None:
        return val
    raise ValueError(f"Could not infer noise_std for {results_path}")


def infer_arg(meta: dict, name: str, default=None):
    try:
        val = meta.get("args", {}).get(name, default)
        return val
    except Exception:
        return default


def combine(paths: List[Path]) -> tuple[pd.DataFrame, dict]:
    rows = []
    meta_summaries = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        meta = load_metadata(p)
        noise = infer_noise(p, meta)
        df = pd.read_csv(p)
        df["noise_std"] = noise
        df["source_dir"] = str(p.parent)
        df["source_file"] = str(p)
        for col, meta_name in [
            ("measurements", "measurements"),
            ("holdout_measurements", "holdout_measurements"),
            ("mask_density", "mask_density"),
            ("mode", "mode"),
            ("n_components", "n_components"),
        ]:
            if col not in df.columns:
                val = infer_arg(meta, meta_name, None)
                if val is not None:
                    df[col] = val
        rows.append(df)
        meta_summaries.append({
            "source_file": str(p),
            "source_dir": str(p.parent),
            "noise_std": noise,
            "measurements": infer_arg(meta, "measurements", None),
            "holdout_measurements": infer_arg(meta, "holdout_measurements", None),
            "n_components": infer_arg(meta, "n_components", None),
            "epsilons": infer_arg(meta, "epsilons", None),
            "mode": infer_arg(meta, "mode", None),
            "mask_density": infer_arg(meta, "mask_density", None),
        })
    combined = pd.concat(rows, ignore_index=True)
    combined["epsilon"] = combined["epsilon"].astype(float)
    combined["noise_std"] = combined["noise_std"].astype(float)
    return combined, {"input_files": meta_summaries}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        c for c in ["heldout_r2", "pair_topk_recall", "confound_abs_coef", "selected_k", "train_r2", "val_r2"]
        if c in df.columns
    ]
    grouped = df.groupby(["noise_std", "epsilon", "method"], dropna=False)
    out = grouped[metrics].agg(["mean", "std", "count"]).reset_index()
    # flatten columns
    out.columns = ["_".join([str(x) for x in tup if x]) for tup in out.columns.to_flat_index()]
    # add sem columns
    for m in metrics:
        std_col = f"{m}_std"
        count_col = f"{m}_count"
        if std_col in out and count_col in out:
            out[f"{m}_sem"] = out[std_col] / np.sqrt(out[count_col].clip(lower=1))
    return out


def threshold_table(summary: pd.DataFrame, r2_threshold: float, pair_threshold: float) -> pd.DataFrame:
    rows = []
    for (method, eps), sub in summary.groupby(["method", "epsilon"]):
        sub = sub.sort_values("noise_std")
        r2_ok = sub[sub.get("heldout_r2_mean", pd.Series(dtype=float)) >= r2_threshold]
        pair_ok = sub[sub.get("pair_topk_recall_mean", pd.Series(dtype=float)) >= pair_threshold]
        both_ok = sub[(sub.get("heldout_r2_mean", pd.Series(dtype=float)) >= r2_threshold) &
                      (sub.get("pair_topk_recall_mean", pd.Series(dtype=float)) >= pair_threshold)]
        rows.append({
            "method": method,
            "epsilon": eps,
            f"max_noise_r2_ge_{r2_threshold:.2f}": r2_ok["noise_std"].max() if len(r2_ok) else np.nan,
            f"max_noise_pair_ge_{pair_threshold:.2f}": pair_ok["noise_std"].max() if len(pair_ok) else np.nan,
            f"max_noise_both_ge_r2_{r2_threshold:.2f}_pair_{pair_threshold:.2f}": both_ok["noise_std"].max() if len(both_ok) else np.nan,
            "max_heldout_r2_mean": sub.get("heldout_r2_mean", pd.Series(dtype=float)).max(),
            "max_pair_topk_recall_mean": sub.get("pair_topk_recall_mean", pd.Series(dtype=float)).max(),
        })
    return pd.DataFrame(rows).sort_values(["epsilon", "method"])


def sem_array(sub: pd.DataFrame, metric: str) -> Optional[np.ndarray]:
    col = f"{metric}_sem"
    if col in sub:
        vals = sub[col].to_numpy(dtype=float)
        if np.all(np.isfinite(vals)):
            return vals
    return None


def plot_metric_vs_noise(summary: pd.DataFrame, metric: str, methods: List[str], epsilons: List[float], outpath: Path, title: str, ylabel: str):
    if f"{metric}_mean" not in summary.columns:
        return
    plt.figure(figsize=(9, 5.4))
    for eps in epsilons:
        for method in methods:
            sub = summary[(summary["epsilon"] == float(eps)) & (summary["method"] == method)].sort_values("noise_std")
            if sub.empty:
                continue
            label = f"{method}, eps={eps:g}"
            x = sub["noise_std"].to_numpy(dtype=float)
            y = sub[f"{metric}_mean"].to_numpy(dtype=float)
            yerr = sem_array(sub, metric)
            plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=label)
    plt.xlabel("measurement noise std")
    plt.ylabel(ylabel)
    plt.title(title)
    if len(summary["noise_std"].unique()) > 2 and summary["noise_std"].min() > 0:
        plt.xscale("log")
    plt.axhline(0, linestyle="--", linewidth=1)
    if metric == "heldout_r2":
        plt.axhline(0.95, linestyle="--", linewidth=1)
    if metric == "pair_topk_recall":
        plt.axhline(0.99, linestyle="--", linewidth=1)
        plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_lifted_summary(summary: pd.DataFrame, epsilons: List[float], outdir: Path):
    # Separate, clean lifted-only plots.
    for metric, ylabel, fname, title in [
        ("heldout_r2", "lifted OMP held-out R²", "claim3_noise_lifted_omp_r2.png", "Lifted OMP noise robustness"),
        ("pair_topk_recall", "top-k pair recovery", "claim3_noise_lifted_omp_pair_recall.png", "Lifted OMP pair recovery vs noise"),
    ]:
        plot_metric_vs_noise(summary, metric, ["lifted_omp"], epsilons, outdir / fname, title, ylabel)


def make_report(summary: pd.DataFrame, thresholds: pd.DataFrame, meta: dict, outdir: Path, r2_threshold: float, pair_threshold: float):
    lines = []
    lines.append("# Claim 3 Noise Sweep Summary\n")
    lines.append("## Inputs\n")
    lines.append(f"- num_input_files: {len(meta['input_files'])}\n")
    noises = sorted(summary["noise_std"].unique())
    eps = sorted(summary["epsilon"].unique())
    lines.append(f"- noise_std values: {noises}\n")
    lines.append(f"- epsilons: {eps}\n")
    # Get common inferred info from first meta.
    first = meta["input_files"][0] if meta["input_files"] else {}
    for k in ["measurements", "holdout_measurements", "n_components", "mode", "mask_density"]:
        if first.get(k) is not None:
            lines.append(f"- {k}: {first.get(k)}\n")
    lines.append("\n## Threshold robustness\n")
    lines.append(f"Thresholds: held-out R² ≥ {r2_threshold}; pair recall ≥ {pair_threshold}.\n\n")
    for eps_val in eps:
        lines.append(f"### epsilon = {eps_val:g}\n\n")
        cols = ["method", "epsilon"] + [c for c in thresholds.columns if c not in {"method", "epsilon"}]
        sub = thresholds[thresholds["epsilon"] == eps_val][cols]
        lines.append(sub.to_markdown(index=False))
        lines.append("\n\n")
    lines.append("## Reading guide\n")
    lines.append("- `lifted_omp` is the key interaction-recovery method.\n")
    lines.append("- If pair recall remains high while R² falls, support is recovered but coefficient/effect-size estimation has degraded.\n")
    lines.append("- If both pair recall and R² collapse, the noise level exceeds the current design/solver's useful regime.\n")
    (outdir / "claim3_noise_report.md").write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Glob(s) or paths to claim3_results.csv files")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--focus-epsilons", nargs="*", type=float, default=[5.0, 8.0])
    ap.add_argument("--r2-threshold", type=float, default=0.95)
    ap.add_argument("--pair-threshold", type=float, default=0.99)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit("No input files matched. Check your --inputs glob and quote it in the shell.")

    df, meta = combine(paths)
    summary = summarize(df)
    thresholds = threshold_table(summary, args.r2_threshold, args.pair_threshold)

    df.to_csv(outdir / "combined_claim3_noise_results.csv", index=False)
    summary.to_csv(outdir / "claim3_noise_summary.csv", index=False)
    thresholds.to_csv(outdir / "claim3_noise_thresholds.csv", index=False)
    with open(outdir / "claim3_noise_metadata_summary.json", "w") as f:
        json.dump(meta, f, indent=2)

    focus_eps = args.focus_epsilons or sorted(df["epsilon"].unique())

    plot_lifted_summary(summary, focus_eps, outdir)
    plot_metric_vs_noise(summary, "heldout_r2", DEFAULT_METHOD_ORDER, focus_eps,
                         outdir / "claim3_noise_methods_r2.png",
                         "Prediction quality vs measurement noise", "held-out subset/intervention R²")
    plot_metric_vs_noise(summary, "pair_topk_recall", ["lifted_omp", "first_order_omp"], focus_eps,
                         outdir / "claim3_noise_pair_recovery.png",
                         "Pair recovery vs measurement noise", "top-k pair recovery")
    plot_metric_vs_noise(summary, "confound_abs_coef", ["lifted_omp", "subset_ridge", "first_order_omp"], focus_eps,
                         outdir / "claim3_noise_confound.png",
                         "Confound false positives vs measurement noise", "absolute off-path confound coefficient")

    make_report(summary, thresholds, meta, outdir, args.r2_threshold, args.pair_threshold)

    print("Wrote:")
    for p in [
        "combined_claim3_noise_results.csv",
        "claim3_noise_summary.csv",
        "claim3_noise_thresholds.csv",
        "claim3_noise_metadata_summary.json",
        "claim3_noise_report.md",
        "claim3_noise_lifted_omp_r2.png",
        "claim3_noise_lifted_omp_pair_recall.png",
        "claim3_noise_methods_r2.png",
        "claim3_noise_pair_recovery.png",
        "claim3_noise_confound.png",
    ]:
        print("  " + str(outdir / p))


if __name__ == "__main__":
    main()
