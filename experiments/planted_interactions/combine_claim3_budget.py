#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
Combine Claim-3 budget-sweep runs.

Expected input layout, e.g.:
  runs/claim3_budget_m16/claim3_results.csv
  runs/claim3_budget_m24/claim3_results.csv
  ...

Each run directory may also contain claim3_metadata.json. If present, this
script reads args.measurements, n_components, true interaction pairs, etc.
If absent, it falls back to parsing the parent directory name or the results
CSV budget_forward column.

Outputs:
  combined_claim3_budget_results.csv
  claim3_budget_summary.csv
  claim3_budget_thresholds.csv
  claim3_budget_metadata_summary.json
  several PNG plots
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Headless plotting; works on Mac, Colab, and servers.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_METHOD_ORDER = [
    "raw_atp",
    "scalar_cal_atp",
    "multigain_atp",
    "finite_single",
    "subset_ridge",
    "first_order_omp",
    "lifted_omp",
]


def _read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse JSON: {path}: {e}") from e


def _extract_measurements_from_dir(path: Path) -> Optional[int]:
    """Try to parse m from parent names like claim3_budget_m64."""
    for part in [path.parent.name, path.name, *[p.name for p in path.parents[:3]]]:
        # Accept m64, _m64, budget64, budget_m64, measurements64.
        m = re.search(r"(?:^|[_\-])(m|budget|measurements)[_\-]?(\d+)(?:$|[_\-])", part)
        if m:
            return int(m.group(2))
        m2 = re.search(r"claim3_budget_m(\d+)", part)
        if m2:
            return int(m2.group(1))
    return None


def _get_nested(d: dict, keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def infer_metadata(results_path: Path, df: pd.DataFrame, metadata_name: str) -> Dict[str, Any]:
    run_dir = results_path.parent
    metadata = _read_json(run_dir / metadata_name) or {}
    args = metadata.get("args", {}) if isinstance(metadata, dict) else {}

    m = args.get("measurements")
    if m is None:
        m = _extract_measurements_from_dir(results_path)
    if m is None:
        # For learned aggregate methods this is usually the run measurement budget.
        learned = df[df["method"].isin(["lifted_omp", "first_order_omp", "subset_ridge"])]
        if "budget_forward" in learned.columns and not learned.empty:
            vals = learned["budget_forward"].dropna().astype(int).unique()
            if len(vals):
                m = int(np.max(vals))
    if m is None:
        raise ValueError(
            f"Could not infer measurement budget for {results_path}. "
            f"Use directory names like claim3_budget_m64 or include claim3_metadata.json."
        )

    n_components = args.get("n_components")
    if n_components is None:
        n_components = _get_nested(metadata, ["metas", 0, "n_components"], None)
    if n_components is None:
        # finite_single budget is often n_components in claim3 script.
        fs = df[df["method"] == "finite_single"]
        if not fs.empty and "budget_forward" in fs.columns:
            vals = fs["budget_forward"].dropna().astype(int).unique()
            if len(vals):
                n_components = int(np.max(vals))
    if n_components is None:
        n_components = int(df.get("n_components", pd.Series([np.nan])).dropna().iloc[0]) if "n_components" in df else None

    interaction_pairs = None
    gt = _get_nested(metadata, ["metas", 0, "ground_truth"], None)
    if isinstance(gt, dict):
        interaction_pairs = gt.get("interaction_pairs")

    n_pairs = None
    if n_components is not None:
        n_pairs = int(n_components * (n_components - 1) // 2)
    lifted_dim = None
    if n_components is not None and n_pairs is not None:
        lifted_dim = int(n_components + n_pairs)

    support_guess = None
    if interaction_pairs is not None:
        # Main effects plus planted pairs. This is only a heuristic for CS reference line.
        main_indices = gt.get("main_indices", []) if isinstance(gt, dict) else []
        support_guess = len(main_indices) + len(interaction_pairs)

    return {
        "source_file": str(results_path),
        "source_dir": str(run_dir),
        "budget_measurements": int(m),
        "n_components": int(n_components) if n_components is not None else None,
        "n_pairs": n_pairs,
        "lifted_dim": lifted_dim,
        "interaction_pairs": interaction_pairs,
        "support_guess": support_guess,
        "args": args,
    }


def expand_inputs(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            # Allow literal file paths that glob did not expand.
            p = Path(pat)
            if p.exists():
                paths.append(p)
    # De-duplicate, stable sorted by path string.
    uniq = sorted({str(p): p for p in paths}.values(), key=lambda p: str(p))
    return uniq


def safe_sem(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) <= 1:
        return 0.0
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "heldout_r2",
        "mae",
        "rmse",
        "pair_topk_recall",
        "confound_abs_coef",
        "true_pair_abs_coef_mean",
        "main_topk_overlap",
        "omp_k",
        "omp_val_r2",
        "budget_forward",
        "budget_backward",
    ]
    metrics = [m for m in metrics if m in df.columns]
    group_cols = ["budget_measurements", "epsilon", "method"]
    agg_spec = {}
    for m in metrics:
        agg_spec[f"{m}_mean"] = (m, "mean")
        agg_spec[f"{m}_std"] = (m, "std")
        agg_spec[f"{m}_sem"] = (m, safe_sem)
    out = df.groupby(group_cols, dropna=False).agg(
        n=("heldout_r2", "count"),
        **agg_spec,
    ).reset_index()
    out = out.sort_values(["epsilon", "method", "budget_measurements"])
    return out


def make_thresholds(summary: pd.DataFrame, r2_threshold: float, pair_threshold: float) -> pd.DataFrame:
    rows = []
    for (eps, method), sub in summary.groupby(["epsilon", "method"]):
        sub = sub.sort_values("budget_measurements")
        r2_cross = sub[sub["heldout_r2_mean"] >= r2_threshold]
        pair_col = "pair_topk_recall_mean"
        pair_cross = sub[sub[pair_col] >= pair_threshold] if pair_col in sub else pd.DataFrame()
        both = sub[(sub["heldout_r2_mean"] >= r2_threshold) & (sub.get(pair_col, pd.Series(0, index=sub.index)) >= pair_threshold)]
        rows.append({
            "epsilon": eps,
            "method": method,
            f"first_budget_r2_ge_{r2_threshold}": int(r2_cross["budget_measurements"].iloc[0]) if not r2_cross.empty else np.nan,
            f"first_budget_pair_ge_{pair_threshold}": int(pair_cross["budget_measurements"].iloc[0]) if not pair_cross.empty else np.nan,
            f"first_budget_both_ge_r2_{r2_threshold}_pair_{pair_threshold}": int(both["budget_measurements"].iloc[0]) if not both.empty else np.nan,
            "max_heldout_r2_mean": float(sub["heldout_r2_mean"].max()),
            "max_pair_topk_recall_mean": float(sub[pair_col].max()) if pair_col in sub else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["epsilon", "method"])


def select_eps(summary: pd.DataFrame, focus_epsilons: Optional[Sequence[float]]) -> List[float]:
    eps = sorted(float(e) for e in summary["epsilon"].dropna().unique())
    if focus_epsilons:
        wanted = set(round(float(e), 10) for e in focus_epsilons)
        got = [e for e in eps if round(e, 10) in wanted]
        if got:
            return got
    # Prefer hard-regime epsilons if present, else all.
    hard = [e for e in eps if e >= 5]
    return hard if hard else eps


def add_reference_lines(ax, n_components: Optional[int], lifted_dim: Optional[int], support_k: int, x_max: float, include_pair_baseline: bool = True):
    if n_components:
        ax.axvline(n_components, linestyle="--", linewidth=1)
        ax.text(n_components, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"n={n_components}", rotation=90, va="bottom", ha="right", fontsize=8)
    if lifted_dim and support_k and support_k > 0:
        cs = support_k * math.log(max(lifted_dim / support_k, math.e))
        ax.axvspan(max(1, cs * 0.75), cs * 1.25, alpha=0.12)
        ax.text(cs, ax.get_ylim()[1] - 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"k log(N/k)≈{cs:.0f}", rotation=90, va="top", ha="center", fontsize=8)
    if include_pair_baseline and n_components:
        pair_budget = n_components * (n_components - 1) // 2
        if pair_budget > x_max:
            ax.annotate(f"exhaustive pairs={pair_budget}",
                        xy=(x_max, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                        xytext=(0.65, 0.08), textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", lw=1), fontsize=9)
        else:
            ax.axvline(pair_budget, linestyle=":", linewidth=1)
            ax.text(pair_budget, ax.get_ylim()[0], f"pairs={pair_budget}", rotation=90, va="bottom", ha="right", fontsize=8)


def plot_metric_vs_budget(
    summary: pd.DataFrame,
    outpath: Path,
    metric: str,
    methods: Sequence[str],
    epsilons: Sequence[float],
    ylabel: str,
    title: str,
    n_components: Optional[int],
    lifted_dim: Optional[int],
    support_k: int,
    ylim: Optional[Tuple[float, float]] = None,
):
    mean_col = f"{metric}_mean"
    sem_col = f"{metric}_sem"
    if mean_col not in summary.columns:
        return
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for eps in epsilons:
        for method in methods:
            sub = summary[(summary["epsilon"].astype(float) == float(eps)) & (summary["method"] == method)].sort_values("budget_measurements")
            if sub.empty:
                continue
            label = f"{method}, eps={eps:g}" if len(epsilons) > 1 else method
            yerr = sub[sem_col] if sem_col in sub else None
            ax.errorbar(sub["budget_measurements"], sub[mean_col], yerr=yerr, marker="o", capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("aggregate measurements used for fitting")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, which="both", alpha=0.25)
    x_max = float(summary["budget_measurements"].max())
    add_reference_lines(ax, n_components, lifted_dim, support_k, x_max)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_lifted_phase(summary: pd.DataFrame, outpath: Path, epsilons: Sequence[float], n_components: Optional[int], lifted_dim: Optional[int], support_k: int, r2_threshold: float, pair_threshold: float):
    fig, ax1 = plt.subplots(figsize=(9.5, 6.0))
    method = "lifted_omp"
    for eps in epsilons:
        sub = summary[(summary["epsilon"].astype(float) == float(eps)) & (summary["method"] == method)].sort_values("budget_measurements")
        if sub.empty:
            continue
        ax1.errorbar(sub["budget_measurements"], sub["heldout_r2_mean"], yerr=sub.get("heldout_r2_sem"), marker="o", capsize=3, label=f"R² eps={eps:g}")
    ax1.axhline(r2_threshold, linestyle="--", linewidth=1)
    ax1.set_xscale("log")
    ax1.set_xlabel("aggregate measurements used for fitting")
    ax1.set_ylabel("lifted OMP held-out R²")
    ax1.set_title("Lifted tomography budget curve")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.set_ylim(-0.05, 1.05)
    x_max = float(summary["budget_measurements"].max())
    add_reference_lines(ax1, n_components, lifted_dim, support_k, x_max)
    ax1.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def write_markdown_report(
    outpath: Path,
    meta_summary: Dict[str, Any],
    thresholds: pd.DataFrame,
    focus_epsilons: Sequence[float],
    r2_threshold: float,
    pair_threshold: float,
):
    lines = []
    lines.append("# Claim 3 Budget Sweep Summary\n")
    lines.append("## Setup inferred from inputs\n")
    for k in ["n_components", "n_pairs", "lifted_dim", "support_k", "cs_heuristic", "exhaustive_pair_budget", "num_input_files"]:
        lines.append(f"- **{k}**: {meta_summary.get(k)}")
    lines.append("\n## Threshold crossings\n")
    lines.append(f"Thresholds: held-out R² ≥ {r2_threshold}; pair recall ≥ {pair_threshold}.\n")
    for eps in focus_epsilons:
        sub = thresholds[thresholds["epsilon"].astype(float) == float(eps)]
        if sub.empty:
            continue
        lines.append(f"### epsilon = {eps:g}\n")
        keep_cols = [c for c in thresholds.columns if c.startswith("first_budget") or c in ["method", "max_heldout_r2_mean", "max_pair_topk_recall_mean"]]
        lines.append(sub[keep_cols].to_markdown(index=False))
        lines.append("")
    lines.append("\n## How to read this\n")
    lines.append("- `lifted_omp` is the key Claim-3 method: it estimates main effects plus pair terms.")
    lines.append("- `first_order_omp`, AtP variants, and `finite_single` cannot recover pure pair terms by construction.")
    lines.append("- Compare the first budget where `lifted_omp` reaches both thresholds to the CS heuristic `k log(N/k)` and the exhaustive-pair baseline `C(n,2)`.\n")
    outpath.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Combine Claim-3 budget-sweep result directories and make budget-curve plots.")
    parser.add_argument("--inputs", nargs="+", default=["runs/claim3_budget_m*/claim3_results.csv"], help="Glob(s) or file paths to claim3_results.csv files. Quote globs in the shell.")
    parser.add_argument("--outdir", default="runs/claim3_budget_combined", help="Output directory for combined CSVs and plots.")
    parser.add_argument("--metadata-name", default="claim3_metadata.json", help="Metadata JSON filename inside each run directory.")
    parser.add_argument("--focus-epsilons", nargs="*", type=float, default=[5.0, 8.0], help="Epsilons to emphasize in plots.")
    parser.add_argument("--methods", nargs="*", default=["lifted_omp", "first_order_omp", "scalar_cal_atp", "finite_single", "subset_ridge"], help="Methods to include in method comparison plots.")
    parser.add_argument("--r2-threshold", type=float, default=0.95)
    parser.add_argument("--pair-threshold", type=float, default=0.99)
    parser.add_argument("--support-k", type=int, default=None, help="Support size k for CS reference line. If omitted, inferred as len(main_indices)+len(interaction_pairs), else 5 fallback.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit(
            "No input files matched. Try, for example:\n"
            "  python combine_claim3_budget.py --inputs 'runs/claim3_budget_m*/claim3_results.csv'"
        )

    frames = []
    run_metas = []
    for p in paths:
        df = pd.read_csv(p)
        required = {"method", "heldout_r2", "seed", "epsilon"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{p} is missing required columns: {sorted(missing)}")
        meta = infer_metadata(p, df, args.metadata_name)
        run_metas.append(meta)
        df = df.copy()
        df["source_file"] = meta["source_file"]
        df["source_dir"] = meta["source_dir"]
        df["budget_measurements"] = int(meta["budget_measurements"])
        df["n_components"] = meta.get("n_components")
        df["n_pairs"] = meta.get("n_pairs")
        df["lifted_dim"] = meta.get("lifted_dim")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["budget_measurements", "epsilon", "seed", "method"])
    combined.to_csv(outdir / "combined_claim3_budget_results.csv", index=False)

    summary = aggregate(combined)
    summary.to_csv(outdir / "claim3_budget_summary.csv", index=False)

    thresholds = make_thresholds(summary, args.r2_threshold, args.pair_threshold)
    thresholds.to_csv(outdir / "claim3_budget_thresholds.csv", index=False)

    # Infer global metadata from run metas.
    n_components_vals = [m.get("n_components") for m in run_metas if m.get("n_components") is not None]
    n_components = int(pd.Series(n_components_vals).mode().iloc[0]) if n_components_vals else None
    n_pairs = n_components * (n_components - 1) // 2 if n_components else None
    lifted_dim = n_components + n_pairs if n_components and n_pairs is not None else None
    support_guesses = [m.get("support_guess") for m in run_metas if m.get("support_guess")]
    support_k = args.support_k or (int(pd.Series(support_guesses).mode().iloc[0]) if support_guesses else 5)
    cs_heuristic = support_k * math.log(max((lifted_dim or support_k) / support_k, math.e)) if lifted_dim else None
    meta_summary = {
        "num_input_files": len(paths),
        "input_files": [str(p) for p in paths],
        "n_components": n_components,
        "n_pairs": n_pairs,
        "lifted_dim": lifted_dim,
        "support_k": support_k,
        "cs_heuristic": cs_heuristic,
        "exhaustive_pair_budget": n_pairs,
        "budgets": sorted(int(x) for x in combined["budget_measurements"].unique()),
        "epsilons": sorted(float(x) for x in combined["epsilon"].unique()),
        "methods": sorted(combined["method"].unique().tolist()),
    }
    with (outdir / "claim3_budget_metadata_summary.json").open("w") as f:
        json.dump(meta_summary, f, indent=2)

    focus_eps = select_eps(summary, args.focus_epsilons)

    plot_lifted_phase(
        summary,
        outdir / "claim3_lifted_omp_budget_r2.png",
        focus_eps,
        n_components,
        lifted_dim,
        support_k,
        args.r2_threshold,
        args.pair_threshold,
    )

    plot_metric_vs_budget(
        summary,
        outdir / "claim3_lifted_omp_pair_recall.png",
        "pair_topk_recall",
        ["lifted_omp", "first_order_omp"],
        focus_eps,
        "top-k pair recovery",
        "Pair recovery vs measurement budget",
        n_components,
        lifted_dim,
        support_k,
        ylim=(-0.05, 1.05),
    )

    plot_metric_vs_budget(
        summary,
        outdir / "claim3_methods_budget_r2.png",
        "heldout_r2",
        args.methods,
        focus_eps,
        "held-out subset/intervention R²",
        "Prediction quality vs measurement budget",
        n_components,
        lifted_dim,
        support_k,
        ylim=(-0.05, 1.05),
    )

    plot_metric_vs_budget(
        summary,
        outdir / "claim3_confound_vs_budget.png",
        "confound_abs_coef",
        args.methods,
        focus_eps,
        "absolute off-path confound coefficient",
        "Confound false positives vs measurement budget",
        n_components,
        lifted_dim,
        support_k,
        ylim=None,
    )

    # A compact table image for the key method in hard epsilons.
    key = summary[(summary["method"] == "lifted_omp") & (summary["epsilon"].astype(float).isin([float(e) for e in focus_eps]))]
    key.to_csv(outdir / "claim3_lifted_omp_key_curve.csv", index=False)

    write_markdown_report(outdir / "claim3_budget_report.md", meta_summary, thresholds, focus_eps, args.r2_threshold, args.pair_threshold)

    print("Wrote:")
    for name in [
        "combined_claim3_budget_results.csv",
        "claim3_budget_summary.csv",
        "claim3_budget_thresholds.csv",
        "claim3_budget_metadata_summary.json",
        "claim3_budget_report.md",
        "claim3_lifted_omp_budget_r2.png",
        "claim3_lifted_omp_pair_recall.png",
        "claim3_methods_budget_r2.png",
        "claim3_confound_vs_budget.png",
    ]:
        print(f"  {outdir / name}")

    # Print the most relevant threshold lines.
    hard = thresholds[(thresholds["method"] == "lifted_omp") & (thresholds["epsilon"].astype(float).isin([float(e) for e in focus_eps]))]
    if not hard.empty:
        print("\nLifted OMP threshold crossings:")
        print(hard.to_string(index=False))

    if cs_heuristic is not None:
        print(f"\nReference: N={lifted_dim}, k≈{support_k}, k log(N/k)≈{cs_heuristic:.1f}, exhaustive pairs={n_pairs}")


if __name__ == "__main__":
    main()
