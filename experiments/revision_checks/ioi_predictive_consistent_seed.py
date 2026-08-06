#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Reanalyse Stage 2c IOI outputs with one fixed cross-validation split.

This script composes the data loading and feature construction primitives from
the packaged Stage 2d analysis.  It makes no model queries.  The fold seed is
held fixed for the displayed point estimate and every paired prompt-bootstrap
replicate; the bootstrap sampling seed controls only prompt resampling.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


MODELS = [
    "additive_head",
    "count_additive",
    "count_plus_PB_count",
    "count_plus_PE_count",
    "count_plus_BE_count",
    "count_plus_all_pairs",
]
PAIR_MODELS = {
    "PB": "count_plus_PB_count",
    "PE": "count_plus_PE_count",
    "BE": "count_plus_BE_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consistent-fold Stage 2d reanalysis of existing Stage 2c outputs"
    )
    parser.add_argument("--stage2d-script", type=Path, required=True)
    parser.add_argument("--input-run", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--fold-seed", type=int, default=777)
    parser.add_argument("--bootstrap-sampling-seed", type=int, default=999)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--fold-audit-count", type=int, default=200)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-6)
    return parser.parse_args()


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("packaged_ioi_stage2d", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Stage 2d script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fold_seeds(count: int, fixed_seed: int) -> List[int]:
    """Return exactly ``count`` nonnegative seeds excluding the fixed seed."""
    seeds: List[int] = []
    candidate = 0
    while len(seeds) < count:
        if candidate != fixed_seed:
            seeds.append(candidate)
        candidate += 1
    return seeds


def make_designs(
    stage2d: ModuleType,
    masks: np.ndarray,
    heads: pd.DataFrame,
    subset: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    designs: Dict[str, np.ndarray] = {}
    columns: Dict[str, List[str]] = {}
    for model in MODELS:
        design, labels = stage2d.build_design(masks, heads, subset, model)
        designs[model] = design
        columns[model] = list(labels)
    return designs, columns


def make_cv_cache(
    stage2d: ModuleType,
    designs: Dict[str, np.ndarray],
    n_rows: int,
    k_folds: int,
    fold_seed: int,
    ridge: float,
) -> Dict[str, List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]]:
    """Precompute linear maps for the fixed ridge fits in each fold."""
    folds = stage2d.kfold_indices(
        n_rows, k_folds, seed=fold_seed, protect_clean=True
    )
    cache: Dict[str, List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = {}
    for model, design in designs.items():
        model_cache = []
        for fold_idx, (train, test) in enumerate(folds):
            x_train = design[train]
            regularizer = ridge * np.eye(design.shape[1], dtype=float)
            regularizer[0, 0] = 0.0
            fit_operator = np.linalg.solve(
                x_train.T @ x_train + regularizer, x_train.T
            )
            prediction_operator = design[test] @ fit_operator
            model_cache.append((fold_idx, train, test, prediction_operator))
        cache[model] = model_cache
    return cache


def predict_from_cache(
    stage2d: ModuleType,
    y: np.ndarray,
    cache: Dict[str, List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    metric_rows = []
    for model in MODELS:
        predictions = np.full(len(y), np.nan, dtype=float)
        for fold_idx, train, test, operator in cache[model]:
            fold_predictions = operator @ y[train]
            predictions[test] = fold_predictions
            prediction_rows.extend(
                {
                    "model": model,
                    "fold": fold_idx,
                    "subset_idx": int(subset_idx),
                    "observed": float(y[subset_idx]),
                    "predicted": float(predicted),
                    "error": float(predicted - y[subset_idx]),
                }
                for subset_idx, predicted in zip(test, fold_predictions)
            )
        mae, rmse, r2 = stage2d.eval_predictions(
            y, predictions, eval_nonclean=True
        )
        metric_rows.append(
            {
                "model": model,
                "n_rows": int(np.isfinite(predictions[1:]).sum()),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def full_fit_coefficients(
    stage2d: ModuleType,
    designs: Dict[str, np.ndarray],
    columns: Dict[str, List[str]],
    y: np.ndarray,
    ridge: float,
) -> pd.DataFrame:
    parts = []
    for model in MODELS:
        coefficients = stage2d.ridge_fit(designs[model], y, ridge)
        parts.append(
            pd.DataFrame(
                {"model": model, "term": columns[model], "coef": coefficients}
            )
        )
    return pd.concat(parts, ignore_index=True)


def bootstrap_metrics(
    stage2d: ModuleType,
    drops: np.ndarray,
    cache: Dict[str, List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]],
    repeats: int,
    sampling_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(sampling_seed)
    rows = []
    for repeat in range(repeats):
        prompt_indices = rng.integers(0, drops.shape[0], size=drops.shape[0])
        resampled_y = drops[prompt_indices].mean(axis=0)
        _, metrics = predict_from_cache(stage2d, resampled_y, cache)
        metrics.insert(0, "bootstrap", repeat)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def interval_summary(
    frame: pd.DataFrame, group_column: str, value_columns: Iterable[str]
) -> pd.DataFrame:
    rows = []
    for group, values in frame.groupby(group_column, sort=False):
        row = {group_column: group, "n": int(len(values))}
        for column in value_columns:
            array = values[column].to_numpy(float)
            row[f"{column}_mean"] = float(np.mean(array))
            row[f"{column}_median"] = float(np.median(array))
            row[f"{column}_q025"] = float(np.quantile(array, 0.025))
            row[f"{column}_q975"] = float(np.quantile(array, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(bootstrap: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pivot = bootstrap.pivot(index="bootstrap", columns="model", values="mae")
    baseline = pivot["count_additive"]
    rows = []
    samples = []
    for model in MODELS:
        if model == "count_additive":
            continue
        delta = (baseline - pivot[model]).to_numpy(float)
        samples.extend(
            {
                "bootstrap": int(repeat),
                "baseline": "count_additive",
                "model": model,
                "delta_mae": float(value),
            }
            for repeat, value in zip(pivot.index, delta)
        )
        rows.append(
            {
                "baseline": "count_additive",
                "model": model,
                "n_bootstrap": int(len(delta)),
                "delta_mae_mean": float(np.mean(delta)),
                "delta_mae_median": float(np.median(delta)),
                "delta_mae_q025": float(np.quantile(delta, 0.025)),
                "delta_mae_q975": float(np.quantile(delta, 0.975)),
                "p_delta_gt_0": float(np.mean(delta > 0)),
                "central_95_interval_excludes_zero": bool(
                    np.quantile(delta, 0.025) > 0
                    or np.quantile(delta, 0.975) < 0
                ),
            }
        )
    return pd.DataFrame(samples), pd.DataFrame(rows)


def audit_fold_seeds(
    stage2d: ModuleType,
    designs: Dict[str, np.ndarray],
    y: np.ndarray,
    seeds: Sequence[int],
    k_folds: int,
    ridge: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_parts = []
    for fold_seed in seeds:
        cache = make_cv_cache(
            stage2d,
            designs,
            len(y),
            k_folds=k_folds,
            fold_seed=fold_seed,
            ridge=ridge,
        )
        _, metrics = predict_from_cache(stage2d, y, cache)
        metrics.insert(0, "fold_seed", fold_seed)
        metric_parts.append(metrics)
    metrics = pd.concat(metric_parts, ignore_index=True)
    metric_summary = interval_summary(metrics, "model", ["mae", "rmse", "r2"])

    pivot = metrics.pivot(index="fold_seed", columns="model", values="mae")
    delta_rows = []
    summary_rows = []
    for pair, model in PAIR_MODELS.items():
        delta = pivot["count_additive"] - pivot[model]
        delta_rows.extend(
            {
                "fold_seed": int(seed),
                "pair": pair,
                "model": model,
                "delta_mae": float(value),
            }
            for seed, value in delta.items()
        )
        sign_flip = bool((delta > 0).any() and (delta <= 0).any())
        positive_fraction = float((delta > 0).mean())
        summary_rows.append(
            {
                "pair": pair,
                "model": model,
                "n_fold_seeds": int(len(delta)),
                "delta_mae_min": float(delta.min()),
                "delta_mae_q025": float(delta.quantile(0.025)),
                "delta_mae_median": float(delta.median()),
                "delta_mae_q975": float(delta.quantile(0.975)),
                "delta_mae_max": float(delta.max()),
                "positive_fraction": positive_fraction,
                "any_sign_flip": sign_flip,
                "material_instability_positive_fraction_lt_0_95": bool(
                    positive_fraction < 0.95
                ),
            }
        )
    return (
        metrics,
        metric_summary,
        pd.DataFrame(delta_rows),
        pd.DataFrame(summary_rows),
    )


def dataframe_text(frame: pd.DataFrame) -> str:
    return "```text\n" + frame.to_string(index=False) + "\n```"


def main() -> None:
    args = parse_args()
    if args.bootstrap_repeats < 1:
        raise ValueError("--bootstrap-repeats must be positive")
    if args.fold_audit_count < 100:
        raise ValueError("--fold-audit-count must be at least 100")

    args.outdir.mkdir(parents=True, exist_ok=True)
    stage2d = load_module(args.stage2d_script)
    heads, subset, masks, drops, y = stage2d.load_stage2c(args.input_run)
    designs, columns = make_designs(stage2d, masks, heads, subset)

    fixed_cache = make_cv_cache(
        stage2d,
        designs,
        len(y),
        k_folds=args.k_folds,
        fold_seed=args.fold_seed,
        ridge=args.ridge,
    )
    predictions, point_metrics = predict_from_cache(stage2d, y, fixed_cache)
    predictions = stage2d.annotate(predictions, subset)
    point_metrics["columns"] = point_metrics["model"].map(
        {model: ",".join(labels) for model, labels in columns.items()}
    )
    point_metrics["n_params"] = point_metrics["model"].map(
        {model: len(labels) for model, labels in columns.items()}
    )
    coefficients = full_fit_coefficients(
        stage2d, designs, columns, y, ridge=args.ridge
    )

    bootstrap = bootstrap_metrics(
        stage2d,
        drops,
        fixed_cache,
        repeats=args.bootstrap_repeats,
        sampling_seed=args.bootstrap_sampling_seed,
    )
    bootstrap_summary = interval_summary(
        bootstrap, "model", ["mae", "rmse", "r2"]
    )
    delta_samples, delta_summary = paired_deltas(bootstrap)

    audit_seeds = fold_seeds(args.fold_audit_count, args.fold_seed)
    (
        seed_metrics,
        seed_metric_summary,
        seed_delta_samples,
        seed_pair_summary,
    ) = audit_fold_seeds(
        stage2d,
        designs,
        y,
        audit_seeds,
        k_folds=args.k_folds,
        ridge=args.ridge,
    )

    point_metrics.to_csv(args.outdir / "point_fit_summary.csv", index=False)
    predictions.to_csv(args.outdir / "point_kfold_predictions.csv", index=False)
    coefficients.to_csv(args.outdir / "full_fit_coefficients.csv", index=False)
    bootstrap.to_csv(args.outdir / "bootstrap_metrics.csv", index=False)
    bootstrap_summary.to_csv(
        args.outdir / "bootstrap_summary_central_95.csv", index=False
    )
    delta_samples.to_csv(
        args.outdir / "paired_delta_mae_samples_vs_count_additive.csv", index=False
    )
    delta_summary.to_csv(
        args.outdir / "paired_delta_mae_summary_central_95.csv", index=False
    )
    seed_metrics.to_csv(args.outdir / "fold_seed_metrics.csv", index=False)
    seed_metric_summary.to_csv(
        args.outdir / "fold_seed_metric_summary_central_95.csv", index=False
    )
    seed_delta_samples.to_csv(
        args.outdir / "fold_seed_pair_delta_mae.csv", index=False
    )
    seed_pair_summary.to_csv(
        args.outdir / "fold_seed_pair_stability_summary.csv", index=False
    )

    fixed_pivot = point_metrics.set_index("model")["mae"]
    fixed_pair_delta = {
        pair: float(fixed_pivot["count_additive"] - fixed_pivot[model])
        for pair, model in PAIR_MODELS.items()
    }
    instability = {
        row.pair: {
            "any_sign_flip": bool(row.any_sign_flip),
            "material_instability": bool(
                row.material_instability_positive_fraction_lt_0_95
            ),
            "positive_fraction": float(row.positive_fraction),
        }
        for row in seed_pair_summary.itertuples(index=False)
    }
    stability_lines = []
    for row in seed_pair_summary.itertuples(index=False):
        reversals = int(round(row.n_fold_seeds * (1.0 - row.positive_fraction)))
        if row.material_instability_positive_fraction_lt_0_95:
            label = "material split instability"
        elif row.any_sign_flip:
            label = "a rare split reversal"
        else:
            label = "no split reversal"
        stability_lines.append(
            f"- {row.pair}: {row.positive_fraction:.1%} of folds improve "
            f"({reversals}/{row.n_fold_seeds} reversals); {label}."
        )
    diagnostics = {
        "input_run": str(args.input_run),
        "packaged_stage2d_script": str(args.stage2d_script),
        "n_subsets": int(len(subset)),
        "n_prompts": int(drops.shape[0]),
        "n_heads": int(len(heads)),
        "fixed_fold_seed": args.fold_seed,
        "bootstrap_sampling_seed": args.bootstrap_sampling_seed,
        "bootstrap_repeats": args.bootstrap_repeats,
        "fold_audit_seeds": audit_seeds,
        "central_interval": [0.025, 0.975],
        "fixed_fold_delta_mae_vs_count_additive": fixed_pair_delta,
        "pair_split_instability": instability,
        "interpretation": {
            "any_sign_flip": "At least one audited fold seed changes whether the pair term improves MAE.",
            "material_instability": "Fewer than 95% of audited fold seeds improve MAE.",
        },
    }
    (args.outdir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    (args.outdir / "configuration.json").write_text(
        json.dumps(
            {
                "stage2d_script": str(args.stage2d_script),
                "input_run": str(args.input_run),
                "fold_seed": args.fold_seed,
                "bootstrap_sampling_seed": args.bootstrap_sampling_seed,
                "bootstrap_repeats": args.bootstrap_repeats,
                "fold_audit_count": args.fold_audit_count,
                "k_folds": args.k_folds,
                "ridge": args.ridge,
                "models": MODELS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = [
        "# IOI predictive reanalysis with a consistent fold seed",
        "",
        "This analysis makes no model queries. It uses fold seed "
        f"{args.fold_seed} for the point estimate and all "
        f"{args.bootstrap_repeats} paired prompt-bootstrap replicates. "
        f"Bootstrap sampling uses the separate seed {args.bootstrap_sampling_seed}.",
        "",
        "Intervals below are central 95% prompt-bootstrap intervals. The fold-seed "
        "audit is a sensitivity analysis, not a confidence interval.",
        "",
        "## Fixed-fold point metrics",
        "",
        dataframe_text(
            point_metrics[["model", "n_rows", "mae", "rmse", "r2", "n_params"]]
        ),
        "",
        "## Paired prompt-bootstrap delta MAE versus count-additive",
        "",
        dataframe_text(delta_summary),
        "",
        f"## Fold sensitivity across {len(audit_seeds)} alternative seeds",
        "",
        dataframe_text(seed_pair_summary),
        "",
        "A positive delta MAE means that adding the named pair term improves "
        "held-out prediction over the count-additive model. `any_sign_flip` "
        "records even a rare reversal. `material_instability` means that fewer "
        "than 95% of audited splits improve.",
        "",
        "## Stability flags",
        "",
        *stability_lines,
    ]
    (args.outdir / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print(point_metrics.to_string(index=False))
    print("\nPaired delta-MAE central 95% intervals:")
    print(delta_summary.to_string(index=False))
    print("\nPair split stability:")
    print(seed_pair_summary.to_string(index=False))
    print(f"\nWrote {args.outdir}")


if __name__ == "__main__":
    main()
