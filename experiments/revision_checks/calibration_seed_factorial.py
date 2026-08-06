#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Separate context variation from mask-and-split variation in Result 3.

This script composes the existing Step-0 HMM harness. It fixes the trained
checkpoint and the estimated belief directions, crosses evaluation-context
seeds with measurement-design seeds, and evaluates:

* raw and scalar-calibrated attribution patching on held-out masks;
* OMP on the same held-out masks;
* small calibration budgets;
* gain transfer to held-out contexts; and
* a descriptive two-way variance decomposition.

It deliberately does not copy or modify the original experimental primitives.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class Design:
    seed: int
    matrix: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    holdout: np.ndarray
    calibration_pool: np.ndarray


def load_step0_module(harness_dir: Path) -> ModuleType:
    """Load the existing harness as a module without changing it."""
    source = harness_dir / "attribution_vs_finite_step0.py"
    if not source.exists():
        raise FileNotFoundError(f"Missing Step-0 harness: {source}")
    sys.path.insert(0, str(harness_dir))
    spec = importlib.util.spec_from_file_location("ntmi_step0_harness", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_numbers(values: Sequence[str], cast: type[int] | type[float]) -> list[int] | list[float]:
    parsed: list[int] | list[float] = []
    for value in values:
        parsed.extend(cast(item.strip()) for item in value.split(",") if item.strip())
    return list(dict.fromkeys(parsed))


def make_design(
    harness: ModuleType,
    seed: int,
    n_measurements: int,
    n_components: int,
    density: float,
    holdout_fraction: float,
) -> Design:
    """Use independent random streams for masks and the data split."""
    mask_seed, split_seed = np.random.SeedSequence(seed).spawn(2)
    matrix = harness.make_random_masks(
        n_measurements,
        n_components,
        "signed",
        density,
        np.random.default_rng(mask_seed),
        normalize=True,
    )
    permutation = np.random.default_rng(split_seed).permutation(n_measurements)
    n_holdout = max(1, int(holdout_fraction * n_measurements))
    holdout = permutation[:n_holdout]
    calibration_pool = permutation[n_holdout:]
    n_validation = max(1, int(0.25 * len(calibration_pool)))
    validation = calibration_pool[:n_validation]
    train = calibration_pool[n_validation:]
    if len(train) < 2:
        raise ValueError("Too few fit masks after the split")
    return Design(seed, matrix, train, validation, holdout, calibration_pool)


def split_name(index: int, design: Design) -> str:
    if index in set(design.holdout.tolist()):
        return "holdout"
    if index in set(design.validation.tolist()):
        return "validation"
    return "train"


def deterministic_rng(*parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(parts))


def two_way_decomposition(frame: pd.DataFrame, metric: str) -> dict[str, float | str]:
    """Descriptive two-way decomposition for one observation per seed cell."""
    table = frame.pivot(index="context_seed", columns="design_seed", values=metric)
    values = table.to_numpy(dtype=float)
    grand = float(values.mean())
    context_means = values.mean(axis=1)
    design_means = values.mean(axis=0)
    ss_total = float(np.square(values - grand).sum())
    ss_context = float(values.shape[1] * np.square(context_means - grand).sum())
    ss_design = float(values.shape[0] * np.square(design_means - grand).sum())
    ss_residual = max(0.0, ss_total - ss_context - ss_design)
    denominator = max(ss_total, 1e-15)
    return {
        "metric": metric,
        "ss_total": ss_total,
        "context_fraction": ss_context / denominator,
        "design_fraction": ss_design / denominator,
        "residual_fraction": ss_residual / denominator,
    }


def summarize_budget_gate(cells: pd.DataFrame, budgets: pd.DataFrame, budget: int = 8) -> pd.DataFrame:
    selected = budgets[budgets["m_gain"] == budget]
    mean_r2 = (
        selected.groupby(["epsilon", "context_seed", "design_seed"], as_index=False)["holdout_r2"]
        .mean()
        .rename(columns={"holdout_r2": "budgeted_atp_r2"})
    )
    merged = mean_r2.merge(
        cells[["epsilon", "context_seed", "design_seed", "omp_holdout_r2"]],
        on=["epsilon", "context_seed", "design_seed"],
        how="left",
    )
    merged["omp_minus_budgeted_atp_r2"] = merged["omp_holdout_r2"] - merged["budgeted_atp_r2"]
    return (
        merged.groupby("epsilon", as_index=False)
        .agg(
            cells=("omp_minus_budgeted_atp_r2", "size"),
            calibrated_r2_mean=("budgeted_atp_r2", "mean"),
            calibrated_r2_min=("budgeted_atp_r2", "min"),
            calibrated_r2_max=("budgeted_atp_r2", "max"),
            omp_r2_mean=("omp_holdout_r2", "mean"),
            gap_mean=("omp_minus_budgeted_atp_r2", "mean"),
            gap_max=("omp_minus_budgeted_atp_r2", "max"),
            cells_within_002=("omp_minus_budgeted_atp_r2", lambda x: int((x <= 0.02).sum())),
        )
    )


def make_budget_figure(cells: pd.DataFrame, budgets: pd.DataFrame, outdir: Path) -> None:
    joined = budgets.merge(
        cells[["epsilon", "context_seed", "design_seed", "omp_holdout_r2"]],
        on=["epsilon", "context_seed", "design_seed"],
        how="left",
    )
    joined["gap"] = joined["omp_holdout_r2"] - joined["holdout_r2"]
    available = sorted(int(value) for value in joined["m_gain"].unique())
    preferred = [0, 4, 8, 16, 96]
    shown = [value for value in preferred if value in available]
    x = np.arange(len(shown))

    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5})
    fig, ax = plt.subplots(figsize=(3.35, 2.35), constrained_layout=True)
    styles = [("#2B6CB0", "o"), ("#C05621", "s")]
    for (epsilon, group), (color, marker) in zip(joined.groupby("epsilon"), styles, strict=False):
        medians: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for value in shown:
            sample = group.loc[group["m_gain"] == value, "gap"].to_numpy(dtype=float)
            medians.append(float(np.median(sample)))
            lower.append(float(np.quantile(sample, 0.10)))
            upper.append(float(np.quantile(sample, 0.90)))
        center = np.asarray(medians)
        ax.plot(x, center, color=color, marker=marker, linewidth=1.35, markersize=3.5, label=rf"$\epsilon={epsilon:g}$")
        ax.fill_between(x, lower, upper, color=color, alpha=0.14, linewidth=0)
    ax.axhline(0.0, color="0.25", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [str(value) for value in shown])
    ax.set_xlabel("finite probes used to fit gain")
    ax.set_ylabel(r"OMP $-$ calibrated AtP held-out $R^2$")
    ax.legend(frameon=False, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(outdir / "calibration_factorial_gap_to_omp.png", dpi=320)
    fig.savefig(outdir / "calibration_factorial_gap_to_omp.pdf")
    plt.close(fig)


def make_gain_figure(cells: pd.DataFrame, outdir: Path) -> None:
    epsilons = sorted(cells["epsilon"].unique())
    fig, axes = plt.subplots(1, len(epsilons), figsize=(3.2 * len(epsilons), 2.6), squeeze=False)
    lower = float(cells["fitted_gain"].min())
    upper = float(cells["fitted_gain"].max())
    image = None
    for axis, epsilon in zip(axes[0], epsilons, strict=False):
        table = cells[cells["epsilon"] == epsilon].pivot(
            index="context_seed", columns="design_seed", values="fitted_gain"
        )
        image = axis.imshow(table.to_numpy(), vmin=lower, vmax=upper, cmap="viridis")
        axis.set_xticks(range(len(table.columns)), [str(x) for x in table.columns])
        axis.set_yticks(range(len(table.index)), [str(x) for x in table.index])
        axis.set_xlabel("measurement-design seed")
        axis.set_ylabel("context seed")
        axis.set_title(rf"$\epsilon={epsilon:g}$")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="fitted gain", shrink=0.82)
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.18, top=0.84, wspace=0.34)
    fig.savefig(outdir / "calibration_factorial_gain_by_seed.png", dpi=240)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--direction-seed", type=int, default=7)
    parser.add_argument("--context-seeds", nargs="+", default=["7", "8", "9"])
    parser.add_argument("--design-seeds", nargs="+", default=["7", "8", "9"])
    parser.add_argument("--gain-subset-seed", type=int, default=1701)
    parser.add_argument("--epsilons", nargs="+", default=["5", "8"])
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--direction-batches", type=int, default=20)
    parser.add_argument("--direction-samples", type=int, default=120000)
    parser.add_argument("--direction-quantile", type=float, default=0.2)
    parser.add_argument("--measurements", type=int, default=128)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--mask-density", type=float, default=0.30)
    parser.add_argument("--omp-max-k", type=int, default=12)
    parser.add_argument("--gain-budgets", nargs="+", default=["4", "8", "16", "96"])
    parser.add_argument("--gain-budget-repeats", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.harness_dir = args.harness_dir.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.outdir = args.outdir.expanduser().resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    context_seeds = [int(x) for x in parse_numbers(args.context_seeds, int)]
    design_seeds = [int(x) for x in parse_numbers(args.design_seeds, int)]
    epsilons = [float(x) for x in parse_numbers(args.epsilons, float)]
    gain_budgets = [int(x) for x in parse_numbers(args.gain_budgets, int)]
    if args.quick:
        context_seeds = context_seeds[:1]
        design_seeds = design_seeds[:1]
        epsilons = epsilons[-1:]
        args.batch_size = min(args.batch_size, 64)
        args.direction_batches = min(args.direction_batches, 2)
        args.direction_samples = min(args.direction_samples, 6000)
        args.measurements = min(args.measurements, 24)
        args.omp_max_k = min(args.omp_max_k, 6)
        args.gain_budget_repeats = min(args.gain_budget_repeats, 3)
        gain_budgets = [value for value in gain_budgets if value <= int(0.75 * args.measurements)]

    harness = load_step0_module(args.harness_dir)
    harness.set_seed(args.direction_seed)
    device = harness.pick_device(args.device)
    model, cfg, params = harness.load_model_from_run(args.run_dir, device)
    sequence_length = cfg.seq_len - 1
    bins = harness.build_time_bins(sequence_length, args.n_bins)
    components = harness.component_index(model.n_layers, args.n_bins)

    print(f"device={device}; contexts={context_seeds}; designs={design_seeds}; epsilons={epsilons}")
    print("Estimating one fixed set of belief directions...")
    directions = harness.compute_layer_directions(
        model,
        cfg,
        params,
        device,
        batches=args.direction_batches,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        samples=args.direction_samples,
        quantile=args.direction_quantile,
    )
    designs = {
        seed: make_design(
            harness,
            seed,
            args.measurements,
            len(components),
            args.mask_density,
            args.holdout_fraction,
        )
        for seed in design_seeds
    }
    for seed, design in designs.items():
        np.save(args.outdir / f"measurement_matrix_design_seed_{seed}.npy", design.matrix)

    cell_rows: list[dict[str, float | int]] = []
    measurement_rows: list[dict[str, float | int | str]] = []
    budget_rows: list[dict[str, float | int]] = []

    for context_seed in context_seeds:
        print(f"\ncontext seed {context_seed}")
        harness.set_seed(context_seed)
        batch = harness.generate_batch(args.batch_size, cfg.seq_len, params, device)
        tokens = batch["tokens"][:, :-1]
        with torch.no_grad():
            baseline_logits = harness.forward_multi_control(model, tokens, {}, directions)
            baseline_z1 = harness.implied_z_from_logits(baseline_logits, params, "z1")[:, -1]
        unit_atp_map = harness.attribution_patch_map(
            model, tokens, params, directions, components, bins, epsilon=1.0
        )

        for epsilon in epsilons:
            atp_map = epsilon * unit_atp_map
            for design_seed, design in designs.items():
                print(f"  epsilon={epsilon:g}, measurement design={design_seed}")
                responses = harness.aggregate_measurements(
                    model,
                    tokens,
                    params,
                    design.matrix,
                    directions,
                    components,
                    bins,
                    epsilon,
                    baseline_z1,
                )
                predictions = design.matrix @ atp_map
                gain = harness.slope_no_intercept(predictions[design.train], responses[design.train])
                omp_map, omp_k, omp_validation_r2 = harness.select_omp_by_validation(
                    design.matrix[design.train],
                    responses[design.train],
                    design.matrix[design.validation],
                    responses[design.validation],
                    max_k=min(args.omp_max_k, len(components)),
                )
                raw_r2 = harness.r2_score(responses[design.holdout], predictions[design.holdout])
                calibrated_r2 = harness.r2_score(
                    responses[design.holdout], gain * predictions[design.holdout]
                )
                omp_r2 = harness.r2_score(
                    responses[design.holdout], design.matrix[design.holdout] @ omp_map
                )
                cell_rows.append(
                    {
                        "epsilon": epsilon,
                        "context_seed": context_seed,
                        "design_seed": design_seed,
                        "fitted_gain": gain,
                        "raw_atp_holdout_r2": raw_r2,
                        "calibrated_atp_holdout_r2": calibrated_r2,
                        "omp_holdout_r2": omp_r2,
                        "omp_k": omp_k,
                        "omp_validation_r2": omp_validation_r2,
                    }
                )
                for index, (response, prediction) in enumerate(zip(responses, predictions, strict=True)):
                    measurement_rows.append(
                        {
                            "epsilon": epsilon,
                            "context_seed": context_seed,
                            "design_seed": design_seed,
                            "mask_index": index,
                            "split": split_name(index, design),
                            "finite_response": float(response),
                            "atp_prediction": float(prediction),
                        }
                    )
                budget_rng = deterministic_rng(
                    args.gain_subset_seed,
                    context_seed,
                    design_seed,
                    int(round(epsilon * 1000)),
                )
                for row in harness.calibration_budget_sweep(
                    design.matrix,
                    responses,
                    atp_map,
                    design.calibration_pool,
                    design.holdout,
                    gain_budgets,
                    args.gain_budget_repeats,
                    budget_rng,
                ):
                    row.update(
                        {
                            "epsilon": epsilon,
                            "context_seed": context_seed,
                            "design_seed": design_seed,
                        }
                    )
                    budget_rows.append(row)

    cells = pd.DataFrame(cell_rows)
    measurements = pd.DataFrame(measurement_rows)
    budgets = pd.DataFrame(budget_rows)
    cells.to_csv(args.outdir / "calibration_factorial_cells.csv", index=False)
    measurements.to_csv(args.outdir / "calibration_factorial_measurements.csv", index=False)
    budgets.to_csv(args.outdir / "calibration_factorial_budgets.csv", index=False)

    cross_rows: list[dict[str, float | int | str]] = []
    for epsilon in epsilons:
        for design_seed, design in designs.items():
            subset = measurements[
                (measurements["epsilon"] == epsilon) & (measurements["design_seed"] == design_seed)
            ]
            for source_seed in context_seeds:
                source = subset[subset["context_seed"] == source_seed]
                source_fit = source[source["split"] == "train"]
                gain = harness.slope_no_intercept(source_fit["atp_prediction"], source_fit["finite_response"])
                for target_seed in context_seeds:
                    target = subset[
                        (subset["context_seed"] == target_seed) & (subset["split"] == "holdout")
                    ]
                    cross_rows.append(
                        {
                            "mode": "single_source",
                            "epsilon": epsilon,
                            "design_seed": design_seed,
                            "source_context": str(source_seed),
                            "target_context": target_seed,
                            "gain": gain,
                            "holdout_r2": harness.r2_score(
                                target["finite_response"], gain * target["atp_prediction"]
                            ),
                        }
                    )
            if len(context_seeds) > 1:
                for target_seed in context_seeds:
                    fit = subset[
                        (subset["context_seed"] != target_seed) & (subset["split"] == "train")
                    ]
                    test = subset[
                        (subset["context_seed"] == target_seed) & (subset["split"] == "holdout")
                    ]
                    gain = harness.slope_no_intercept(fit["atp_prediction"], fit["finite_response"])
                    cross_rows.append(
                        {
                            "mode": "leave_one_context_out",
                            "epsilon": epsilon,
                            "design_seed": design_seed,
                            "source_context": "all_other_contexts",
                            "target_context": target_seed,
                            "gain": gain,
                            "holdout_r2": harness.r2_score(
                                test["finite_response"], gain * test["atp_prediction"]
                            ),
                        }
                    )
    cross_context = pd.DataFrame(cross_rows)
    cross_context.to_csv(args.outdir / "calibration_cross_context_transfer.csv", index=False)

    budget_gate = summarize_budget_gate(cells, budgets, budget=8)
    budget_gate.to_csv(args.outdir / "calibration_eight_probe_gate.csv", index=False)
    decompositions = []
    for epsilon, frame in cells.groupby("epsilon"):
        for metric in [
            "fitted_gain",
            "raw_atp_holdout_r2",
            "calibrated_atp_holdout_r2",
            "omp_holdout_r2",
        ]:
            row = two_way_decomposition(frame, metric)
            row["epsilon"] = float(epsilon)
            decompositions.append(row)
    decomposition_frame = pd.DataFrame(decompositions)
    decomposition_frame.to_csv(args.outdir / "calibration_variance_decomposition.csv", index=False)

    make_budget_figure(cells, budgets, args.outdir)
    make_gain_figure(cells, args.outdir)
    summary = {
        "checkpoint": str(args.run_dir / "model.pt"),
        "fixed_direction_seed": args.direction_seed,
        "context_seeds": context_seeds,
        "measurement_design_seeds": design_seeds,
        "gain_subset_seed": args.gain_subset_seed,
        "epsilons": epsilons,
        "n_components": len(components),
        "n_measurements": args.measurements,
        "split_counts": {
            "train": int(len(next(iter(designs.values())).train)),
            "validation": int(len(next(iter(designs.values())).validation)),
            "holdout": int(len(next(iter(designs.values())).holdout)),
            "calibration_pool": int(len(next(iter(designs.values())).calibration_pool)),
        },
        "config": asdict(cfg),
        "eight_probe_gate": budget_gate.to_dict(orient="records"),
        "cross_context_summary": (
            cross_context.groupby(["mode", "epsilon"], as_index=False)["holdout_r2"]
            .agg(["mean", "min", "max"])
            .reset_index()
            .to_dict(orient="records")
            if not cross_context.empty
            else []
        ),
    }
    (args.outdir / "calibration_factorial_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nEight-probe gate (OMP minus mean calibrated AtP):")
    print(budget_gate.to_string(index=False))
    if not cross_context.empty:
        print("\nCross-context gain transfer:")
        print(
            cross_context.groupby(["mode", "epsilon"])["holdout_r2"]
            .agg(["mean", "min", "max"])
            .to_string()
        )
    print(f"\nWrote results to {args.outdir}")


if __name__ == "__main__":
    main()
