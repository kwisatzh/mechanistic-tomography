# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mechtomo.statistics import mae, r2, rmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Break the frozen Qwen held-out predictions down by mask density."
    )
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def metric_row(
    label: str,
    density: float | None,
    scale: float,
    observed: np.ndarray,
    additive: np.ndarray,
    lifted: np.ndarray,
) -> dict[str, float | int | str | None]:
    additive_mae = mae(observed, additive)
    lifted_mae = mae(observed, lifted)
    difference = additive_mae - lifted_mae
    return {
        "stratum": label,
        "density": density,
        "scale": scale,
        "n_actions": int(len(observed)),
        "calibrated_additive_mae": additive_mae,
        "calibrated_additive_rmse": rmse(observed, additive),
        "calibrated_additive_r2": r2(observed, additive),
        "lifted_mae": lifted_mae,
        "lifted_rmse": rmse(observed, lifted),
        "lifted_r2": r2(observed, lifted),
        "additive_mae_minus_lifted_mae": difference,
        "relative_lifted_mae_improvement": difference / additive_mae,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    analysis_dir = run_dir / "analysis"
    surface_path = run_dir / "surface_measurements.npz"
    predictions_path = analysis_dir / "test_predictions.csv"

    with np.load(surface_path, allow_pickle=False) as surface:
        densities = np.asarray(surface["action_densities"], dtype=float)
        scales = np.asarray(surface["action_scales"], dtype=float)
        splits = np.asarray(surface["action_splits"]).astype("U16")

    with predictions_path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))

    mask_indices = np.asarray([int(row["mask_index"]) for row in records], dtype=int)
    observed = np.asarray([float(row["observed"]) for row in records], dtype=float)
    additive = np.asarray([float(row["calibrated_additive"]) for row in records], dtype=float)
    lifted = np.asarray([float(row["lifted"]) for row in records], dtype=float)

    if not np.all(splits[mask_indices] == "test"):
        raise ValueError("test_predictions.csv contains a non-test action")
    test_scales = np.unique(scales[mask_indices])
    if len(test_scales) != 1:
        raise ValueError("expected one held-out test scale")
    test_scale = float(test_scales[0])

    rows = [
        metric_row("all_test_actions", None, test_scale, observed, additive, lifted)
    ]
    for density in sorted(np.unique(densities[mask_indices])):
        selected = densities[mask_indices] == density
        rows.append(
            metric_row(
                f"density_{density:g}",
                float(density),
                test_scale,
                observed[selected],
                additive[selected],
                lifted[selected],
            )
        )

    json_path = analysis_dir / "stratified_metrics.json"
    json_path.write_text(json.dumps({"schema_version": 1, "rows": rows}, indent=2) + "\n")

    csv_path = analysis_dir / "stratified_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
