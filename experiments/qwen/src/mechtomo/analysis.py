# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np

from .design import ActionDesign
from .observers import FittedObserver, select_ridge
from .statistics import Interval, mae, paired_two_way_bootstrap_mae_improvement, r2, rmse


@dataclass(frozen=True)
class SurfaceMeasurements:
    design: ActionDesign
    fit_effects: np.ndarray
    fit_groups: np.ndarray
    test_effects: np.ndarray
    test_groups: np.ndarray
    test_collateral: np.ndarray | None = None

    def validate(self) -> None:
        self.design.validate()
        n_masks = len(self.design.actions)
        for name, values, groups in (
            ("fit", self.fit_effects, self.fit_groups),
            ("test", self.test_effects, self.test_groups),
        ):
            if values.ndim != 2 or values.shape[1] != n_masks:
                raise ValueError(f"{name}_effects must have shape prompts x {n_masks}")
            if len(groups) != values.shape[0]:
                raise ValueError(f"{name} group count does not match prompts")
            if not np.isfinite(values).all():
                raise ValueError(f"{name}_effects contain non-finite values")
        if self.test_collateral is not None:
            if self.test_collateral.ndim != 2 or self.test_collateral.shape[1] != n_masks:
                raise ValueError("test_collateral must have one column per action")
            if not np.isfinite(self.test_collateral).all():
                raise ValueError("test_collateral contains non-finite values")


@dataclass(frozen=True)
class AnalysisConfig:
    ridge_grid: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)
    bootstrap_repeats: int = 2000
    bootstrap_seed: int = 29
    alpha: float = 0.05
    practical_relative_improvement: float = 0.05
    selector_setpoint_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)


def save_surface(path: str | Path, surface: SurfaceMeasurements) -> None:
    surface.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    collateral = (
        surface.test_collateral
        if surface.test_collateral is not None
        else np.empty((0, 0), dtype=float)
    )
    np.savez_compressed(
        path,
        actions=surface.design.actions,
        action_splits=surface.design.splits,
        action_densities=surface.design.densities,
        action_scales=surface.design.scales,
        design_seed=np.asarray([surface.design.seed], dtype=int),
        fit_effects=surface.fit_effects,
        fit_groups=surface.fit_groups,
        test_effects=surface.test_effects,
        test_groups=surface.test_groups,
        test_collateral=collateral,
    )


def load_surface(path: str | Path) -> SurfaceMeasurements:
    with np.load(path, allow_pickle=False) as data:
        design = ActionDesign(
            actions=np.asarray(data["actions"], dtype=float),
            splits=np.asarray(data["action_splits"]).astype("U16"),
            densities=np.asarray(data["action_densities"], dtype=float),
            scales=np.asarray(data["action_scales"], dtype=float),
            seed=int(np.asarray(data["design_seed"])[0]),
        )
        collateral = np.asarray(data["test_collateral"], dtype=float)
        surface = SurfaceMeasurements(
            design=design,
            fit_effects=np.asarray(data["fit_effects"], dtype=float),
            fit_groups=np.asarray(data["fit_groups"]).astype("U128"),
            test_effects=np.asarray(data["test_effects"], dtype=float),
            test_groups=np.asarray(data["test_groups"]).astype("U128"),
            test_collateral=None if collateral.size == 0 else collateral,
        )
    surface.validate()
    return surface


def _fit_observers(surface: SurfaceMeasurements, config: AnalysisConfig) -> dict[str, FittedObserver]:
    design = surface.design
    calibration = design.indices("calibration")
    validation = design.indices("validation")
    if len(validation) == 0:
        raise ValueError("the action design requires a validation split")
    fit_mean = surface.fit_effects.mean(axis=0)
    observers = {}
    for family in ("calibrated_additive", "lifted"):
        observers[family] = select_ridge(
            design.actions[calibration],
            fit_mean[calibration],
            design.actions[validation],
            fit_mean[validation],
            family,
            config.ridge_grid,
        )
    return observers


def _prediction_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(observed, predicted),
        "rmse": rmse(observed, predicted),
        "r2": r2(observed, predicted),
    }


def _selector_rows(
    surface: SurfaceMeasurements,
    observers: dict[str, FittedObserver],
    config: AnalysisConfig,
) -> list[dict[str, float | int | str]]:
    calibration = surface.design.indices("calibration")
    test = surface.design.indices("test")
    fit_mean = surface.fit_effects.mean(axis=0)
    test_mean = surface.test_effects.mean(axis=0)
    positive_calibration = fit_mean[calibration][fit_mean[calibration] > 1e-12]
    if len(positive_calibration) == 0:
        return []
    setpoints = np.quantile(positive_calibration, config.selector_setpoint_quantiles)
    predictions = {name: observer.predict(surface.design.actions[test]) for name, observer in observers.items()}
    predictions["oracle_diagnostic"] = test_mean[test]
    rows: list[dict[str, float | int | str]] = []
    for setpoint_index, setpoint in enumerate(setpoints):
        for name, prediction in predictions.items():
            objective = (prediction - setpoint) ** 2
            local_index = int(np.argmin(objective))
            mask_index = int(test[local_index])
            actual = float(test_mean[mask_index])
            row: dict[str, float | int | str] = {
                "setpoint_index": setpoint_index,
                "setpoint": float(setpoint),
                "surrogate": name,
                "mask_index": mask_index,
                "predicted_effect": float(prediction[local_index]),
                "actual_effect": actual,
                "absolute_setpoint_error": float(abs(actual - setpoint)),
                "action_energy": float(np.sum(surface.design.actions[mask_index] ** 2)),
            }
            if surface.test_collateral is not None:
                row["mean_benign_refusal_margin_change"] = float(
                    surface.test_collateral[:, mask_index].mean()
                )
            rows.append(row)
    return rows


def _primary_status(relative: Interval, practical_threshold: float) -> str:
    if not 0.0 < practical_threshold < 1.0:
        raise ValueError("practical relative-improvement threshold must lie in (0, 1)")
    if relative.low > practical_threshold:
        return "lifted advantage detected"
    if relative.low >= -practical_threshold and relative.high <= practical_threshold:
        return "practical equivalence established"
    if relative.high < practical_threshold:
        return "practically meaningful lifted advantage ruled out"
    return "no lifted advantage detected"


def _primary_interpretation(status: str, practical_threshold: float) -> str:
    percentage = 100.0 * practical_threshold
    if status == "lifted advantage detected":
        return (
            f"The lifted surrogate improves held-out MAE by more than the preregistered "
            f"{percentage:g}% relative threshold on this fixed model, task, basis, scale, "
            "and design."
        )
    if status == "practical equivalence established":
        return (
            f"The relative-improvement interval lies entirely within +/-{percentage:g}%; "
            "the two response-surrogate families are practically equivalent under the preregistered margin."
        )
    if status == "practically meaningful lifted advantage ruled out":
        return (
            f"The upper confidence bound is below the preregistered {percentage:g}% threshold, "
            "ruling out a practically meaningful lifted advantage without claiming equivalence."
        )
    return (
        "No lifted advantage was detected, but the confidence interval does not rule out the "
        f"preregistered {percentage:g}% practically meaningful improvement."
    )


def analyze_surface(
    surface: SurfaceMeasurements,
    outdir: str | Path,
    config: AnalysisConfig = AnalysisConfig(),
) -> dict:
    surface.validate()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    observers = _fit_observers(surface, config)
    test_masks = surface.design.indices("test")
    observed = surface.test_effects[:, test_masks].mean(axis=0)
    predictions = {
        name: observer.predict(surface.design.actions[test_masks])
        for name, observer in observers.items()
    }
    metrics = {}
    for name, prediction in predictions.items():
        observer = observers[name]
        metrics[name] = {
            **_prediction_metrics(observed, prediction),
            "ridge": observer.ridge,
            "n_parameters": len(observer.coefficients),
        }
    comparison = paired_two_way_bootstrap_mae_improvement(
        surface.test_effects,
        test_masks,
        predictions["calibrated_additive"],
        predictions["lifted"],
        surface.test_groups,
        repeats=config.bootstrap_repeats,
        seed=config.bootstrap_seed,
        alpha=config.alpha,
    )
    status = _primary_status(
        comparison.relative,
        config.practical_relative_improvement,
    )
    primary = {
        "contrast": "calibrated_additive_mae_minus_lifted_mae",
        **asdict(comparison.absolute),
        "absolute_mae_improvement": asdict(comparison.absolute),
        "relative_mae_improvement": asdict(comparison.relative),
        "practical_relative_improvement_threshold": config.practical_relative_improvement,
        "status": status,
        "success": status == "lifted advantage detected",
        "practical_equivalence": status == "practical equivalence established",
        "meaningful_lifted_advantage_ruled_out": status in {
            "practical equivalence established",
            "practically meaningful lifted advantage ruled out",
        },
        "interpretation": _primary_interpretation(
            status,
            config.practical_relative_improvement,
        ),
        "bootstrap": {
            "scheme": "paired two-way percentile bootstrap",
            "prompt_resampling_unit": "prompt-family cluster",
            "action_resampling_unit": "test action row",
        },
    }
    selector_rows = _selector_rows(surface, observers, config)
    selector_status = (
        "selector results available"
        if selector_rows
        else "no positive calibration effects; no selector result"
    )
    summary = {
        "schema_version": 1,
        "n_fit_prompts": int(surface.fit_effects.shape[0]),
        "n_test_prompts": int(surface.test_effects.shape[0]),
        "n_actions": int(surface.design.actions.shape[0]),
        "n_test_actions": int(len(test_masks)),
        "metrics": metrics,
        "primary": primary,
        "selector": {
            "type": "fixed-budget finite-action selector",
            "status": selector_status,
            "setpoint_source": "positive calibration effects only",
            "only_surrogate_varies": True,
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.save(outdir / "bootstrap_delta_mae.npy", comparison.absolute_draws)
    np.save(outdir / "bootstrap_relative_mae_improvement.npy", comparison.relative_draws)
    with (outdir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mask_index", "observed", "calibrated_additive", "lifted"],
        )
        writer.writeheader()
        for offset, mask_index in enumerate(test_masks):
            writer.writerow({
                "mask_index": int(mask_index),
                "observed": float(observed[offset]),
                "calibrated_additive": float(predictions["calibrated_additive"][offset]),
                "lifted": float(predictions["lifted"][offset]),
            })
    selector_fields = list(selector_rows[0]) if selector_rows else []
    with (outdir / "selector_choices.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selector_fields)
        if selector_fields:
            writer.writeheader()
            writer.writerows(selector_rows)
    return summary
