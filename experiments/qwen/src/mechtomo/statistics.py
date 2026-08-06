# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    probability_positive: float
    repeats: int


@dataclass(frozen=True)
class MAEImprovementBootstrap:
    """Paired absolute and relative MAE-improvement bootstrap results."""

    absolute: Interval
    relative: Interval
    absolute_draws: np.ndarray
    relative_draws: np.ndarray


def mae(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(observed) - np.asarray(predicted))))


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    delta = np.asarray(observed) - np.asarray(predicted)
    return float(np.sqrt(np.mean(delta**2)))


def r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    if denominator <= 1e-15:
        return float("nan")
    return float(1.0 - np.sum((observed - predicted) ** 2) / denominator)


def grouped_bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) == 0:
        raise ValueError("at least one prompt-family group is required")
    sampled = rng.choice(unique, size=len(unique), replace=True)
    chunks = [np.flatnonzero(groups == group) for group in sampled]
    return np.concatenate(chunks)


def _relative_mae_improvement(baseline_mae: float, comparison_mae: float) -> float:
    if baseline_mae <= np.finfo(float).eps:
        raise ValueError("relative MAE improvement is undefined when baseline MAE is zero")
    return float((baseline_mae - comparison_mae) / baseline_mae)


def _interval(estimate: float, draws: np.ndarray, alpha: float) -> Interval:
    return Interval(
        estimate=float(estimate),
        low=float(np.quantile(draws, alpha / 2.0)),
        high=float(np.quantile(draws, 1.0 - alpha / 2.0)),
        probability_positive=float(np.mean(draws > 0.0)),
        repeats=len(draws),
    )


def paired_two_way_bootstrap_mae_improvement(
    per_prompt_effects: np.ndarray,
    mask_indices: np.ndarray,
    baseline_prediction: np.ndarray,
    comparison_prediction: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
    alpha: float = 0.05,
) -> MAEImprovementBootstrap:
    """Bootstrap paired MAE improvement over prompt families and actions.

    The locked point estimates are computed once from the complete test sample.
    Each bootstrap draw independently resamples prompt-family clusters and test
    action rows. Relative improvement is
    ``(MAE_baseline - MAE_comparison) / MAE_baseline``.
    """

    effects = np.asarray(per_prompt_effects, dtype=float)
    mask_indices = np.asarray(mask_indices, dtype=int)
    baseline_prediction = np.asarray(baseline_prediction, dtype=float)
    comparison_prediction = np.asarray(comparison_prediction, dtype=float)
    if effects.ndim != 2:
        raise ValueError("per_prompt_effects must have shape prompts x masks")
    if effects.shape[0] == 0:
        raise ValueError("at least one prompt is required")
    if len(groups) != effects.shape[0]:
        raise ValueError("one group label is required per prompt")
    if not (len(mask_indices) == len(baseline_prediction) == len(comparison_prediction)):
        raise ValueError("mask and prediction lengths differ")
    if len(mask_indices) == 0:
        raise ValueError("at least one test action is required")
    if np.any(mask_indices < 0) or np.any(mask_indices >= effects.shape[1]):
        raise ValueError("mask index is outside the effect matrix")
    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not (
        np.isfinite(effects).all()
        and np.isfinite(baseline_prediction).all()
        and np.isfinite(comparison_prediction).all()
    ):
        raise ValueError("bootstrap inputs must be finite")

    locked_observed = effects[:, mask_indices].mean(axis=0)
    locked_baseline_mae = mae(locked_observed, baseline_prediction)
    locked_comparison_mae = mae(locked_observed, comparison_prediction)
    absolute_estimate = locked_baseline_mae - locked_comparison_mae
    relative_estimate = _relative_mae_improvement(
        locked_baseline_mae,
        locked_comparison_mae,
    )

    rng = np.random.default_rng(seed)
    absolute_draws = np.empty(repeats, dtype=float)
    relative_draws = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        prompt_indices = grouped_bootstrap_indices(groups, rng)
        action_positions = rng.integers(0, len(mask_indices), size=len(mask_indices))
        observed = effects[prompt_indices][:, mask_indices].mean(axis=0)[action_positions]
        baseline_mae = mae(observed, baseline_prediction[action_positions])
        comparison_mae = mae(observed, comparison_prediction[action_positions])
        absolute_draws[repeat] = baseline_mae - comparison_mae
        relative_draws[repeat] = _relative_mae_improvement(baseline_mae, comparison_mae)

    return MAEImprovementBootstrap(
        absolute=_interval(absolute_estimate, absolute_draws, alpha),
        relative=_interval(relative_estimate, relative_draws, alpha),
        absolute_draws=absolute_draws,
        relative_draws=relative_draws,
    )


def paired_prompt_bootstrap_delta_mae(
    per_prompt_effects: np.ndarray,
    mask_indices: np.ndarray,
    baseline_prediction: np.ndarray,
    comparison_prediction: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[Interval, np.ndarray]:
    """Compatibility wrapper returning the two-way absolute comparison."""

    result = paired_two_way_bootstrap_mae_improvement(
        per_prompt_effects,
        mask_indices,
        baseline_prediction,
        comparison_prediction,
        groups,
        repeats,
        seed,
        alpha,
    )
    return result.absolute, result.absolute_draws
