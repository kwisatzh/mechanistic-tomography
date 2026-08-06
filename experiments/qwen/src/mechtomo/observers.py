# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .design import design_matrix


@dataclass(frozen=True)
class FittedObserver:
    family: str
    ridge: float
    coefficients: np.ndarray
    feature_names: tuple[str, ...]

    def predict(self, actions: np.ndarray) -> np.ndarray:
        matrix, names = design_matrix(actions, self.family)
        if tuple(names) != self.feature_names:
            raise ValueError("observer feature schema changed")
        return matrix @ self.coefficients


def ridge_fit(matrix: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    target = np.asarray(target, dtype=float)
    if matrix.ndim != 2 or target.ndim != 1 or len(matrix) != len(target):
        raise ValueError("invalid ridge inputs")
    penalty = ridge * np.eye(matrix.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    lhs = matrix.T @ matrix + penalty
    rhs = matrix.T @ target
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def fit_observer_family(
    actions: np.ndarray,
    target: np.ndarray,
    family: str,
    ridge: float,
) -> FittedObserver:
    matrix, names = design_matrix(actions, family)
    coefficients = ridge_fit(matrix, target, ridge)
    return FittedObserver(family, float(ridge), coefficients, tuple(names))


def select_ridge(
    train_actions: np.ndarray,
    train_target: np.ndarray,
    validation_actions: np.ndarray,
    validation_target: np.ndarray,
    family: str,
    ridge_grid: tuple[float, ...],
) -> FittedObserver:
    if not ridge_grid:
        raise ValueError("ridge_grid must not be empty")
    candidates = [
        fit_observer_family(train_actions, train_target, family, ridge)
        for ridge in ridge_grid
    ]
    errors = [
        float(np.mean(np.abs(candidate.predict(validation_actions) - validation_target)))
        for candidate in candidates
    ]
    best = min(range(len(candidates)), key=lambda index: (errors[index], ridge_grid[index]))
    combined_actions = np.concatenate([train_actions, validation_actions], axis=0)
    combined_target = np.concatenate([train_target, validation_target], axis=0)
    return fit_observer_family(combined_actions, combined_target, family, candidates[best].ridge)
