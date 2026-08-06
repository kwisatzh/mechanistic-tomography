# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ActionDesign:
    """A fixed, split intervention library.

    Rows are layerwise steering amplitudes.  Nonzero rows are L2-normalized
    before the requested physical scale is applied, so density does not
    silently change the intervention budget.
    """

    actions: np.ndarray
    splits: np.ndarray
    densities: np.ndarray
    scales: np.ndarray
    seed: int

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.splits == split)

    def validate(self) -> None:
        if self.actions.ndim != 2:
            raise ValueError("actions must be a two-dimensional matrix")
        n = self.actions.shape[0]
        if not (len(self.splits) == len(self.densities) == len(self.scales) == n):
            raise ValueError("design metadata length mismatch")
        if not np.isfinite(self.actions).all():
            raise ValueError("actions contain non-finite values")
        norms = np.linalg.norm(self.actions, axis=1)
        nonzero = norms > 0
        if not np.allclose(norms[nonzero], self.scales[nonzero], atol=1e-10):
            raise ValueError("nonzero action rows must match their declared L2 scale")
        patterns = [tuple(np.sign(row).astype(int).tolist()) for row in self.actions]
        if len(set(patterns)) != len(patterns):
            raise ValueError("action support/sign patterns must be unique across all splits")


def _balanced_densities(n_rows: int, densities: Iterable[float], rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(tuple(densities), dtype=float)
    if len(values) == 0 or np.any((values <= 0) | (values > 1)):
        raise ValueError("densities must lie in (0, 1]")
    out = np.resize(values, n_rows).copy()
    rng.shuffle(out)
    return out


def _balanced_scales(n_rows: int, scales: Iterable[float], rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(tuple(scales), dtype=float)
    if len(values) == 0 or np.any(values <= 0):
        raise ValueError("scales must be positive")
    out = np.resize(values, n_rows).copy()
    rng.shuffle(out)
    return out


def _sample_rows(
    n_rows: int,
    n_sites: int,
    densities: Iterable[float],
    scales: Iterable[float],
    rng: np.random.Generator,
    used_patterns: set[tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    density_rows = _balanced_densities(n_rows, densities, rng)
    scale_rows = _balanced_scales(n_rows, scales, rng)
    actions = np.zeros((n_rows, n_sites), dtype=float)
    for row, (density, scale) in enumerate(zip(density_rows, scale_rows)):
        count = max(1, min(n_sites, int(round(density * n_sites))))
        for _attempt in range(10000):
            candidate = np.zeros(n_sites, dtype=float)
            sites = rng.choice(n_sites, size=count, replace=False)
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=count)
            candidate[sites] = signs
            pattern = tuple(np.sign(candidate).astype(int).tolist())
            if pattern in used_patterns:
                continue
            used_patterns.add(pattern)
            actions[row] = candidate * (scale / np.linalg.norm(candidate))
            break
        else:
            raise RuntimeError("could not sample enough unique action support/sign patterns")
    return actions, density_rows, scale_rows


def make_action_design(
    n_sites: int,
    split_sizes: dict[str, int],
    densities: Iterable[float] = (0.25, 0.5, 0.75, 1.0),
    fit_scales: Iterable[float] = (0.5, 1.0),
    validation_scales: Iterable[float] | None = None,
    heldout_scales: Iterable[float] = (0.75,),
    seed: int = 7,
    include_zero: bool = True,
    include_singletons: bool = True,
) -> ActionDesign:
    """Create the preregistered random action library.

    Calibration and hyperparameter validation use endpoint scales while test
    uses held-out within-range scales. Zero and signed singleton rows are
    calibration-only.
    """

    if n_sites < 2:
        raise ValueError("at least two intervention sites are required")
    if "calibration" not in split_sizes or "test" not in split_sizes:
        raise ValueError("split_sizes must include calibration and test")
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    split_labels: list[str] = []
    density_meta: list[float] = []
    scale_meta: list[float] = []
    used_patterns: set[tuple[int, ...]] = set()

    if include_zero:
        blocks.append(np.zeros((1, n_sites), dtype=float))
        split_labels.append("calibration")
        density_meta.append(0.0)
        scale_meta.append(0.0)
        used_patterns.add(tuple([0] * n_sites))

    if include_singletons:
        singletons = np.concatenate([np.eye(n_sites), -np.eye(n_sites)], axis=0)
        blocks.append(singletons)
        split_labels.extend(["calibration"] * len(singletons))
        density_meta.extend([1.0 / n_sites] * len(singletons))
        scale_meta.extend([1.0] * len(singletons))
        used_patterns.update(tuple(np.sign(row).astype(int).tolist()) for row in singletons)

    for split, count in split_sizes.items():
        if count < 0:
            raise ValueError("split sizes must be nonnegative")
        if split == "calibration":
            scales = fit_scales
        elif split == "validation":
            scales = fit_scales if validation_scales is None else validation_scales
        else:
            scales = heldout_scales
        rows, row_density, row_scale = _sample_rows(
            count, n_sites, densities, scales, rng, used_patterns
        )
        blocks.append(rows)
        split_labels.extend([split] * count)
        density_meta.extend(row_density.tolist())
        scale_meta.extend(row_scale.tolist())

    design = ActionDesign(
        actions=np.concatenate(blocks, axis=0),
        splits=np.asarray(split_labels, dtype="U16"),
        densities=np.asarray(density_meta, dtype=float),
        scales=np.asarray(scale_meta, dtype=float),
        seed=seed,
    )
    design.validate()
    return design


def pair_indices(n_sites: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n_sites, k=1)


def design_matrix(actions: np.ndarray, family: str) -> tuple[np.ndarray, list[str]]:
    """Compose observer-family features from the same physical actions."""

    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 2:
        raise ValueError("actions must be two-dimensional")
    n_rows, n_sites = actions.shape
    density = np.count_nonzero(actions, axis=1) / float(n_sites)
    scale = np.linalg.norm(actions, axis=1)
    main_names = [f"u{i}" for i in range(n_sites)]
    squares = actions**2
    square_names = [f"u{i}^2" for i in range(n_sites)]
    base = np.column_stack([np.ones(n_rows), actions, squares, density, density**2, scale])
    names = ["intercept", *main_names, *square_names, "density", "density2", "scale"]
    if family in {"linear", "count_additive", "calibrated_additive"}:
        return base, names
    if family == "lifted":
        left, right = pair_indices(n_sites)
        pairs = actions[:, left] * actions[:, right]
        pair_names = [f"u{i}*u{j}" for i, j in zip(left, right)]
        return np.column_stack([base, pairs]), [*names, *pair_names]
    raise ValueError(f"unknown observer family: {family}")
