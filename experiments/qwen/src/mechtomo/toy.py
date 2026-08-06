# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .design import ActionDesign


@dataclass(frozen=True)
class ToySurface:
    effects: np.ndarray
    groups: np.ndarray
    main: np.ndarray
    interactions: np.ndarray


def simulate_surface(
    design: ActionDesign,
    n_prompts: int = 80,
    interaction_strength: float = 0.8,
    noise: float = 0.05,
    seed: int = 19,
) -> ToySurface:
    """Small deterministic falsifier for the complete analysis path."""

    rng = np.random.default_rng(seed)
    n_sites = design.actions.shape[1]
    main = rng.normal(0.0, 0.5, size=n_sites)
    interactions = np.zeros((n_sites, n_sites), dtype=float)
    interactions[0, 1] = interaction_strength
    if n_sites > 3:
        interactions[2, 3] = -0.65 * interaction_strength
    mean_effect = design.actions @ main
    for left in range(n_sites):
        for right in range(left + 1, n_sites):
            mean_effect += interactions[left, right] * design.actions[:, left] * design.actions[:, right]
    prompt_offsets = rng.normal(0.0, 0.08, size=(n_prompts, 1))
    prompt_noise = rng.normal(0.0, noise, size=(n_prompts, len(mean_effect)))
    effects = mean_effect[None, :] + prompt_offsets + prompt_noise
    groups = np.asarray([f"family_{index // 4:03d}" for index in range(n_prompts)])
    return ToySurface(effects, groups, main, interactions)
