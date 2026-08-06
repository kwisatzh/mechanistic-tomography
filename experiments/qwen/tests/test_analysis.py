# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
import csv
import json

import numpy as np

from mechtomo.analysis import (
    AnalysisConfig,
    SurfaceMeasurements,
    _primary_status,
    analyze_surface,
    load_surface,
    save_surface,
)
from mechtomo.design import make_action_design
from mechtomo.statistics import Interval
from mechtomo.toy import simulate_surface


def _surface():
    design = make_action_design(
        n_sites=6,
        split_sizes={"calibration": 80, "validation": 40, "test": 80},
        seed=23,
    )
    simulated = simulate_surface(
        design,
        n_prompts=120,
        interaction_strength=1.2,
        noise=0.02,
        seed=31,
    )
    return SurfaceMeasurements(
        design=design,
        fit_effects=simulated.effects[:60],
        fit_groups=simulated.groups[:60],
        test_effects=simulated.effects[60:],
        test_groups=simulated.groups[60:],
        test_collateral=abs(simulated.effects[70:]) * 0.01,
    )


def test_surface_round_trip_allows_distinct_collateral_prompt_count(tmp_path):
    surface = _surface()
    path = tmp_path / "surface.npz"
    save_surface(path, surface)
    loaded = load_surface(path)
    assert loaded.test_collateral.shape[0] != loaded.test_effects.shape[0]
    assert loaded.test_collateral.shape[1] == loaded.test_effects.shape[1]


def test_lifted_observer_wins_on_planted_pair_surface(tmp_path):
    summary = analyze_surface(
        _surface(),
        tmp_path,
        AnalysisConfig(bootstrap_repeats=150, bootstrap_seed=41),
    )
    additive_mae = summary["metrics"]["calibrated_additive"]["mae"]
    lifted_mae = summary["metrics"]["lifted"]["mae"]
    assert lifted_mae < additive_mae
    assert summary["primary"]["estimate"] == additive_mae - lifted_mae
    assert summary["primary"]["relative_mae_improvement"]["estimate"] == (
        additive_mae - lifted_mae
    ) / additive_mae
    assert summary["primary"]["status"] == "lifted advantage detected"
    disk = json.loads((tmp_path / "summary.json").read_text())
    assert disk["selector"]["type"] == "fixed-budget finite-action selector"
    assert disk["selector"]["only_surrogate_varies"] is True
    with (tmp_path / "selector_choices.csv").open(newline="") as handle:
        selector_rows = list(csv.DictReader(handle))
    assert selector_rows
    assert all(float(row["setpoint"]) > 0.0 for row in selector_rows)


def test_primary_status_reserves_equivalence_for_two_sided_practical_interval():
    def interval(low, high):
        return Interval(estimate=(low + high) / 2.0, low=low, high=high, probability_positive=0.5, repeats=100)

    assert _primary_status(interval(0.06, 0.10), 0.05) == "lifted advantage detected"
    assert _primary_status(interval(-0.04, 0.04), 0.05) == "practical equivalence established"
    assert _primary_status(interval(-0.20, 0.04), 0.05) == (
        "practically meaningful lifted advantage ruled out"
    )
    assert _primary_status(interval(-0.02, 0.07), 0.05) == "no lifted advantage detected"


def test_no_positive_calibration_effect_reports_no_selector_result(tmp_path):
    design = make_action_design(
        n_sites=6,
        split_sizes={"calibration": 40, "validation": 20, "test": 40},
        seed=71,
    )
    actions = design.actions
    interaction = actions[:, 0] * actions[:, 1]
    mean_effect = -10.0 + interaction
    rng = np.random.default_rng(73)
    fit_effects = mean_effect[None, :] + rng.normal(0.0, 0.01, size=(30, len(actions)))
    test_effects = mean_effect[None, :] + rng.normal(0.0, 0.01, size=(30, len(actions)))
    surface = SurfaceMeasurements(
        design=design,
        fit_effects=fit_effects,
        fit_groups=np.asarray([f"fit_{index}" for index in range(len(fit_effects))]),
        test_effects=test_effects,
        test_groups=np.asarray([f"test_{index}" for index in range(len(test_effects))]),
    )
    summary = analyze_surface(
        surface,
        tmp_path,
        AnalysisConfig(bootstrap_repeats=60, bootstrap_seed=79),
    )
    assert summary["selector"]["status"] == (
        "no positive calibration effects; no selector result"
    )
    assert (tmp_path / "selector_choices.csv").read_text() == ""
