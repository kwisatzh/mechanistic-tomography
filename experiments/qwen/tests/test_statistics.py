# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
import numpy as np

from mechtomo.statistics import mae, paired_two_way_bootstrap_mae_improvement


def test_two_way_bootstrap_uses_locked_point_estimate_and_resamples_actions():
    effects = np.asarray(
        [
            [0.0, 2.0, 4.0],
            [0.0, 2.0, 4.0],
            [0.0, 2.0, 4.0],
        ]
    )
    masks = np.asarray([0, 1, 2])
    baseline = np.asarray([1.0, 0.0, 6.0])
    comparison = np.asarray([0.5, 1.5, 5.0])
    result = paired_two_way_bootstrap_mae_improvement(
        effects,
        masks,
        baseline,
        comparison,
        np.asarray(["family_a", "family_a", "family_a"]),
        repeats=200,
        seed=101,
    )

    observed = effects.mean(axis=0)
    baseline_mae = mae(observed, baseline)
    comparison_mae = mae(observed, comparison)
    expected_absolute = baseline_mae - comparison_mae
    assert result.absolute.estimate == expected_absolute
    assert result.relative.estimate == expected_absolute / baseline_mae
    assert np.std(result.absolute_draws) > 0.0
    assert np.std(result.relative_draws) > 0.0
