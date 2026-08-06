# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
import numpy as np

from mechtomo.design import design_matrix, make_action_design


def test_action_rows_are_budget_normalized_and_split():
    design = make_action_design(
        n_sites=8,
        split_sizes={"calibration": 20, "validation": 8, "test": 10},
        seed=11,
    )
    design.validate()
    assert len(design.indices("test")) == 10
    norms = np.linalg.norm(design.actions, axis=1)
    assert np.allclose(norms[norms > 0], design.scales[norms > 0])
    patterns = [tuple(np.sign(row).astype(int)) for row in design.actions]
    assert len(patterns) == len(set(patterns))


def test_lifted_schema_adds_every_unique_pair():
    actions = np.eye(5)
    linear, linear_names = design_matrix(actions, "calibrated_additive")
    lifted, lifted_names = design_matrix(actions, "lifted")
    assert lifted.shape[1] - linear.shape[1] == 10
    assert len(set(lifted_names)) == len(lifted_names)
    assert lifted_names[: len(linear_names)] == linear_names
