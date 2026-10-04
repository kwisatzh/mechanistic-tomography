# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Outcome-independent designs and transparent unbatched cost accounting."""
import hashlib
import json
from pathlib import Path
import numpy as np


def dense_rows(count, dimensions, seed):
    rng = np.random.default_rng(seed)
    rows = []
    seen = set()
    while len(rows) < count:
        signs = rng.choice([-1.0, 1.0], size=dimensions)
        key = tuple(signs)
        if key not in seen:
            seen.add(key)
            rows.append(signs / np.sqrt(dimensions))
    return np.asarray(rows)


def designs(config):
    n = len(config["layers"])
    result = {
        "menu": dense_rows(config["menu_actions"], n, config["menu_seed"]),
        "validation": dense_rows(config["validation_actions"], n, config["validation_seed"]),
    }
    for seed in config["design_seeds"]:
        order = np.random.default_rng(seed).permutation(n)
        coordinates = np.eye(n)[order]
        result[f"coordinate_{seed}"] = np.stack([coordinates, -coordinates], axis=1).reshape(-1, n)
        result[f"aggregate_{seed}"] = dense_rows(max(config["budgets"]), n, seed)
        result[f"search_order_{seed}"] = np.random.default_rng(seed).permutation(config["menu_actions"])
    # Pool overlap must fail, not be silently repaired after a seed is frozen.
    groups = {k: {tuple(row) for row in v} for k, v in result.items() if not k.startswith("search_order")}
    for name, rows in groups.items():
        if name.startswith("coordinate"):
            continue  # Same coordinate pool, intentionally reordered across seeds.
        for other in ("menu", "validation"):
            if name != other and rows & groups[other]:
                raise ValueError(f"Overlapping scientific designs: {name} and {other}")
    return result


def costs(config):
    c = config["prompt_counts"]
    stems = len(config["refusal_stems"]) + len(config["compliance_stems"])
    scales = len(config["scales"])
    seeds = len(config["design_seeds"])
    # Coordinate pool is identical across seeds, so acquire it once.
    fit_actions = 2 * len(config["layers"]) + seeds * max(config["budgets"]) + config["validation_actions"] + config["menu_actions"]
    fit = fit_actions * c["fit_harmful"] * stems * scales
    test = config["menu_actions"] * (c["test_harmful"] + c["collateral_benign"]) * stems * scales
    clean = (c["fit_harmful"] + c["test_harmful"] + c["collateral_benign"]) * stems
    direction = 2 * c["direction_per_label"]
    return {"unit": "unbatched prompt continuation evaluations; direction uses prompt only",
            "fit_and_selection": fit, "heldout_evaluation": test, "clean": clean,
            "direction_prompt_forwards": direction,
            "total": fit + test + clean + direction,
            "measured_gpu_seconds": None,
            "note": "Planning count only; no constant conversion to GPU time or backward cost."}


def audit(config):
    d = designs(config)
    return {"cost_plan": costs(config), "designs": {
        k: {"shape": list(v.shape), "sha256": hashlib.sha256(v.tobytes()).hexdigest(),
            **({"rank": int(np.linalg.matrix_rank(v)),
                "max_norm_error": float(np.max(np.abs(np.linalg.norm(v, axis=1)-1)))}
               if v.ndim == 2 else {})}
        for k, v in d.items()}}


if __name__ == "__main__":
    print(json.dumps(audit(json.loads((Path(__file__).parent / "config.json").read_text())), indent=2))
