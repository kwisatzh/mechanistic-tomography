# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
import json

import numpy as np
import pytest

from mechtomo.qwen_refusal import DIRECTION_NAME, DirectionBundle, PromptRecord
from mechtomo.runner import (
    _analysis_config,
    _canonical_hash,
    _environment_fingerprint,
    _establish_measurement_inputs,
    _file_hash,
    _records_for_experiment,
    _validate_direction_bundle,
    _verify_analysis_inputs,
    _write_json,
    _write_measurement_completion,
)


def _config(prepared_path="prompts.jsonl"):
    return {
        "schema_version": 1,
        "model": {"id": "model", "revision": "commit"},
        "data": {
            "prepared_jsonl": str(prepared_path),
            "splits": {
                "direction": "direction",
                "fit": "fit",
                "test": "test_id",
                "collateral": "collateral_id",
            },
        },
        "task": {
            "system_prompt": "system",
            "refusal_stems": ["no"],
            "compliance_stems": ["yes"],
        },
        "actuator": {
            "layers": [1, 2],
            "base_fraction": 0.05,
            "position": "last_prompt_token",
        },
        "design": {},
        "analysis": {
            "ridge_grid": [0.1],
            "bootstrap_repeats": 10,
            "bootstrap_seed": 3,
            "alpha": 0.05,
            "practical_relative_improvement": 0.07,
        },
        "selector": {"setpoint_quantiles": [0.5]},
        "runtime": {"device": "cpu", "dtype": "float32", "batch_size": 2, "max_length": 32},
    }


def _record(prompt_id, text, label, family, split):
    return PromptRecord(prompt_id, text, label, family, split)


def _valid_records():
    return [
        _record("dh", "direction harmful", "harmful", "fdh", "direction"),
        _record("db", "direction benign", "benign", "fdb", "direction"),
        _record("f", "fit prompt", "harmful", "ff", "fit"),
        _record("t", "test prompt", "harmful", "ft", "test_id"),
        _record("c", "collateral prompt", "benign", "fc", "collateral_id"),
    ]


def test_runner_rejects_family_and_normalized_text_leakage():
    config = _config()
    records = _valid_records()
    records[2] = _record("f", "fit prompt", "harmful", "fdh", "fit")
    with pytest.raises(ValueError, match="family"):
        _records_for_experiment(records, config)

    records = _valid_records()
    records[3] = _record("t", "  DIRECTION   HARMFUL ", "harmful", "ft", "test_id")
    with pytest.raises(ValueError, match="normalized text"):
        _records_for_experiment(records, config)


def test_stale_measurement_cache_without_fingerprint_is_refused(tmp_path):
    (tmp_path / "effects_all.npy").write_bytes(b"legacy")
    with pytest.raises(RuntimeError, match="without an input fingerprint"):
        _establish_measurement_inputs(tmp_path, {"schema_version": 2})

    (tmp_path / "effects_all.npy").unlink()
    path = _establish_measurement_inputs(tmp_path, {"schema_version": 2})
    assert json.loads(path.read_text()) == {"schema_version": 2}
    with pytest.raises(RuntimeError, match="inputs changed"):
        _establish_measurement_inputs(tmp_path, {"schema_version": 3})


def test_analysis_verifies_config_sources_and_surface_hash(tmp_path):
    prepared = tmp_path / "prompts.jsonl"
    prepared.write_text("locked prompts\n", encoding="utf-8")
    config = _config(prepared)
    (tmp_path / "directions.npz").write_bytes(b"directions")
    (tmp_path / "directions_inputs.json").write_text("{}\n", encoding="utf-8")
    surface = tmp_path / "surface_measurements.npz"
    surface.write_bytes(b"surface")
    inputs = {
        "schema_version": 2,
        "config_sha256": _canonical_hash(config),
        "prepared_jsonl_sha256": _file_hash(prepared),
        "directions_sha256": _file_hash(tmp_path / "directions.npz"),
        "directions_inputs_sha256": _file_hash(tmp_path / "directions_inputs.json"),
    }
    _write_json(tmp_path / "measurement_inputs.json", inputs)
    _write_measurement_completion(tmp_path, config)

    _verify_analysis_inputs(config, tmp_path)
    surface.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash verification"):
        _verify_analysis_inputs(config, tmp_path)


def test_direction_bundle_and_analysis_threshold_match_config():
    config = _config()
    bundle = DirectionBundle(
        layers=(1, 2),
        directions=np.asarray([[0.05, 0.0], [0.0, 0.05]]),
        residual_norms=np.ones(2),
        base_fraction=0.05,
        model_id="model",
        model_revision="commit",
        direction_name=DIRECTION_NAME,
    )
    _validate_direction_bundle(bundle, config)
    assert _analysis_config(config).practical_relative_improvement == 0.07
    changed = {**config, "actuator": {**config["actuator"], "base_fraction": 0.04}}
    with pytest.raises(ValueError, match="base fraction"):
        _validate_direction_bundle(bundle, changed)


def test_environment_fingerprint_records_accelerator_resolution():
    environment = _environment_fingerprint()
    assert set(environment["packages"]) == {"numpy", "torch", "transformers", "accelerate"}
    assert "resolved" in environment["accelerator"]
    assert "cuda_available" in environment["accelerator"]
