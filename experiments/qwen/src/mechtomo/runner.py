# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Sequence
import unicodedata

import numpy as np

from .analysis import AnalysisConfig, SurfaceMeasurements, analyze_surface, load_surface, save_surface
from .design import make_action_design
from .qwen_refusal import (
    DIRECTION_NAME,
    DirectionBundle,
    PromptRecord,
    QwenRefusalPlant,
    load_directions,
    load_prompt_records,
    save_directions,
    select_records,
)


_MEASUREMENT_CACHE_NAMES = (
    "clean_refusal_margin.npy",
    "effects_all.npy",
    "measurement_progress.json",
    "surface_measurements.npz",
    "measurement_complete.json",
)


def load_config(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported config schema")
    return value


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _records_hash(records: Sequence[PromptRecord]) -> str:
    return _canonical_hash([
        {
            "id": record.prompt_id,
            "text": record.text,
            "normalized_text": _normalized_text(record.text),
            "label": record.label,
            "family": record.family,
            "split": record.split,
            "source": record.source,
        }
        for record in records
    ])


def _environment_fingerprint() -> dict:
    packages = {}
    for name in ("numpy", "torch", "transformers", "accelerate"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    accelerator = {
        "resolved": "cpu",
        "torch_cuda_version": None,
        "cuda_available": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": 0,
        "cuda_current_device": None,
        "cuda_device_name": None,
        "cuda_compute_capability": None,
        "mps_available": False,
    }
    if packages["torch"] is not None:
        try:
            import torch

            accelerator["torch_cuda_version"] = torch.version.cuda
            accelerator["cuda_available"] = bool(torch.cuda.is_available())
            accelerator["cuda_device_count"] = int(torch.cuda.device_count())
            if accelerator["cuda_available"]:
                device_index = int(torch.cuda.current_device())
                accelerator["resolved"] = "cuda"
                accelerator["cuda_current_device"] = device_index
                accelerator["cuda_device_name"] = torch.cuda.get_device_name(device_index)
                accelerator["cuda_compute_capability"] = list(
                    torch.cuda.get_device_capability(device_index)
                )
            mps_backend = getattr(torch.backends, "mps", None)
            accelerator["mps_available"] = bool(
                mps_backend is not None and mps_backend.is_available()
            )
            if accelerator["resolved"] == "cpu" and accelerator["mps_available"]:
                accelerator["resolved"] = "mps"
        except Exception as exc:
            accelerator["introspection_error"] = type(exc).__name__
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "accelerator": accelerator,
    }


def _source_fingerprint() -> dict:
    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parents[1]
    paths = sorted(package_dir.glob("*.py"))
    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        paths.append(pyproject)
    digest = hashlib.sha256()
    names = []
    for path in sorted(paths):
        try:
            name = str(path.relative_to(project_root))
        except ValueError:
            name = path.name
        names.append(name)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"sha256": digest.hexdigest(), "files": names}


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _write_metadata(outdir: Path, config: dict, stage: str) -> None:
    metadata = {
        "schema_version": 2,
        "stage": stage,
        "config_sha256": _canonical_hash(config),
        "environment": _environment_fingerprint(),
        "source": _source_fingerprint(),
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "claim_scope": (
            "Private weak-ground-truth behavioral finite-effect prediction and fixed-budget "
            "action selection in the declared Qwen/refusal/layerwise-steering basis; not "
            "mechanistic identification, Paper 1 evidence, modern-scale validation, or an arXiv gate."
        ),
    }
    _write_json(outdir / "run_metadata.json", metadata)


def _build_plant(config: dict) -> QwenRefusalPlant:
    model = config["model"]
    task = config["task"]
    actuator = config["actuator"]
    runtime = config["runtime"]
    return QwenRefusalPlant(
        model_id=model["id"],
        revision=model["revision"],
        layers=actuator["layers"],
        refusal_stems=task["refusal_stems"],
        compliance_stems=task["compliance_stems"],
        system_prompt=task["system_prompt"],
        device=runtime["device"],
        dtype=runtime["dtype"],
        batch_size=runtime["batch_size"],
        max_length=runtime["max_length"],
        position=actuator.get("position", "last_prompt_token"),
    )


def _records_for_experiment(records: Sequence[PromptRecord], config: dict):
    splits = config["data"]["splits"]
    direction = select_records(records, splits["direction"])
    fit = select_records(records, splits["fit"], label="harmful")
    test = select_records(records, splits["test"], label="harmful")
    collateral = select_records(records, splits["collateral"], label="benign")
    for name, selected in (
        ("direction", direction),
        ("fit", fit),
        ("test", test),
        ("collateral", collateral),
    ):
        if not selected:
            raise ValueError(f"no prompts selected for {name}")
    named = {
        "direction": direction,
        "fit": fit,
        "test": test,
        "collateral": collateral,
    }
    keys = {
        name: {
            "ID": {record.prompt_id for record in selected},
            "family": {record.family for record in selected},
            "normalized text": {_normalized_text(record.text) for record in selected},
        }
        for name, selected in named.items()
    }
    names = tuple(named)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            for key_name in ("ID", "family", "normalized text"):
                overlap = keys[left][key_name] & keys[right][key_name]
                if overlap:
                    example = sorted(overlap)[0]
                    raise ValueError(
                        f"{left} and {right} prompts overlap by {key_name}: {example!r}"
                    )
    return direction, fit, test, collateral


def _direction_inputs(config: dict, direction_records: Sequence[PromptRecord]) -> dict:
    model = config["model"]
    actuator = config["actuator"]
    runtime = config["runtime"]
    transformers_version = _environment_fingerprint()["packages"]["transformers"]
    return {
        "schema_version": 2,
        "direction_name": DIRECTION_NAME,
        "model": {
            "id": model["id"],
            "revision": model["revision"],
        },
        "tokenizer": {
            "id": model["id"],
            "revision": model["revision"],
            "transformers_version": transformers_version,
        },
        "actuator": {
            "layers": [int(layer) for layer in actuator["layers"]],
            "base_fraction": float(actuator["base_fraction"]),
            "position": actuator.get("position", "last_prompt_token"),
        },
        "system_prompt_sha256": hashlib.sha256(
            config["task"]["system_prompt"].encode("utf-8")
        ).hexdigest(),
        "construction": {
            "prepared_jsonl_sha256": _file_hash(config["data"]["prepared_jsonl"]),
            "records_sha256": _records_hash(direction_records),
            "ordered_prompt_ids_sha256": hashlib.sha256(
                "\n".join(record.prompt_id for record in direction_records).encode("utf-8")
            ).hexdigest(),
            "n_records": len(direction_records),
        },
        "capture": {
            "device": runtime["device"],
            "dtype": runtime["dtype"],
            "batch_size": int(runtime["batch_size"]),
            "max_length": int(runtime["max_length"]),
        },
        "environment": _environment_fingerprint(),
        "source": _source_fingerprint(),
    }


def _validate_direction_bundle(bundle: DirectionBundle, config: dict) -> None:
    bundle.validate()
    expected_layers = tuple(int(layer) for layer in config["actuator"]["layers"])
    expected_fraction = float(config["actuator"]["base_fraction"])
    if bundle.model_id != config["model"]["id"] or bundle.model_revision != config["model"]["revision"]:
        raise ValueError("direction checkpoint was produced by a different model revision")
    if tuple(bundle.layers) != expected_layers:
        raise ValueError("direction checkpoint uses different actuator layers")
    if not np.isclose(bundle.base_fraction, expected_fraction, rtol=0.0, atol=1e-12):
        raise ValueError("direction checkpoint uses a different base fraction")
    if bundle.direction_name != DIRECTION_NAME:
        raise ValueError("direction checkpoint has a different semantic name")


def _load_validated_directions(
    config: dict,
    outdir: Path,
    direction_records: Sequence[PromptRecord],
) -> DirectionBundle:
    direction_path = outdir / "directions.npz"
    inputs_path = outdir / "directions_inputs.json"
    if not direction_path.exists() or not inputs_path.exists():
        if direction_path.exists() or inputs_path.exists():
            raise RuntimeError(
                "incomplete direction cache provenance; use a fresh outdir"
            )
        raise FileNotFoundError("run the directions stage first")
    expected = _direction_inputs(config, direction_records)
    if _read_json(inputs_path) != expected:
        raise RuntimeError(
            "direction inputs changed in a checkpointed run directory; use a fresh outdir"
        )
    bundle = load_directions(direction_path)
    _validate_direction_bundle(bundle, config)
    return bundle


def construct_directions(config: dict, outdir: Path) -> DirectionBundle:
    records = load_prompt_records(config["data"]["prepared_jsonl"])
    direction_records, _fit, _test, _collateral = _records_for_experiment(records, config)
    direction_path = outdir / "directions.npz"
    inputs_path = outdir / "directions_inputs.json"
    if direction_path.exists() or inputs_path.exists():
        return _load_validated_directions(config, outdir, direction_records)
    expected_inputs = _direction_inputs(config, direction_records)
    plant = _build_plant(config)
    bundle = plant.capture_directions(
        direction_records,
        base_fraction=float(config["actuator"]["base_fraction"]),
    )
    _validate_direction_bundle(bundle, config)
    save_directions(direction_path, bundle)
    _write_json(inputs_path, expected_inputs)
    return bundle


def _measure_matrix(
    plant: QwenRefusalPlant,
    records: Sequence[PromptRecord],
    bundle: DirectionBundle,
    actions: np.ndarray,
    outdir: Path,
) -> np.ndarray:
    clean_path = outdir / "clean_refusal_margin.npy"
    if clean_path.exists():
        clean = np.load(clean_path)
    else:
        clean = plant.refusal_margin(records)
        np.save(clean_path, clean)
    if clean.shape != (len(records),):
        raise ValueError("clean-score checkpoint shape changed")
    matrix_path = outdir / "effects_all.npy"
    expected_shape = (len(records), len(actions))
    if matrix_path.exists():
        effects = np.lib.format.open_memmap(matrix_path, mode="r+")
        if effects.shape != expected_shape:
            raise ValueError("effect checkpoint shape changed")
    else:
        effects = np.lib.format.open_memmap(matrix_path, mode="w+", dtype="float64", shape=expected_shape)
        effects[:] = np.nan
        effects.flush()
    for mask_index, action in enumerate(actions):
        if np.isfinite(effects[:, mask_index]).all():
            continue
        if np.linalg.norm(action) <= 1e-15:
            effects[:, mask_index] = 0.0
        else:
            effects[:, mask_index] = plant.refusal_margin(records, bundle=bundle, action=action) - clean
        effects.flush()
        (outdir / "measurement_progress.json").write_text(
            json.dumps({"last_completed_action": mask_index, "n_actions": len(actions)}, indent=2) + "\n",
            encoding="utf-8",
        )
    return np.asarray(effects)


def _measurement_inputs(
    config: dict,
    records: Sequence[PromptRecord],
    actions: np.ndarray,
    outdir: Path,
) -> dict:
    return {
        "schema_version": 2,
        "config_sha256": _canonical_hash(config),
        "prepared_jsonl_sha256": _file_hash(config["data"]["prepared_jsonl"]),
        "directions_sha256": _file_hash(outdir / "directions.npz"),
        "directions_inputs_sha256": _file_hash(outdir / "directions_inputs.json"),
        "actions_sha256": _array_hash(actions),
        "records_sha256": _records_hash(records),
        "ordered_prompt_ids_sha256": hashlib.sha256(
            "\n".join(record.prompt_id for record in records).encode("utf-8")
        ).hexdigest(),
        "environment": _environment_fingerprint(),
        "source": _source_fingerprint(),
    }


def _establish_measurement_inputs(outdir: Path, expected: dict) -> Path:
    path = outdir / "measurement_inputs.json"
    if path.exists():
        if _read_json(path) != expected:
            raise RuntimeError(
                "measurement inputs changed in a checkpointed run directory; use a fresh outdir"
            )
        return path
    stale = [name for name in _MEASUREMENT_CACHE_NAMES if (outdir / name).exists()]
    if stale:
        raise RuntimeError(
            "measurement cache exists without an input fingerprint "
            f"({', '.join(stale)}); use a fresh outdir"
        )
    _write_json(path, expected)
    return path


def _write_measurement_completion(outdir: Path, config: dict) -> None:
    inputs_path = outdir / "measurement_inputs.json"
    surface_path = outdir / "surface_measurements.npz"
    completion = {
        "schema_version": 1,
        "config_sha256": _canonical_hash(config),
        "measurement_inputs_sha256": _file_hash(inputs_path),
        "surface_measurements_sha256": _file_hash(surface_path),
    }
    _write_json(outdir / "measurement_complete.json", completion)


def _verify_analysis_inputs(config: dict, outdir: Path) -> None:
    inputs_path = outdir / "measurement_inputs.json"
    completion_path = outdir / "measurement_complete.json"
    surface_path = outdir / "surface_measurements.npz"
    for path in (inputs_path, completion_path, surface_path):
        if not path.exists():
            raise RuntimeError(f"analysis requires verified measurement artifact {path.name}")
    inputs = _read_json(inputs_path)
    completion = _read_json(completion_path)
    expected_config_hash = _canonical_hash(config)
    if inputs.get("config_sha256") != expected_config_hash:
        raise RuntimeError("analysis config does not match the measured surface")
    expected_completion = {
        "schema_version": 1,
        "config_sha256": expected_config_hash,
        "measurement_inputs_sha256": _file_hash(inputs_path),
        "surface_measurements_sha256": _file_hash(surface_path),
    }
    if completion != expected_completion:
        raise RuntimeError("measurement completion or surface hash verification failed")
    prepared_path = Path(config["data"]["prepared_jsonl"])
    if not prepared_path.exists() or _file_hash(prepared_path) != inputs.get("prepared_jsonl_sha256"):
        raise RuntimeError("prepared prompt source no longer matches the measured surface")
    for filename, key in (
        ("directions.npz", "directions_sha256"),
        ("directions_inputs.json", "directions_inputs_sha256"),
    ):
        path = outdir / filename
        if not path.exists() or _file_hash(path) != inputs.get(key):
            raise RuntimeError(f"{filename} no longer matches the measured surface")


def measure_surface(config: dict, outdir: Path) -> SurfaceMeasurements:
    records = load_prompt_records(config["data"]["prepared_jsonl"])
    direction, fit, test, collateral = _records_for_experiment(records, config)
    bundle = _load_validated_directions(config, outdir, direction)
    design_config = config["design"]
    design = make_action_design(
        n_sites=len(config["actuator"]["layers"]),
        split_sizes={
            "calibration": int(design_config["n_calibration_actions"]),
            "validation": int(design_config["n_validation_actions"]),
            "test": int(design_config["n_test_actions"]),
        },
        densities=tuple(design_config["densities"]),
        fit_scales=tuple(design_config["fit_scales"]),
        validation_scales=tuple(design_config.get("validation_scales", design_config["fit_scales"])),
        heldout_scales=tuple(design_config["heldout_scales"]),
        seed=int(design_config["seed"]),
    )
    all_records = [*fit, *test, *collateral]
    _establish_measurement_inputs(
        outdir,
        _measurement_inputs(config, all_records, design.actions, outdir),
    )
    plant = _build_plant(config)
    all_effects = _measure_matrix(plant, all_records, bundle, design.actions, outdir)
    fit_end = len(fit)
    test_end = fit_end + len(test)
    surface = SurfaceMeasurements(
        design=design,
        fit_effects=all_effects[:fit_end],
        fit_groups=np.asarray([record.family for record in fit]),
        test_effects=all_effects[fit_end:test_end],
        test_groups=np.asarray([record.family for record in test]),
        test_collateral=all_effects[test_end:],
    )
    save_surface(outdir / "surface_measurements.npz", surface)
    _write_measurement_completion(outdir, config)
    return surface


def _analysis_config(config: dict) -> AnalysisConfig:
    value = config["analysis"]
    selector = config["selector"]
    return AnalysisConfig(
        ridge_grid=tuple(float(item) for item in value["ridge_grid"]),
        bootstrap_repeats=int(value["bootstrap_repeats"]),
        bootstrap_seed=int(value["bootstrap_seed"]),
        alpha=float(value["alpha"]),
        practical_relative_improvement=float(
            value.get("practical_relative_improvement", 0.05)
        ),
        selector_setpoint_quantiles=tuple(float(item) for item in selector["setpoint_quantiles"]),
    )


def run_experiment(config_path: str | Path, outdir: str | Path, stage: str) -> dict | None:
    config = load_config(config_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if stage not in {"directions", "measure", "analyze", "all"}:
        raise ValueError(f"unknown stage: {stage}")
    if stage in {"directions", "all"}:
        construct_directions(config, outdir)
    if stage in {"measure", "all"}:
        measure_surface(config, outdir)
    result = None
    if stage in {"analyze", "all"}:
        surface_path = outdir / "surface_measurements.npz"
        _verify_analysis_inputs(config, outdir)
        surface = load_surface(surface_path)
        result = analyze_surface(surface, outdir / "analysis", _analysis_config(config))
    _write_metadata(outdir, config, stage)
    return result
