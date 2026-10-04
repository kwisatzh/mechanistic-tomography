# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Bounded engineering checks only. Never loads the scientific prompt file."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from adapter import make_plant, retained
from design import costs

ROOT = Path(__file__).resolve().parent
FIXTURES = [
    "Name the months of the year.", "Explain why ice floats in water.",
    "Give a short description of a library.", "What is the difference between a square and a rectangle?",
    "Describe the steps for watering a houseplant.", "Explain how a bicycle bell makes a sound.",
    "Summarize this repeated sentence briefly. " + "The garden contains red flowers and tall trees. " * 80,
    "Count the main ideas in this description. " + "A public library lends books and provides reading space. " * 80,
]


def verify_freeze():
    freeze = json.loads((ROOT / "preflight_freeze.json").read_text())
    for name, expected in freeze["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Preflight input changed: " + name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_freeze()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "config.json").read_text())
    started = time.monotonic()
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA and native BF16 are required")
    plant = make_plant(config)
    if time.monotonic()-started > config["preflight"]["max_setup_seconds"]:
        raise TimeoutError("Setup allowance exceeded")
    rows = [retained.PromptRecord(f"engineering-{i}", text, "benign", f"engineering-{i}", "engineering")
            for i, text in enumerate(FIXTURES)]
    # Direction norms are measured on engineering prompts, not scientific cases.
    captures = {layer: [] for layer in plant.layers}
    seq = [plant._prefix_ids(r.text) for r in rows]
    ids, mask = plant._padded_batch(seq)
    plant._positions = plant._last_prompt_positions(ids.shape[1], [0]*len(rows))
    torch.cuda.synchronize()
    capture_started = time.monotonic()
    with torch.inference_mode(), plant._capture_hooks(captures):
        plant.model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
    torch.cuda.synchronize()
    capture_seconds = time.monotonic()-capture_started
    norms = np.asarray([np.median(np.linalg.norm(np.concatenate(captures[layer]), axis=1)) for layer in plant.layers])
    random = np.random.default_rng(10399).normal(size=(len(plant.layers), config["runtime"]["expected_hidden_size"]))
    vectors = random / np.linalg.norm(random, axis=1)[:, None] * (config["base_fraction"]*norms[:, None])
    bundle = retained.DirectionBundle(plant.layers, vectors, norms, config["base_fraction"], plant.model_id, plant.revision)
    bundle.validate()
    zero = np.zeros(len(plant.layers))
    coordinate = zero.copy(); coordinate[0] = min(config["scales"])
    dense = np.full(len(plant.layers), min(config["scales"])/np.sqrt(len(plant.layers)))
    # Inspect actual BF16 displacements on a short batch. Merely seeing hooks
    # run or a changed final logit does not verify their intervention location.
    actual_edits = []
    before = {}
    first = [plant._prefix_ids(r.text) for r in rows[:2]]
    edit_ids, edit_mask = plant._padded_batch(first)
    plant._positions = plant._last_prompt_positions(edit_ids.shape[1], [0, 0])
    observers = []
    try:
        for offset, layer in enumerate(plant.layers):
            def remember(_m, _a, output, layer=layer):
                h = output[0] if isinstance(output, tuple) else output
                before[layer] = h.detach().clone()
            observers.append(plant.blocks[layer].register_forward_hook(remember))
        with plant._edit_hooks(bundle, dense):
            for offset, layer in enumerate(plant.layers):
                def inspect(_m, _a, output, offset=offset, layer=layer):
                    h = output[0] if isinstance(output, tuple) else output
                    delta = h.float()-before[layer].float()
                    indices = torch.arange(h.shape[0], device=h.device)
                    observed = delta[indices, plant._positions].clone()
                    delta[indices, plant._positions] = 0
                    intended = torch.as_tensor(vectors[offset]*dense[offset], device=h.device, dtype=torch.float32)
                    relative = (observed-intended).norm(dim=1)/intended.norm()
                    actual_edits.append({"layer":layer, "max_relative_rounding_error":float(relative.max().item()),
                                         "off_position_max":float(delta.abs().max().item())})
                observers.append(plant.blocks[layer].register_forward_hook(inspect))
            with torch.inference_mode():
                plant.model(input_ids=edit_ids, attention_mask=edit_mask, use_cache=False, logits_to_keep=1)
    finally:
        for observer in observers:
            observer.remove()
    before.clear()
    records = []
    forward_counts = {"calls": 0, "sequences": 0, "input_tokens": 0}
    def counter(_module, _args, kwargs):
        forward_counts["calls"] += 1
        forward_counts["sequences"] += int(kwargs["input_ids"].shape[0])
        forward_counts["input_tokens"] += int(kwargs["attention_mask"].sum().item())
    handle = plant.model.register_forward_pre_hook(counter, with_kwargs=True)
    measurement_started = time.monotonic()
    def measure(name, action=None, batch=None):
        if time.monotonic()-measurement_started > config["preflight"]["max_measurement_seconds"]:
            raise TimeoutError("Engineering measurement allowance exceeded")
        previous = plant.batch_size
        plant.batch_size = batch or previous
        counts_before = dict(forward_counts)
        torch.cuda.synchronize(); began = time.monotonic()
        try:
            value = plant.refusal_margin(rows, bundle if action is not None else None, action)
            torch.cuda.synchronize()
        finally:
            plant.batch_size = previous
        records.append({"name": name, "seconds": time.monotonic()-began,
                        **{k: forward_counts[k]-counts_before[k] for k in forward_counts},
                        "margin": value.tolist()})
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite response")
        return value
    try:
        clean = measure("clean")
        no_op = measure("zero_edit", zero)
        edited = measure("dense_edit", dense)
        restored = measure("restored_clean")
        single = measure("dense_edit_batch_one", dense, batch=1)
        measure("coordinate_edit", coordinate)
        # Repeated timings are retained individually, not selected by speed.
        for i in range(2):
            measure(f"dense_timing_{i}", dense)
    finally:
        handle.remove()
    checks = {
        "zero_edit": bool(np.allclose(clean, no_op, atol=config["preflight"]["repeat_margin_atol"], rtol=0)),
        "restored": bool(np.allclose(clean, restored, atol=config["preflight"]["repeat_margin_atol"], rtol=0)),
        "batch_invariance": bool(np.allclose(edited, single, atol=config["preflight"]["batch_margin_atol"], rtol=0)),
        "all_layers_captured": all(len(captures[layer]) == 1 for layer in plant.layers),
        "context_bound": all(len(s)+plant._max_stem_len <= plant.max_length for s in seq),
        "no_leaked_hooks": all(not block._forward_hooks for block in plant.blocks),
        "edit_locations": len(actual_edits) == len(plant.layers) and all(r["off_position_max"] == 0 for r in actual_edits),
        "edit_rounding": all(r["max_relative_rounding_error"] <= config["preflight"]["edit_rounding_rtol"] for r in actual_edits),
    }
    dense_times = [r["seconds"]/r["sequences"] for r in records if r["name"] in {"dense_timing_0", "dense_timing_1"}]
    projected_seconds = max(dense_times) * costs(config)["total"]
    report = {
        "scope": "engineering only; no scientific outcomes collected",
        "checks": checks, "passed": all(checks.values()), "records": records,
        "max_batch_margin_difference": float(np.max(np.abs(edited-single))),
        "direction_capture_seconds": capture_seconds,
        "actual_edit_checks": actual_edits,
        "planned_evaluations": costs(config), "rough_projection_seconds": projected_seconds,
        "planning_margin_seconds": 2*projected_seconds,
        "projection_limit": "Engineering prompt mixture; does not measure the actual study length distribution.",
        "token_lengths": [len(s) for s in seq], "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "device": torch.cuda.get_device_name(), "elapsed_seconds": time.monotonic()-started,
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "numpy", "accelerate"]},
        "scientific_launch_authorized_by_worker": False,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"passed": report["passed"], "report": str(out/"report.json")}))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
