# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Prepare fixed public inputs and a preflight freeze, without model calls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import urllib.request
import numpy as np
from design import audit, designs

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "harmbench.csv": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/8e1604d1171fe8a48d8febecd22f600e462bdcdd/data/behavior_datasets/harmbench_behaviors_text_all.csv",
    "xstest.csv": "https://raw.githubusercontent.com/paul-rottger/xstest/d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d/xstest_prompts.csv",
}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text())
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    for name, url in SOURCES.items():
        path = data / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read()
            with path.open("xb") as handle:
                handle.write(content)
    spec = importlib.util.spec_from_file_location("retained_preparation", ROOT / "vendor/prepare_prompts.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    output = data / "prompts.jsonl"
    if not output.exists():
        prep.prepare(data / "harmbench.csv", data / "xstest.csv", output,
                     seed=config["prompt_seed"], profile="custom", **config["prompt_counts"])
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    prep._assert_disjoint_outputs(prep._split_rows(rows))
    if not (data / "designs.npz").exists():
        np.savez_compressed(data / "designs.npz", **designs(config))
    # Re-read the retained design rather than trust an existing file by name.
    with np.load(data / "designs.npz", allow_pickle=False) as saved:
        for key, value in designs(config).items():
            if not np.array_equal(saved[key], value):
                raise ValueError("Saved design changed")
    if not (ROOT / "design_audit.json").exists():
        dump(ROOT / "design_audit.json", audit(config))
    if args.freeze:
        paths = [*ROOT.glob("*.py"), ROOT / "config.json", ROOT / "PROTOCOL.md",
                 ROOT / "README.md", ROOT / "requirements-preflight.txt",
                 *ROOT.glob("vendor/*.py"), *data.glob("*"), ROOT / "design_audit.json"]
        dump(ROOT / "preflight_freeze.json", {
            "scope": "protocol and engineering preflight only; full comparison not launched",
            "files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(paths) if p.is_file()},
        })
    print(json.dumps({"prepared_prompt_count": len(rows), "costs": audit(config)["cost_plan"], "frozen": args.freeze}))


if __name__ == "__main__":
    main()
