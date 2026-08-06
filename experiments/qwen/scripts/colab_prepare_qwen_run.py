# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import urllib.request


REPOSITORY = "https://github.com/kwisatzh/mechanistic-tomography.git"
REPOSITORY_ROOT = Path("/content/mechanistic-tomography")
ROOT = REPOSITORY_ROOT / "experiments" / "qwen"


def run(*args: str) -> None:
    subprocess.run(list(args), check=True)


if not REPOSITORY_ROOT.exists():
    run("git", "clone", "--depth", "1", REPOSITORY, str(REPOSITORY_ROOT))

run(sys.executable, "-m", "pip", "install", "-q", "-e", f"{ROOT}[qwen,analysis]")

raw = ROOT / "data" / "raw"
prepared = ROOT / "data" / "prepared"
raw.mkdir(parents=True, exist_ok=True)
prepared.mkdir(parents=True, exist_ok=True)

sources = {
    raw / "harmbench_behaviors_text_all.csv": (
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
        "8e1604d1171fe8a48d8febecd22f600e462bdcdd/"
        "data/behavior_datasets/harmbench_behaviors_text_all.csv"
    ),
    raw / "xstest_prompts.csv": (
        "https://raw.githubusercontent.com/paul-rottger/xstest/"
        "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d/xstest_prompts.csv"
    ),
}
for destination, url in sources.items():
    if not destination.exists():
        urllib.request.urlretrieve(url, destination)

profiles = {
    "pilot": {
        "out": prepared / "prompts_pilot.jsonl",
        "direction": "16",
        "fit": "8",
        "test": "8",
        "collateral": "25",
    },
    "full": {
        "out": prepared / "prompts_full.jsonl",
        "direction": "32",
        "fit": "112",
        "test": "224",
        "collateral": "150",
    },
}
for profile, values in profiles.items():
    run(
        sys.executable,
        "-m",
        "mechtomo.cli",
        "prepare-data",
        "--harmbench",
        str(raw / "harmbench_behaviors_text_all.csv"),
        "--xstest",
        str(raw / "xstest_prompts.csv"),
        "--out",
        str(values["out"]),
        "--profile",
        profile,
        "--direction-per-label",
        values["direction"],
        "--fit-harmful",
        values["fit"],
        "--test-harmful",
        values["test"],
        "--collateral-benign",
        values["collateral"],
    )

import torch

hardware = {
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
}
if not hardware["cuda_available"] or not hardware["bf16_supported"]:
    raise RuntimeError(f"Qwen scientific run requires CUDA BF16: {hardware}")

print(json.dumps({"root": str(ROOT), "hardware": hardware}, indent=2))
