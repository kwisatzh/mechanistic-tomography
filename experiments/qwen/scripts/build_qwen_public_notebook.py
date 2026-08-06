# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "mechanistic_tomography_qwen_colab.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    notebook["metadata"].update(
        {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        }
    )
    notebook["cells"] = [
        markdown(
            """# Mechanistic Tomography on Qwen-2.5-7B

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kwisatzh/mechanistic-tomography/blob/main/experiments/qwen/notebooks/mechanistic_tomography_qwen_colab.ipynb)

This notebook reproduces the Qwen result from the frozen measurements in a few minutes on CPU. It also provides an optional, resumable GPU path for rerunning the measurements.

**Question.** In a fixed Qwen-2.5-7B refusal-steering regime, do pairwise interaction features predict held-out finite interventions better than a calibrated additive map?

**Answer in this experiment.** Both maps predict the held-out surface well. The calibrated additive map reaches $R^2=0.983$ and MAE $0.00379$. Pairwise lifting does not improve held-out MAE. This supports the mechanistic-tomography decision rule: begin with the cheapest adequate measurement family and add interactions only when held-out residuals require them.

This is a weak-ground-truth study. It tests prediction of a declared behavioral intervention surface. It does not identify Qwen's true refusal mechanism, establish that interactions never matter, or support a scaling law."""
        ),
        markdown(
            """## Audience, prerequisites, and outline

This walkthrough is for mechanistic-interpretability, control, networking, and systems researchers who want to inspect or reproduce the measurement decision without first learning the repository internals.

You need Python for the frozen analysis. The optional rerun requires a Colab A100/H100-class GPU, a Hugging Face connection, and roughly two hours for the full A100 profile.

1. Install the tested package.
2. Verify and unpack the frozen result.
3. Reproduce the primary comparison.
4. Inspect a descriptive mask-density breakdown.
5. Optionally rerun the finite measurements with Drive checkpoints.
6. Interpret what the result does and does not say about mechanistic tomography."""
        ),
        code(
            """# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

RUN_GPU_EXPERIMENT = False
PROFILE = "h200_pilot"  # choose h200_pilot or h200_full when RUN_GPU_EXPERIMENT=True
REPOSITORY_URL = "https://github.com/kwisatzh/mechanistic-tomography.git"
EXPECTED_ARCHIVE_SHA256 = "aca53bf0c108a0de1812edbbbf98ece0612a304a151f50cf3f13e109ac01544e"

def find_repository() -> Path:
    working_directory = Path.cwd().resolve()
    candidates = [working_directory, *working_directory.parents, Path("/content/mechanistic-tomography")]
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "mechtomo").is_dir():
            return candidate.resolve()
        qwen_project = candidate / "experiments" / "qwen"
        if (qwen_project / "pyproject.toml").exists():
            return qwen_project.resolve()
    destination = Path("/content/mechanistic-tomography")
    subprocess.run(["git", "clone", "--depth", "1", REPOSITORY_URL, str(destination)], check=True)
    return (destination / "experiments" / "qwen").resolve()

REPO = find_repository()
REPO"""
        ),
        markdown(
            """## 1. Install the analysis path

The default route does not load Qwen or use a GPU. It installs the repository's analysis dependencies and reads the frozen, minimally processed intervention measurements."""
        ),
        code(
            """subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO}[analysis]"],
    check=True,
)

import matplotlib.pyplot as plt
import numpy as np

print({"repository": str(REPO), "python": sys.version.split()[0]})"""
        ),
        markdown(
            """## 2. Verify and unpack the frozen result

The archive contains the 401-action design, prompt-level effects, environment and source fingerprints, model directions, predictions, bootstrap draws, and reported summaries. The hash below prevents a silently changed archive from being treated as the published result."""
        ),
        code(
            """archive = REPO / "artifacts" / "frozen" / "qwen2_5_7b_a100_full_results.zip"
assert archive.exists(), f"Missing frozen archive: {archive}"
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
assert digest == EXPECTED_ARCHIVE_SHA256, (digest, EXPECTED_ARCHIVE_SHA256)

unpack_root = REPO / "artifacts" / "notebook"
run_dir = unpack_root / "qwen2_5_7b_a100_full"
if not (run_dir / "analysis" / "summary.json").exists():
    unpack_root.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(archive, unpack_root)

summary = json.loads((run_dir / "analysis" / "summary.json").read_text())
completion = json.loads((run_dir / "measurement_complete.json").read_text())
{
    "archive_sha256": digest,
    "actions": summary["n_actions"],
    "test_actions": summary["n_test_actions"],
    "test_prompts": summary["n_test_prompts"],
    "measurement_complete": completion,
}"""
        ),
        markdown(
            """## 3. Reproduce the primary comparison

The estimand is the mean finite change in a deterministic refusal-stem margin under a held-out layerwise intervention. The additive and lifted observers see the same calibration and validation measurements. Only their response-model family differs.

The primary contrast is

$$\\operatorname{MAE}(\\text{calibrated additive})-\\operatorname{MAE}(\\text{lifted}).$$

A positive value favors lifting. The preregistered success gate requires the lower confidence bound on the *relative* improvement to exceed 5%."""
        ),
        code(
            """metrics = summary["metrics"]
primary = summary["primary"]

header = "| Observer | Parameters | MAE | RMSE | R² |\\n|---|---:|---:|---:|---:|"
rows = []
for name in ("calibrated_additive", "lifted"):
    values = metrics[name]
    rows.append(
        f"| {name.replace('_', ' ')} | {values['n_parameters']} | "
        f"{values['mae']:.6f} | {values['rmse']:.6f} | {values['r2']:.4f} |"
    )
print(header + "\\n" + "\\n".join(rows))
print()
print("Status:", primary["status"])
print("Relative lifted MAE improvement:", f"{100 * primary['relative_mae_improvement']['estimate']:.2f}%")
print(
    "95% interval:",
    f"[{100 * primary['relative_mae_improvement']['low']:.2f}%, "
    f"{100 * primary['relative_mae_improvement']['high']:.2f}%]",
)"""
        ),
        markdown(
            """The result is not "MT failed on a larger model." The calibrated additive map is the MT estimate, and it predicts the held-out finite surface accurately. The diagnostic says that pairwise escalation is not supported in this declared operating region.

The interval reaches slightly above the 5% practical threshold, so this run does **not** establish formal practical equivalence. The correct statement is narrower: no lifted advantage was detected."""
        ),
        code(
            """prediction_path = run_dir / "analysis" / "test_predictions.csv"
with prediction_path.open(newline="", encoding="utf-8") as handle:
    prediction_rows = list(csv.DictReader(handle))

observed = np.asarray([float(row["observed"]) for row in prediction_rows])
additive = np.asarray([float(row["calibrated_additive"]) for row in prediction_rows])
lifted = np.asarray([float(row["lifted"]) for row in prediction_rows])

assert np.isclose(np.mean(np.abs(observed - additive)), metrics["calibrated_additive"]["mae"])
assert np.isclose(np.mean(np.abs(observed - lifted)), metrics["lifted"]["mae"])

limits = [min(observed.min(), additive.min(), lifted.min()), max(observed.max(), additive.max(), lifted.max())]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
for axis, values, title in zip(axes, [additive, lifted], ["Calibrated additive", "Lifted"]):
    axis.scatter(observed, values, s=18, alpha=0.7)
    axis.plot(limits, limits, "k--", linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("Observed held-out effect")
    axis.set_ylabel("Predicted held-out effect")
fig.suptitle("Qwen-2.5-7B finite-effect prediction")
fig.tight_layout()"""
        ),
        markdown(
            """## 4. Descriptive breakdown by mask density

All 128 test actions use the held-out intervention scale $0.75$. They are balanced across four mask densities. This breakdown was not the preregistered primary comparison; use it to understand the result, not to replace the primary endpoint."""
        ),
        code(
            """subprocess.run(
    [
        sys.executable,
        str(REPO / "scripts" / "analyze_qwen_strata.py"),
        str(run_dir),
    ],
    cwd=REPO,
    check=True,
    env={**__import__('os').environ, "PYTHONPATH": str(REPO / "src")},
    capture_output=True,
    text=True,
)
strata = json.loads((run_dir / "analysis" / "stratified_metrics.json").read_text())["rows"]

print("| Density | Actions | Additive MAE | Lifted MAE | Relative lifted improvement |")
print("|---:|---:|---:|---:|---:|")
for row in strata[1:]:
    print(
        f"| {row['density']:.2f} | {row['n_actions']} | "
        f"{row['calibrated_additive_mae']:.6f} | {row['lifted_mae']:.6f} | "
        f"{100 * row['relative_lifted_mae_improvement']:.2f}% |"
    )"""
        ),
        markdown(
            """The descriptive pattern is small and non-monotone: lifting helps by about 1.4% at densities 0.5 and 0.75, is nearly tied at 0.25, and is worse at density 1.0. That is consistent with the aggregate null. It does not support a post-hoc density-specific claim.

## 5. Optional GPU rerun

Leave `RUN_GPU_EXPERIMENT=False` to reproduce the published analysis without spending GPU time. To rerun measurements, select an A100/H100 runtime, set the flag to `True`, and choose the pilot before the full profile. The output directory lives in Google Drive, and measurement progress is flushed after every action.

The prompt sources and Qwen revision are pinned. The refusal margin uses fixed refusal and compliance stems rather than an LLM judge. Do not print or publish the harmful prompt text."""
        ),
        code(
            """if RUN_GPU_EXPERIMENT:
    import torch
    assert torch.cuda.is_available(), "Select a GPU runtime before rerunning measurements."
    assert torch.cuda.is_bf16_supported(), "The Qwen profiles require CUDA BF16."

    try:
        from google.colab import drive
        drive.mount("/content/drive")
        output_root = Path("/content/drive/MyDrive/MechanisticTomography")
    except ImportError:
        output_root = REPO / "artifacts" / "runs"

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO}[qwen,analysis]"],
        check=True,
    )
    print({"device": torch.cuda.get_device_name(0), "profile": PROFILE, "output_root": str(output_root)})
else:
    print("Frozen-analysis mode: no model loaded and no GPU used.")"""
        ),
        code(
            """if RUN_GPU_EXPERIMENT:
    raw = REPO / "data" / "raw"
    prepared = REPO / "data" / "prepared"
    raw.mkdir(parents=True, exist_ok=True)
    prepared.mkdir(parents=True, exist_ok=True)
    sources = {
        raw / "harmbench_behaviors_text_all.csv": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/8e1604d1171fe8a48d8febecd22f600e462bdcdd/data/behavior_datasets/harmbench_behaviors_text_all.csv",
        raw / "xstest_prompts.csv": "https://raw.githubusercontent.com/paul-rottger/xstest/d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d/xstest_prompts.csv",
    }
    for destination, url in sources.items():
        if not destination.exists():
            urllib.request.urlretrieve(url, destination)

    profiles = {
        "h200_pilot": {
            "config": "configs/qwen2_5_7b_h200_pilot.json",
            "data": "prompts_pilot.jsonl",
            "prepare_profile": "pilot",
            "direction": 16,
            "fit": 8,
            "test": 8,
            "collateral": 25,
        },
        "h200_full": {
            "config": "configs/qwen2_5_7b_h200_full.json",
            "data": "prompts_full.jsonl",
            "prepare_profile": "full",
            "direction": 32,
            "fit": 112,
            "test": 224,
            "collateral": 150,
        },
    }
    profile = profiles[PROFILE]
    prompt_path = prepared / profile["data"]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mechtomo.cli",
            "prepare-data",
            "--harmbench",
            str(raw / "harmbench_behaviors_text_all.csv"),
            "--xstest",
            str(raw / "xstest_prompts.csv"),
            "--out",
            str(prompt_path),
            "--profile",
            profile["prepare_profile"],
            "--direction-per-label",
            str(profile["direction"]),
            "--fit-harmful",
            str(profile["fit"]),
            "--test-harmful",
            str(profile["test"]),
            "--collateral-benign",
            str(profile["collateral"]),
        ],
        cwd=REPO,
        check=True,
    )
    print(json.loads(prompt_path.with_suffix(prompt_path.suffix + ".manifest.json").read_text()))"""
        ),
        code(
            """if RUN_GPU_EXPERIMENT:
    config_path = REPO / profile["config"]
    gpu_run_dir = output_root / f"qwen2_5_7b_{PROFILE}"
    gpu_run_dir.mkdir(parents=True, exist_ok=True)
    for stage in ("directions", "measure", "analyze"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mechtomo.cli",
                "qwen",
                "--config",
                str(config_path),
                "--outdir",
                str(gpu_run_dir),
                "--stage",
                stage,
            ],
            cwd=REPO,
            check=True,
        )
        print({"completed_stage": stage, "run_dir": str(gpu_run_dir)})

    rerun_summary = json.loads((gpu_run_dir / "analysis" / "summary.json").read_text())
    print(rerun_summary["primary"])
else:
    print("Set RUN_GPU_EXPERIMENT=True to execute this section.")"""
        ),
        markdown(
            """## Exercise: apply the MT decision rule

Suppose a future model produces a relative lifted-improvement interval of $[7\\%, 13\\%]$ under the same preregistered 5% threshold. Should the next observer keep the additive map or escalate to the lifted map?

Use the cell below as an answer scaffold. Then compare it with this run's interval."""
        ),
        code(
            """candidate_interval = {"low": 0.07, "high": 0.13}
threshold = 0.05

if candidate_interval["low"] > threshold:
    decision = "Escalate: the held-out evidence clears the practical interaction threshold."
else:
    decision = "Do not escalate on this evidence."
decision"""
        ),
        markdown(
            """## Pitfalls and extensions

- **Do not turn a null into a universal claim.** This result is conditional on Qwen-2.5-7B, this layerwise basis, refusal-margin readout, prompt distribution, action library, and intervention scale.
- **Do not infer a scaling law from GPT-2 and one Qwen model.** A scaling claim requires a matched sweep across a model family.
- **Do not confuse behavioral prediction with mechanism identification.** Qwen supplies no analytic posterior, planted support, compiler label, or complete circuit ground truth.
- **Do not select a favorable density after seeing the result.** The density table is descriptive; the aggregate paired bootstrap remains primary.

A useful next extension is a matched Qwen-family sweep with the task, basis, split, and action design held fixed. Another is a different behavioral surface where the additive residual remains structured on held-out actions.

## Take-away

Mechanistic tomography did not promise that every transformer requires pairwise recovery. It supplied the test that decides whether the extra measurements earn their cost. On this 7B model and declared surface, finite calibration carries most of the predictive work, and pairwise lifting does not improve the preregistered held-out MAE endpoint.

Paper and project page: [Mechanistic Tomography](https://kwisatzh.github.io/mechanistic-tomography/)."""
        ),
    ]
    for index, cell in enumerate(notebook["cells"], start=1):
        cell["id"] = f"mt-qwen-{index:02d}"
    NOTEBOOK.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
