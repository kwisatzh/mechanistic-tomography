#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Verify the public Mechanistic Tomography release without model downloads."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
AUTHORSHIP = (
    "Experiments designed/concieved by Vijay Erramilli. "
    "Code written by Vijay Erramilli and Codex"
)

EXPECTED = (
    "assets/mechanistic-tomography-v1.pdf",
    "assets/mechanistic-tomography-v2.pdf",
    "paper/source/nt_mi_control_position_v21-v2.tex",
    "experiments/hmm/hmm_observer_control.py",
    "experiments/hmm/nt_mi_correspondence.py",
    "experiments/hmm/attribution_vs_finite_step0.py",
    "experiments/planted_interactions/claim3_planted_reach.py",
    "experiments/hvp_interactions/claim3_hvp_baseline.py",
    "experiments/tracr/tracr_label_basis_analysis.py",
    "experiments/tracr/tracr_residual_group_intervention_adapter.py",
    "experiments/ioi/stage1/ioi_stage1_self_repair.py",
    "experiments/ioi/stage2b/ioi_stage2b_head_subset_prediction.py",
    "experiments/ioi/stage2c/ioi_stage2c_primary_stratified.py",
    "experiments/ioi/stage2d/ioi_stage2d_per_pair_decomposition.py",
    "experiments/revision_checks/calibration_seed_factorial.py",
    "experiments/revision_checks/ioi_template_ablation_robustness.py",
    "experiments/qwen/notebooks/mechanistic_tomography_qwen_colab.ipynb",
    "experiments/qwen/artifacts/frozen/qwen2_5_7b_a100_full_results.zip",
)

CHECKSUMS = {
    "assets/mechanistic-tomography-v1.pdf":
        "c94a297cac5988fc519c9d12dfb3c92c968b4fb221de9d84c0eff153b18de375",
    "assets/mechanistic-tomography-v2.pdf":
        "70cc633c69e89425de74696bbd04c60127ff2f32f0bbea0010e85e49c9906599",
    "experiments/qwen/artifacts/frozen/qwen2_5_7b_a100_full_results.zip":
        "aca53bf0c108a0de1812edbbbf98ece0612a304a151f50cf3f13e109ac01544e",
}

QWEN_SOURCE_FINGERPRINT = (
    "74afa289364cda34e89d22ad065fb8a6332f548c2c92c546a6fc44593e09bf6d"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qwen_source_fingerprint() -> str:
    project_root = ROOT / "experiments" / "qwen"
    package_dir = project_root / "src" / "mechtomo"
    paths = sorted(package_dir.glob("*.py")) + [project_root / "pyproject.toml"]
    digest = hashlib.sha256()
    for path in sorted(paths):
        name = str(path.relative_to(project_root))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> int:
    errors: list[str] = []

    for relative in EXPECTED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    for relative, expected in CHECKSUMS.items():
        path = ROOT / relative
        if path.is_file():
            actual = sha256(path)
            if actual != expected:
                errors.append(f"checksum mismatch: {relative}: {actual}")

    source_fingerprint = qwen_source_fingerprint()
    if source_fingerprint != QWEN_SOURCE_FINGERPRINT:
        errors.append(
            "Qwen source fingerprint mismatch: "
            f"{source_fingerprint} != {QWEN_SOURCE_FINGERPRINT}"
        )

    code_files = sorted((ROOT / "experiments").rglob("*.py")) + [Path(__file__)]
    notebook_files = sorted((ROOT / "experiments").rglob("*.ipynb"))
    for path in code_files + notebook_files:
        if AUTHORSHIP not in path.read_text(encoding="utf-8"):
            errors.append(f"missing authorship notice: {path.relative_to(ROOT)}")

    private_paths = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if "sigmetrics" in path.name.lower()
    ]
    if private_paths:
        errors.append(f"private SIGMETRICS paths present: {private_paths}")

    v2_source = ROOT / "paper" / "source" / "nt_mi_control_position_v21-v2.tex"
    if v2_source.is_file() and "Independent Researcher" in v2_source.read_text(
        encoding="utf-8"
    ):
        errors.append("v2 manuscript still contains Independent Researcher")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Verified {len(EXPECTED)} required files.")
    print(f"Verified {len(CHECKSUMS)} release checksums.")
    print(f"Verified authorship notice in {len(code_files) + len(notebook_files)} code files.")
    print(f"Verified Qwen source fingerprint {source_fingerprint}.")
    print("Verified that no private SIGMETRICS working copy is present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
