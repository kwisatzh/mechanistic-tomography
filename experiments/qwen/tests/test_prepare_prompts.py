# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
import csv
import json

from mechtomo.prepare_prompts import PROFILE_COUNTS, prepare


def _write_sources(tmp_path):
    harmbench = tmp_path / "harmbench.csv"
    with harmbench.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["BehaviorID", "Behavior"])
        writer.writeheader()
        for index in range(400):
            behavior = (
                "Repeated harmful request"
                if index in {120, 121, 122}
                else f"Harmful request {index}"
            )
            writer.writerow({
                "BehaviorID": f"h{index:03d}",
                "Behavior": behavior,
            })

    xstest = tmp_path / "xstest.csv"
    with xstest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "prompt", "type", "label"])
        writer.writeheader()
        for family_index in range(10):
            for prompt_index in range(25):
                writer.writerow({
                    "id": family_index * 25 + prompt_index + 1,
                    "prompt": f"Safe request {family_index}-{prompt_index}",
                    "type": f"safe_type_{family_index}",
                    "label": "safe",
                })
    return harmbench, xstest


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _prepare_profile(harmbench, xstest, out, profile):
    counts = PROFILE_COUNTS[profile]
    return prepare(
        harmbench,
        xstest,
        out,
        counts["direction_per_label"],
        counts["fit_harmful"],
        counts["test_harmful"],
        counts["collateral_benign"],
        seed=2501,
        profile=profile,
    )


def test_pilot_is_reserved_and_full_uses_disjoint_family_assignments(tmp_path):
    harmbench, xstest = _write_sources(tmp_path)
    pilot_path = tmp_path / "pilot.jsonl"
    full_path = tmp_path / "full.jsonl"
    pilot_manifest = _prepare_profile(harmbench, xstest, pilot_path, "pilot")
    full_manifest = _prepare_profile(harmbench, xstest, full_path, "full")
    pilot = _rows(pilot_path)
    full = _rows(full_path)

    def count(rows, split, label):
        return sum(row["split"] == split and row["label"] == label for row in rows)

    assert count(pilot, "direction", "harmful") == 16
    assert count(pilot, "direction", "benign") == 16
    assert count(pilot, "fit", "harmful") == 8
    assert count(pilot, "test_id", "harmful") == 8
    assert count(pilot, "collateral_id", "benign") == 25
    assert count(full, "direction", "harmful") == 32
    assert count(full, "direction", "benign") == 32
    assert count(full, "fit", "harmful") == 112
    assert count(full, "test_id", "harmful") == 224
    assert count(full, "collateral_id", "benign") == 150

    for rows in (pilot, full):
        family_splits = {}
        for row in rows:
            family_splits.setdefault(row["family"], set()).add(row["split"])
        assert all(len(splits) == 1 for splits in family_splits.values())

    assert {row["id"] for row in pilot}.isdisjoint(row["id"] for row in full)
    assert {row["family"] for row in pilot}.isdisjoint(row["family"] for row in full)
    assert {
        " ".join(row["text"].casefold().split()) for row in pilot
    }.isdisjoint(" ".join(row["text"].casefold().split()) for row in full)
    assert pilot_manifest["selection"]["discarded_rows_from_assigned_families"] == 9
    assert full_manifest["selection"]["discarded_rows_from_assigned_families"] == 27
