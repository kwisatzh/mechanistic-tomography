# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable
import unicodedata


PROFILE_COUNTS = {
    "pilot": {
        "direction_per_label": 16,
        "fit_harmful": 8,
        "test_harmful": 8,
        "collateral_benign": 25,
    },
    "full": {
        "direction_per_label": 32,
        "fit_harmful": 112,
        "test_harmful": 224,
        "collateral_benign": 150,
    },
}

CUSTOM_DEFAULT_COUNTS = {
    "direction_per_label": 16,
    "fit_harmful": 32,
    "test_harmful": 64,
    "collateral_benign": 64,
}


def _first(row: dict[str, str], names: Iterable[str]) -> str:
    lowered = {key.lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower(), "")
        if value and value.strip():
            return value.strip()
    return ""


def _stable_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _harmbench_records(path: Path, seed: int) -> list[dict]:
    records = []
    for index, row in enumerate(_read_csv(path)):
        text = _first(row, ("Behavior", "prompt", "goal", "request"))
        if not text:
            continue
        prompt_id = _first(row, ("BehaviorID", "behavior_id", "id")) or f"harmbench-{index:05d}"
        # HarmBench contains a few distinct BehaviorIDs with exactly repeated
        # behavior text. Treat those rows as one leakage family while retaining
        # every canonical ID in the assigned split.
        normalized = _normalized_text(text)
        family = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
        records.append({
            "id": f"harmbench:{prompt_id}",
            "text": text,
            "label": "harmful",
            "family": f"harmbench:text:{family}",
            "source": "HarmBench",
        })
    return sorted(records, key=lambda row: _stable_order(row["family"], seed))


def _xstest_benign_records(path: Path, seed: int) -> list[dict]:
    records = []
    for index, row in enumerate(_read_csv(path)):
        prompt_type = _first(row, ("type", "prompt_type", "category"))
        if prompt_type.lower().startswith("contrast_"):
            continue
        text = _first(row, ("prompt", "text", "goal"))
        if not text:
            continue
        prompt_id = _first(row, ("id", "prompt_id")) or f"xstest-{index:04d}"
        family = prompt_type or prompt_id
        records.append({
            "id": f"xstest:{prompt_id}",
            "text": text,
            "label": "benign",
            "family": f"xstest:{family}",
            "source": "XSTest",
        })
    return sorted(
        records,
        key=lambda row: (
            _stable_order(row["family"], seed),
            _stable_order(row["id"], seed),
        ),
    )


def _take_whole_families(
    records: list[dict],
    used_families: set[str],
    count: int,
    split: str,
    allow_partial_family: bool,
) -> list[dict]:
    if count < 0:
        raise ValueError(f"requested a negative record count for {split}")
    if count == 0:
        return []
    grouped: dict[str, list[dict]] = {}
    for row in records:
        if row["family"] not in used_families:
            grouped.setdefault(row["family"], []).append(row)

    if not allow_partial_family:
        # HarmBench duplicate-text clusters must remain intact. A deterministic
        # subset-sum is cheap here because almost every family is a singleton.
        solutions: dict[int, tuple[str, ...]] = {0: ()}
        for family, family_rows in grouped.items():
            size = len(family_rows)
            additions: dict[int, tuple[str, ...]] = {}
            for total, selected_families in tuple(solutions.items()):
                new_total = total + size
                if new_total <= count and new_total not in solutions and new_total not in additions:
                    additions[new_total] = (*selected_families, family)
            solutions.update(additions)
            if count in solutions:
                selected_families = solutions[count]
                used_families.update(selected_families)
                return [
                    {**row, "split": split}
                    for selected_family in selected_families
                    for row in grouped[selected_family]
                ]
    else:
        selected: list[dict] = []
        for family, family_rows in grouped.items():
            # An XSTest type belongs to only one split/profile. If the exact
            # target is reached partway through it, unused rows are discarded.
            used_families.add(family)
            needed = count - len(selected)
            selected.extend({**row, "split": split} for row in family_rows[:needed])
            if len(selected) == count:
                return selected
    available = sum(len(rows) for rows in grouped.values())
    raise ValueError(
        f"requested {count} records for {split}, but only {available} records "
        "remain in unassigned families"
    )


def _select_prompt_splits(
    harmful: list[dict],
    benign: list[dict],
    counts: dict[str, int],
) -> tuple[list[dict], set[str], set[str]]:
    harmful_families: set[str] = set()
    benign_families: set[str] = set()
    output = [
        *_take_whole_families(
            harmful,
            harmful_families,
            counts["direction_per_label"],
            "direction",
            allow_partial_family=False,
        ),
        *_take_whole_families(
            benign,
            benign_families,
            counts["direction_per_label"],
            "direction",
            allow_partial_family=True,
        ),
        *_take_whole_families(
            harmful,
            harmful_families,
            counts["fit_harmful"],
            "fit",
            allow_partial_family=False,
        ),
        *_take_whole_families(
            harmful,
            harmful_families,
            counts["test_harmful"],
            "test_id",
            allow_partial_family=False,
        ),
        *_take_whole_families(
            benign,
            benign_families,
            counts["collateral_benign"],
            "collateral_id",
            allow_partial_family=True,
        ),
    ]
    return output, harmful_families, benign_families


def _reserve_families(
    harmful: list[dict],
    benign: list[dict],
    counts: dict[str, int],
) -> tuple[list[dict], list[dict]]:
    selected, harmful_families, benign_families = _select_prompt_splits(harmful, benign, counts)
    selected_ids = {row["id"] for row in selected}
    reserved_harmful = [
        row for row in harmful if row["family"] in harmful_families and row["id"] in selected_ids
    ]
    reserved_benign = [
        row for row in benign if row["family"] in benign_families and row["id"] in selected_ids
    ]
    return reserved_harmful, reserved_benign


def _assert_disjoint_outputs(named_outputs: dict[str, list[dict]]) -> None:
    names = tuple(named_outputs)
    keys = {
        name: {
            "id": {row["id"] for row in rows},
            "family": {row["family"] for row in rows},
            "normalized text": {_normalized_text(row["text"]) for row in rows},
        }
        for name, rows in named_outputs.items()
    }
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            for key_name in ("id", "family", "normalized text"):
                overlap = keys[left][key_name] & keys[right][key_name]
                if overlap:
                    example = sorted(overlap)[0]
                    raise ValueError(
                        f"{left} and {right} overlap by {key_name}: {example!r}"
                    )


def _split_rows(output: list[dict]) -> dict[str, list[dict]]:
    return {
        "direction": [row for row in output if row["split"] == "direction"],
        "fit": [row for row in output if row["split"] == "fit"],
        "test": [row for row in output if row["split"] == "test_id"],
        "collateral": [row for row in output if row["split"] == "collateral_id"],
    }


def prepare(
    harmbench: Path,
    xstest: Path,
    out: Path,
    direction_per_label: int,
    fit_harmful: int,
    test_harmful: int,
    collateral_benign: int,
    seed: int,
    profile: str = "custom",
) -> dict:
    if profile not in {"custom", *PROFILE_COUNTS}:
        raise ValueError(f"unknown preparation profile: {profile}")
    counts = {
        "direction_per_label": int(direction_per_label),
        "fit_harmful": int(fit_harmful),
        "test_harmful": int(test_harmful),
        "collateral_benign": int(collateral_benign),
    }
    if profile in PROFILE_COUNTS and counts != PROFILE_COUNTS[profile]:
        raise ValueError(f"{profile} profile requires counts {PROFILE_COUNTS[profile]}")
    harmful = _harmbench_records(harmbench, seed)
    benign = _xstest_benign_records(xstest, seed)
    all_source_rows = [*harmful, *benign]
    reserved_harmful: list[dict] = []
    reserved_benign: list[dict] = []
    reserved_harmful_families: set[str] = set()
    reserved_benign_families: set[str] = set()
    if profile == "full":
        reserved_harmful, reserved_benign = _reserve_families(
            harmful,
            benign,
            PROFILE_COUNTS["pilot"],
        )
        reserved_harmful_families = {row["family"] for row in reserved_harmful}
        reserved_benign_families = {row["family"] for row in reserved_benign}
        harmful = [row for row in harmful if row["family"] not in reserved_harmful_families]
        benign = [row for row in benign if row["family"] not in reserved_benign_families]

    output, harmful_families, benign_families = _select_prompt_splits(
        harmful,
        benign,
        counts,
    )
    _assert_disjoint_outputs(_split_rows(output))
    if profile == "full":
        _assert_disjoint_outputs(
            {
                "pilot reservation": [*reserved_harmful, *reserved_benign],
                "full": output,
            }
        )
    selected_or_reserved_ids = {
        row["id"] for row in [*output, *reserved_harmful, *reserved_benign]
    }
    assigned_families = {
        *harmful_families,
        *benign_families,
        *reserved_harmful_families,
        *reserved_benign_families,
    }
    discarded_assigned_ids = sorted(
        row["id"]
        for row in all_source_rows
        if row["family"] in assigned_families and row["id"] not in selected_or_reserved_ids
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 2,
        "profile": profile,
        "seed": seed,
        "inputs": {
            "harmbench_file": harmbench.name,
            "harmbench_sha256": hashlib.sha256(harmbench.read_bytes()).hexdigest(),
            "xstest_file": xstest.name,
            "xstest_sha256": hashlib.sha256(xstest.read_bytes()).hexdigest(),
        },
        "output_file": out.name,
        "output_sha256": digest,
        "counts": {
            "direction_harmful": counts["direction_per_label"],
            "direction_benign": counts["direction_per_label"],
            "fit_harmful": counts["fit_harmful"],
            "test_harmful": counts["test_harmful"],
            "collateral_benign": counts["collateral_benign"],
        },
        "selection": {
            "family_atomic": True,
            "pilot_reserved": profile == "full",
            "reserved_pilot_harmful": len(reserved_harmful),
            "reserved_pilot_benign": len(reserved_benign),
            "reserved_pilot_ids_sha256": hashlib.sha256(
                "\n".join(
                    sorted(row["id"] for row in [*reserved_harmful, *reserved_benign])
                ).encode("utf-8")
            ).hexdigest(),
            "selected_harmful_families": len(harmful_families),
            "selected_benign_families": len(benign_families),
            "selected_families_sha256": hashlib.sha256(
                "\n".join(sorted({*harmful_families, *benign_families})).encode("utf-8")
            ).hexdigest(),
            "discarded_rows_from_assigned_families": len(discarded_assigned_ids),
            "discarded_assigned_ids_sha256": hashlib.sha256(
                "\n".join(discarded_assigned_ids).encode("utf-8")
            ).hexdigest(),
        },
    }
    out.with_suffix(out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--harmbench", type=Path, required=True)
    parser.add_argument("--xstest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=("custom", "pilot", "full"), default="custom")
    parser.add_argument("--direction-per-label", type=int)
    parser.add_argument("--fit-harmful", type=int)
    parser.add_argument("--test-harmful", type=int)
    parser.add_argument("--collateral-benign", type=int)
    parser.add_argument("--seed", type=int, default=2501)


def run_from_args(args: argparse.Namespace) -> None:
    requested = PROFILE_COUNTS.get(args.profile, CUSTOM_DEFAULT_COUNTS)
    counts = {
        name: requested[name] if getattr(args, name) is None else getattr(args, name)
        for name in CUSTOM_DEFAULT_COUNTS
    }
    manifest = prepare(
        args.harmbench,
        args.xstest,
        args.out,
        counts["direction_per_label"],
        counts["fit_harmful"],
        counts["test_harmful"],
        counts["collateral_benign"],
        args.seed,
        profile=args.profile,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
