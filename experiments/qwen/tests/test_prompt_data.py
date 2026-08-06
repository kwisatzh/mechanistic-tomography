# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
import json

import pytest

from mechtomo.qwen_refusal import load_prompt_records, select_records


def test_prompt_contract_and_selection(tmp_path):
    path = tmp_path / "prompts.jsonl"
    rows = [
        {"id": "a", "text": "alpha", "label": "harmful", "family": "f1", "split": "fit"},
        {"id": "b", "text": "beta", "label": "benign", "family": "f2", "split": "direction"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    records = load_prompt_records(path)
    assert [record.prompt_id for record in select_records(records, "fit", "harmful")] == ["a"]


def test_duplicate_prompt_ids_fail(tmp_path):
    path = tmp_path / "prompts.jsonl"
    row = {"id": "same", "text": "x", "label": "benign", "family": "f", "split": "direction"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="unique"):
        load_prompt_records(path)
