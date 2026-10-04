# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Load only prompt identities and families for CPU replay; no prompt text."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parent / 'packet'

def load_inputs():
    config = json.loads((ROOT / 'config.json').read_text())
    with np.load(ROOT / 'data/designs.npz', allow_pickle=False) as z:
        designs = {k:z[k].copy() for k in z.files}
    records = [SimpleNamespace(**json.loads(s)) for s in (ROOT / 'data/prompt_metadata.jsonl').read_text().splitlines()]
    groups = {key:[r for r in records if r.split == split] for key,split in
              [('direction','direction'),('fit','fit'),('test','test_id'),('collateral','collateral_id')]}
    assert list(map(len,groups.values())) == [32,32,96,32]
    assert len({r.id for r in records}) == len(records)
    return config, designs, groups
