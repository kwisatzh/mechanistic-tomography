# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Check the retained specificity records without importing a model library."""
import csv
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
def load(name):
    with (root / "frozen" / name / "gate_d_rotating_control.csv").open() as f:
        return {(r["variant"], float(r["alpha"])): r for r in csv.DictReader(f)}

old = load("gate_d_rotating_v1")
new = load("specificity_cpu_verification")
assert len(old) == len(new) == 8 and old.keys() == new.keys()
oracle, mixed = ("oracle_z1", 0.0), ("entangled_rotated", 0.5)
assert [round(float(old[mixed][key]), 2) for key in ("base_target_mse", "control_target_mse")] == [25.02, 4.81]
assert round(float(old[oracle]["control_target_mse"]), 2) == 3.56
assert [round(float(old[key]["collateral_q2_abs"]), 3) for key in (oracle, mixed)] == [0.037, 0.079]
for data in (old, new):
    assert float(data[mixed]["control_target_mse"]) < float(data[mixed]["base_target_mse"])
    assert float(data[mixed]["collateral_q2_abs"]) > 2 * float(data[oracle]["collateral_q2_abs"])
    assert float(data[mixed]["collateral_z2_abs"]) < float(data[oracle]["collateral_z2_abs"])
metadata = json.loads((root / "frozen/specificity_cpu_verification/execution.json").read_text())
for name, expected in metadata["input_sha256"].items():
    path = root / name if name.startswith("frozen/") else root / Path(name).name
    data = path.read_bytes()
    if name.startswith("code/"):
        # The execution receipt hashes the anonymous distribution. Only its
        # authorship comment differs from the public source; verify all other bytes.
        public = b"# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex"
        anonymous = b"# Experiments designed/concieved by Anonymous authors. Code written by Anonymous authors and Codex"
        assert data.count(public) == 1, name
        data = data.replace(public, anonymous)
    assert hashlib.sha256(data).hexdigest() == expected, name
print(json.dumps({"passed": True, "historical_rounded_values_verified": True,
                  "cpu_probability_pattern_verified": True,
                  "log_odds_movement_does_not_increase": True,
                  "model_executed": False}, indent=2))
