# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/content/mechanistic-tomography-qwen-followup")
CONFIG = ROOT / "configs" / "qwen2_5_7b_h200_pilot.json"
OUTDIR = ROOT / "artifacts" / "runs" / "qwen2_5_7b_a100_pilot"

os.chdir(ROOT)
started = time.monotonic()
for stage in ("directions", "measure", "analyze"):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mechtomo.cli",
            "qwen",
            "--config",
            str(CONFIG),
            "--outdir",
            str(OUTDIR),
            "--stage",
            stage,
        ],
        check=True,
    )
    print(json.dumps({"stage": stage, "elapsed_seconds": time.monotonic() - started}))

summary = json.loads((OUTDIR / "analysis" / "summary.json").read_text())
print(json.dumps({"outdir": str(OUTDIR), "primary": summary["primary"], "metrics": summary["metrics"]}, indent=2))
