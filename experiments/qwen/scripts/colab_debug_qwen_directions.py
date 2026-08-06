# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


root = Path("/content/mechanistic-tomography-qwen-followup")
os.chdir(root)
command = [
    sys.executable,
    "-m",
    "mechtomo.cli",
    "qwen",
    "--config",
    str(root / "configs" / "qwen2_5_7b_h200_pilot.json"),
    "--outdir",
    str(root / "artifacts" / "runs" / "qwen2_5_7b_a100_pilot"),
    "--stage",
    "directions",
]
result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result.stdout)
print(f"exit_code={result.returncode}")
