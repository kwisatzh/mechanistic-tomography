# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Set explicit FP32 arithmetic, then compose the unchanged preflight."""
import json
import os
from pathlib import Path
import sys

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


def set_precision(torch):
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    settings = {"float32_matmul_precision":torch.get_float32_matmul_precision(),
                "cuda_matmul_tf32":torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32":torch.backends.cudnn.allow_tf32,
                "NVIDIA_TF32_OVERRIDE":os.environ["NVIDIA_TF32_OVERRIDE"]}
    if settings["float32_matmul_precision"] != "highest" or settings["cuda_matmul_tf32"] or settings["cudnn_tf32"]:
        raise RuntimeError("Full FP32 arithmetic was not enabled")
    return settings


def main():
    import torch
    import preflight
    settings=set_precision(torch)
    config=json.loads((preflight.ROOT/"config.json").read_text())
    if config["runtime"]["dtype"] != "float32":
        raise ValueError("This entrypoint requires the frozen FP32 configuration")
    build=preflight.make_plant
    def checked_plant(config):
        plant=build(config)
        types=sorted({str(p.dtype) for p in plant.model.parameters() if p.is_floating_point()})
        if types != ["torch.float32"]:
            raise ValueError("Model contains non-FP32 parameters")
        settings["floating_parameter_dtypes"]=types
        return plant
    preflight.make_plant=checked_plant
    try:
        preflight.main()
    finally:
        output=Path(sys.argv[sys.argv.index("--output")+1])
        if output.exists():
            (output/"precision.json").write_text(json.dumps(settings,indent=2)+"\n")


if __name__=="__main__": main()
