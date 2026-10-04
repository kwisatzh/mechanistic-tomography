# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Compose the retained MT plant; only the declared Gemma rendering changes."""
from collections.abc import Mapping
import importlib.util
from pathlib import Path
import sys


SOURCE = Path(__file__).parent / "vendor" / "qwen_refusal.py"
spec = importlib.util.spec_from_file_location("mt_retained_qwen_plant", SOURCE)
retained = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = retained
spec.loader.exec_module(retained)


class GemmaPlant(retained.QwenRefusalPlant):
    def _chat_prefix_ids(self, user_text):
        # Gemma's official template has no system role. The neutral instruction
        # is explicit content of the same user turn, not a changed template.
        content = self.system_prompt + "\n\n" + user_text
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True,
        )
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], (tuple, list)):
            if len(ids) != 1:
                raise ValueError("Expected one templated prompt")
            ids = ids[0]
        if not ids:
            raise ValueError("Empty templated prompt")
        return [int(x) for x in ids]


def make_plant(config):
    runtime = config["runtime"]
    plant = GemmaPlant(
        model_id=config["model_id"], revision=config["revision"],
        layers=config["layers"], refusal_stems=config["refusal_stems"],
        compliance_stems=config["compliance_stems"],
        system_prompt=config["system_prompt"], device=runtime["device"],
        dtype=runtime["dtype"], batch_size=runtime["batch_size"],
        max_length=runtime["max_length"],
    )
    cfg = plant.model.config
    if cfg.model_type != "gemma2":
        raise ValueError("Unexpected model architecture")
    if len(plant.blocks) != runtime["expected_layers"]:
        raise ValueError("Unexpected layer count")
    if cfg.hidden_size != runtime["expected_hidden_size"]:
        raise ValueError("Unexpected residual width")
    return plant
