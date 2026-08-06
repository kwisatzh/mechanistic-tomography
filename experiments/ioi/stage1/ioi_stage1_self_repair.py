#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
Stage-1 IOI self-repair feasibility test.

Goal: measure whether published IOI primary Name Mover heads and Backup Name
Mover heads exhibit a non-additive compensation term under group ablation.

Effects are measured as positive drops in IOI logit difference:
    drop(G) = LD_clean - LD_ablate(G)
so the self-repair interaction is
    interaction = drop(P+B) - drop(P) - drop(B).

A positive interaction means that ablating both groups hurts more than the sum
of ablating each group in isolation: the backup group matters conditionally on
primary name movers being removed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

try:
    from transformer_lens import HookedTransformer, utils
except Exception as e:  # pragma: no cover - helpful runtime error
    HookedTransformer = None
    utils = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


DEFAULT_PRIMARY_HEADS = "9.9,9.6,10.0"
# Wang et al. IOI Figure 2 / Appendix backup name mover list.
DEFAULT_BACKUP_HEADS = "9.0,9.7,10.1,10.2,10.6,10.10,11.2,11.9"

# Names chosen to usually be single GPT-2 tokens when preceded by a space.
DEFAULT_NAMES = [
    "John", "Mary", "Bob", "Alice", "Tom", "Sarah", "James", "Emily",
    "Michael", "Jessica", "David", "Laura", "Daniel", "Emma", "Robert",
    "Anna", "Jacob", "Scott", "Kevin", "Lisa", "Mark", "Susan", "Steven",
    "Rachel", "Brian", "Karen", "Jason", "Nancy", "Eric", "Helen",
]

DEFAULT_TEMPLATE = "When {io} and {s} went to the store, {s} gave a bottle of milk to"


@dataclass
class PromptRecord:
    prompt: str
    io_name: str
    s_name: str
    io_token: int
    s_token: int


@dataclass
class ConditionResult:
    ablation: str
    position_mode: str
    condition: str
    heads: str
    logit_diff_mean: float
    logit_diff_std: float
    drop_from_clean: float
    n_prompts: int


def parse_heads(spec: str) -> List[Tuple[int, int]]:
    if not spec.strip():
        return []
    heads: List[Tuple[int, int]] = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if '.' not in item:
            raise ValueError(f"Head '{item}' must use layer.head notation, e.g. 9.9")
        layer_s, head_s = item.split('.', 1)
        heads.append((int(layer_s), int(head_s)))
    return heads


def format_heads(heads: Sequence[Tuple[int, int]]) -> str:
    return ','.join(f"{l}.{h}" for l, h in heads)


def choose_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str, device: str):
    if HookedTransformer is None:
        raise ImportError(
            "Could not import transformer_lens. Install with: pip install transformer-lens\n"
            f"Original import error: {_IMPORT_ERR!r}"
        )
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    return model


def single_token_names(model, names: Sequence[str]) -> Tuple[List[str], Dict[str, int]]:
    good: List[str] = []
    toks: Dict[str, int] = {}
    for name in names:
        try:
            tok = int(model.to_single_token(" " + name))
        except Exception:
            continue
        good.append(name)
        toks[name] = tok
    if len(good) < 4:
        raise RuntimeError(
            "Too few single-token names found. Try passing --names with GPT-2 single-token names."
        )
    return good, toks


def make_prompts(
    model,
    n: int,
    seed: int,
    template: str = DEFAULT_TEMPLATE,
    names: Optional[Sequence[str]] = None,
) -> List[PromptRecord]:
    rng = random.Random(seed)
    names = list(names or DEFAULT_NAMES)
    names, toks = single_token_names(model, names)
    out: List[PromptRecord] = []
    seen = set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        io, s = rng.sample(names, 2)
        prompt = template.format(io=io, s=s)
        key = (prompt, io, s)
        if key in seen:
            continue
        seen.add(key)
        out.append(PromptRecord(prompt=prompt, io_name=io, s_name=s, io_token=toks[io], s_token=toks[s]))
    if len(out) < n:
        # Duplicates are fine; random sample with replacement if needed.
        while len(out) < n:
            io, s = rng.sample(names, 2)
            prompt = template.format(io=io, s=s)
            out.append(PromptRecord(prompt=prompt, io_name=io, s_name=s, io_token=toks[io], s_token=toks[s]))
    return out


def tokens_for_records(model, records: Sequence[PromptRecord], device: str):
    prompts = [r.prompt for r in records]
    toks = model.to_tokens(prompts, prepend_bos=False).to(device)
    return toks


def answer_token_tensors(records: Sequence[PromptRecord], device: str):
    io = torch.tensor([r.io_token for r in records], dtype=torch.long, device=device)
    s = torch.tensor([r.s_token for r in records], dtype=torch.long, device=device)
    return io, s


def chunk_indices(n: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


def compute_mean_z(model, tokens: torch.Tensor, batch_size: int) -> Dict[int, torch.Tensor]:
    """Return per-layer mean hook_z, shape [pos, n_heads, d_head]."""
    sums: Dict[int, torch.Tensor] = {}
    count = 0
    names_filter = lambda name: name.endswith("hook_z")
    for a, b in tqdm(list(chunk_indices(tokens.shape[0], batch_size)), desc="Computing reference means"):
        batch = tokens[a:b]
        with torch.no_grad():
            _, cache = model.run_with_cache(batch, names_filter=names_filter, return_type="logits")
        bs = batch.shape[0]
        count += bs
        for layer in range(model.cfg.n_layers):
            name = utils.get_act_name("z", layer)
            z = cache[name].detach().float().cpu()  # [batch,pos,heads,d_head]
            if layer not in sums:
                sums[layer] = z.sum(dim=0)
            else:
                sums[layer] += z.sum(dim=0)
        del cache
    means = {layer: s / float(count) for layer, s in sums.items()}
    return means


def make_ablation_hooks(
    model,
    heads: Sequence[Tuple[int, int]],
    ablation: str,
    mean_z: Optional[Dict[int, torch.Tensor]],
    position_mode: str,
):
    by_layer: Dict[int, List[int]] = {}
    for layer, head in heads:
        if layer < 0 or layer >= model.cfg.n_layers:
            raise ValueError(f"Layer {layer} out of range for model with {model.cfg.n_layers} layers")
        if head < 0 or head >= model.cfg.n_heads:
            raise ValueError(f"Head {head} out of range for model with {model.cfg.n_heads} heads")
        by_layer.setdefault(layer, []).append(head)

    hooks = []
    for layer, head_list in by_layer.items():
        hook_name = utils.get_act_name("z", layer)

        def hook_fn(z, hook, layer=layer, head_list=head_list):
            z = z.clone()
            pos_len = z.shape[1]
            for h in head_list:
                if ablation == "zero":
                    if position_mode == "end":
                        z[:, -1, h, :] = 0.0
                    elif position_mode == "all":
                        z[:, :, h, :] = 0.0
                    else:
                        raise ValueError(position_mode)
                elif ablation == "mean":
                    if mean_z is None:
                        raise ValueError("mean_z required for mean ablation")
                    ref = mean_z[layer].to(z.device, dtype=z.dtype)
                    if ref.shape[0] < pos_len:
                        raise ValueError(
                            f"Reference mean has pos length {ref.shape[0]}, but batch has {pos_len}. "
                            "Use fixed-length prompts/templates."
                        )
                    if position_mode == "end":
                        z[:, -1, h, :] = ref[pos_len - 1, h, :]
                    elif position_mode == "all":
                        z[:, :, h, :] = ref[:pos_len, h, :].unsqueeze(0)
                    else:
                        raise ValueError(position_mode)
                else:
                    raise ValueError(ablation)
            return z

        hooks.append((hook_name, hook_fn))
    return hooks


def compute_logit_diffs(
    model,
    tokens: torch.Tensor,
    io_tokens: torch.Tensor,
    s_tokens: torch.Tensor,
    batch_size: int,
    heads: Optional[Sequence[Tuple[int, int]]] = None,
    ablation: str = "none",
    mean_z: Optional[Dict[int, torch.Tensor]] = None,
    position_mode: str = "end",
) -> np.ndarray:
    vals: List[np.ndarray] = []
    hooks = []
    if heads:
        hooks = make_ablation_hooks(model, heads, ablation=ablation, mean_z=mean_z, position_mode=position_mode)
    for a, b in tqdm(list(chunk_indices(tokens.shape[0], batch_size)), desc=f"Run {ablation}:{format_heads(heads or []) or 'clean'}"):
        batch = tokens[a:b]
        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(batch, return_type="logits", fwd_hooks=hooks)
            else:
                logits = model(batch)
            final_logits = logits[:, -1, :]
            diff = final_logits.gather(1, io_tokens[a:b, None]).squeeze(1) - final_logits.gather(1, s_tokens[a:b, None]).squeeze(1)
            vals.append(diff.detach().float().cpu().numpy())
    return np.concatenate(vals, axis=0)


def summarize_condition(ablation: str, position_mode: str, condition: str, heads: Sequence[Tuple[int, int]], ld: np.ndarray, clean_mean: float) -> ConditionResult:
    return ConditionResult(
        ablation=ablation,
        position_mode=position_mode,
        condition=condition,
        heads=format_heads(heads),
        logit_diff_mean=float(np.mean(ld)),
        logit_diff_std=float(np.std(ld, ddof=1)) if len(ld) > 1 else 0.0,
        drop_from_clean=float(clean_mean - np.mean(ld)),
        n_prompts=int(len(ld)),
    )


def go_nogo(relative_interaction: float, go_threshold: float, diagnose_threshold: float) -> str:
    if relative_interaction >= go_threshold:
        return "GO: build Stage-2 IOI ObserverBench task"
    if relative_interaction >= diagnose_threshold:
        return "DIAGNOSE: signal present but weak; inspect heads/ablation/prompts"
    return "STOP/REDESIGN: interaction too small for Stage-2"


def plot_stage1(summary_df: pd.DataFrame, outpath: Path):
    import matplotlib.pyplot as plt

    ablations = list(summary_df["ablation"].unique())
    n = len(ablations)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 4), squeeze=False)
    for ax, ablation in zip(axes[0], ablations):
        row = summary_df[summary_df["ablation"] == ablation].iloc[0]
        labels = ["primary", "backup", "both", "additive"]
        values = [row["drop_primary"], row["drop_backup"], row["drop_both"], row["drop_primary"] + row["drop_backup"]]
        ax.bar(labels, values)
        ax.set_title(f"{ablation} ablation")
        ax.set_ylabel("drop in IOI logit diff")
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.text(0.02, 0.95, f"interaction={row['interaction']:.3f}\nrel={row['relative_interaction']:.2f}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def run_one_ablation(
    model,
    tokens: torch.Tensor,
    io_tokens: torch.Tensor,
    s_tokens: torch.Tensor,
    primary: Sequence[Tuple[int, int]],
    backup: Sequence[Tuple[int, int]],
    ablation: str,
    mean_z: Optional[Dict[int, torch.Tensor]],
    position_mode: str,
    batch_size: int,
    go_threshold: float,
    diagnose_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    clean = compute_logit_diffs(model, tokens, io_tokens, s_tokens, batch_size=batch_size)
    clean_mean = float(np.mean(clean))
    primary_ld = compute_logit_diffs(model, tokens, io_tokens, s_tokens, batch_size=batch_size, heads=primary, ablation=ablation, mean_z=mean_z, position_mode=position_mode)
    backup_ld = compute_logit_diffs(model, tokens, io_tokens, s_tokens, batch_size=batch_size, heads=backup, ablation=ablation, mean_z=mean_z, position_mode=position_mode)
    both_heads = list(primary) + list(backup)
    both_ld = compute_logit_diffs(model, tokens, io_tokens, s_tokens, batch_size=batch_size, heads=both_heads, ablation=ablation, mean_z=mean_z, position_mode=position_mode)

    conds = [
        summarize_condition(ablation, position_mode, "clean", [], clean, clean_mean),
        summarize_condition(ablation, position_mode, "primary_ablated", primary, primary_ld, clean_mean),
        summarize_condition(ablation, position_mode, "backup_ablated", backup, backup_ld, clean_mean),
        summarize_condition(ablation, position_mode, "primary_plus_backup_ablated", both_heads, both_ld, clean_mean),
    ]
    cond_df = pd.DataFrame([asdict(c) for c in conds])

    A = float(clean_mean - np.mean(primary_ld))
    B = float(clean_mean - np.mean(backup_ld))
    AB = float(clean_mean - np.mean(both_ld))
    interaction = AB - A - B
    relative = interaction / max(abs(A), 1e-12)
    summary = {
        "ablation": ablation,
        "position_mode": position_mode,
        "clean_logit_diff": clean_mean,
        "drop_primary": A,
        "drop_backup": B,
        "drop_both": AB,
        "additive_prediction": A + B,
        "interaction": interaction,
        "relative_interaction": relative,
        "go_threshold": go_threshold,
        "diagnose_threshold": diagnose_threshold,
        "decision": go_nogo(relative, go_threshold, diagnose_threshold),
    }
    return cond_df, summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Stage-1 IOI self-repair feasibility test")
    p.add_argument("--outdir", type=str, default="runs/ioi_stage1_self_repair")
    p.add_argument("--model", type=str, default="gpt2-small")
    p.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, mps")
    p.add_argument("--n-prompts", type=int, default=256)
    p.add_argument("--n-reference", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--template", type=str, default=DEFAULT_TEMPLATE)
    p.add_argument("--primary-heads", type=str, default=DEFAULT_PRIMARY_HEADS)
    p.add_argument("--backup-heads", type=str, default=DEFAULT_BACKUP_HEADS)
    p.add_argument("--ablation", type=str, default="mean", choices=["mean", "zero", "both"])
    p.add_argument("--positions", type=str, default="end", choices=["end", "all"], help="ablate only final prediction position or all positions")
    p.add_argument("--go-threshold", type=float, default=0.30)
    p.add_argument("--diagnose-threshold", type=float, default=0.10)
    p.add_argument("--names", type=str, default="", help="Optional comma-separated names to use")
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    torch.set_grad_enabled(False)
    print(f"Loading {args.model} on {device}...")
    model = load_model(args.model, device)

    primary = parse_heads(args.primary_heads)
    backup = parse_heads(args.backup_heads)
    names = [x.strip() for x in args.names.split(',') if x.strip()] if args.names else None

    records = make_prompts(model, args.n_prompts, seed=args.seed, template=args.template, names=names)
    ref_records = make_prompts(model, args.n_reference, seed=args.seed + 10_000, template=args.template, names=names)

    # Validate fixed token length; mean ablation with per-position means assumes this.
    tokens = tokens_for_records(model, records, device)
    ref_tokens = tokens_for_records(model, ref_records, device)
    if tokens.shape[1] != ref_tokens.shape[1]:
        raise RuntimeError(f"Eval token length {tokens.shape[1]} != reference token length {ref_tokens.shape[1]}")

    io_tokens, s_tokens = answer_token_tensors(records, device)

    pd.DataFrame([asdict(r) for r in records]).to_csv(outdir / "ioi_stage1_prompts.csv", index=False)

    mean_z = None
    ablations = [args.ablation]
    if args.ablation == "both":
        ablations = ["mean", "zero"]
    if "mean" in ablations:
        mean_z = compute_mean_z(model, ref_tokens, batch_size=args.batch_size)

    all_conditions = []
    summaries = []
    for ablation in ablations:
        cond_df, summary = run_one_ablation(
            model=model,
            tokens=tokens,
            io_tokens=io_tokens,
            s_tokens=s_tokens,
            primary=primary,
            backup=backup,
            ablation=ablation,
            mean_z=mean_z,
            position_mode=args.positions,
            batch_size=args.batch_size,
            go_threshold=args.go_threshold,
            diagnose_threshold=args.diagnose_threshold,
        )
        all_conditions.append(cond_df)
        summaries.append(summary)

    conditions_df = pd.concat(all_conditions, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    conditions_df.to_csv(outdir / "ioi_stage1_condition_results.csv", index=False)
    summary_df.to_csv(outdir / "ioi_stage1_summary.csv", index=False)

    metadata = {
        "args": vars(args),
        "device": device,
        "model": args.model,
        "primary_heads": primary,
        "backup_heads": backup,
        "n_prompts": len(records),
        "n_reference": len(ref_records),
        "token_length": int(tokens.shape[1]),
        "sign_convention": "effects are positive drops: clean_logit_diff - ablated_logit_diff",
        "go_rule": {
            "go": f"relative_interaction >= {args.go_threshold}",
            "diagnose": f"{args.diagnose_threshold} <= relative_interaction < {args.go_threshold}",
            "stop_redesign": f"relative_interaction < {args.diagnose_threshold}",
        },
    }
    with open(outdir / "ioi_stage1_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    plot_stage1(summary_df, outdir / "ioi_stage1_bar.png")

    print("\n=== Stage-1 IOI self-repair summary ===")
    print(summary_df.to_string(index=False))
    print("\nWrote:")
    for fn in ["ioi_stage1_condition_results.csv", "ioi_stage1_summary.csv", "ioi_stage1_metadata.json", "ioi_stage1_bar.png", "ioi_stage1_prompts.csv"]:
        print(f"  {outdir / fn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
