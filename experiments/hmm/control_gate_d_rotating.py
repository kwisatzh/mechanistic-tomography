#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
Gate-D specificity experiment with rotated/confounded actuation directions.

The original observer->control harness changed only the scalar control strength
while keeping the actuation direction fixed. That is a good observer-fidelity
experiment, but it underpowers the nuisance/specificity test: a confounded
observer cannot move the nuisance channel unless its error also rotates the
control direction.

This script reuses a trained hmm_observer_control.py run and evaluates:
  - clean z1 observers steering along z1 direction
  - scalar-entangled observer steering along z1 direction
  - rotated-entangled observers steering along z1 + alpha*z2 direction

Expected use:
  python control_gate_d_rotating.py \
    --run-dir runs/m4_baseline_seed7 \
    --device mps \
    --outdir runs/m4_baseline_seed7/gate_d_rotating_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from hmm_observer_control import (  # type: ignore
    ExperimentConfig,
    HMMParams,
    RidgeProbe,
    TinyCausalTransformer,
    bern_kl,
    collect_activations,
    generate_batch,
    invert_next_obs_prob_to_current_p,
    logit,
    marginal_probs_from_logits,
    next_obs_prob_from_current_p,
    pick_device,
    set_seed,
)


def safe_torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model_from_run(run_dir: Path, device: torch.device) -> Tuple[TinyCausalTransformer, ExperimentConfig, HMMParams]:
    ckpt = safe_torch_load(run_dir / "model.pt", device)
    cfg = ExperimentConfig(**ckpt["cfg"])
    params = HMMParams(**ckpt["params"])
    model = TinyCausalTransformer(
        vocab_size=4,
        seq_len=cfg.seq_len - 1,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_mlp=cfg.d_mlp,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, params


def implied_z_from_logits(logits: torch.Tensor, params: HMMParams, which: str) -> torch.Tensor:
    q1, q2 = marginal_probs_from_logits(logits)
    if which == "z1":
        p = invert_next_obs_prob_to_current_p(q1, params.p_stay1, params.p_emit1)
    elif which == "z2":
        p = invert_next_obs_prob_to_current_p(q2, params.p_stay2, params.p_emit2)
    else:
        raise ValueError(which)
    return logit(p)


def high_low_direction(
    model: TinyCausalTransformer,
    cfg: ExperimentConfig,
    params: HMMParams,
    device: torch.device,
    layer: int,
    target: str,
    batches: int,
    samples: int,
    quantile: float,
) -> torch.Tensor:
    data = collect_activations(model, params, device, batches, cfg.batch_size, cfg.seq_len, samples)
    X = data["X_layers"][layer].float()
    y = data[target].float()
    lo = torch.quantile(y, quantile)
    hi = torch.quantile(y, 1.0 - quantile)
    direction = X[y >= hi].mean(dim=0) - X[y <= lo].mean(dim=0)
    direction = direction / direction.norm().clamp_min(1e-8)
    return direction.detach().to(device)


def probe_direction_gain(probe: RidgeProbe, direction: torch.Tensor) -> float:
    d = direction.detach().cpu().float()
    std = probe.train_std.detach().cpu().float().clamp_min(1e-6)
    w = probe.w.detach().cpu().float()
    return float((w * (d / std)).sum().item())


def forward_one_control(
    model: TinyCausalTransformer,
    idx: torch.Tensor,
    layer: int,
    direction: torch.Tensor,
    strengths: torch.Tensor,
) -> torch.Tensor:
    control = {"layer": int(layer), "direction": direction, "strengths": strengths}
    logits, _ = model(idx, return_acts=False, control=control)
    return logits


def observer_value(
    name: str,
    z1: torch.Tensor,
    z2: torch.Tensor,
    act: torch.Tensor,
    probe_z1: RidgeProbe,
    alpha: float = 0.0,
) -> torch.Tensor:
    if name == "oracle_z1":
        return z1
    if name == "linear_probe_z1":
        return probe_z1.predict(act)
    if name == "last_obs_proxy_z1":
        return 1.5 * torch.sign(z1)
    if name in {"entangled_scalar", "entangled_rotated"}:
        return z1 + alpha * z2
    raise ValueError(f"unknown observer variant {name}")


def parse_alpha_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def plot_results(df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(7.2, 5.0))
    for _, row in df.iterrows():
        label = row["variant"]
        if row["alpha"] != 0:
            label += f" α={row['alpha']:.2g}"
        plt.scatter(row["control_target_mse"], row["collateral_z2_abs"])
        plt.text(row["control_target_mse"] + 0.02, row["collateral_z2_abs"], label, fontsize=8)
    plt.xlabel("closed-loop target MSE on z₁")
    plt.ylabel("collateral movement in implied z₂")
    plt.title("Gate D with direction-rotating confounded observers")
    plt.tight_layout()
    plt.savefig(outdir / "gate_d_target_vs_collateral_z2.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 5.0))
    for _, row in df.iterrows():
        label = row["variant"]
        if row["alpha"] != 0:
            label += f" α={row['alpha']:.2g}"
        plt.scatter(row["mean_abs_strength"], row["collateral_z2_abs"])
        plt.text(row["mean_abs_strength"] + 0.01, row["collateral_z2_abs"], label, fontsize=8)
    plt.xlabel("mean |control strength|")
    plt.ylabel("collateral movement in implied z₂")
    plt.title("Collateral after separating strength from direction error")
    plt.tight_layout()
    plt.savefig(outdir / "gate_d_strength_vs_collateral_z2.png", dpi=180)
    plt.close()

    order = df.sort_values("collateral_z2_abs", ascending=False)
    plt.figure(figsize=(8.5, 4.6))
    labels = [f"{r.variant}\nα={r.alpha:.2g}" if r.alpha != 0 else r.variant for r in order.itertuples()]
    x = np.arange(len(order))
    plt.bar(x, order["collateral_z2_abs"].to_numpy())
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("collateral movement in implied z₂")
    plt.title("Nuisance damage by observer/actuator variant")
    plt.tight_layout()
    plt.savefig(outdir / "gate_d_collateral_bar.png", dpi=180)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Rotating-direction Gate-D specificity test")
    p.add_argument("--run-dir", required=True, type=str)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--control-batches", type=int, default=20)
    p.add_argument("--control-batch-size", type=int, default=256)
    p.add_argument("--direction-batches", type=int, default=20)
    p.add_argument("--direction-samples", type=int, default=120000)
    p.add_argument("--quantile", type=float, default=0.20)
    p.add_argument("--alphas", type=str, default="0.5,0.9,1.5,2.0")
    p.add_argument("--controller-gain", type=float, default=None)
    p.add_argument("--max-strength", type=float, default=None)
    p.add_argument("--no-orthogonalize-z2", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else run_dir / "gate_d_rotating"
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = pick_device(args.device)

    print(f"Using device: {device}")
    print(f"Loading run: {run_dir}")
    print(f"Writing results: {outdir}")

    model, cfg, params = load_model_from_run(run_dir, device)
    if args.controller_gain is None:
        args.controller_gain = cfg.controller_gain
    if args.max_strength is None:
        args.max_strength = cfg.max_strength

    probe_pack = safe_torch_load(run_dir / "probes.pt", torch.device("cpu"))
    best_layer = int(probe_pack["best_layer"])
    probe_z1: RidgeProbe = probe_pack["probes_z1"][best_layer]
    probe_z2: RidgeProbe = probe_pack["probes_z2"][best_layer]
    d1 = probe_pack["direction"].float().to(device)
    d1 = d1 / d1.norm().clamp_min(1e-8)

    print("Computing z2 nuisance direction...")
    d2 = high_low_direction(
        model, cfg, params, device, best_layer, "z2",
        batches=args.direction_batches,
        samples=args.direction_samples,
        quantile=args.quantile,
    )
    if not args.no_orthogonalize_z2:
        d2 = d2 - torch.dot(d2, d1) * d1
        d2 = d2 / d2.norm().clamp_min(1e-8)

    g_z1_d1 = probe_direction_gain(probe_z1, d1)
    g_z1_d2 = probe_direction_gain(probe_z1, d2)
    g_z2_d1 = probe_direction_gain(probe_z2, d1)
    g_z2_d2 = probe_direction_gain(probe_z2, d2)
    direction_info = {
        "best_layer": best_layer,
        "z1_gain_d1": g_z1_d1,
        "z1_gain_d2": g_z1_d2,
        "z2_gain_d1": g_z2_d1,
        "z2_gain_d2": g_z2_d2,
        "dot_d1_d2": float(torch.dot(d1, d2).detach().cpu()),
        "orthogonalized_z2": not args.no_orthogonalize_z2,
    }
    with open(outdir / "direction_info.json", "w") as f:
        json.dump(direction_info, f, indent=2)

    variants: List[Tuple[str, float, torch.Tensor, float]] = []
    # name, alpha, direction, controller_gain_denominator
    variants.append(("oracle_z1", 0.0, d1, g_z1_d1))
    variants.append(("linear_probe_z1", 0.0, d1, g_z1_d1))
    variants.append(("last_obs_proxy_z1", 0.0, d1, g_z1_d1))
    variants.append(("entangled_scalar", 0.9, d1, g_z1_d1))

    for alpha in parse_alpha_list(args.alphas):
        dmix = d1 + alpha * d2
        dmix = dmix / dmix.norm().clamp_min(1e-8)
        # This is the readout gain of the mixed observed variable z1 + alpha*z2
        # along the mixed actuator direction.
        g_mix = probe_direction_gain(probe_z1, dmix) + alpha * probe_direction_gain(probe_z2, dmix)
        variants.append(("entangled_rotated", alpha, dmix, g_mix))

    rows = []
    for variant, alpha, direction, gain in tqdm(variants, desc="Gate-D variants"):
        if abs(gain) < 1e-6:
            gain = 1e-6 if gain >= 0 else -1e-6
        accum: Dict[str, List[float]] = {k: [] for k in [
            "observer_rmse_z1", "base_target_mse", "control_target_mse", "control_target_kl",
            "natural_ce_base", "natural_ce_control", "collateral_q2_abs", "collateral_q2_kl",
            "collateral_z2_abs", "collateral_z2_mse", "mean_abs_strength",
        ]}
        for _ in range(args.control_batches):
            batch = generate_batch(args.control_batch_size, cfg.seq_len, params, device)
            idx = batch["tokens"][:, :-1]
            target = batch["tokens"][:, 1:]
            z1_true = batch["z1"][:, :-1]
            z2_true = batch["z2"][:, :-1]
            target_z = torch.full_like(z1_true, cfg.target_z)
            p_target = torch.sigmoid(target_z)
            q1_target = next_obs_prob_from_current_p(p_target, params.p_stay1, params.p_emit1)

            with torch.no_grad():
                base_logits, acts = model(idx, return_acts=True)
                assert acts is not None
                act = acts[best_layer]
                zhat = observer_value(variant, z1_true, z2_true, act, probe_z1, alpha=alpha)
                observer_rmse_z1 = torch.sqrt(torch.mean((zhat - z1_true) ** 2))

                strengths = args.controller_gain * (target_z - zhat) / gain
                strengths = strengths.clamp(-args.max_strength, args.max_strength)
                ctrl_logits = forward_one_control(model, idx, best_layer, direction, strengths)

                q1_base, q2_base = marginal_probs_from_logits(base_logits)
                q1_ctrl, q2_ctrl = marginal_probs_from_logits(ctrl_logits)
                z1_base = implied_z_from_logits(base_logits, params, "z1")
                z1_ctrl = implied_z_from_logits(ctrl_logits, params, "z1")
                z2_base = implied_z_from_logits(base_logits, params, "z2")
                z2_ctrl = implied_z_from_logits(ctrl_logits, params, "z2")

                base_target_mse = torch.mean((z1_base - target_z) ** 2)
                control_target_mse = torch.mean((z1_ctrl - target_z) ** 2)
                control_target_kl = torch.mean(bern_kl(q1_target, q1_ctrl))
                ce_base = F.cross_entropy(base_logits.reshape(-1, 4), target.reshape(-1))
                ce_ctrl = F.cross_entropy(ctrl_logits.reshape(-1, 4), target.reshape(-1))
                collateral_q2_abs = torch.mean(torch.abs(q2_ctrl - q2_base))
                collateral_q2_kl = torch.mean(bern_kl(q2_base, q2_ctrl))
                collateral_z2_abs = torch.mean(torch.abs(z2_ctrl - z2_base))
                collateral_z2_mse = torch.mean((z2_ctrl - z2_base) ** 2)
                mean_abs_strength = torch.mean(torch.abs(strengths))

            values = {
                "observer_rmse_z1": observer_rmse_z1,
                "base_target_mse": base_target_mse,
                "control_target_mse": control_target_mse,
                "control_target_kl": control_target_kl,
                "natural_ce_base": ce_base,
                "natural_ce_control": ce_ctrl,
                "collateral_q2_abs": collateral_q2_abs,
                "collateral_q2_kl": collateral_q2_kl,
                "collateral_z2_abs": collateral_z2_abs,
                "collateral_z2_mse": collateral_z2_mse,
                "mean_abs_strength": mean_abs_strength,
            }
            for k, v in values.items():
                accum[k].append(float(v.detach().cpu()))

        row = {
            "variant": variant,
            "alpha": float(alpha),
            "layer": best_layer,
            "gain_used": float(gain),
            "direction_dot_z1": float(torch.dot(direction, d1).detach().cpu()),
            "direction_dot_z2": float(torch.dot(direction, d2).detach().cpu()),
        }
        row.update({k: float(np.mean(v)) for k, v in accum.items()})
        row["control_improvement_mse"] = row["base_target_mse"] - row["control_target_mse"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "gate_d_rotating_control.csv", index=False)
    plot_results(df, outdir)

    summary = {
        "run_dir": str(run_dir),
        "config": asdict(cfg),
        "hmm_params": asdict(params),
        "direction_info": direction_info,
        "results": df.to_dict(orient="records"),
        "interpretation": (
            "This test gives Gate D teeth by allowing confounded observers to rotate the actuation direction, not merely "
            "scale a fixed z1 direction. A genuine identification/specificity failure should show higher nuisance z2 "
            "movement for entangled_rotated variants than for oracle_z1/linear_probe_z1, even when mean strength is similar."
        ),
    }
    with open(outdir / "gate_d_rotating_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. Key files:")
    for name in [
        "direction_info.json",
        "gate_d_rotating_control.csv",
        "gate_d_rotating_summary.json",
        "gate_d_target_vs_collateral_z2.png",
        "gate_d_strength_vs_collateral_z2.png",
        "gate_d_collateral_bar.png",
    ]:
        print(f"  {outdir / name}")
    print("\nResults:")
    cols = ["variant", "alpha", "observer_rmse_z1", "control_target_mse", "collateral_z2_abs", "collateral_q2_abs", "mean_abs_strength"]
    print(df[cols].sort_values(["variant", "alpha"]).to_string(index=False))


if __name__ == "__main__":
    main()
