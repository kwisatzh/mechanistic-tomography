#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
Mechanistic Tomography correspondence experiment for the HMM belief wind tunnel.

This is Experiment Set 1, not the observer->control falsifier.

Goal:
  Compare a direct MI mechanism map against a tomography-recovered mechanism map.

Direct MI map:
  For each candidate component c=(residual layer, time bin), add a small perturbation
  along a belief direction and measure the end-to-end change in the final-token
  implied belief z1. This is the single-component causal/patching estimate.

Tomography map:
  Apply random aggregate perturbation masks over many components, observe only the
  end-to-end aggregate change, and solve an inverse problem y ≈ A x to recover
  per-component effects. Compare x_hat to the direct MI map.

Interpretation:
  If x_hat agrees with the direct MI map and predicts held-out aggregate effects,
  then this wind tunnel supports the MI<->tomography correspondence: the same
  hidden causal structure recoverable by direct patching is also recoverable from
  designed aggregate measurements.

Expected usage from the belief_tomography_harness directory:
  python nt_mi_correspondence.py \
    --run-dir runs/m4_baseline_seed7 \
    --device mps \
    --outdir runs/m4_baseline_seed7/nt_mi_set1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

# Allow running either after copying this file into the harness directory,
# or from another directory with the harness as the current working directory.
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Import objects from the original harness. This script should live next to
# hmm_observer_control.py or be run from that directory.
from hmm_observer_control import (  # type: ignore
    ExperimentConfig,
    HMMParams,
    TinyCausalTransformer,
    collect_activations,
    generate_batch,
    invert_next_obs_prob_to_current_p,
    logit,
    marginal_probs_from_logits,
    pick_device,
    set_seed,
)


def safe_torch_load(path: Path, device: torch.device) -> dict:
    """Load PyTorch checkpoints across newer/older torch defaults."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model_from_run(run_dir: Path, device: torch.device) -> Tuple[TinyCausalTransformer, ExperimentConfig, HMMParams]:
    ckpt_path = run_dir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Could not find {ckpt_path}. Run hmm_observer_control.py first.")
    ckpt = safe_torch_load(ckpt_path, device)
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


def implied_z_from_logits(logits: torch.Tensor, params: HMMParams, which: str = "z1") -> torch.Tensor:
    """Convert next-token logits to implied current posterior logit for HMM #1 or #2."""
    q1, q2 = marginal_probs_from_logits(logits)
    if which == "z1":
        p = invert_next_obs_prob_to_current_p(q1, params.p_stay1, params.p_emit1)
    elif which == "z2":
        p = invert_next_obs_prob_to_current_p(q2, params.p_stay2, params.p_emit2)
    else:
        raise ValueError("which must be z1 or z2")
    return logit(p)


def build_time_bins(T: int, n_bins: int) -> List[Tuple[int, int]]:
    edges = np.linspace(0, T, n_bins + 1, dtype=int)
    bins: List[Tuple[int, int]] = []
    for i in range(n_bins):
        start = int(edges[i])
        end = int(edges[i + 1])
        if end <= start:
            end = min(T, start + 1)
        bins.append((start, end))
    return bins


def forward_multi_control(
    model: TinyCausalTransformer,
    idx: torch.Tensor,
    layer_strengths: Dict[int, torch.Tensor],
    layer_dirs: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """Forward pass with multiple layer controls.

    layer_strengths[li]: tensor [B,T] or [T], multiplied by layer_dirs[li]: [C].
    """
    B, T = idx.shape
    x = model.tok_emb(idx) + model.pos_emb[:, :T, :]
    for li, block in enumerate(model.blocks):
        x = block(x)
        if li in layer_strengths:
            strengths = layer_strengths[li].to(x.device, x.dtype)
            if strengths.ndim == 1:
                strengths = strengths.view(1, T).expand(B, T)
            direction = layer_dirs[li].to(x.device, x.dtype)
            x = x + strengths.unsqueeze(-1) * direction.view(1, 1, -1)
    logits = model.head(model.ln_f(x))
    return logits


def compute_layer_directions(
    model: TinyCausalTransformer,
    cfg: ExperimentConfig,
    params: HMMParams,
    device: torch.device,
    batches: int,
    batch_size: int,
    seq_len: int,
    samples: int,
    quantile: float = 0.2,
) -> Dict[int, torch.Tensor]:
    """High-z1 minus low-z1 residual direction for every layer."""
    data = collect_activations(model, params, device, batches, batch_size, seq_len, samples)
    z1 = data["z1"].float()
    lo = torch.quantile(z1, quantile)
    hi = torch.quantile(z1, 1.0 - quantile)
    dirs: Dict[int, torch.Tensor] = {}
    for li in range(model.n_layers):
        X = data["X_layers"][li].float()
        direction = X[z1 >= hi].mean(dim=0) - X[z1 <= lo].mean(dim=0)
        direction = direction / direction.norm().clamp_min(1e-8)
        dirs[li] = direction.detach().to(device)
    return dirs


def component_index(n_layers: int, n_bins: int) -> List[Tuple[int, int]]:
    return [(li, bi) for li in range(n_layers) for bi in range(n_bins)]


def strengths_from_mask(
    mask: np.ndarray,
    comps: List[Tuple[int, int]],
    bins: List[Tuple[int, int]],
    n_layers: int,
    T: int,
    batch_size: int,
    epsilon: float,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    layer_strengths: Dict[int, torch.Tensor] = {}
    for li in range(n_layers):
        layer_strengths[li] = torch.zeros((batch_size, T), device=device)
    for m, coeff in enumerate(mask):
        if coeff == 0:
            continue
        li, bi = comps[m]
        start, end = bins[bi]
        layer_strengths[li][:, start:end] += float(coeff) * epsilon
    # Drop layers with all-zero controls to save a little work.
    return {li: s for li, s in layer_strengths.items() if torch.any(s != 0)}


def measure_effect(
    model: TinyCausalTransformer,
    idx: torch.Tensor,
    params: HMMParams,
    layer_strengths: Dict[int, torch.Tensor],
    layer_dirs: Dict[int, torch.Tensor],
    base_z1_final: torch.Tensor,
    base_z2_final: torch.Tensor,
) -> Tuple[float, float]:
    logits = forward_multi_control(model, idx, layer_strengths, layer_dirs)
    z1 = implied_z_from_logits(logits, params, "z1")[:, -1]
    z2 = implied_z_from_logits(logits, params, "z2")[:, -1]
    return (
        float((z1 - base_z1_final).mean().detach().cpu()),
        float((z2 - base_z2_final).mean().detach().cpu()),
    )


def fit_ridge_centered(A: np.ndarray, y: np.ndarray, ridge: float) -> Tuple[np.ndarray, float]:
    """Centered ridge with intercept. Returns 1D coefficients and scalar intercept.

    This intentionally flattens y and A_mean. Newer NumPy versions are
    stricter about converting size-1 arrays to Python scalars, and older
    versions only warned. Keeping both beta and intercept explicitly scalar/1D
    also prevents subtle shape bugs in downstream plotting and JSON summaries.
    """
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")
    if y.ndim != 1 or y.shape[0] != A.shape[0]:
        raise ValueError(f"y must be length {A.shape[0]}, got shape {y.shape}")

    A_mean = A.mean(axis=0)          # shape: (M,)
    y_mean = float(y.mean())         # scalar
    Ac = A - A_mean                  # broadcasts over rows
    yc = y - y_mean
    M = A.shape[1]
    beta = np.linalg.solve(Ac.T @ Ac + ridge * np.eye(M), Ac.T @ yc).reshape(-1)
    intercept = y_mean - float(A_mean.dot(beta))
    return beta, intercept


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom < 1e-12:
        return float("nan")
    return 1.0 - float(np.sum((y - yhat) ** 2)) / denom


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    return corrcoef(ar, br)


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = min(k, len(a))
    ia = set(np.argsort(-np.abs(a))[:k].tolist())
    ib = set(np.argsort(-np.abs(b))[:k].tolist())
    return len(ia & ib) / max(1, k)


def make_random_masks(n: int, M: int, kind: str, density: float, rng: np.random.Generator) -> np.ndarray:
    if kind == "signed":
        active = rng.random((n, M)) < density
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n, M))
        return active.astype(float) * signs
    if kind == "bernoulli":
        return (rng.random((n, M)) < density).astype(float)
    raise ValueError("mask kind must be signed or bernoulli")


def plot_heatmaps(df: pd.DataFrame, n_layers: int, n_bins: int, outdir: Path) -> None:
    mi = df.pivot(index="layer", columns="bin", values="mi_effect_z1").sort_index().to_numpy()
    nt = df.pivot(index="layer", columns="bin", values="nt_effect_z1").sort_index().to_numpy()
    diff = nt - mi
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)
    for ax, mat, title in zip(axes, [mi, nt, diff], ["Direct MI effect", "Tomography recovered", "NT - MI"]):
        im = ax.imshow(mat, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("time bin")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(outdir / "mi_nt_component_heatmaps.png", dpi=180)
    plt.close(fig)


def plot_scatter(mi: np.ndarray, nt: np.ndarray, outdir: Path) -> None:
    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(mi, nt)
    lo = min(float(mi.min()), float(nt.min()))
    hi = max(float(mi.max()), float(nt.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("direct MI single-component effect")
    plt.ylabel("tomography-recovered effect")
    plt.title("Does tomography recover the MI mechanism map?")
    plt.tight_layout()
    plt.savefig(outdir / "mi_vs_nt_scatter.png", dpi=180)
    plt.close()


def plot_sample_efficiency(curve: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(curve["n_train"], curve["pearson_vs_mi"], marker="o", label="Pearson vs MI")
    plt.plot(curve["n_train"], curve["heldout_r2"], marker="o", label="held-out aggregate R²")
    plt.xlabel("number of aggregate measurements used")
    plt.ylabel("score")
    plt.title("Sample efficiency of tomography recovery")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "nt_sample_efficiency.png", dpi=180)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Mechanistic tomography correspondence experiment")
    p.add_argument("--run-dir", required=True, type=str, help="Existing hmm_observer_control.py run directory")
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-bins", type=int, default=8)
    p.add_argument("--epsilon", type=float, default=0.6)
    p.add_argument("--batch-size", type=int, default=384)
    p.add_argument("--direction-batches", type=int, default=20)
    p.add_argument("--direction-samples", type=int, default=120000)
    p.add_argument("--measurements", type=int, default=256)
    p.add_argument("--holdout-frac", type=float, default=0.25)
    p.add_argument("--mask-kind", choices=["signed", "bernoulli"], default="signed")
    p.add_argument("--mask-density", type=float, default=0.30)
    p.add_argument("--ridge", type=float, default=1e-2)
    args = p.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else run_dir / "nt_mi_set1"
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)

    print(f"Using device: {device}")
    print(f"Loading run: {run_dir}")
    print(f"Writing results: {outdir}")

    model, cfg, params = load_model_from_run(run_dir, device)
    T = cfg.seq_len - 1
    bins = build_time_bins(T, args.n_bins)
    comps = component_index(model.n_layers, args.n_bins)
    M = len(comps)

    print("Computing layer belief directions...")
    layer_dirs = compute_layer_directions(
        model, cfg, params, device,
        batches=args.direction_batches,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        samples=args.direction_samples,
    )

    print("Generating fixed evaluation batch...")
    batch = generate_batch(args.batch_size, cfg.seq_len, params, device)
    idx = batch["tokens"][:, :-1]
    with torch.no_grad():
        base_logits, _ = model(idx)
        base_z1_final = implied_z_from_logits(base_logits, params, "z1")[:, -1]
        base_z2_final = implied_z_from_logits(base_logits, params, "z2")[:, -1]

    # Direct MI: single-component patching effects.
    print(f"Direct MI patching over {M} components...")
    mi_z1 = np.zeros(M, dtype=np.float64)
    mi_z2 = np.zeros(M, dtype=np.float64)
    for m in range(M):
        mask = np.zeros(M, dtype=np.float64)
        mask[m] = 1.0
        strengths = strengths_from_mask(mask, comps, bins, model.n_layers, T, args.batch_size, args.epsilon, device)
        with torch.no_grad():
            dz1, dz2 = measure_effect(model, idx, params, strengths, layer_dirs, base_z1_final, base_z2_final)
        mi_z1[m] = dz1
        mi_z2[m] = dz2

    # Tomography measurements: aggregate random masks.
    print(f"Collecting {args.measurements} aggregate tomography measurements...")
    A = make_random_masks(args.measurements, M, args.mask_kind, args.mask_density, rng)
    y_z1 = np.zeros(args.measurements, dtype=np.float64)
    y_z2 = np.zeros(args.measurements, dtype=np.float64)
    for j in range(args.measurements):
        strengths = strengths_from_mask(A[j], comps, bins, model.n_layers, T, args.batch_size, args.epsilon, device)
        with torch.no_grad():
            dz1, dz2 = measure_effect(model, idx, params, strengths, layer_dirs, base_z1_final, base_z2_final)
        y_z1[j] = dz1
        y_z2[j] = dz2
        if (j + 1) % max(1, args.measurements // 10) == 0:
            print(f"  {j + 1}/{args.measurements}")

    # Save raw measurement data before fitting. This makes reruns/debugging safer if
    # a later inverse-solver or plotting step fails.
    raw_meas_df = pd.DataFrame({"measurement": np.arange(args.measurements), "y_z1": y_z1, "y_z2": y_z2})
    raw_meas_df.to_csv(outdir / "tomography_measurements.csv", index=False)
    np.save(outdir / "measurement_matrix_A.npy", A)
    np.save(outdir / "direct_mi_z1.npy", mi_z1)
    np.save(outdir / "direct_mi_z2.npy", mi_z2)

    # Train/held-out split over aggregate measurements.
    perm = rng.permutation(args.measurements)
    n_holdout = max(1, int(args.holdout_frac * args.measurements))
    hold = perm[:n_holdout]
    train = perm[n_holdout:]
    beta, intercept = fit_ridge_centered(A[train], y_z1[train], ridge=args.ridge)
    yhat_train = A[train] @ beta + intercept
    yhat_hold = A[hold] @ beta + intercept

    component_rows = []
    for m, (li, bi) in enumerate(comps):
        start, end = bins[bi]
        component_rows.append({
            "component": m,
            "layer": li,
            "bin": bi,
            "token_start": start,
            "token_end_exclusive": end,
            "mi_effect_z1": mi_z1[m],
            "mi_effect_z2_collateral": mi_z2[m],
            "nt_effect_z1": beta[m],
            "abs_error": abs(beta[m] - mi_z1[m]),
        })
    comp_df = pd.DataFrame(component_rows)
    comp_df.to_csv(outdir / "component_effects.csv", index=False)

    # Sample-efficiency curve.
    train_sizes = sorted(set([max(8, M // 2), M, 2 * M, 4 * M, len(train)]))
    train_sizes = [n for n in train_sizes if n <= len(train)]
    curve_rows = []
    for n in train_sizes:
        subset = train[:n]
        b, c = fit_ridge_centered(A[subset], y_z1[subset], ridge=args.ridge)
        yh = A[hold] @ b + c
        curve_rows.append({
            "n_train": int(n),
            "pearson_vs_mi": corrcoef(mi_z1, b),
            "spearman_vs_mi": spearman_corr(mi_z1, b),
            "top5_overlap": topk_overlap(mi_z1, b, k=min(5, M)),
            "heldout_r2": r2_score(y_z1[hold], yh),
        })
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(outdir / "sample_efficiency.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "config": asdict(cfg),
        "hmm_params": asdict(params),
        "args": vars(args),
        "n_components": M,
        "n_measurements": int(args.measurements),
        "direct_mi_norm": float(np.linalg.norm(mi_z1)),
        "nt_norm": float(np.linalg.norm(beta)),
        "pearson_nt_vs_mi": corrcoef(mi_z1, beta),
        "spearman_nt_vs_mi": spearman_corr(mi_z1, beta),
        "top5_overlap_abs_effect": topk_overlap(mi_z1, beta, k=min(5, M)),
        "train_aggregate_r2": r2_score(y_z1[train], yhat_train),
        "heldout_aggregate_r2": r2_score(y_z1[hold], yhat_hold),
        "nt_vs_mi_mean_abs_error": float(np.mean(np.abs(beta - mi_z1))),
        "interpretation": (
            "Positive signal: high heldout_aggregate_r2 plus positive Pearson/Spearman/top-k overlap "
            "between nt_effect_z1 and direct mi_effect_z1. That means designed aggregate measurements "
            "recover the same layer/bin mechanism map that direct patching finds. If aggregate R2 is high "
            "but NT-vs-MI correlation is low, the aggregate predictor may be exploiting a confounded basis. "
            "If both are low, the linear tomography scaffold is not adequate at this epsilon/basis/task."
        ),
    }
    with open(outdir / "mt_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_heatmaps(comp_df, model.n_layers, args.n_bins, outdir)
    plot_scatter(mi_z1, beta, outdir)
    plot_sample_efficiency(curve_df, outdir)

    print("\nDone. Key files:")
    for name in [
        "mt_summary.json",
        "component_effects.csv",
        "mi_vs_nt_scatter.png",
        "mi_nt_component_heatmaps.png",
        "nt_sample_efficiency.png",
        "sample_efficiency.csv",
    ]:
        print(f"  {outdir / name}")
    print("\nSummary:")
    for k in ["pearson_nt_vs_mi", "spearman_nt_vs_mi", "top5_overlap_abs_effect", "train_aggregate_r2", "heldout_aggregate_r2"]:
        print(f"  {k}: {summary[k]:.4f}")


if __name__ == "__main__":
    main()
