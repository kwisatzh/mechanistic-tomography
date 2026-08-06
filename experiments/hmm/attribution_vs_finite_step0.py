#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
Step 0: Attribution patching vs finite-epsilon mechanistic tomography.

Goal
----
This script tests whether the current HMM wind tunnel is truly a finite-epsilon
regime or merely a first-order/gradient regime.

It compares four maps over the same component basis (layer x time-bin):
  1. direct finite patching map: single component interventions at scale epsilon
  2. attribution patching map: one backward pass, infinitesimal derivative scaled by epsilon
  3. ridge tomography map: aggregate finite interventions + ridge inverse
  4. OMP tomography map: aggregate finite interventions + sparse inverse

The key diagnostic is held-out aggregate prediction across epsilon:
  - If AtP predicts held-out finite aggregate effects as well as finite tomography,
    then this harness is first-order at that epsilon.
  - If AtP works at small epsilon but fails at the operating epsilon where finite
    tomography still predicts aggregate effects, that is evidence for a genuine
    finite-epsilon gap.

Expected usage from the belief_tomography_harness directory:
  python attribution_vs_finite_step0.py \
    --run-dir runs/m4_baseline_seed7 \
    --device mps \
    --outdir runs/m4_baseline_seed7/step0_atp_vs_finite

Use --quick for a smoke test. Use --h100 for a larger Colab run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

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


def component_index(n_layers: int, n_bins: int) -> List[Tuple[int, int]]:
    return [(li, bi) for li in range(n_layers) for bi in range(n_bins)]


def make_random_masks(n_measurements: int, n_components: int, kind: str, density: float, rng: np.random.Generator, normalize: bool) -> np.ndarray:
    if kind == "signed":
        active = rng.random((n_measurements, n_components)) < density
        # Ensure each row has at least one active coordinate.
        for i in range(n_measurements):
            if not active[i].any():
                active[i, rng.integers(0, n_components)] = True
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_measurements, n_components))
        A = active.astype(np.float64) * signs
    elif kind == "bernoulli":
        A = (rng.random((n_measurements, n_components)) < density).astype(np.float64)
        for i in range(n_measurements):
            if not A[i].any():
                A[i, rng.integers(0, n_components)] = 1.0
    else:
        raise ValueError("mask kind must be signed or bernoulli")
    if normalize:
        norms = np.linalg.norm(A, axis=1, keepdims=True)
        A = A / np.maximum(norms, 1e-12)
    return A


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    yhat = np.asarray(yhat, dtype=np.float64).reshape(-1)
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom < 1e-12:
        return float("nan")
    return 1.0 - float(np.sum((y - yhat) ** 2)) / denom


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    return corrcoef(ar, br)


def slope_no_intercept(x: np.ndarray, y: np.ndarray) -> float:
    """Return scalar g minimizing ||y - g*x||_2. Interpreted as y-on-x slope."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    denom = float(np.dot(x, x))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denom)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def fit_ridge_centered(A: np.ndarray, y: np.ndarray, ridge: float) -> Tuple[np.ndarray, float]:
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    A_mean = A.mean(axis=0)
    y_mean = float(y.mean())
    Ac = A - A_mean
    yc = y - y_mean
    beta = np.linalg.solve(Ac.T @ Ac + ridge * np.eye(A.shape[1]), Ac.T @ yc).reshape(-1)
    intercept = y_mean - float(A_mean.dot(beta))
    return beta, intercept


def omp_fit(A: np.ndarray, y: np.ndarray, max_k: int, ridge: float = 1e-8) -> np.ndarray:
    """Simple OMP fit with exactly up to max_k selected columns."""
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    residual = y.copy()
    selected: List[int] = []
    beta = np.zeros(A.shape[1], dtype=np.float64)
    for _ in range(max_k):
        corr = A.T @ residual
        if selected:
            corr[selected] = 0.0
        j = int(np.argmax(np.abs(corr)))
        if abs(corr[j]) < 1e-12:
            break
        selected.append(j)
        As = A[:, selected]
        coef = np.linalg.solve(As.T @ As + ridge * np.eye(len(selected)), As.T @ y)
        residual = y - As @ coef
    if selected:
        beta[selected] = coef
    return beta


def select_omp_by_validation(Atr: np.ndarray, ytr: np.ndarray, Aval: np.ndarray, yval: np.ndarray, max_k: int) -> Tuple[np.ndarray, int, float]:
    best_beta: np.ndarray | None = None
    best_k = 0
    best_r2 = -1e18
    max_k = max(1, min(max_k, Atr.shape[1]))
    for k in range(1, max_k + 1):
        beta = omp_fit(Atr, ytr, max_k=k)
        yhat = Aval @ beta
        score = r2_score(yval, yhat)
        if score > best_r2:
            best_r2 = score
            best_beta = beta
            best_k = k
    assert best_beta is not None
    return best_beta, best_k, best_r2


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


def forward_multi_control(
    model: TinyCausalTransformer,
    idx: torch.Tensor,
    layer_strengths: Dict[int, torch.Tensor],
    layer_dirs: Dict[int, torch.Tensor],
) -> torch.Tensor:
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
    return model.head(model.ln_f(x))


def forward_capture_acts(model: TinyCausalTransformer, idx: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    B, T = idx.shape
    x = model.tok_emb(idx) + model.pos_emb[:, :T, :]
    acts: List[torch.Tensor] = []
    for block in model.blocks:
        x = block(x)
        x.retain_grad()
        acts.append(x)
    logits = model.head(model.ln_f(x))
    return logits, acts


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
    layer_strengths: Dict[int, torch.Tensor] = {li: torch.zeros((batch_size, T), device=device) for li in range(n_layers)}
    for m, coeff in enumerate(mask):
        if abs(float(coeff)) < 1e-12:
            continue
        li, bi = comps[m]
        start, end = bins[bi]
        layer_strengths[li][:, start:end] += float(coeff) * epsilon
    return {li: s for li, s in layer_strengths.items() if torch.any(s != 0)}


def measure_effect(
    model: TinyCausalTransformer,
    idx: torch.Tensor,
    params: HMMParams,
    layer_strengths: Dict[int, torch.Tensor],
    layer_dirs: Dict[int, torch.Tensor],
    base_z1_final: torch.Tensor,
) -> float:
    logits = forward_multi_control(model, idx, layer_strengths, layer_dirs)
    z1 = implied_z_from_logits(logits, params, "z1")[:, -1]
    return float((z1 - base_z1_final).mean().detach().cpu())


def direct_finite_map(
    model: TinyCausalTransformer,
    idx: torch.Tensor,
    params: HMMParams,
    layer_dirs: Dict[int, torch.Tensor],
    comps: List[Tuple[int, int]],
    bins: List[Tuple[int, int]],
    epsilon: float,
    base_z1_final: torch.Tensor,
) -> np.ndarray:
    B, T = idx.shape
    out = np.zeros(len(comps), dtype=np.float64)
    for m in tqdm(range(len(comps)), desc=f"direct finite eps={epsilon:g}", leave=False):
        mask = np.zeros(len(comps), dtype=np.float64)
        mask[m] = 1.0
        strengths = strengths_from_mask(mask, comps, bins, model.n_layers, T, B, epsilon, idx.device)
        with torch.no_grad():
            out[m] = measure_effect(model, idx, params, strengths, layer_dirs, base_z1_final)
    return out


def attribution_patch_map(
    model: TinyCausalTransformer,
    idx: torch.Tensor,
    params: HMMParams,
    layer_dirs: Dict[int, torch.Tensor],
    comps: List[Tuple[int, int]],
    bins: List[Tuple[int, int]],
    epsilon: float,
) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    logits, acts = forward_capture_acts(model, idx)
    readout = implied_z_from_logits(logits, params, "z1")[:, -1].mean()
    readout.backward()
    out = np.zeros(len(comps), dtype=np.float64)
    for m, (li, bi) in enumerate(comps):
        grad = acts[li].grad
        if grad is None:
            raise RuntimeError(f"Missing grad for layer {li}")
        start, end = bins[bi]
        d = layer_dirs[li].to(grad.device, grad.dtype)
        local = (grad[:, start:end, :] * d.view(1, 1, -1)).sum()
        out[m] = float((epsilon * local).detach().cpu())
    model.zero_grad(set_to_none=True)
    return out


def aggregate_measurements(
    model: TinyCausalTransformer,
    idx: torch.Tensor,
    params: HMMParams,
    A: np.ndarray,
    layer_dirs: Dict[int, torch.Tensor],
    comps: List[Tuple[int, int]],
    bins: List[Tuple[int, int]],
    epsilon: float,
    base_z1_final: torch.Tensor,
) -> np.ndarray:
    B, T = idx.shape
    y = np.zeros(A.shape[0], dtype=np.float64)
    for j in tqdm(range(A.shape[0]), desc=f"aggregate eps={epsilon:g}", leave=False):
        strengths = strengths_from_mask(A[j], comps, bins, model.n_layers, T, B, epsilon, idx.device)
        with torch.no_grad():
            y[j] = measure_effect(model, idx, params, strengths, layer_dirs, base_z1_final)
    return y


def plot_results(df: pd.DataFrame, outdir: Path) -> None:
    # Primary diagnostic: scale/gain drift. Pearson can stay near 1 when finite effects
    # are merely a rescaled version of infinitesimal AtP.
    plt.figure(figsize=(7.0, 4.5))
    if "finite_on_atp_slope" in df:
        plt.plot(df["epsilon"], df["finite_on_atp_slope"], marker="o", label="slope: finite ≈ g · AtP")
    if "finite_to_atp_norm_ratio" in df:
        plt.plot(df["epsilon"], df["finite_to_atp_norm_ratio"], marker="o", label="norm ratio ||finite|| / ||AtP||")
    if "atp_calibration_gain" in df:
        plt.plot(df["epsilon"], df["atp_calibration_gain"], marker="o", label="gain fit on aggregate train")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("finite intervention scale epsilon")
    plt.ylabel("scale / calibration factor")
    plt.title("Step 0 primary diagnostic: finite effect scale vs AtP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "step0_atp_scale_vs_epsilon.png", dpi=180)
    plt.close()

    # Error decomposition: finite_single R2 < 1 is aggregate non-additivity/interaction;
    # finite_single R2 - AtP R2 is the finite-vs-infinitesimal map gap.
    plt.figure(figsize=(7.0, 4.5))
    for col, label in [
        ("aggregate_additivity_gap", "1 - finite-single R² (aggregate non-additivity)"),
        ("atp_finite_calibration_gap", "finite-single R² - AtP R²"),
        ("atp_calibrated_gap", "finite-single R² - calibrated AtP R²"),
    ]:
        if col in df:
            plt.plot(df["epsilon"], df[col], marker="o", label=label)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("finite intervention scale epsilon")
    plt.ylabel("R² gap on held-out aggregate measurements")
    plt.title("Step 0 error decomposition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "step0_error_decomposition_vs_epsilon.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.5))
    for col, label in [
        ("atp_holdout_r2", "AtP gradient map"),
        ("atp_calibrated_holdout_r2", "AtP + scalar calibration"),
        ("finite_single_holdout_r2", "single finite map"),
        ("ridge_holdout_r2", "ridge tomography"),
        ("omp_holdout_r2", "OMP tomography"),
    ]:
        if col in df:
            plt.plot(df["epsilon"], df[col], marker="o", label=label)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("finite intervention scale epsilon")
    plt.ylabel("held-out aggregate R²")
    plt.title("Step 0: held-out finite aggregate prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "step0_heldout_r2_vs_epsilon.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.5))
    for col, label in [
        ("atp_vs_finite_pearson", "AtP vs finite patch"),
        ("ridge_vs_finite_pearson", "ridge vs finite patch"),
        ("omp_vs_finite_pearson", "OMP vs finite patch"),
    ]:
        if col in df:
            plt.plot(df["epsilon"], df[col], marker="o", label=label)
    plt.xlabel("finite intervention scale epsilon")
    plt.ylabel("map Pearson correlation")
    plt.title("Secondary diagnostic: map ranking agreement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "step0_map_corr_vs_epsilon.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(df["epsilon"], df["omp_k"], marker="o", label="selected OMP support size")
    plt.xlabel("finite intervention scale epsilon")
    plt.ylabel("OMP k selected by validation")
    plt.title("Sparse tomography support size across epsilon")
    plt.tight_layout()
    plt.savefig(outdir / "step0_omp_k_vs_epsilon.png", dpi=180)
    plt.close()


def parse_epsilons(values: List[str]) -> List[float]:
    eps: List[float] = []
    for v in values:
        if "," in v:
            eps.extend(float(x) for x in v.split(",") if x.strip())
        else:
            eps.append(float(v))
    return eps


def parse_ints(values: List[str]) -> List[int]:
    out: List[int] = []
    for v in values:
        if "," in v:
            out.extend(int(x) for x in v.split(",") if x.strip())
        else:
            out.append(int(v))
    return sorted(set(out))


def calibration_budget_sweep(
    A: np.ndarray,
    y: np.ndarray,
    atp_map: np.ndarray,
    train_indices: np.ndarray,
    hold_indices: np.ndarray,
    budgets: List[int],
    repeats: int,
    rng: np.random.Generator,
) -> List[dict]:
    """Fit scalar gains for AtP using small calibration subsets and score on holdout.

    This is the m_gain experiment: with one backward pass we get the full AtP map;
    with m_gain aggregate forward measurements we estimate a single scalar gain.
    """
    train_indices = np.asarray(train_indices, dtype=int)
    hold_indices = np.asarray(hold_indices, dtype=int)
    pred = A @ atp_map
    y_hold = y[hold_indices]
    pred_hold = pred[hold_indices]
    rows: List[dict] = []

    # budget 0 means raw, uncalibrated AtP.
    raw_r2 = r2_score(y_hold, pred_hold)
    rows.append({
        "m_gain": 0,
        "repeat": 0,
        "gain": 1.0,
        "holdout_r2": float(raw_r2),
        "n_train_available": int(len(train_indices)),
    })

    max_train = int(len(train_indices))
    for budget in budgets:
        budget = int(budget)
        if budget <= 0:
            continue
        if budget > max_train:
            continue
        reps = repeats if budget < max_train else 1
        for rep in range(reps):
            if budget == max_train:
                chosen = train_indices
            else:
                chosen = rng.choice(train_indices, size=budget, replace=False)
            gain = slope_no_intercept(pred[chosen], y[chosen])
            r2 = r2_score(y_hold, gain * pred_hold)
            rows.append({
                "m_gain": budget,
                "repeat": rep,
                "gain": float(gain),
                "holdout_r2": float(r2),
                "n_train_available": max_train,
            })
    return rows


def summarize_gain_budget(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby(["epsilon", "m_gain"], as_index=False)
        .agg(
            holdout_r2_mean=("holdout_r2", "mean"),
            holdout_r2_std=("holdout_r2", "std"),
            gain_mean=("gain", "mean"),
            gain_std=("gain", "std"),
            repeats=("holdout_r2", "count"),
        )
    )


def plot_gain_budget(budget_df: pd.DataFrame, results_df: pd.DataFrame, outdir: Path) -> None:
    if budget_df.empty:
        return
    summary = summarize_gain_budget(budget_df)
    summary.to_csv(outdir / "step0_gain_budget_summary.csv", index=False)

    # Plot selected epsilons so the figure is readable. Prefer controller-scale values.
    available = sorted(float(x) for x in summary["epsilon"].unique())
    preferred = [0.6, 1.2, 2.0, 5.0, 8.0]
    selected: List[float] = []
    for target in preferred:
        if any(abs(e - target) < 1e-9 for e in available):
            selected.append(target)
    if not selected:
        selected = available[-min(5, len(available)):]

    plt.figure(figsize=(7.0, 4.5))
    for eps in selected:
        sub = summary[np.isclose(summary["epsilon"], eps)].sort_values("m_gain")
        yerr = sub["holdout_r2_std"].fillna(0.0).to_numpy()
        plt.errorbar(sub["m_gain"], sub["holdout_r2_mean"], yerr=yerr, marker="o", capsize=3, label=f"eps={eps:g}")
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("aggregate measurements used only to calibrate AtP gain (m_gain)")
    plt.ylabel("held-out aggregate R²")
    plt.title("Step 0: calibrated AtP sample efficiency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "step0_gain_budget_r2_vs_m_gain.png", dpi=180)
    plt.close()

    # Gap-to-tomography plot: positive means OMP beats budgeted calibrated AtP.
    ref = results_df[["epsilon", "omp_holdout_r2", "ridge_holdout_r2", "atp_calibrated_holdout_r2"]].copy()
    merged = summary.merge(ref, on="epsilon", how="left")
    merged["omp_minus_budgeted_atp_r2"] = merged["omp_holdout_r2"] - merged["holdout_r2_mean"]
    merged["ridge_minus_budgeted_atp_r2"] = merged["ridge_holdout_r2"] - merged["holdout_r2_mean"]
    merged.to_csv(outdir / "step0_gain_budget_summary_with_refs.csv", index=False)

    plt.figure(figsize=(7.0, 4.5))
    for eps in selected:
        sub = merged[np.isclose(merged["epsilon"], eps)].sort_values("m_gain")
        plt.plot(sub["m_gain"], sub["omp_minus_budgeted_atp_r2"], marker="o", label=f"OMP - cal AtP, eps={eps:g}")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("m_gain")
    plt.ylabel("R² gap on held-out aggregate measurements")
    plt.title("Does tomography beat small-budget calibrated AtP?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "step0_gain_budget_gap_to_omp.png", dpi=180)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Step 0 AtP-vs-finite-epsilon mechanistic tomography diagnostic")
    p.add_argument("--run-dir", required=True, type=str, help="Existing hmm_observer_control.py run directory")
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-bins", type=int, default=8)
    p.add_argument(
        "--epsilons",
        nargs="+",
        default=["0.05", "0.1", "0.2", "0.4", "0.6", "0.9", "1.2", "2.0", "3.0", "5.0", "8.0"],
        help="Intervention scales to test. Defaults now include the controller-scale regime."
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--direction-batches", type=int, default=20)
    p.add_argument("--direction-samples", type=int, default=120000)
    p.add_argument("--direction-quantile", type=float, default=0.2, help="Quantile used to estimate high-minus-low belief directions.")
    p.add_argument("--measurements", type=int, default=128)
    p.add_argument("--holdout-frac", type=float, default=0.25)
    p.add_argument("--mask-kind", choices=["signed", "bernoulli"], default="signed")
    p.add_argument("--mask-density", type=float, default=0.30)
    p.add_argument("--raw-masks", action="store_true", help="Do not normalize aggregate mask rows. Default uses unit-norm rows.")
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--omp-max-k", type=int, default=12)
    p.add_argument(
        "--gain-budgets",
        nargs="+",
        default=["1", "2", "4", "8", "16", "32", "64"],
        help="Calibration-set sizes for AtP scalar-gain budget sweep. Use 0 for raw AtP; budgets above available train measurements are skipped.",
    )
    p.add_argument("--gain-budget-repeats", type=int, default=50, help="Random subset repeats for each m_gain budget.")
    p.add_argument("--no-gain-budget-sweep", action="store_true", help="Disable the m_gain calibrated-AtP budget experiment.")
    p.add_argument("--quick", action="store_true", help="Cheap smoke settings")
    p.add_argument("--h100", action="store_true", help="Larger Colab/H100 settings")
    args = p.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else run_dir / "step0_atp_vs_finite"
    outdir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.epsilons = ["0.1", "0.6", "5.0"]
        args.batch_size = min(args.batch_size, 96)
        args.direction_batches = min(args.direction_batches, 6)
        args.direction_samples = min(args.direction_samples, 30000)
        args.measurements = min(args.measurements, 32)
        args.omp_max_k = min(args.omp_max_k, 8)
        args.gain_budget_repeats = min(args.gain_budget_repeats, 8)
    if args.h100:
        args.batch_size = max(args.batch_size, 1024)
        args.direction_batches = max(args.direction_batches, 60)
        args.direction_samples = max(args.direction_samples, 500000)
        args.measurements = max(args.measurements, 512)
        args.omp_max_k = max(args.omp_max_k, 32)
        # Keep the wide epsilon range on H100 unless explicitly overridden.

    epsilons = parse_epsilons(args.epsilons)
    gain_budgets = parse_ints(args.gain_budgets)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)

    print(f"Using device: {device}")
    print(f"Loading run: {run_dir}")
    print(f"Writing results: {outdir}")
    print(f"epsilons: {epsilons}")

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
        quantile=args.direction_quantile,
    )

    print("Generating fixed evaluation batch...")
    batch = generate_batch(args.batch_size, cfg.seq_len, params, device)
    idx = batch["tokens"][:, :-1]
    with torch.no_grad():
        base_logits_model, _ = model(idx)
        base_logits_manual = forward_multi_control(model, idx, {}, layer_dirs)
        base_forward_max_abs_diff = float((base_logits_model - base_logits_manual).abs().max().detach().cpu())
        if base_forward_max_abs_diff > 1e-4:
            print(f"WARNING: model(idx) and manual forward differ at zero intervention: max_abs_diff={base_forward_max_abs_diff:.3e}")
        else:
            print(f"Manual forward zero-intervention check passed: max_abs_diff={base_forward_max_abs_diff:.3e}")
        # Use the same manual-forward path used by perturbed measurements to avoid constant offsets.
        base_z1_final = implied_z_from_logits(base_logits_manual, params, "z1")[:, -1]

    print("Computing attribution-patching unit map once; maps at epsilon are epsilon * unit_gradient_map...")
    atp_unit_map = attribution_patch_map(model, idx, params, layer_dirs, comps, bins, epsilon=1.0)

    # Reuse one measurement design across epsilons to isolate epsilon effects.
    A = make_random_masks(args.measurements, M, args.mask_kind, args.mask_density, rng, normalize=not args.raw_masks)
    perm = rng.permutation(args.measurements)
    n_hold = max(1, int(args.holdout_frac * args.measurements))
    hold = perm[:n_hold]
    train_all = perm[n_hold:]
    # Split train into fit/validation for OMP k selection.
    n_val = max(1, int(0.25 * len(train_all)))
    val = train_all[:n_val]
    train = train_all[n_val:]
    if len(train) < 2:
        raise ValueError("Too few measurements after split; increase --measurements or decrease --holdout-frac.")
    np.save(outdir / "measurement_matrix_A.npy", A)

    result_rows = []
    map_rows = []
    gain_budget_rows = []
    for eps in epsilons:
        print(f"\n=== epsilon={eps:g} ===")
        finite_map = direct_finite_map(model, idx, params, layer_dirs, comps, bins, eps, base_z1_final)
        atp_map = eps * atp_unit_map
        y = aggregate_measurements(model, idx, params, A, layer_dirs, comps, bins, eps, base_z1_final)

        ridge_beta, ridge_intercept = fit_ridge_centered(A[train], y[train], ridge=args.ridge)
        omp_beta, omp_k, omp_val_r2 = select_omp_by_validation(A[train], y[train], A[val], y[val], max_k=min(args.omp_max_k, M))

        # Calibrate AtP by a single no-intercept scalar gain on aggregate train measurements.
        atp_train_pred = A[train] @ atp_map
        atp_cal_gain = slope_no_intercept(atp_train_pred, y[train])
        atp_cal_map = atp_cal_gain * atp_map

        # Optional m_gain sweep: how many aggregate forward measurements are needed
        # to calibrate AtP's scalar gain? This is the practical baseline against
        # full finite tomography in the white-box regime.
        if not args.no_gain_budget_sweep:
            budget_rng = np.random.default_rng(args.seed + int(round(float(eps) * 1000003)))
            for b_row in calibration_budget_sweep(
                A=A,
                y=y,
                atp_map=atp_map,
                train_indices=train_all,
                hold_indices=hold,
                budgets=gain_budgets,
                repeats=args.gain_budget_repeats,
                rng=budget_rng,
            ):
                b_row["epsilon"] = float(eps)
                gain_budget_rows.append(b_row)

        # Decompose errors using held-out aggregate measurements:
        #   finite_single_holdout_r2 < 1: finite map is not additive under aggregate interventions (interaction/nonlinearity of combined masks)
        #   finite_single_holdout_r2 - atp_holdout_r2: finite per-component map beats infinitesimal AtP map (per-component curvature/calibration gap)
        finite_single_r2 = r2_score(y[hold], A[hold] @ finite_map)
        atp_r2 = r2_score(y[hold], A[hold] @ atp_map)
        atp_cal_r2 = r2_score(y[hold], A[hold] @ atp_cal_map)
        ridge_r2 = r2_score(y[hold], A[hold] @ ridge_beta + ridge_intercept)
        omp_r2 = r2_score(y[hold], A[hold] @ omp_beta)

        row = {
            "epsilon": float(eps),
            "n_components": int(M),
            "n_measurements": int(args.measurements),
            "mask_kind": args.mask_kind,
            "mask_density": float(args.mask_density),
            "mask_rows_normalized": bool(not args.raw_masks),
            "finite_map_norm": float(np.linalg.norm(finite_map)),
            "atp_map_norm": float(np.linalg.norm(atp_map)),
            "atp_calibrated_map_norm": float(np.linalg.norm(atp_cal_map)),
            "ridge_map_norm": float(np.linalg.norm(ridge_beta)),
            "omp_map_norm": float(np.linalg.norm(omp_beta)),
            "finite_on_atp_slope": slope_no_intercept(atp_map, finite_map),
            "atp_on_finite_slope": slope_no_intercept(finite_map, atp_map),
            "finite_to_atp_norm_ratio": float(np.linalg.norm(finite_map) / max(np.linalg.norm(atp_map), 1e-12)),
            "atp_vs_finite_cosine": cosine_sim(atp_map, finite_map),
            "atp_calibration_gain": float(atp_cal_gain),
            "atp_vs_finite_pearson": corrcoef(atp_map, finite_map),
            "atp_vs_finite_spearman": spearman_corr(atp_map, finite_map),
            "ridge_vs_finite_pearson": corrcoef(ridge_beta, finite_map),
            "ridge_vs_finite_spearman": spearman_corr(ridge_beta, finite_map),
            "omp_vs_finite_pearson": corrcoef(omp_beta, finite_map),
            "omp_vs_finite_spearman": spearman_corr(omp_beta, finite_map),
            "finite_single_holdout_r2": float(finite_single_r2),
            "atp_holdout_r2": float(atp_r2),
            "atp_calibrated_holdout_r2": float(atp_cal_r2),
            "ridge_holdout_r2": float(ridge_r2),
            "omp_holdout_r2": float(omp_r2),
            "ridge_train_r2": r2_score(y[train], A[train] @ ridge_beta + ridge_intercept),
            "omp_val_r2": float(omp_val_r2),
            "omp_k": int(omp_k),
            "aggregate_additivity_gap": float(1.0 - finite_single_r2),
            "atp_finite_calibration_gap": float(finite_single_r2 - atp_r2),
            "atp_calibrated_gap": float(finite_single_r2 - atp_cal_r2),
            "atp_minus_finite_mae": float(np.mean(np.abs(atp_map - finite_map))),
            "atp_calibrated_minus_finite_mae": float(np.mean(np.abs(atp_cal_map - finite_map))),
            "ridge_minus_finite_mae": float(np.mean(np.abs(ridge_beta - finite_map))),
            "omp_minus_finite_mae": float(np.mean(np.abs(omp_beta - finite_map))),
        }
        result_rows.append(row)

        for m, (li, bi) in enumerate(comps):
            start, end = bins[bi]
            map_rows.append({
                "epsilon": float(eps),
                "component": int(m),
                "layer": int(li),
                "bin": int(bi),
                "token_start": int(start),
                "token_end_exclusive": int(end),
                "finite_effect": float(finite_map[m]),
                "atp_effect": float(atp_map[m]),
                "atp_calibrated_effect": float(atp_cal_map[m]),
                "ridge_effect": float(ridge_beta[m]),
                "omp_effect": float(omp_beta[m]),
            })

        print(pd.DataFrame([row])[[
            "epsilon", "finite_on_atp_slope", "finite_to_atp_norm_ratio", "atp_vs_finite_pearson",
            "finite_single_holdout_r2", "atp_holdout_r2", "atp_calibrated_holdout_r2",
            "ridge_holdout_r2", "omp_holdout_r2", "omp_k"
        ]].to_string(index=False))

    results_df = pd.DataFrame(result_rows)
    maps_df = pd.DataFrame(map_rows)
    gain_budget_df = pd.DataFrame(gain_budget_rows)
    results_df.to_csv(outdir / "step0_results.csv", index=False)
    maps_df.to_csv(outdir / "step0_component_maps.csv", index=False)
    if not gain_budget_df.empty:
        gain_budget_df.to_csv(outdir / "step0_gain_budget_results.csv", index=False)
    plot_results(results_df, outdir)
    if not gain_budget_df.empty:
        plot_gain_budget(gain_budget_df, results_df, outdir)

    summary = {
        "run_dir": str(run_dir),
        "args": vars(args),
        "config": asdict(cfg),
        "hmm_params": asdict(params),
        "n_components": int(M),
        "base_forward_max_abs_diff": float(base_forward_max_abs_diff),
        "cost_model": {
            "atp_per_eval_batch": "1 forward + 1 backward for the full first-order map; reused across epsilon by scaling.",
            "calibrated_atp_per_epsilon": "m_gain aggregate forward measurements to fit one scalar gain, plus the AtP backward pass.",
            "finite_single_per_epsilon": f"{M} forward interventions for the full coordinate finite map.",
            "tomography_per_epsilon": f"{args.measurements} aggregate forward interventions plus sparse/ridge solve.",
        },
        "interpretation": (
            "If AtP held-out R2 tracks finite/tomography across epsilon, this harness is effectively first-order. "
            "If AtP works at small epsilon but degrades at larger epsilon where ridge/OMP finite tomography remains predictive, "
            "that is evidence that finite-epsilon effect maps differ from infinitesimal attribution maps."
        ),
        "results": results_df.to_dict(orient="records"),
    }
    with open(outdir / "step0_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. Key files:")
    for name in [
        "step0_summary.json",
        "step0_results.csv",
        "step0_component_maps.csv",
        "step0_gain_budget_results.csv",
        "step0_gain_budget_summary.csv",
        "step0_gain_budget_r2_vs_m_gain.png",
        "step0_gain_budget_gap_to_omp.png",
        "step0_heldout_r2_vs_epsilon.png",
        "step0_map_corr_vs_epsilon.png",
        "step0_omp_k_vs_epsilon.png",
        "measurement_matrix_A.npy",
    ]:
        print(f"  {outdir / name}")

    print("\nFinal results:")
    print(results_df[[
        "epsilon", "atp_vs_finite_pearson", "finite_single_holdout_r2", "atp_holdout_r2",
        "ridge_holdout_r2", "omp_holdout_r2", "omp_k"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
