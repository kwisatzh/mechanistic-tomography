#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
Post-hoc sparse mechanistic tomography recovery.

Run this after nt_mi_correspondence.py has produced:
  measurement_matrix_A.npy
  tomography_measurements.csv
  direct_mi_z1.npy

Purpose:
  Ridge recovery validates local additive tomography, but it does not test the
  measurement-efficiency advantage. This script compares ridge to sparse
  recovery (OMP and optional LASSO) on the same aggregate measurements, using
  only aggregate y-values for fitting/validation and the direct MI map only for
  final diagnostics.

Example:
  python sparse_tomography_posthoc.py \
    --mt-dir runs/m4_baseline_seed7/nt_mi_set1_v2 \
    --outdir runs/m4_baseline_seed7/nt_mi_sparse_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ar = pd.Series(np.asarray(a).reshape(-1)).rank(method="average").to_numpy()
    br = pd.Series(np.asarray(b).reshape(-1)).rank(method="average").to_numpy()
    return corrcoef(ar, br)


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y).reshape(-1)
    yhat = np.asarray(yhat).reshape(-1)
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom < 1e-12:
        return float("nan")
    return 1.0 - float(np.sum((y - yhat) ** 2)) / denom


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = min(k, len(a), len(b))
    ia = set(np.argsort(-np.abs(a))[:k].tolist())
    ib = set(np.argsort(-np.abs(b))[:k].tolist())
    return len(ia & ib) / max(1, k)


def fit_ridge_centered(A: np.ndarray, y: np.ndarray, ridge: float) -> Tuple[np.ndarray, float, Dict[str, float]]:
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    A_mean = A.mean(axis=0)
    y_mean = float(y.mean())
    Ac = A - A_mean
    yc = y - y_mean
    M = A.shape[1]
    beta = np.linalg.solve(Ac.T @ Ac + ridge * np.eye(M), Ac.T @ yc).reshape(-1)
    intercept = y_mean - float(A_mean.dot(beta))
    info = {"support_size": float(np.sum(np.abs(beta) > 1e-8)), "selected_k": float(M), "alpha": float(ridge)}
    return beta, intercept, info


def fit_omp_centered(A: np.ndarray, y: np.ndarray, k_max: int, ridge: float = 1e-10, tol: float = 1e-12) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """Orthogonal matching pursuit with centered data and an intercept.

    Selection uses column-normalized correlations; coefficient refits use the
    original centered columns. This keeps the returned beta in the original
    measurement units.
    """
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, M = A.shape
    k_max = max(0, min(int(k_max), M, max(0, n - 1)))
    A_mean = A.mean(axis=0)
    y_mean = float(y.mean())
    Ac = A - A_mean
    yc = y - y_mean
    col_norm = np.linalg.norm(Ac, axis=0)
    safe_norm = np.where(col_norm > 1e-12, col_norm, 1.0)
    Z = Ac / safe_norm
    residual = yc.copy()
    active: List[int] = []
    coef_active = np.zeros(0, dtype=np.float64)

    for _ in range(k_max):
        corr = Z.T @ residual
        if active:
            corr[np.array(active, dtype=int)] = 0.0
        j = int(np.argmax(np.abs(corr)))
        if not np.isfinite(corr[j]) or abs(corr[j]) < tol:
            break
        active.append(j)
        Xs = Ac[:, active]
        if ridge > 0:
            coef_active = np.linalg.solve(Xs.T @ Xs + ridge * np.eye(len(active)), Xs.T @ yc)
        else:
            coef_active = np.linalg.lstsq(Xs, yc, rcond=None)[0]
        residual = yc - Xs @ coef_active
        if np.linalg.norm(residual) <= tol * max(1.0, np.linalg.norm(yc)):
            break

    beta = np.zeros(M, dtype=np.float64)
    if active:
        beta[np.array(active, dtype=int)] = coef_active
    intercept = y_mean - float(A_mean.dot(beta))
    info = {"support_size": float(len(active)), "selected_k": float(len(active)), "alpha": float(k_max)}
    return beta, intercept, info


def soft_threshold(x: float, lam: float) -> float:
    if x > lam:
        return x - lam
    if x < -lam:
        return x + lam
    return 0.0


def fit_lasso_cd_centered(A: np.ndarray, y: np.ndarray, alpha: float, max_iter: int = 4000, tol: float = 1e-7) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """Small dependency-free LASSO coordinate descent.

    Objective: (1/(2n)) ||yc - Z b||^2 + alpha ||b||_1,
    where Z is centered and standardized. Returned beta is converted back to
    original A units.
    """
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, M = A.shape
    A_mean = A.mean(axis=0)
    y_mean = float(y.mean())
    Ac = A - A_mean
    yc = y - y_mean
    scale = np.sqrt(np.mean(Ac ** 2, axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    Z = Ac / scale
    z2 = np.mean(Z ** 2, axis=0)
    b = np.zeros(M, dtype=np.float64)
    pred = np.zeros(n, dtype=np.float64)

    for _ in range(max_iter):
        max_delta = 0.0
        for j in range(M):
            # residual plus current contribution of feature j
            r_j = yc - pred + Z[:, j] * b[j]
            rho = float(np.mean(Z[:, j] * r_j))
            new_b = soft_threshold(rho, alpha) / max(z2[j], 1e-12)
            delta = new_b - b[j]
            if delta != 0.0:
                pred += Z[:, j] * delta
                b[j] = new_b
                max_delta = max(max_delta, abs(delta))
        if max_delta < tol:
            break

    beta = b / scale
    intercept = y_mean - float(A_mean.dot(beta))
    info = {"support_size": float(np.sum(np.abs(beta) > 1e-8)), "selected_k": float(np.sum(np.abs(beta) > 1e-8)), "alpha": float(alpha)}
    return beta, intercept, info


def lasso_alpha_grid(A: np.ndarray, y: np.ndarray, n_grid: int = 24, min_frac: float = 1e-3) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    Ac = A - A.mean(axis=0)
    yc = y - float(y.mean())
    scale = np.sqrt(np.mean(Ac ** 2, axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    Z = Ac / scale
    alpha_max = float(np.max(np.abs(Z.T @ yc)) / max(1, len(y)))
    if not np.isfinite(alpha_max) or alpha_max <= 1e-12:
        alpha_max = 1.0
    return np.geomspace(alpha_max, alpha_max * min_frac, n_grid)


def evaluate_fit(method: str, A_fit: np.ndarray, y_fit: np.ndarray, A_val: np.ndarray, y_val: np.ndarray,
                 A_hold: np.ndarray, y_hold: np.ndarray, mi: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """Fit one method, using validation y only for hyperparameter selection."""
    candidates: List[Tuple[np.ndarray, float, Dict[str, float], float]] = []

    if method == "ridge":
        beta, intercept, info = fit_ridge_centered(A_fit, y_fit, ridge=args.ridge)
        val_r2 = r2_score(y_val, A_val @ beta + intercept)
        candidates.append((beta, intercept, info, val_r2))
    elif method == "omp":
        for k in args.omp_k_grid:
            if k <= 0 or k >= len(y_fit):
                continue
            beta, intercept, info = fit_omp_centered(A_fit, y_fit, k_max=k, ridge=args.omp_refit_ridge)
            val_r2 = r2_score(y_val, A_val @ beta + intercept)
            candidates.append((beta, intercept, info, val_r2))
    elif method == "lasso":
        for alpha in lasso_alpha_grid(A_fit, y_fit, n_grid=args.lasso_grid):
            beta, intercept, info = fit_lasso_cd_centered(A_fit, y_fit, alpha=alpha, max_iter=args.lasso_iter)
            val_r2 = r2_score(y_val, A_val @ beta + intercept)
            candidates.append((beta, intercept, info, val_r2))
    else:
        raise ValueError(f"unknown method {method}")

    if not candidates:
        beta = np.zeros(A_fit.shape[1], dtype=np.float64)
        intercept = float(np.mean(y_fit))
        info = {"support_size": 0.0, "selected_k": 0.0, "alpha": 0.0}
        return beta, intercept, {**info, "val_r2": float("nan"), "holdout_r2": r2_score(y_hold, A_hold @ beta + intercept)}

    # Hyperparameters are chosen by validation aggregate prediction, not MI-map agreement.
    beta, intercept, info, val_r2 = max(candidates, key=lambda t: (-np.inf if np.isnan(t[3]) else t[3]))
    info = dict(info)
    info["val_r2"] = float(val_r2)
    info["holdout_r2"] = r2_score(y_hold, A_hold @ beta + intercept)
    info["train_r2"] = r2_score(y_fit, A_fit @ beta + intercept)
    info["pearson_vs_mi"] = corrcoef(mi, beta)
    info["spearman_vs_mi"] = spearman_corr(mi, beta)
    info["top5_overlap"] = topk_overlap(mi, beta, k=min(5, len(mi)))
    info["mae_vs_mi"] = float(np.mean(np.abs(beta - mi)))
    info["beta_norm"] = float(np.linalg.norm(beta))
    return beta, intercept, info


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def plot_curves(df: pd.DataFrame, outdir: Path, y_col: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(6.8, 4.4))
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method].sort_values("n_train")
        plt.plot(sub["n_train"], sub[y_col], marker="o", label=method)
    plt.axvline(float(df["n_components"].iloc[0]), linestyle="--", linewidth=1, label="n_components")
    plt.xlabel("aggregate measurements used for fitting")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs measurement budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=180)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Sparse post-hoc recovery from existing mechanistic tomography measurements")
    p.add_argument("--mt-dir", required=True, type=str, help="Directory from nt_mi_correspondence.py")
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train-sizes", type=str, default="8,12,16,24,32,48,64,96,128,192")
    p.add_argument("--holdout-frac", type=float, default=0.25)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--methods", type=str, default="ridge,omp,lasso")
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--omp-k-grid", type=str, default="1,2,3,4,5,6,8,10,12,16")
    p.add_argument("--omp-refit-ridge", type=float, default=1e-10)
    p.add_argument("--lasso-grid", type=int, default=24)
    p.add_argument("--lasso-iter", type=int, default=4000)
    args = p.parse_args()

    mt_dir = Path(args.mt_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else mt_dir / "sparse_recovery"
    outdir.mkdir(parents=True, exist_ok=True)

    A = np.load(mt_dir / "measurement_matrix_A.npy")
    meas_df = pd.read_csv(mt_dir / "tomography_measurements.csv")
    y = meas_df["y_z1"].to_numpy(dtype=np.float64)
    mi = np.load(mt_dir / "direct_mi_z1.npy")
    M = A.shape[1]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(A.shape[0])
    n_hold = max(1, int(args.holdout_frac * len(perm)))
    n_val = max(1, int(args.val_frac * len(perm)))
    hold_idx = perm[:n_hold]
    val_idx = perm[n_hold:n_hold + n_val]
    pool_idx = perm[n_hold + n_val:]

    requested_sizes = parse_int_list(args.train_sizes)
    train_sizes = [n for n in requested_sizes if n <= len(pool_idx)]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    args.omp_k_grid = parse_int_list(args.omp_k_grid)

    rows = []
    betas: Dict[Tuple[str, int], np.ndarray] = {}
    for n_train in train_sizes:
        fit_idx = pool_idx[:n_train]
        for method in methods:
            beta, intercept, info = evaluate_fit(
                method,
                A[fit_idx], y[fit_idx],
                A[val_idx], y[val_idx],
                A[hold_idx], y[hold_idx],
                mi,
                args,
            )
            row = {
                "method": method,
                "n_train": int(n_train),
                "n_components": int(M),
                "n_holdout": int(len(hold_idx)),
                "n_validation": int(len(val_idx)),
                **info,
            }
            rows.append(row)
            betas[(method, n_train)] = beta

    df = pd.DataFrame(rows).sort_values(["method", "n_train"])
    df.to_csv(outdir / "sparse_recovery_sample_efficiency.csv", index=False)
    np.savez(outdir / "sparse_recovery_betas.npz", **{f"{m}_{n}": b for (m, n), b in betas.items()})

    plot_curves(df, outdir, "pearson_vs_mi", "Pearson vs direct MI map", "sparse_pearson_vs_budget.png")
    plot_curves(df, outdir, "holdout_r2", "held-out aggregate R²", "sparse_holdout_r2_vs_budget.png")
    plot_curves(df, outdir, "top5_overlap", "top-5 overlap with direct MI", "sparse_top5_vs_budget.png")

    # A compact summary of the first budget where each method crosses a useful threshold.
    thresholds = {"pearson_0.90": ("pearson_vs_mi", 0.90), "pearson_0.95": ("pearson_vs_mi", 0.95), "r2_0.90": ("holdout_r2", 0.90)}
    summary = {
        "mt_dir": str(mt_dir),
        "n_measurements_total": int(A.shape[0]),
        "n_components": int(M),
        "train_sizes": train_sizes,
        "methods": methods,
        "threshold_crossings": {},
        "interpretation": (
            "Ridge validates local additive recovery. OMP/LASSO test the actual tomography advantage claim: "
            "whether a sparse mechanism map can be recovered with fewer aggregate measurements than exhaustive "
            "single-component patching. Hyperparameters are selected by aggregate-measurement validation R2, not by "
            "direct-MI agreement. Direct MI is used only for final diagnostics."
        ),
    }
    for method in methods:
        sub = df[df["method"] == method].sort_values("n_train")
        summary["threshold_crossings"][method] = {}
        for name, (col, th) in thresholds.items():
            hit = sub[sub[col] >= th]
            summary["threshold_crossings"][method][name] = None if hit.empty else int(hit.iloc[0]["n_train"])
    with open(outdir / "sparse_recovery_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done. Key files:")
    for name in [
        "sparse_recovery_summary.json",
        "sparse_recovery_sample_efficiency.csv",
        "sparse_pearson_vs_budget.png",
        "sparse_holdout_r2_vs_budget.png",
        "sparse_top5_vs_budget.png",
    ]:
        print(f"  {outdir / name}")
    print("\nThreshold crossings:")
    print(json.dumps(summary["threshold_crossings"], indent=2))


if __name__ == "__main__":
    main()
