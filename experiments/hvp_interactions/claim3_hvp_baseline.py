#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
Claim 3 designed-HVP baseline.

White-box reviewers will ask: if the target is a sparse second-order / pair map,
why not use Hessian-vector products instead of forward subset measurements?

This script implements that baseline for the Claim-3 planted plant. It recovers
sparse pair coefficients from designed HVP queries and evaluates the recovered
pair map on the same held-out finite-intervention masks used by lifted forward
measurement.

Outputs:
  - claim3_hvp_results.csv
  - claim3_hvp_summary.csv
  - claim3_hvp_metadata.json
  - claim3_hvp_pair_recall.png
  - claim3_hvp_r2.png
  - claim3_hvp_selected_k.png
  - claim3_hvp_report.md

Interpretation:
  One backward pass gives the first-order gradient, but not the sparse pair map.
  HVPs are a white-box designed-measurement primitive for that pair map.
  If a few HVPs recover pairs, the right paper claim is not "forward tomography
  beats all white-box methods," but "interaction maps still require designed
  measurements; the primitive is HVPs when gradients are available and forward
  subset measurements when they are not."
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((y - yhat) ** 2) / denom)


def pair_indices(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def pair_features(A: np.ndarray, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if len(pairs) == 0:
        return np.zeros((A.shape[0], 0), dtype=float)
    X = np.empty((A.shape[0], len(pairs)), dtype=float)
    for k, (i, j) in enumerate(pairs):
        X[:, k] = A[:, i] * A[:, j]
    return X


def make_masks(rng: np.random.Generator, m: int, n: int, density: float, kind: str = "signed", normalize_rows: bool = True) -> np.ndarray:
    active = rng.random((m, n)) < float(density)
    if kind == "signed":
        A = active.astype(float) * rng.choice([-1.0, 1.0], size=(m, n))
    elif kind == "binary":
        A = active.astype(float)
    elif kind == "gaussian":
        A = rng.normal(0.0, 1.0, size=(m, n))
        if density < 1.0:
            A *= active.astype(float)
    else:
        raise ValueError(f"unknown kind {kind}")
    for i in range(m):
        if np.all(A[i] == 0):
            j = rng.integers(0, n)
            A[i, j] = rng.choice([-1.0, 1.0]) if kind in ("signed", "gaussian") else 1.0
    if normalize_rows:
        norms = np.linalg.norm(A, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        A = A / norms
    return A


class PlantedPlant:
    def __init__(
        self,
        n: int = 64,
        mode: str = "combined",
        beta1: float = 1.4,
        beta2: float = -1.1,
        lambda1: float = 0.18,
        lambda2: float = 0.75,
        gamma_interaction: float = 1.4,
        gamma_redundant: float = -1.2,
    ):
        if n < 8:
            raise ValueError("n must be at least 8")
        self.n = int(n)
        self.mode = mode
        self.main_indices = [0, 1]
        self.interaction_pair = (2, 3)
        self.redundant_pair = (4, 5)
        self.confound_idx = 6
        self.beta = np.zeros(n, dtype=float)
        self.lam = np.ones(n, dtype=float) * lambda1
        self.pairs: Dict[Tuple[int, int], float] = {}
        if mode in ("two_coord_saturation", "combined", "confound"):
            self.beta[0] = beta1
            self.beta[1] = beta2
            self.lam[0] = lambda1
            self.lam[1] = lambda2
        if mode in ("pure_interaction", "combined"):
            self.pairs[self.interaction_pair] = gamma_interaction
        if mode in ("redundant_pair", "combined"):
            self.pairs[self.redundant_pair] = gamma_redundant
        if mode == "confound":
            self.beta[0] = beta1
            self.beta[1] = 0.0
            self.lam[0] = lambda1

    def atp_map(self) -> np.ndarray:
        return self.beta.copy()

    def true_linear_finite_map(self, epsilon: float) -> np.ndarray:
        out = np.zeros(self.n, dtype=float)
        for i, b in enumerate(self.beta):
            if abs(b) > 0:
                lam = self.lam[i]
                out[i] = b * math.tanh(lam * epsilon) / (lam * epsilon)
        return out

    def finite_single_map(self, epsilon: float) -> np.ndarray:
        # Same as true_linear_finite_map because planted pair terms vanish on singletons.
        return self.true_linear_finite_map(epsilon)

    def hessian_pair_matrix(self, epsilon: float) -> np.ndarray:
        """Hessian of normalized finite readout y_tilde(a) with respect to mask a.

        For pair term gamma * epsilon * a_i * a_j, H_ij = H_ji = gamma * epsilon.
        Saturating main effects have zero second derivative at the origin; this baseline
        is intentionally about the sparse interaction map, not finite main calibration.
        """
        H = np.zeros((self.n, self.n), dtype=float)
        for (i, j), g in self.pairs.items():
            H[i, j] = g * float(epsilon)
            H[j, i] = g * float(epsilon)
        return H

    def pair_coeff_vector(self, epsilon: float, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
        coeff = np.zeros(len(pairs), dtype=float)
        for k, p in enumerate(pairs):
            if p in self.pairs:
                coeff[k] = self.pairs[p] * float(epsilon)
        return coeff

    def measure(self, A: np.ndarray, epsilon: float, noise_std: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        eps = float(epsilon)
        y = np.zeros(A.shape[0], dtype=float)
        for i, b in enumerate(self.beta):
            if abs(b) > 0:
                lam = self.lam[i]
                y += b * np.tanh(lam * eps * A[:, i]) / (lam * eps)
        for (i, j), g in self.pairs.items():
            y += g * eps * A[:, i] * A[:, j]
        if noise_std > 0:
            if rng is None:
                rng = np.random.default_rng(0)
            y += rng.normal(0.0, float(noise_std), size=A.shape[0])
        return y

    def true_pairs(self) -> List[Tuple[int, int]]:
        return list(self.pairs.keys())


def hvp_design_matrix(V: np.ndarray, pairs: Sequence[Tuple[int, int]], n: int) -> np.ndarray:
    """Build X such that vec(H V_t)_i = X @ pair_coeffs.

    Unknown pair coefficient c_ij equals H_ij = H_ji. For each query vector v and
    output coordinate r, (H v)_r = sum_j c_{rj} v_j. Each HVP query contributes n
    scalar equations.
    """
    V = np.asarray(V, dtype=float)
    q = V.shape[0]
    X = np.zeros((q * n, len(pairs)), dtype=float)
    for col, (i, j) in enumerate(pairs):
        # Row for output i gets v_j; row for output j gets v_i.
        X[np.arange(q) * n + i, col] = V[:, j]
        X[np.arange(q) * n + j, col] = V[:, i]
    return X


def fit_omp_path(X: np.ndarray, y: np.ndarray, max_k: int, ridge: float = 1e-8) -> List[Tuple[List[int], np.ndarray, float]]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    # Center y but not X; HVP equations have no intercept. Keep columns normalized for selection.
    col_norms = np.linalg.norm(X, axis=0)
    col_norms[col_norms < 1e-12] = 1.0
    Xn = X / col_norms
    residual = y.copy()
    active: List[int] = []
    path: List[Tuple[List[int], np.ndarray, float]] = []
    max_k = int(min(max_k, X.shape[1], max(1, X.shape[0] - 1)))
    for _ in range(max_k):
        corr = Xn.T @ residual
        if active:
            corr[np.array(active, dtype=int)] = 0.0
        j = int(np.argmax(np.abs(corr)))
        active.append(j)
        Xa = Xn[:, active]
        coef_scaled = np.linalg.solve(Xa.T @ Xa + ridge * np.eye(len(active)), Xa.T @ y)
        residual = y - Xa @ coef_scaled
        beta = np.zeros(X.shape[1], dtype=float)
        beta[np.array(active)] = coef_scaled / col_norms[np.array(active)]
        train_r2 = r2_score(y, X @ beta)
        path.append((list(active), beta, train_r2))
    return path


def fit_omp_with_hvp_validation(
    V: np.ndarray,
    Y: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    max_k: int,
    val_frac: float,
    rng: np.random.Generator,
    ridge: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, object]]:
    q, n = V.shape
    if q < 4:
        # Too little budget for reliable validation; use a small default path length.
        q_train = q
        train_idx = np.arange(q)
        val_idx = np.array([], dtype=int)
    else:
        perm = rng.permutation(q)
        q_val = max(1, int(round(val_frac * q)))
        val_idx = perm[:q_val]
        train_idx = perm[q_val:]
        q_train = len(train_idx)
        if q_train < 1:
            train_idx = perm
            val_idx = np.array([], dtype=int)
    Xtr = hvp_design_matrix(V[train_idx], pairs, n)
    ytr = Y[train_idx].reshape(-1)
    path = fit_omp_path(Xtr, ytr, max_k=max_k, ridge=ridge)
    if len(val_idx) > 0:
        Xval = hvp_design_matrix(V[val_idx], pairs, n)
        yval = Y[val_idx].reshape(-1)
        scores = [r2_score(yval, Xval @ beta) for _, beta, _ in path]
        best_idx = int(np.nanargmax(scores))
        best_k = best_idx + 1
        best_val_r2 = float(scores[best_idx])
    else:
        # Fallback: choose min(max_k, 2 * expected planted pair count if available later).
        best_k = min(max_k, 4)
        best_val_r2 = float("nan")
    # Refit on all queries with selected k by taking the path on all data.
    Xall = hvp_design_matrix(V, pairs, n)
    yall = Y.reshape(-1)
    path_all = fit_omp_path(Xall, yall, max_k=max(best_k, 1), ridge=ridge)
    beta = path_all[best_k - 1][1]
    return beta, {
        "best_k": int(best_k),
        "best_val_r2": best_val_r2,
        "train_queries": int(q_train),
        "val_queries": int(len(val_idx)),
    }


def pair_recall(beta_pair: np.ndarray, pairs: Sequence[Tuple[int, int]], true_pairs: Sequence[Tuple[int, int]]) -> float:
    true_set = set(tuple(p) for p in true_pairs)
    if not true_set:
        return float("nan")
    k = len(true_set)
    order = np.argsort(-np.abs(beta_pair))[:k]
    pred = {tuple(pairs[i]) for i in order}
    return float(len(pred.intersection(true_set)) / k)


def run_one(args: argparse.Namespace, seed: int, epsilon: float, hvp_budget: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + int(1000 * epsilon) + 17 * hvp_budget)
    plant = PlantedPlant(
        n=args.n_components,
        mode=args.mode,
        beta1=args.beta1,
        beta2=args.beta2,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        gamma_interaction=args.gamma_interaction,
        gamma_redundant=args.gamma_redundant,
    )
    n = plant.n
    pairs = pair_indices(n)
    H = plant.hessian_pair_matrix(epsilon)
    V = make_masks(rng, hvp_budget, n, density=args.hvp_density, kind=args.hvp_kind, normalize_rows=not args.raw_hvp_vectors)
    Y = V @ H.T
    if args.hvp_noise_std > 0:
        Y = Y + rng.normal(0.0, args.hvp_noise_std, size=Y.shape)

    beta_pair, info = fit_omp_with_hvp_validation(
        V, Y, pairs, max_k=args.hvp_omp_max_k, val_frac=args.val_frac, rng=rng, ridge=args.ridge
    )

    A_test = make_masks(rng, args.holdout_measurements, n, density=args.mask_density, kind=args.mask_kind, normalize_rows=not args.raw_masks)
    y_test = plant.measure(A_test, epsilon=epsilon, noise_std=args.forward_noise_std, rng=rng)
    pair_X = pair_features(A_test, pairs)

    main_atp = plant.atp_map()
    main_finite = plant.true_linear_finite_map(epsilon)
    pred_atp_main = A_test @ main_atp + pair_X @ beta_pair
    pred_finite_main = A_test @ main_finite + pair_X @ beta_pair
    pred_pairs_only = pair_X @ beta_pair

    true_pair_beta = plant.pair_coeff_vector(epsilon, pairs)
    pair_rmse = float(np.sqrt(np.mean((beta_pair - true_pair_beta) ** 2)))
    pair_l1 = float(np.sum(np.abs(beta_pair - true_pair_beta)))
    pair_rec = pair_recall(beta_pair, pairs, plant.true_pairs())
    selected = np.argsort(-np.abs(beta_pair))[:max(1, info["best_k"])]
    false_selected = [pairs[i] for i in selected if tuple(pairs[i]) not in set(plant.true_pairs())]

    rows = []
    for method, pred, main_source in [
        ("hvp_pairs_atp_main", pred_atp_main, "atp"),
        ("hvp_pairs_finite_main", pred_finite_main, "finite_main_oracle"),
        ("hvp_pairs_only", pred_pairs_only, "none"),
    ]:
        rows.append({
            "seed": seed,
            "epsilon": epsilon,
            "hvp_budget": hvp_budget,
            "method": method,
            "main_source": main_source,
            "heldout_r2": r2_score(y_test, pred),
            "pair_topk_recall": pair_rec,
            "pair_rmse": pair_rmse,
            "pair_l1": pair_l1,
            "selected_k": info["best_k"],
            "hvp_val_r2": info["best_val_r2"],
            "false_pair_count_in_selected": len(false_selected),
            "budget_backward": hvp_budget + (1 if method != "hvp_pairs_only" else 0),
            "budget_forward": 0,
            "train_queries": info["train_queries"],
            "val_queries": info["val_queries"],
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["heldout_r2", "pair_topk_recall", "pair_rmse", "pair_l1", "selected_k", "hvp_val_r2", "false_pair_count_in_selected"]
    return df.groupby(["epsilon", "hvp_budget", "method"])[metrics].agg(["mean", "std", "count"]).reset_index()


def threshold_table(summary_flat: pd.DataFrame, r2_thr: float, pair_thr: float) -> pd.DataFrame:
    rows = []
    for (eps, method), sub in summary_flat.groupby(["epsilon", "method"]):
        sub = sub.sort_values("hvp_budget")
        r2_budget = sub.loc[sub["heldout_r2_mean"] >= r2_thr, "hvp_budget"]
        pair_budget = sub.loc[sub["pair_topk_recall_mean"] >= pair_thr, "hvp_budget"]
        both_budget = sub.loc[(sub["heldout_r2_mean"] >= r2_thr) & (sub["pair_topk_recall_mean"] >= pair_thr), "hvp_budget"]
        rows.append({
            "epsilon": eps,
            "method": method,
            f"first_hvp_budget_r2_ge_{r2_thr}": float(r2_budget.iloc[0]) if len(r2_budget) else np.nan,
            f"first_hvp_budget_pair_ge_{pair_thr}": float(pair_budget.iloc[0]) if len(pair_budget) else np.nan,
            f"first_hvp_budget_both_ge_r2_{r2_thr}_pair_{pair_thr}": float(both_budget.iloc[0]) if len(both_budget) else np.nan,
            "max_heldout_r2_mean": float(sub["heldout_r2_mean"].max()),
            "max_pair_topk_recall_mean": float(sub["pair_topk_recall_mean"].max()),
        })
    return pd.DataFrame(rows)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = ["_".join([str(c) for c in col if c != ""]).strip("_") if isinstance(col, tuple) else str(col) for col in df.columns]
    return df


def plot_results(summary_flat: pd.DataFrame, outdir: Path, r2_thr: float, pair_thr: float):
    outdir.mkdir(parents=True, exist_ok=True)
    methods = ["hvp_pairs_atp_main", "hvp_pairs_finite_main"]
    # R2 vs HVP budget.
    plt.figure(figsize=(9.2, 5.4))
    for method in methods:
        for eps in sorted(summary_flat["epsilon"].unique()):
            sub = summary_flat[(summary_flat.method == method) & (np.isclose(summary_flat.epsilon, eps))].sort_values("hvp_budget")
            if sub.empty:
                continue
            label = f"{method}, eps={eps:g}"
            plt.errorbar(sub["hvp_budget"], sub["heldout_r2_mean"], yerr=sub["heldout_r2_std"].fillna(0), marker="o", capsize=3, label=label)
    plt.axhline(r2_thr, linestyle="--", linewidth=1)
    plt.xscale("log", base=2)
    plt.xlabel("HVP queries / backward-pass budget")
    plt.ylabel("held-out finite-intervention R²")
    plt.title("Designed-HVP baseline: prediction quality")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "claim3_hvp_r2.png", dpi=180)
    plt.close()

    # Pair recall.
    plt.figure(figsize=(9.2, 5.4))
    for eps in sorted(summary_flat["epsilon"].unique()):
        sub = summary_flat[(summary_flat.method == "hvp_pairs_finite_main") & (np.isclose(summary_flat.epsilon, eps))].sort_values("hvp_budget")
        if sub.empty:
            continue
        plt.errorbar(sub["hvp_budget"], sub["pair_topk_recall_mean"], yerr=sub["pair_topk_recall_std"].fillna(0), marker="o", capsize=3, label=f"eps={eps:g}")
    plt.axhline(pair_thr, linestyle="--", linewidth=1)
    plt.xscale("log", base=2)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("HVP queries / backward-pass budget")
    plt.ylabel("top-k pair recovery")
    plt.title("Designed-HVP baseline: pair recovery")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "claim3_hvp_pair_recall.png", dpi=180)
    plt.close()

    # Selected k.
    plt.figure(figsize=(9.2, 5.0))
    for eps in sorted(summary_flat["epsilon"].unique()):
        sub = summary_flat[(summary_flat.method == "hvp_pairs_finite_main") & (np.isclose(summary_flat.epsilon, eps))].sort_values("hvp_budget")
        if sub.empty:
            continue
        plt.errorbar(sub["hvp_budget"], sub["selected_k_mean"], yerr=sub["selected_k_std"].fillna(0), marker="o", capsize=3, label=f"eps={eps:g}")
    plt.xscale("log", base=2)
    plt.xlabel("HVP queries / backward-pass budget")
    plt.ylabel("selected OMP support size")
    plt.title("Designed-HVP baseline: selected support size")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "claim3_hvp_selected_k.png", dpi=180)
    plt.close()


def write_report(outdir: Path, metadata: Dict[str, object], thresholds: pd.DataFrame):
    lines = []
    lines.append("# Claim 3 Designed-HVP Baseline\n")
    lines.append("## Setup\n")
    for k in ["n_components", "n_pairs", "lifted_dim", "support_k", "hvp_budgets", "epsilons", "seeds"]:
        lines.append(f"- **{k}**: {metadata.get(k)}")
    lines.append("\n## Thresholds\n")
    lines.append("```text")
    lines.append(thresholds.to_string(index=False))
    lines.append("```")
    lines.append("\n## Reading guide\n")
    lines.append("- `hvp_pairs_atp_main` uses AtP first-order main effects plus HVP-recovered pair terms.")
    lines.append("- `hvp_pairs_finite_main` is an upper-bound diagnostic: finite-calibrated main effects plus HVP-recovered pair terms. It isolates pair recovery from first-order finite calibration.")
    lines.append("- If HVP recovers pairs in a few backward passes, the white-box interactional regime still needs designed measurements, but the primitive is HVP rather than forward subset masks.")
    (outdir / "claim3_hvp_report.md").write_text("\n".join(lines))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Claim 3 HVP baseline")
    p.add_argument("--outdir", type=str, default="runs/claim3_hvp_baseline")
    p.add_argument("--mode", type=str, default="combined", choices=["combined", "two_coord_saturation", "pure_interaction", "redundant_pair", "confound"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-components", type=int, default=64)
    p.add_argument("--epsilons", type=float, nargs="+", default=[5.0, 8.0])
    p.add_argument("--hvp-budgets", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 24, 32, 48, 64])
    p.add_argument("--holdout-measurements", type=int, default=512)
    p.add_argument("--mask-kind", type=str, default="signed", choices=["signed", "binary", "gaussian"], help="heldout forward mask kind")
    p.add_argument("--mask-density", type=float, default=0.30)
    p.add_argument("--raw-masks", action="store_true")
    p.add_argument("--hvp-kind", type=str, default="signed", choices=["signed", "binary", "gaussian"], help="designed HVP vector family")
    p.add_argument("--hvp-density", type=float, default=1.0, help="HVP vectors default dense; try 0.3 to match forward masks")
    p.add_argument("--raw-hvp-vectors", action="store_true")
    p.add_argument("--hvp-noise-std", type=float, default=0.01)
    p.add_argument("--forward-noise-std", type=float, default=0.01)
    p.add_argument("--ridge", type=float, default=1e-8)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--hvp-omp-max-k", type=int, default=16)
    p.add_argument("--r2-threshold", type=float, default=0.95)
    p.add_argument("--pair-threshold", type=float, default=0.99)
    p.add_argument("--beta1", type=float, default=1.4)
    p.add_argument("--beta2", type=float, default=-1.1)
    p.add_argument("--lambda1", type=float, default=0.18)
    p.add_argument("--lambda2", type=float, default=0.75)
    p.add_argument("--gamma-interaction", type=float, default=1.4)
    p.add_argument("--gamma-redundant", type=float, default=-1.2)
    p.add_argument("--quick", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    if args.quick:
        args.seeds = args.seeds[:1]
        args.hvp_budgets = [1, 2, 4, 8]
        args.epsilons = [5.0, 8.0]
        args.holdout_measurements = min(args.holdout_measurements, 128)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for budget in args.hvp_budgets:
        for seed in args.seeds:
            for eps in args.epsilons:
                print(f"Running HVP baseline seed={seed} eps={eps} hvp_budget={budget}")
                all_rows.append(run_one(args, seed=seed, epsilon=float(eps), hvp_budget=int(budget)))
    results = pd.concat(all_rows, ignore_index=True)
    summary = summarize(results)
    summary_flat = flatten_columns(summary)
    thresholds = threshold_table(summary_flat, args.r2_threshold, args.pair_threshold)

    n = args.n_components
    metadata = {
        "args": vars(args),
        "n_components": n,
        "n_pairs": n * (n - 1) // 2,
        "lifted_dim": n + n * (n - 1) // 2,
        "support_k": 5,
        "hvp_budgets": args.hvp_budgets,
        "epsilons": args.epsilons,
        "seeds": args.seeds,
        "interpretation": "HVP recovers the sparse pair map in the white-box second-order regime. It is the correct reviewer baseline for interactional estimands."
    }
    results.to_csv(outdir / "claim3_hvp_results.csv", index=False)
    summary.to_csv(outdir / "claim3_hvp_summary.csv", index=False)
    summary_flat.to_csv(outdir / "claim3_hvp_summary_flat.csv", index=False)
    thresholds.to_csv(outdir / "claim3_hvp_thresholds.csv", index=False)
    with open(outdir / "claim3_hvp_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    plot_results(summary_flat, outdir, args.r2_threshold, args.pair_threshold)
    write_report(outdir, metadata, thresholds)
    print("\nWrote:")
    for name in [
        "claim3_hvp_results.csv", "claim3_hvp_summary.csv", "claim3_hvp_summary_flat.csv",
        "claim3_hvp_thresholds.csv", "claim3_hvp_metadata.json", "claim3_hvp_report.md",
        "claim3_hvp_r2.png", "claim3_hvp_pair_recall.png", "claim3_hvp_selected_k.png",
    ]:
        print(f"  {outdir / name}")
    print("\nThresholds:")
    print(thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
