#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
Claim 3 planted-reach experiment for Mechanistic Tomography.

This is a standalone synthetic wind-tunnel for the regime Step 0 says is needed:
rank > 1 finite-epsilon misspecification, pure interactions, redundancy/self-repair,
and confounded/off-path coordinates. It is intentionally small enough to run on an
Apple M4 CPU/MPS machine and does not require a GPU.

The script compares:
  - singleton finite patching (identity design)
  - raw AtP (infinitesimal first-order map)
  - scalar-calibrated AtP
  - two-gain/multi-gain calibrated AtP for known main-effect groups
  - random subset regression / KernelSHAP-style linear ridge
  - sparse first-order tomography (OMP)
  - lifted sparse tomography over pair terms (OMP on [A, pair(A)])

The default "combined" plant includes:
  (a) two main-effect coordinates with heterogeneous saturation;
  (b) a pure interaction pair with zero first-order map;
  (c) a redundant/self-repair pair visible only jointly;
  (d) an off-path confound coordinate, optionally correlated with a causal coordinate
      in the training measurement design.

The readout is a finite normalized effect y_tilde = (f(h + eps*D a)-f(h))/eps.
The infinitesimal AtP map is the derivative at eps=0. In the one-coordinate smooth
case, scalar calibration can fix AtP. In this planted setting, that should fail
because the finite correction is multi-rank and interactional.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((y - yhat) ** 2) / denom)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def make_masks(
    rng: np.random.Generator,
    m: int,
    n: int,
    density: float,
    kind: str = "signed",
    normalize_rows: bool = True,
    confound_pair: Tuple[int, int] | None = None,
    confound_prob: float = 0.0,
) -> np.ndarray:
    """Create random aggregate measurement masks.

    If confound_pair=(causal, confound) and confound_prob>0, then with that
    probability the confound column is copied from the causal column. This simulates
    a bad measurement design where a non-causal coordinate is aliased with a causal
    one in train measurements. Heldout designs should usually set confound_prob=0.
    """
    density = float(density)
    if not (0.0 < density <= 1.0):
        raise ValueError("density must be in (0, 1]")
    active = rng.random((m, n)) < density
    if kind == "signed":
        signs = rng.choice([-1.0, 1.0], size=(m, n))
        A = active.astype(float) * signs
    elif kind == "binary":
        A = active.astype(float)
    else:
        raise ValueError(f"unknown mask kind {kind}")

    # Avoid all-zero rows.
    for i in range(m):
        if np.all(A[i] == 0):
            j = rng.integers(0, n)
            A[i, j] = rng.choice([-1.0, 1.0]) if kind == "signed" else 1.0

    if confound_pair is not None and confound_prob > 0:
        c, q = confound_pair
        flip = rng.random(m) < confound_prob
        A[flip, q] = A[flip, c]

    if normalize_rows:
        norms = np.linalg.norm(A, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        A = A / norms
    return A


def pair_indices(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def pair_features(A: np.ndarray, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    if len(pairs) == 0:
        return np.zeros((A.shape[0], 0), dtype=float)
    X = np.empty((A.shape[0], len(pairs)), dtype=float)
    for k, (i, j) in enumerate(pairs):
        X[:, k] = A[:, i] * A[:, j]
    return X


def center_fit_intercept(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    xm = X.mean(axis=0)
    ym = float(y.mean())
    return X - xm, y - ym, ym - float(np.dot(xm, np.zeros(X.shape[1])))


def fit_ridge(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> Tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    xm = X.mean(axis=0)
    ym = y.mean()
    Xc = X - xm
    yc = y - ym
    G = Xc.T @ Xc + ridge * np.eye(X.shape[1])
    beta = np.linalg.solve(G, Xc.T @ yc)
    intercept = float(ym - xm @ beta)
    return beta, intercept


def predict_linear(X: np.ndarray, beta: np.ndarray, intercept: float = 0.0) -> np.ndarray:
    return np.asarray(X) @ np.asarray(beta) + float(intercept)


def fit_scalar_gain(feature: np.ndarray, y: np.ndarray, ridge: float = 1e-9) -> float:
    f = np.asarray(feature, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    denom = float(np.dot(f, f) + ridge)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(f, y) / denom)


def fit_omp(
    X: np.ndarray,
    y: np.ndarray,
    max_k: int,
    val_frac: float = 0.25,
    ridge: float = 1e-8,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """OMP with validation-selected k.

    Returns beta, intercept, info. Columns are standardized internally for selection,
    then coefficients are mapped back to original scale.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    m, p = X.shape
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.permutation(m)
    n_val = max(1, int(round(val_frac * m)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    Xtr_raw, ytr = X[tr_idx], y[tr_idx]
    Xval_raw, yval = X[val_idx], y[val_idx]

    xm = Xtr_raw.mean(axis=0)
    xs = Xtr_raw.std(axis=0)
    xs[xs < 1e-10] = 1.0
    ym = ytr.mean()
    Xtr = (Xtr_raw - xm) / xs
    Xval = (Xval_raw - xm) / xs
    yc = ytr - ym

    residual = yc.copy()
    active: List[int] = []
    best = None
    hist = []
    max_k = int(min(max_k, p, max(1, len(tr_idx) - 1)))

    for k in range(1, max_k + 1):
        corr = Xtr.T @ residual
        if active:
            corr[np.array(active, dtype=int)] = 0.0
        j = int(np.argmax(np.abs(corr)))
        active.append(j)
        Xa = Xtr[:, active]
        coef_a = np.linalg.solve(Xa.T @ Xa + ridge * np.eye(len(active)), Xa.T @ yc)
        residual = yc - Xa @ coef_a
        pred_val = ym + Xval[:, active] @ coef_a
        val_r2 = r2_score(yval, pred_val)
        hist.append({"k": k, "val_r2": val_r2})
        if best is None or val_r2 > best[0]:
            best = (val_r2, list(active), coef_a.copy(), k)

    assert best is not None
    _, active_best, _, best_k = best

    # Refit on all data with selected support.
    xm_all = X.mean(axis=0)
    xs_all = X.std(axis=0)
    xs_all[xs_all < 1e-10] = 1.0
    ym_all = y.mean()
    Xc_all = (X - xm_all) / xs_all
    yc_all = y - ym_all
    Xa_all = Xc_all[:, active_best]
    coef_scaled = np.linalg.solve(
        Xa_all.T @ Xa_all + ridge * np.eye(len(active_best)), Xa_all.T @ yc_all
    )
    beta = np.zeros(p, dtype=float)
    beta[np.array(active_best)] = coef_scaled / xs_all[np.array(active_best)]
    intercept = float(ym_all - xm_all @ beta)
    info = {
        "best_k": int(best_k),
        "best_val_r2": float(best[0]),
        "active": active_best,
        "history": hist,
    }
    return beta, intercept, info


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
            # Main effect exists on 0; confound coordinate 6 has no causal effect.
            self.beta[0] = beta1
            self.beta[1] = 0.0
            self.lam[0] = lambda1
        # In combined mode, confound has no causal effect but can be aliased by train design.

    def atp_map(self) -> np.ndarray:
        """Infinitesimal first-order map."""
        return self.beta.copy()

    def finite_single_map(self, epsilon: float) -> np.ndarray:
        n = self.n
        out = np.zeros(n, dtype=float)
        for i in range(n):
            a = np.zeros(n, dtype=float)
            a[i] = 1.0
            out[i] = self.measure(a[None, :], epsilon=epsilon, noise_std=0.0)[0]
        return out

    def true_linear_finite_map(self, epsilon: float) -> np.ndarray:
        """Main-effect finite map, excluding interactions."""
        out = np.zeros(self.n, dtype=float)
        for i, b in enumerate(self.beta):
            if abs(b) > 0:
                lam = self.lam[i]
                out[i] = b * math.tanh(lam * epsilon) / (lam * epsilon)
        return out

    def measure(
        self,
        A: np.ndarray,
        epsilon: float,
        noise_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Normalized finite effect y_tilde for rows of A."""
        A = np.asarray(A, dtype=float)
        eps = float(epsilon)
        y = np.zeros(A.shape[0], dtype=float)
        # Heterogeneous saturating main effects. f_i(t)=beta_i*tanh(lambda_i*t)/lambda_i.
        # y_tilde = f_i(eps*a_i)/eps.
        for i, b in enumerate(self.beta):
            if abs(b) > 0:
                lam = self.lam[i]
                y += b * np.tanh(lam * eps * A[:, i]) / (lam * eps)
        # Pure/redundant interactions: f_ij(t_i,t_j)=gamma*t_i*t_j, normalized by eps.
        # y_tilde = gamma * eps * a_i*a_j.
        for (i, j), g in self.pairs.items():
            y += g * eps * A[:, i] * A[:, j]
        if noise_std > 0:
            if rng is None:
                rng = np.random.default_rng(0)
            y += rng.normal(0.0, float(noise_std), size=A.shape[0])
        return y

    def ground_truth_support(self) -> Dict[str, object]:
        return {
            "main_indices": [i for i, b in enumerate(self.beta) if abs(b) > 0],
            "interaction_pairs": list(self.pairs.keys()),
            "confound_idx": self.confound_idx,
        }


def evaluate_method(name: str, y_test: np.ndarray, pred: np.ndarray) -> Dict[str, float | str]:
    return {
        "method": name,
        "heldout_r2": r2_score(y_test, pred),
        "mae": float(np.mean(np.abs(y_test - pred))),
        "rmse": float(np.sqrt(np.mean((y_test - pred) ** 2))),
    }


def topk_overlap(scores: np.ndarray, support: Sequence[int], k: int | None = None) -> float:
    support = list(support)
    if not support:
        return float("nan")
    if k is None:
        k = len(support)
    order = np.argsort(-np.abs(scores))[:k]
    return float(len(set(order).intersection(support)) / len(set(support)))


def run_one(args: argparse.Namespace, seed: int, epsilon: float, outdir: Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    rng = set_seed(seed)
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
    conf_pair = (0, plant.confound_idx) if args.confound_correlation > 0 else None
    A_train = make_masks(
        rng,
        args.measurements,
        n,
        args.mask_density,
        kind=args.mask_kind,
        normalize_rows=not args.raw_masks,
        confound_pair=conf_pair,
        confound_prob=args.confound_correlation,
    )
    A_test = make_masks(
        rng,
        args.holdout_measurements,
        n,
        args.mask_density,
        kind=args.mask_kind,
        normalize_rows=not args.raw_masks,
        confound_pair=None,
        confound_prob=0.0,
    )
    y_train = plant.measure(A_train, epsilon=epsilon, noise_std=args.noise_std, rng=rng)
    y_test = plant.measure(A_test, epsilon=epsilon, noise_std=args.noise_std, rng=rng)

    atp = plant.atp_map()
    finite_single = plant.finite_single_map(epsilon=epsilon)

    rows: List[Dict[str, object]] = []
    coef_rows: List[Dict[str, object]] = []

    def add_coefficients(method: str, main_coef: np.ndarray, pair_coef: Dict[Tuple[int, int], float] | None = None):
        gt = plant.ground_truth_support()
        for i, val in enumerate(main_coef):
            if abs(val) > args.coeff_report_threshold or i in gt["main_indices"] or i == gt["confound_idx"]:
                coef_rows.append({
                    "seed": seed,
                    "epsilon": epsilon,
                    "method": method,
                    "term_type": "main",
                    "i": i,
                    "j": -1,
                    "coef": float(val),
                    "is_true_main": int(i in gt["main_indices"]),
                    "is_confound": int(i == gt["confound_idx"]),
                    "is_true_pair": 0,
                })
        if pair_coef:
            true_pairs = set(tuple(p) for p in gt["interaction_pairs"])
            for (i, j), val in pair_coef.items():
                if abs(val) > args.coeff_report_threshold or (i, j) in true_pairs:
                    coef_rows.append({
                        "seed": seed,
                        "epsilon": epsilon,
                        "method": method,
                        "term_type": "pair",
                        "i": i,
                        "j": j,
                        "coef": float(val),
                        "is_true_main": 0,
                        "is_confound": 0,
                        "is_true_pair": int((i, j) in true_pairs),
                    })

    # Method 1: raw AtP.
    pred = A_test @ atp
    rec = evaluate_method("raw_atp", y_test, pred)
    rec.update({"seed": seed, "epsilon": epsilon, "budget_forward": 0, "budget_backward": 1})
    rows.append(rec)
    add_coefficients("raw_atp", atp)

    # Method 2: scalar calibrated AtP.
    train_feat = A_train @ atp
    gain = fit_scalar_gain(train_feat, y_train)
    pred = gain * (A_test @ atp)
    rec = evaluate_method("scalar_cal_atp", y_test, pred)
    rec.update({"seed": seed, "epsilon": epsilon, "gain": gain, "budget_forward": args.measurements, "budget_backward": 1})
    rows.append(rec)
    add_coefficients("scalar_cal_atp", gain * atp)

    # Method 3: multi-gain AtP for known causal main-effect coordinates.
    # This intentionally gives AtP a strong baseline: if the only misspecification is heterogeneous
    # saturation of known coordinates, two gains should fix it. It still cannot see interactions.
    group_cols = []
    group_names = []
    for idx in plant.main_indices:
        if abs(atp[idx]) > 0:
            group_cols.append(A_train[:, idx] * atp[idx])
            group_names.append(idx)
    if group_cols:
        Xg = np.vstack(group_cols).T
        beta_g, intercept_g = fit_ridge(Xg, y_train, ridge=args.ridge)
        Xg_test = np.vstack([A_test[:, idx] * atp[idx] for idx in group_names]).T
        pred = predict_linear(Xg_test, beta_g, intercept_g)
        main_coef = np.zeros(n, dtype=float)
        for idx, g in zip(group_names, beta_g):
            main_coef[idx] = atp[idx] * g
        rec = evaluate_method("multigain_atp", y_test, pred)
        rec.update({"seed": seed, "epsilon": epsilon, "budget_forward": args.measurements, "budget_backward": 1})
        rows.append(rec)
        add_coefficients("multigain_atp", main_coef)

    # Method 4: finite singleton patching, additive prediction.
    pred = A_test @ finite_single
    rec = evaluate_method("finite_single", y_test, pred)
    rec.update({"seed": seed, "epsilon": epsilon, "budget_forward": n, "budget_backward": 0})
    rows.append(rec)
    add_coefficients("finite_single", finite_single)

    # Method 5: random subset regression / ridge on main effects.
    beta_ridge, intercept_ridge = fit_ridge(A_train, y_train, ridge=args.ridge)
    pred = predict_linear(A_test, beta_ridge, intercept_ridge)
    rec = evaluate_method("subset_ridge", y_test, pred)
    rec.update({"seed": seed, "epsilon": epsilon, "budget_forward": args.measurements, "budget_backward": 0})
    rows.append(rec)
    add_coefficients("subset_ridge", beta_ridge)

    # Method 6: sparse first-order tomography via OMP.
    beta_omp, intercept_omp, info_omp = fit_omp(A_train, y_train, max_k=args.omp_max_k, val_frac=args.val_frac, rng=rng)
    pred = predict_linear(A_test, beta_omp, intercept_omp)
    rec = evaluate_method("first_order_omp", y_test, pred)
    rec.update({
        "seed": seed,
        "epsilon": epsilon,
        "omp_k": info_omp["best_k"],
        "omp_val_r2": info_omp["best_val_r2"],
        "budget_forward": args.measurements,
        "budget_backward": 0,
    })
    rows.append(rec)
    add_coefficients("first_order_omp", beta_omp)

    # Method 7: lifted sparse tomography over main + pair terms.
    all_pairs = pair_indices(n)
    if args.max_lifted_pairs and args.max_lifted_pairs < len(all_pairs):
        # Always include planted pairs; fill rest randomly.
        planted = set(plant.ground_truth_support()["interaction_pairs"])
        remaining = [p for p in all_pairs if p not in planted]
        rng.shuffle(remaining)
        selected_pairs = list(planted) + remaining[: max(0, args.max_lifted_pairs - len(planted))]
        selected_pairs = sorted(selected_pairs)
    else:
        selected_pairs = all_pairs
    X_train_lift = np.hstack([A_train, pair_features(A_train, selected_pairs)])
    X_test_lift = np.hstack([A_test, pair_features(A_test, selected_pairs)])
    beta_lift, intercept_lift, info_lift = fit_omp(
        X_train_lift, y_train, max_k=args.lifted_omp_max_k, val_frac=args.val_frac, rng=rng
    )
    pred = predict_linear(X_test_lift, beta_lift, intercept_lift)
    rec = evaluate_method("lifted_omp", y_test, pred)
    rec.update({
        "seed": seed,
        "epsilon": epsilon,
        "omp_k": info_lift["best_k"],
        "omp_val_r2": info_lift["best_val_r2"],
        "budget_forward": args.measurements,
        "budget_backward": 0,
    })
    rows.append(rec)
    main_lift = beta_lift[:n]
    pair_coef = {p: beta_lift[n + k] for k, p in enumerate(selected_pairs)}
    add_coefficients("lifted_omp", main_lift, pair_coef)

    # Support/reach diagnostics added to rows.
    gt = plant.ground_truth_support()
    true_main = gt["main_indices"]
    true_pairs = set(tuple(p) for p in gt["interaction_pairs"])
    confound_idx = int(gt["confound_idx"])
    df = pd.DataFrame(rows)
    # Add support metrics by method from coefficient rows.
    coef_df = pd.DataFrame(coef_rows)
    support_metrics = []
    for method in df["method"].unique():
        main_coef = np.zeros(n, dtype=float)
        pair_scores: Dict[Tuple[int, int], float] = {}
        sub = coef_df[(coef_df.method == method) & (coef_df.seed == seed) & (coef_df.epsilon == epsilon)]
        for _, r in sub.iterrows():
            if r.term_type == "main":
                main_coef[int(r.i)] = float(r.coef)
            elif r.term_type == "pair":
                pair_scores[(int(r.i), int(r.j))] = float(r.coef)
        true_pair_score = 0.0
        pair_recall = float("nan")
        if true_pairs:
            sorted_pairs = sorted(pair_scores.items(), key=lambda kv: -abs(kv[1]))
            top_pairs = {p for p, _ in sorted_pairs[: len(true_pairs)]}
            pair_recall = len(top_pairs.intersection(true_pairs)) / len(true_pairs)
            true_pair_score = float(np.mean([abs(pair_scores.get(p, 0.0)) for p in true_pairs]))
        support_metrics.append({
            "method": method,
            "main_topk_overlap": topk_overlap(main_coef, true_main),
            "confound_abs_coef": float(abs(main_coef[confound_idx])) if confound_idx < n else float("nan"),
            "pair_topk_recall": pair_recall,
            "true_pair_abs_coef_mean": true_pair_score,
        })
    support_df = pd.DataFrame(support_metrics)
    df = df.merge(support_df, on="method", how="left")
    meta = {
        "seed": seed,
        "epsilon": epsilon,
        "n_components": n,
        "mode": args.mode,
        "mask_kind": args.mask_kind,
        "mask_density": args.mask_density,
        "rows_normalized": not args.raw_masks,
        "measurements": args.measurements,
        "holdout_measurements": args.holdout_measurements,
        "confound_correlation_train": args.confound_correlation,
        "ground_truth": {
            "main_indices": true_main,
            "interaction_pairs": [list(p) for p in true_pairs],
            "confound_idx": confound_idx,
            "atp_map": plant.atp_map().tolist(),
            "finite_single_map": finite_single.tolist(),
            "true_linear_finite_map": plant.true_linear_finite_map(epsilon).tolist(),
        },
    }
    return df, coef_df, meta


def plot_results(results: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # R2 vs epsilon for methods.
    methods_order = [
        "raw_atp", "scalar_cal_atp", "multigain_atp", "finite_single",
        "subset_ridge", "first_order_omp", "lifted_omp",
    ]
    plt.figure(figsize=(9, 5.2))
    for method in methods_order:
        sub = results[results.method == method]
        if sub.empty:
            continue
        g = sub.groupby("epsilon")["heldout_r2"].agg(["mean", "std"]).reset_index()
        plt.errorbar(g["epsilon"], g["mean"], yerr=g["std"].fillna(0.0), marker="o", capsize=3, label=method)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("finite intervention scale epsilon")
    plt.ylabel("held-out subset/intervention R2")
    plt.title("Claim 3 planted reach: prediction quality")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outdir / "claim3_heldout_r2_by_epsilon.png", dpi=180)
    plt.close()

    # Pair recovery.
    plt.figure(figsize=(9, 5.2))
    for method in methods_order:
        sub = results[results.method == method]
        if sub.empty or sub["pair_topk_recall"].isna().all():
            continue
        g = sub.groupby("epsilon")["pair_topk_recall"].agg(["mean", "std"]).reset_index()
        plt.errorbar(g["epsilon"], g["mean"], yerr=g["std"].fillna(0.0), marker="o", capsize=3, label=method)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("finite intervention scale epsilon")
    plt.ylabel("top-k pair recovery")
    plt.title("Pure interactions/redundancy: pair recovery")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outdir / "claim3_pair_recovery.png", dpi=180)
    plt.close()

    # Confound false-positive.
    plt.figure(figsize=(9, 5.2))
    for method in methods_order:
        sub = results[results.method == method]
        if sub.empty:
            continue
        g = sub.groupby("epsilon")["confound_abs_coef"].agg(["mean", "std"]).reset_index()
        plt.errorbar(g["epsilon"], g["mean"], yerr=g["std"].fillna(0.0), marker="o", capsize=3, label=method)
    plt.xlabel("finite intervention scale epsilon")
    plt.ylabel("absolute off-path confound coefficient")
    plt.title("Confound false-positive rate")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outdir / "claim3_confound_false_positive.png", dpi=180)
    plt.close()

    # Bar plot at largest epsilon.
    eps_max = float(np.max(results["epsilon"]))
    sub = results[np.isclose(results["epsilon"], eps_max)]
    g = sub.groupby("method")["heldout_r2"].agg(["mean", "std"]).reindex(methods_order).dropna(how="all")
    plt.figure(figsize=(10, 5.0))
    x = np.arange(len(g))
    plt.bar(x, g["mean"], yerr=g["std"].fillna(0.0), capsize=4)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, g.index, rotation=35, ha="right")
    plt.ylabel("held-out R2")
    plt.title(f"Claim 3 planted reach at epsilon={eps_max:g}")
    plt.tight_layout()
    plt.savefig(outdir / "claim3_final_epsilon_bar.png", dpi=180)
    plt.close()


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["heldout_r2", "mae", "rmse", "main_topk_overlap", "pair_topk_recall", "confound_abs_coef"]
    return results.groupby(["epsilon", "method"])[metrics].agg(["mean", "std", "count"]).reset_index()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Claim 3 planted reach experiment")
    p.add_argument("--outdir", type=str, default="runs/claim3_planted_reach")
    p.add_argument("--mode", type=str, default="combined", choices=["combined", "two_coord_saturation", "pure_interaction", "redundant_pair", "confound"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-components", type=int, default=64)
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.6, 1.2, 2.0, 5.0, 8.0])
    p.add_argument("--measurements", type=int, default=256)
    p.add_argument("--holdout-measurements", type=int, default=512)
    p.add_argument("--mask-kind", type=str, default="signed", choices=["signed", "binary"])
    p.add_argument("--mask-density", type=float, default=0.30)
    p.add_argument("--raw-masks", action="store_true", help="do not normalize mask rows")
    p.add_argument("--confound-correlation", type=float, default=0.0, help="probability train confound column copies causal column; use 0.85 for bad-design confound test")
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--omp-max-k", type=int, default=12)
    p.add_argument("--lifted-omp-max-k", type=int, default=16)
    p.add_argument("--max-lifted-pairs", type=int, default=0, help="0 means all n choose 2 pairs")
    p.add_argument("--coeff-report-threshold", type=float, default=0.05)
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
        args.n_components = min(args.n_components, 32)
        args.seeds = args.seeds[:1]
        args.epsilons = [1.2, 5.0]
        args.measurements = min(args.measurements, 96)
        args.holdout_measurements = min(args.holdout_measurements, 128)
        args.max_lifted_pairs = min(args.max_lifted_pairs or 400, 400)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_coefs = []
    metas = []
    for seed in args.seeds:
        for eps in args.epsilons:
            print(f"Running seed={seed} epsilon={eps} mode={args.mode}")
            res, coefs, meta = run_one(args, seed=seed, epsilon=eps, outdir=outdir)
            all_results.append(res)
            all_coefs.append(coefs)
            metas.append(meta)
    results = pd.concat(all_results, ignore_index=True)
    coefs = pd.concat(all_coefs, ignore_index=True)
    summary = summarize(results)

    results.to_csv(outdir / "claim3_results.csv", index=False)
    coefs.to_csv(outdir / "claim3_coefficients.csv", index=False)
    summary.to_csv(outdir / "claim3_summary.csv", index=False)
    with open(outdir / "claim3_metadata.json", "w") as f:
        json.dump({
            "args": vars(args),
            "metas": metas,
            "interpretation": {
                "two_coord_saturation": "Scalar-calibrated AtP should fail when two coordinates saturate differently; multi-gain or tomography should improve.",
                "pure_interaction_redundancy": "First-order maps are blind; lifted subset measurements should recover pair terms if conditioning is adequate.",
                "confound": "A correlated training design can produce off-path false positives; independent designed masks should suppress them.",
            }
        }, f, indent=2)

    plot_results(results, outdir)
    print("\nWrote:")
    for name in ["claim3_results.csv", "claim3_coefficients.csv", "claim3_summary.csv", "claim3_metadata.json", "claim3_heldout_r2_by_epsilon.png", "claim3_pair_recovery.png", "claim3_confound_false_positive.png", "claim3_final_epsilon_bar.png"]:
        print(f"  {outdir/name}")
    print("\nHeadline by largest epsilon:")
    eps_max = max(args.epsilons)
    print(results[np.isclose(results.epsilon, eps_max)].groupby("method")[["heldout_r2", "pair_topk_recall", "confound_abs_coef"]].mean().sort_values("heldout_r2", ascending=False).to_string())


if __name__ == "__main__":
    main()
