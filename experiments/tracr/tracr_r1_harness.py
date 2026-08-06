#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
Tracr mechanistic-tomography harness, v0.

This is intentionally split into two backends:

1. --backend tracr
   First-day feasibility check against google-deepmind/tracr:
   - compile a first-order predicate program and a bigram/interaction program;
   - run a few sequences;
   - verify compiled outputs;
   - inventory residual/layer/attention artifacts exposed by Tracr.

   This does NOT yet implement residual-stream patching inside Tracr's Haiku graph.
   It tells us whether Tracr is usable for the R1/R2 experiments and what adapter
   work is needed.

2. --backend surrogate
   A fully runnable known-circuit surrogate using the same measurement API. It
   exercises the method comparison we want for Tracr: AtP / finite singleton /
   first-order OMP / lifted OMP / HVP-style second-order probes. This is a code
   path sanity check, not a Tracr result.

The next concrete adapter milestone is to implement TracrActivationAdapter.patch,
using the residual/layer outputs discovered by --backend tracr.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# ----------------------------- generic utilities -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom < 1e-12:
        return float("nan")
    return 1.0 - float(np.sum((y - yhat) ** 2)) / denom


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def make_signed_masks(m: int, n: int, density: float, *, seed: int, normalize: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = np.zeros((m, n), dtype=float)
    for i in range(m):
        k = max(1, int(round(density * n)))
        idx = rng.choice(n, size=k, replace=False)
        A[i, idx] = rng.choice([-1.0, 1.0], size=k)
    if normalize:
        norms = np.linalg.norm(A, axis=1, keepdims=True)
        A = A / np.maximum(norms, 1e-12)
    return A


def ridge_fit(A: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> Tuple[np.ndarray, float]:
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    A_mean = A.mean(axis=0, keepdims=True)
    y_mean = float(y.mean())
    Ac = A - A_mean
    yc = y - y_mean
    beta = np.linalg.solve(Ac.T @ Ac + ridge * np.eye(A.shape[1]), Ac.T @ yc)
    intercept = y_mean - float(A_mean.reshape(-1) @ beta)
    return beta, intercept


def omp_fit(A: np.ndarray, y: np.ndarray, max_k: int, val_frac: float = 0.25, seed: int = 0) -> Tuple[np.ndarray, float, int, float]:
    """Simple OMP with validation-picked support size."""
    rng = np.random.default_rng(seed)
    m, n = A.shape
    perm = rng.permutation(m)
    n_val = max(1, int(round(val_frac * m)))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    At = A[tr_idx]
    yt = y[tr_idx]
    Av = A[val_idx]
    yv = y[val_idx]

    support: List[int] = []
    residual = yt.copy()
    best = (None, -np.inf, 0, 0.0)  # beta, r2, k, intercept

    for k in range(1, max_k + 1):
        corr = At.T @ residual
        order = np.argsort(-np.abs(corr))
        next_j = None
        for j in order:
            if int(j) not in support:
                next_j = int(j)
                break
        if next_j is None:
            break
        support.append(next_j)
        beta_s, intercept = ridge_fit(At[:, support], yt, ridge=1e-8)
        pred_t = At[:, support] @ beta_s + intercept
        residual = yt - pred_t
        pred_v = Av[:, support] @ beta_s + intercept
        rv = r2_score(yv, pred_v)
        full_beta = np.zeros(n, dtype=float)
        full_beta[support] = beta_s
        if rv > best[1]:
            best = (full_beta.copy(), rv, k, intercept)
    if best[0] is None:
        return np.zeros(n), float(y.mean()), 0, float("nan")
    return best  # type: ignore


def topk_pair_recall(coef: np.ndarray, n_main: int, true_pairs: List[Tuple[int, int]], top_k: Optional[int] = None) -> float:
    if top_k is None:
        top_k = len(true_pairs)
    pair_coefs = coef[n_main:]
    if len(pair_coefs) == 0:
        return 0.0
    # pair index mapping: lexicographic i<j
    pairs: List[Tuple[int, int]] = []
    for i in range(n_main):
        for j in range(i + 1, n_main):
            pairs.append((i, j))
    top = np.argsort(-np.abs(pair_coefs))[:top_k]
    found = {pairs[int(t)] for t in top}
    truth = set(tuple(p) for p in true_pairs)
    return len(found & truth) / max(1, len(truth))


def lift_masks(A: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    n = A.shape[1]
    cols = [A]
    pairs: List[Tuple[int, int]] = []
    pair_cols = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            pair_cols.append((A[:, i] * A[:, j])[:, None])
    if pair_cols:
        cols.append(np.hstack(pair_cols))
    return np.hstack(cols), pairs


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=str)


# ----------------------------- Tracr feasibility ------------------------------

def try_import_tracr():
    try:
        from tracr.compiler import compiling  # type: ignore
        from tracr.compiler import lib  # type: ignore
        from tracr.rasp import rasp  # type: ignore
        return compiling, lib, rasp, None
    except Exception as e:
        return None, None, None, e


def _expected_first_order(seq):
    # Tracr decodes the BOS position as "BOS"; leave it as BOS to avoid a false mismatch.
    return ["BOS" if i == 0 and t == "BOS" else (1.0 if t == "A" else 0.0) for i, t in enumerate(seq)]


def _expected_bigram_numeric(seq):
    return ["BOS" if i == 0 and t == "BOS" else (1.0 if i > 0 and seq[i-1] == "A" and t == "B" else 0.0) for i, t in enumerate(seq)]


def _expected_bigram_categorical(seq):
    return ["BOS" if i == 0 and t == "BOS" else ("AB" if i > 0 and seq[i-1] == "A" and t == "B" else "OTHER") for i, t in enumerate(seq)]


def _expected_prev_token(seq):
    return ["BOS" if i == 0 and t == "BOS" else seq[i-1] for i, t in enumerate(seq)]


def build_rasp_program(name: str, lib: Any, rasp: Any):
    """Return (program, vocab, max_seq_len, examples, expected_fn).

    v1 deliberately includes several Tracr-compatible candidate constructions for
    the interaction program. Tracr rejects many non-linear numerical SequenceMaps;
    these candidates keep the interaction categorical as long as possible.
    """
    bos = "BOS"
    vocab = {"A", "B", "C"}
    max_seq_len = 6
    examples = [[bos, "A", "B", "C"], [bos, "B", "A", "B", "A"], [bos, "A", "C", "B"]]

    if name == "first_order_is_A":
        program = rasp.numerical(rasp.tokens == "A").named("is_A")
        examples_fo = [[bos, "A", "B", "A"], [bos, "B", "C", "A", "A"]]
        return program, vocab, max_seq_len, examples_fo, _expected_first_order

    if name == "prev_token":
        # Categorical attention/aggregate sanity check: output the previous token.
        program = lib.shift_by(1, rasp.tokens).named("prev_token")
        return program, vocab, max_seq_len, examples, _expected_prev_token

    if name == "paircat_AB":
        # Keep the pair detector categorical. This should compile if Tracr sees
        # both inputs/outputs as categorical lookup-table variables.
        prev = lib.shift_by(1, rasp.tokens).named("prev_token")
        paircat = rasp.SequenceMap(
            lambda p, t: "AB" if p == "A" and t == "B" else "OTHER",
            prev,
            rasp.tokens,
        ).named("paircat_AB")
        return paircat, vocab, max_seq_len, examples, _expected_bigram_categorical

    if name == "paircat_AB_to_numeric":
        # Same as paircat_AB but convert the categorical pair label to numeric.
        prev = lib.shift_by(1, rasp.tokens).named("prev_token")
        paircat = rasp.SequenceMap(
            lambda p, t: "AB" if p == "A" and t == "B" else "OTHER",
            prev,
            rasp.tokens,
        ).named("paircat_AB")
        program = rasp.numerical(paircat == "AB").named("detect_AB_num")
        return program, vocab, max_seq_len, examples, _expected_bigram_numeric

    if name == "direct_bigram_AB":
        # Direct categorical inputs -> boolean output SequenceMap. This may or may
        # not compile depending on Tracr's internal type inference, so it is useful
        # as a probe.
        prev = lib.shift_by(1, rasp.tokens).named("prev_token")
        program = rasp.numerical(rasp.SequenceMap(
            lambda p, t: (p == "A" and t == "B"),
            prev,
            rasp.tokens,
        ).named("direct_bigram_AB"))
        return program, vocab, max_seq_len, examples, _expected_bigram_numeric

    if name == "detect_pattern_AB_raw":
        # Tracr's built-in detect_pattern returns a boolean SOp. In some installs,
        # compiling it directly gives better type diagnostics than wrapping it in
        # rasp.numerical.
        program = lib.detect_pattern(rasp.tokens, ["A", "B"]).named("detect_pattern_AB_raw")
        return program, vocab, max_seq_len, examples, _expected_bigram_numeric

    if name == "detect_pattern_AB_numeric":
        program = rasp.numerical(lib.detect_pattern(rasp.tokens, ["A", "B"])).named("detect_pattern_AB_numeric")
        return program, vocab, max_seq_len, examples, _expected_bigram_numeric

    if name == "shuffle_dyck2":
        # Retained as an optional diagnostic. It often fails on newer installs
        # because Tracr's compiler support for nonlinear numeric SequenceMaps is narrow.
        program = rasp.numerical(lib.make_shuffle_dyck2()).named("shuffle_dyck2_num")
        vocab2 = {"(", ")", "{", "}", "x"}
        max_seq_len2 = 8
        examples2 = [[bos, "(", ")", "{", "}"], [bos, "(", "{", ")", "}"]]
        def expected(seq):
            return None
        return program, vocab2, max_seq_len2, examples2, expected

    raise ValueError(f"unknown program {name}")

def run_tracr_feasibility(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    compiling, lib, rasp, err = try_import_tracr()
    report: Dict[str, Any] = {
        "backend": "tracr",
        "status": "not_run",
        "programs": [],
        "install_hint": "git clone https://github.com/google-deepmind/tracr && cd tracr && pip install .",
    }
    if err is not None:
        report["status"] = "missing_tracr"
        report["error"] = repr(err)
        write_json(outdir / "tracr_feasibility_report.json", report)
        print("Tracr is not importable. Wrote report with install hints:", outdir / "tracr_feasibility_report.json")
        return

    for name in args.programs:
        item: Dict[str, Any] = {"name": name}
        try:
            program, vocab, max_seq_len, examples, expected = build_rasp_program(name, lib, rasp)
            model = compiling.compile_rasp_to_model(
                program,
                vocab=vocab,
                max_seq_len=max_seq_len,
                compiler_bos="BOS",
            )
            ex_reports = []
            for seq in examples:
                out = model.apply(seq)
                decoded = getattr(out, "decoded", None)
                residuals = getattr(out, "residuals", None)
                layer_outputs = getattr(out, "layer_outputs", None)
                attn_logits = getattr(out, "attn_logits", None)
                ex_reports.append({
                    "input": seq,
                    "decoded": decoded,
                    "expected": expected(seq) if expected is not None else None,
                    "residuals_type": type(residuals).__name__,
                    "layer_outputs_type": type(layer_outputs).__name__,
                    "attn_logits_type": type(attn_logits).__name__,
                    "residuals_keys_or_len": list(residuals.keys())[:10] if hasattr(residuals, "keys") else (len(residuals) if residuals is not None and hasattr(residuals, "__len__") else None),
                    "layer_outputs_keys_or_len": list(layer_outputs.keys())[:10] if hasattr(layer_outputs, "keys") else (len(layer_outputs) if layer_outputs is not None and hasattr(layer_outputs, "__len__") else None),
                })
            item.update({
                "status": "ok",
                "vocab": sorted(list(vocab)),
                "max_seq_len": max_seq_len,
                "examples": ex_reports,
            })
        except Exception as e:
            item.update({"status": "failed", "error": repr(e)})
        report["programs"].append(item)
    report["status"] = "ok"
    write_json(outdir / "tracr_feasibility_report.json", report)
    print("Wrote", outdir / "tracr_feasibility_report.json")


# ----------------------- known-circuit surrogate experiment -------------------

@dataclass
class SurrogatePlant:
    n: int = 64
    beta0: float = 1.4
    beta1: float = -1.1
    gamma_interaction: float = 1.4
    gamma_redundant: float = -1.2
    lambda0: float = 0.18
    lambda1: float = 0.75
    noise_std: float = 0.01
    main_indices: Tuple[int, int] = (0, 1)
    true_pairs: Tuple[Tuple[int, int], Tuple[int, int]] = ((2, 3), (4, 5))

    def atp_main(self) -> np.ndarray:
        x = np.zeros(self.n)
        x[0] = self.beta0
        x[1] = self.beta1
        return x

    def finite_main(self, eps: float) -> np.ndarray:
        x = np.zeros(self.n)
        # saturating singleton effects, matching the Claim-3 plant idea
        x[0] = self.beta0 * (1.0 - self.lambda0 * np.log1p(eps) / 2.0)
        x[1] = self.beta1 * (1.0 - self.lambda1 * np.log1p(eps) / 2.0)
        return x

    def response(self, A: np.ndarray, eps: float, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = self.finite_main(eps)
        y = A @ x
        # pair terms: normalized masks already have small coefficients; use eps scale
        i, j = self.true_pairs[0]
        y = y + self.gamma_interaction * eps * A[:, i] * A[:, j]
        i, j = self.true_pairs[1]
        y = y + self.gamma_redundant * eps * A[:, i] * A[:, j]
        if self.noise_std > 0:
            y = y + rng.normal(0, self.noise_std, size=len(y))
        return y


def run_surrogate(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    plant = SurrogatePlant(n=args.n_components, noise_std=args.noise_std)
    for seed in args.seeds:
        for eps in args.epsilons:
            A_train = make_signed_masks(args.measurements, plant.n, args.mask_density, seed=seed + 1000, normalize=True)
            A_hold = make_signed_masks(args.holdout_measurements, plant.n, args.mask_density, seed=seed + 2000, normalize=True)
            y_train = plant.response(A_train, eps, seed=seed + 3000)
            y_hold = plant.response(A_hold, eps, seed=seed + 4000)

            # Baselines: raw AtP, scalar-calibrated AtP, finite singleton, first-order OMP
            atp = plant.atp_main()
            finite = plant.finite_main(eps)
            gain_num = float((A_train @ atp) @ y_train)
            gain_den = float((A_train @ atp) @ (A_train @ atp) + 1e-12)
            gain = gain_num / gain_den
            cal_atp = gain * atp
            fo_coef, fo_intercept, fo_k, fo_val = omp_fit(A_train, y_train, max_k=args.omp_max_k, seed=seed)

            # Lifted OMP
            Phi_train, pairs = lift_masks(A_train)
            Phi_hold, _ = lift_masks(A_hold)
            lifted_coef, lifted_intercept, lifted_k, lifted_val = omp_fit(Phi_train, y_train, max_k=args.lifted_omp_max_k, seed=seed)

            methods = {
                "raw_atp": (atp, 0.0, A_hold),
                "scalar_cal_atp": (cal_atp, 0.0, A_hold),
                "finite_single": (finite, 0.0, A_hold),
                "first_order_omp": (fo_coef, fo_intercept, A_hold),
                "lifted_omp": (lifted_coef, lifted_intercept, Phi_hold),
            }
            for method, (coef, intercept, design_hold) in methods.items():
                pred = design_hold @ coef + intercept
                pair_recall = 0.0
                if method == "lifted_omp":
                    pair_recall = topk_pair_recall(coef, plant.n, list(plant.true_pairs))
                rows.append({
                    "backend": "surrogate",
                    "seed": seed,
                    "epsilon": eps,
                    "method": method,
                    "heldout_r2": r2_score(y_hold, pred),
                    "pair_topk_recall": pair_recall,
                    "selected_k": int(lifted_k if method == "lifted_omp" else (fo_k if method == "first_order_omp" else 0)),
                    "gain": gain if method == "scalar_cal_atp" else np.nan,
                })
    if pd is not None:
        df = pd.DataFrame(rows)
        df.to_csv(outdir / "surrogate_results.csv", index=False)
        summ = df.groupby(["epsilon", "method"]).agg(
            heldout_r2_mean=("heldout_r2", "mean"),
            heldout_r2_std=("heldout_r2", "std"),
            pair_recall_mean=("pair_topk_recall", "mean"),
        ).reset_index()
        summ.to_csv(outdir / "surrogate_summary.csv", index=False)
        if plt is not None:
            for metric, fname, ylabel in [
                ("heldout_r2", "surrogate_r2.png", "held-out R2"),
                ("pair_topk_recall", "surrogate_pair_recall.png", "pair recall"),
            ]:
                plt.figure(figsize=(8,5))
                for method in sorted(df.method.unique()):
                    sub = df[df.method == method].groupby("epsilon")[metric].agg(["mean", "std"]).reset_index()
                    plt.errorbar(sub["epsilon"], sub["mean"], yerr=sub["std"].fillna(0), marker="o", label=method)
                plt.axhline(0.95 if metric == "heldout_r2" else 0.99, ls="--")
                plt.xlabel("epsilon")
                plt.ylabel(ylabel)
                plt.title(f"Surrogate {metric}")
                plt.legend(fontsize=8)
                plt.tight_layout()
                plt.savefig(outdir / fname, dpi=160)
                plt.close()
    else:
        write_json(outdir / "surrogate_results.json", rows)
    write_json(outdir / "surrogate_metadata.json", {"plant": asdict(plant), "args": vars(args)})
    print("Wrote surrogate results to", outdir)


# ----------------------------- CLI -------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tracr/MT R1 harness v0")
    p.add_argument("--backend", choices=["tracr", "surrogate"], default="surrogate")
    p.add_argument("--outdir", default="runs/tracr_mt_harness")
    p.add_argument("--programs", nargs="+", default=["first_order_is_A", "prev_token", "paircat_AB", "paircat_AB_to_numeric", "direct_bigram_AB", "detect_pattern_AB_raw", "detect_pattern_AB_numeric"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    p.add_argument("--epsilons", nargs="+", type=float, default=[1.0, 2.0, 5.0, 8.0])
    p.add_argument("--n-components", type=int, default=64)
    p.add_argument("--measurements", type=int, default=128)
    p.add_argument("--holdout-measurements", type=int, default=512)
    p.add_argument("--mask-density", type=float, default=0.30)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--omp-max-k", type=int, default=12)
    p.add_argument("--lifted-omp-max-k", type=int, default=16)
    args = p.parse_args()
    return args


def main() -> None:
    args = parse_args()
    if args.backend == "tracr":
        run_tracr_feasibility(args)
    else:
        run_surrogate(args)

if __name__ == "__main__":
    main()
