#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
Tracr R1/R2 label-basis analysis.

Goal: a fast ground-truth compiled-circuit bridge for the mechanistic-tomography
paper. This is intentionally NOT yet a full causal residual-patching adapter.
It asks whether Tracr's native residual basis linearizes an interaction, and
whether a restricted pre-interaction basis needs lifted pair features.

Primary targets:
  R2 null: first_order_is_A
    - full/native and token-only bases should be linearly predictive.
    - lifted features should not materially improve.

  R1 reach: paircat_AB_to_numeric
    - full/native basis likely contains an explicit AB/detect feature, so linear works.
    - restricted pre-interaction basis keeps only current-token and prev-token labels;
      first-order linear should struggle on AB, lifted pair features should recover
      prev_token:A x tokens:B.

Outputs:
  tracr_label_basis_results.csv
  tracr_label_basis_summary.json
  tracr_label_basis_top_terms.csv
  optional plots if matplotlib is available.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def try_import_tracr():
    try:
        from tracr.compiler import compiling  # type: ignore
        from tracr.compiler import lib  # type: ignore
        from tracr.rasp import rasp  # type: ignore
        return compiling, lib, rasp, None
    except Exception as e:
        return None, None, None, e


def _expected_first_order(seq: Sequence[str]) -> List[Any]:
    return ["BOS" if i == 0 else (1.0 if t == "A" else 0.0) for i, t in enumerate(seq)]


def _expected_bigram_numeric(seq: Sequence[str]) -> List[Any]:
    out: List[Any] = []
    for i, t in enumerate(seq):
        if i == 0:
            out.append("BOS")
        else:
            out.append(1.0 if (i > 0 and seq[i - 1] == "A" and t == "B") else 0.0)
    return out


def build_rasp_program(name: str, lib: Any, rasp: Any):
    bos = "BOS"
    vocab = {"A", "B", "C"}
    max_seq_len = 6
    if name == "first_order_is_A":
        program = rasp.numerical(rasp.tokens == "A").named("is_A")
        return program, vocab, max_seq_len, _expected_first_order
    if name == "paircat_AB_to_numeric":
        prev = lib.shift_by(1, rasp.tokens).named("prev_token")
        paircat = rasp.SequenceMap(
            lambda p, t: "AB" if p == "A" and t == "B" else "OTHER",
            prev,
            rasp.tokens,
        ).named("paircat_AB")
        program = rasp.numerical(paircat == "AB").named("detect_AB_num")
        return program, vocab, max_seq_len, _expected_bigram_numeric
    if name == "detect_pattern_AB_raw":
        program = lib.detect_pattern(rasp.tokens, ["A", "B"]).named("detect_pattern_AB_raw")
        return program, vocab, max_seq_len, _expected_bigram_numeric
    raise ValueError(f"unknown program {name}")


def to_float_target(v: Any) -> Optional[float]:
    if v == "BOS":
        return None
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    return float(v)


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom < 1e-12:
        return float("nan")
    return 1.0 - float(np.sum((y - yhat) ** 2)) / denom


def ridge_fit(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> Tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    xm = X.mean(axis=0, keepdims=True)
    ym = float(y.mean())
    Xc = X - xm
    yc = y - ym
    beta = np.linalg.solve(Xc.T @ Xc + ridge * np.eye(X.shape[1]), Xc.T @ yc)
    intercept = ym - float(xm.reshape(-1) @ beta)
    return beta, intercept


def omp_fit(X: np.ndarray, y: np.ndarray, max_k: int, ridge: float = 1e-8) -> Tuple[np.ndarray, float, List[int]]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    residual = y - y.mean()
    support: List[int] = []
    beta_full = np.zeros(X.shape[1], dtype=float)
    intercept = float(y.mean())
    for _ in range(max_k):
        corr = X.T @ residual
        order = np.argsort(-np.abs(corr))
        nxt = None
        for j in order:
            jj = int(j)
            if jj not in support and np.std(X[:, jj]) > 1e-12:
                nxt = jj
                break
        if nxt is None:
            break
        support.append(nxt)
        beta_s, intercept = ridge_fit(X[:, support], y, ridge=ridge)
        pred = X[:, support] @ beta_s + intercept
        residual = y - pred
    beta_full[:] = 0.0
    if support:
        beta_s, intercept = ridge_fit(X[:, support], y, ridge=ridge)
        beta_full[support] = beta_s
    return beta_full, intercept, support


def train_test_split(n: int, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(test_frac * n)))
    return idx[n_test:], idx[:n_test]


def make_sequences(vocab: Sequence[str], max_seq_len: int, exact_len: int, max_examples: Optional[int], seed: int) -> List[List[str]]:
    toks = [t for t in vocab if t != "BOS"]
    L = min(exact_len, max_seq_len)
    inner_len = L - 1
    all_inner = list(itertools.product(toks, repeat=inner_len))
    seqs = [["BOS", *x] for x in all_inner]
    if max_examples is not None and len(seqs) > max_examples:
        rng = np.random.default_rng(seed)
        take = rng.choice(len(seqs), size=max_examples, replace=False)
        seqs = [seqs[int(i)] for i in take]
    return seqs


def collect_dataset(model: Any, seqs: List[List[str]], expected_fn: Callable[[Sequence[str]], List[Any]], stage: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
    labels = list(getattr(model, "residual_labels", []))
    X_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    meta: List[Dict[str, Any]] = []
    for si, seq in enumerate(seqs):
        out = model.apply(seq)
        if stage == "final_residual":
            arr = np.asarray(out.residuals[-1][0], dtype=float)  # [T, d]
        elif stage == "pre_final_residual":
            arr = np.asarray(out.residuals[-2][0], dtype=float)
        else:
            raise ValueError(f"unknown stage {stage}")
        expected = expected_fn(seq)
        for pos, exp in enumerate(expected):
            target = to_float_target(exp)
            if target is None:
                continue
            X_rows.append(arr[pos].copy())
            y_rows.append(float(target))
            meta.append({"seq_idx": si, "pos": pos, "token": seq[pos], "prev_token": seq[pos-1] if pos > 0 else None, "seq": " ".join(seq)})
    return np.vstack(X_rows), np.asarray(y_rows, dtype=float), labels, meta


def basis_indices(labels: List[str], program: str, basis: str) -> List[int]:
    keep: List[int] = []
    for i, lab in enumerate(labels):
        l = str(lab)
        if "compiler_pad" in l:
            continue
        if basis == "full_native":
            keep.append(i)
        elif basis == "tokens_only":
            if l.startswith("tokens:") and l not in {"tokens:BOS", "tokens:compiler_pad"}:
                keep.append(i)
        elif basis == "restricted_pre_interaction":
            # Keep current-token and previous-token/shifted-token features, exclude explicit AB/detect/map outputs and indices.
            if l.startswith("indices:") or l == "one":
                continue
            banned = ["detect", "paircat", "map_", "compiler_pad"]
            if any(b in l for b in banned):
                continue
            if l.startswith("tokens:") and l not in {"tokens:BOS", "tokens:compiler_pad"}:
                keep.append(i)
            elif "prev_token" in l or "shift_by" in l:
                keep.append(i)
        else:
            raise ValueError(f"unknown basis {basis}")
    # de-duplicate while preserving order
    seen = set(); out = []
    for i in keep:
        if i not in seen:
            out.append(i); seen.add(i)
    return out


def lift_features(X: np.ndarray, names: List[str], max_pairs: int = 0) -> Tuple[np.ndarray, List[str]]:
    cols = [X]
    new_names = list(names)
    pair_cols = []
    pair_names = []
    n = X.shape[1]
    # Only make pairs between non-constant columns.
    valid = [i for i in range(n) if np.std(X[:, i]) > 1e-12]
    pairs = []
    for a, i in enumerate(valid):
        for j in valid[a+1:]:
            pairs.append((i, j))
    if max_pairs and len(pairs) > max_pairs:
        # Keep all pairs among sparse Tracr bases by default; max_pairs is a safety valve.
        pairs = pairs[:max_pairs]
    for i, j in pairs:
        pair_cols.append((X[:, i] * X[:, j])[:, None])
        pair_names.append(f"{names[i]} × {names[j]}")
    if pair_cols:
        cols.append(np.hstack(pair_cols))
        new_names.extend(pair_names)
    return np.hstack(cols), new_names


def top_terms(beta: np.ndarray, names: List[str], k: int = 12) -> List[Dict[str, Any]]:
    order = np.argsort(-np.abs(beta))[:k]
    return [{"rank": r+1, "term": names[int(i)], "coef": float(beta[int(i)]), "abs_coef": float(abs(beta[int(i)]))} for r, i in enumerate(order) if abs(beta[int(i)]) > 1e-10]


def pair_recall(selected_names: Sequence[str], program: str) -> float:
    # Ground-truth pair for paircat is prev_token:A AND current token:B.
    if program == "paircat_AB_to_numeric":
        targets = [("prev_token", ":A", "tokens:B")]
    elif program == "detect_pattern_AB_raw":
        # Built-in detect_pattern uses shifted/boolean features; this is less exact, so use a loose AB criterion.
        targets = [("shift_by", "True", "tokens:B")]
    else:
        return float("nan")
    for name in selected_names:
        for req1, req2, req3 in targets:
            s = str(name)
            if "×" in s and req1 in s and req2 in s and req3 in s:
                return 1.0
    return 0.0


def analyze_program(program_name: str, args: argparse.Namespace, compiling: Any, lib: Any, rasp: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    program, vocab, max_seq_len, expected_fn = build_rasp_program(program_name, lib, rasp)
    model = compiling.compile_rasp_to_model(program, vocab=vocab, max_seq_len=max_seq_len, compiler_bos="BOS")
    seqs = make_sequences(sorted(vocab), max_seq_len, args.seq_len, args.max_sequences, args.seed)
    X_full, y, labels, meta = collect_dataset(model, seqs, expected_fn, args.stage)
    tr, te = train_test_split(len(y), args.test_frac, args.seed)

    summary = {
        "program": program_name,
        "n_sequences": len(seqs),
        "n_examples": int(len(y)),
        "stage": args.stage,
        "n_residual_labels": len(labels),
        "residual_labels": labels,
        "positive_rate": float(y.mean()),
    }
    rows: List[Dict[str, Any]] = []
    term_rows: List[Dict[str, Any]] = []
    bases = ["full_native", "tokens_only"]
    if program_name != "first_order_is_A":
        bases.append("restricted_pre_interaction")

    for basis in bases:
        idx = basis_indices(labels, program_name, basis)
        if not idx:
            continue
        X = X_full[:, idx]
        names = [labels[i] for i in idx]
        for method in ["ridge", "omp"]:
            if method == "ridge":
                beta, intercept = ridge_fit(X[tr], y[tr], ridge=args.ridge)
                selected = list(np.argsort(-np.abs(beta))[:min(args.top_k_terms, len(beta))])
            else:
                beta, intercept, selected = omp_fit(X[tr], y[tr], max_k=min(args.omp_max_k, X.shape[1]), ridge=args.ridge)
            pred = X[te] @ beta + intercept
            pred_tr = X[tr] @ beta + intercept
            sel_names = [names[i] for i in selected]
            row = {
                "program": program_name,
                "basis": basis,
                "method": method,
                "feature_family": "first_order",
                "n_features": int(X.shape[1]),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "train_r2": r2_score(y[tr], pred_tr),
                "test_r2": r2_score(y[te], pred),
                "selected_k": int(len(selected)),
                "selected_terms": ";".join(sel_names),
                "pair_recall": pair_recall(sel_names, program_name),
            }
            rows.append(row)
            for t in top_terms(beta, names, args.top_k_terms):
                t.update({"program": program_name, "basis": basis, "method": method, "feature_family": "first_order"})
                term_rows.append(t)

        if basis == "restricted_pre_interaction":
            XL, namesL = lift_features(X, names, max_pairs=args.max_lifted_pairs)
            for method in ["ridge", "omp"]:
                if method == "ridge":
                    beta, intercept = ridge_fit(XL[tr], y[tr], ridge=args.ridge)
                    selected = list(np.argsort(-np.abs(beta))[:min(args.top_k_terms, len(beta))])
                else:
                    beta, intercept, selected = omp_fit(XL[tr], y[tr], max_k=min(args.lifted_omp_max_k, XL.shape[1]), ridge=args.ridge)
                pred = XL[te] @ beta + intercept
                pred_tr = XL[tr] @ beta + intercept
                sel_names = [namesL[i] for i in selected]
                row = {
                    "program": program_name,
                    "basis": basis,
                    "method": method,
                    "feature_family": "lifted_pairs",
                    "n_features": int(XL.shape[1]),
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "train_r2": r2_score(y[tr], pred_tr),
                    "test_r2": r2_score(y[te], pred),
                    "selected_k": int(len(selected)),
                    "selected_terms": ";".join(sel_names),
                    "pair_recall": pair_recall(sel_names, program_name),
                }
                rows.append(row)
                for t in top_terms(beta, namesL, args.top_k_terms):
                    t.update({"program": program_name, "basis": basis, "method": method, "feature_family": "lifted_pairs"})
                    term_rows.append(t)
    return rows, term_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--programs", nargs="+", default=["first_order_is_A", "paircat_AB_to_numeric", "detect_pattern_AB_raw"])
    parser.add_argument("--outdir", type=str, default="runs/tracr_label_basis_v6")
    parser.add_argument("--seq-len", type=int, default=6, help="Total sequence length including BOS; capped by Tracr max_seq_len.")
    parser.add_argument("--max-sequences", type=int, default=0, help="Optional random subsample of input sequences; 0 means all.")
    parser.add_argument("--stage", type=str, default="final_residual", choices=["final_residual", "pre_final_residual"])
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--omp-max-k", type=int, default=4)
    parser.add_argument("--lifted-omp-max-k", type=int, default=6)
    parser.add_argument("--max-lifted-pairs", type=int, default=0, help="0 means all pairwise products.")
    parser.add_argument("--top-k-terms", type=int, default=12)
    args = parser.parse_args()
    if args.max_sequences <= 0:
        args.max_sequences = None
    set_seed(args.seed)

    compiling, lib, rasp, err = try_import_tracr()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if err is not None:
        raise RuntimeError(f"Could not import Tracr: {err!r}")

    rows: List[Dict[str, Any]] = []
    term_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for p in args.programs:
        print(f"Analyzing {p}...")
        try:
            r, t, s = analyze_program(p, args, compiling, lib, rasp)
            rows.extend(r); term_rows.extend(t); summaries.append(s)
        except Exception as e:
            summaries.append({"program": p, "status": "failed", "error": repr(e)})
            print(f"  failed: {e!r}")

    df = pd.DataFrame(rows)
    terms = pd.DataFrame(term_rows)
    df.to_csv(outdir / "tracr_label_basis_results.csv", index=False)
    terms.to_csv(outdir / "tracr_label_basis_top_terms.csv", index=False)
    with (outdir / "tracr_label_basis_summary.json").open("w") as f:
        json.dump({"args": vars(args), "program_summaries": summaries}, f, indent=2, default=str)

    if plt is not None and len(df):
        # Simple bar plot: test R2 by program/basis/method/family.
        plot_df = df.copy()
        plot_df["label"] = plot_df["program"] + "\n" + plot_df["basis"] + "\n" + plot_df["method"] + "/" + plot_df["feature_family"]
        fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(plot_df)), 5))
        ax.bar(np.arange(len(plot_df)), plot_df["test_r2"].values)
        ax.set_xticks(np.arange(len(plot_df)))
        ax.set_xticklabels(plot_df["label"].values, rotation=75, ha="right", fontsize=8)
        ax.axhline(0, ls="--", lw=1)
        ax.set_ylabel("held-out position-level R²")
        ax.set_title("Tracr label-basis analysis: first-order vs lifted features")
        fig.tight_layout()
        fig.savefig(outdir / "tracr_label_basis_r2.png", dpi=160)
        plt.close(fig)

    print("Wrote:")
    print(" ", outdir / "tracr_label_basis_results.csv")
    print(" ", outdir / "tracr_label_basis_top_terms.csv")
    print(" ", outdir / "tracr_label_basis_summary.json")


if __name__ == "__main__":
    main()
