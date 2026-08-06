#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
Tracr residual-label GROUP intervention adapter (v8).

This extends the v7 final-residual intervention test.  v7 ablated/reconstructed
only a single detector coordinate (e.g. detect_AB_num_2 or the True detector).
For Tracr circuits the final readout often uses a *group* of explicit detector
labels: e.g. paircat_AB_3:AB, paircat_AB_3:OTHER, detect_AB_num_2, or both
True/False detector labels plus compiler map labels.

This script tests group-level causal readout interventions:

  1. Fit the final readout from final residual labels to the target output.
  2. Ablate the full explicit-detector group.
  3. Reconstruct that group from a restricted pre-interaction basis using
     (a) first-order restricted labels only and
     (b) lifted pair features over the restricted labels.
  4. Optionally also write a semantic hard-coded reconstruction for the simple
     AB circuits, as a sanity check.

Scope: this is still a final-residual causal readout intervention, not an
intermediate patch-and-continue through the Tracr graph.

Outputs
-------
  tracr_residual_group_intervention_results.csv
  tracr_residual_group_reconstruction_quality.csv
  tracr_residual_label_ablation.csv
  tracr_residual_group_intervention_summary.json
  tracr_residual_group_intervention_core.png
  tracr_detector_reconstruction_quality.png
  tracr_residual_label_ablation_top.png

Example
-------
python tracr_residual_group_intervention_adapter.py \
  --programs first_order_is_A paircat_AB_to_numeric detect_pattern_AB_raw \
  --seq-len 6 \
  --outdir runs/tracr_residual_group_intervention_v8
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# ------------------------------ utilities ------------------------------

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
            out.append(1.0 if (seq[i - 1] == "A" and t == "B") else 0.0)
    return out


def build_rasp_program(name: str, lib: Any, rasp: Any):
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


def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def ridge_fit(X: np.ndarray, y: np.ndarray, ridge: float = 1e-8) -> Tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    xm = X.mean(axis=0, keepdims=True)
    ym = float(y.mean())
    Xc = X - xm
    yc = y - ym
    beta = np.linalg.solve(Xc.T @ Xc + ridge * np.eye(X.shape[1]), Xc.T @ yc)
    intercept = ym - float(xm.reshape(-1) @ beta)
    return beta, intercept


def ridge_fit_multi(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Centered multi-output ridge. Returns beta [p, q], intercept [q]."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    xm = X.mean(axis=0, keepdims=True)
    ym = Y.mean(axis=0, keepdims=True)
    Xc = X - xm
    Yc = Y - ym
    beta = np.linalg.solve(Xc.T @ Xc + ridge * np.eye(X.shape[1]), Xc.T @ Yc)
    intercept = (ym - xm @ beta).reshape(-1)
    return beta, intercept


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


def collect_final_residual_dataset(model: Any, seqs: List[List[str]], expected_fn: Callable[[Sequence[str]], List[Any]]) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
    labels = list(getattr(model, "residual_labels", []))
    X_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    meta: List[Dict[str, Any]] = []
    for si, seq in enumerate(seqs):
        out = model.apply(seq)
        arr = np.asarray(out.residuals[-1][0], dtype=float)  # [T, d_model]
        expected = expected_fn(seq)
        for pos, exp in enumerate(expected):
            target = to_float_target(exp)
            if target is None:
                continue
            X_rows.append(arr[pos].copy())
            y_rows.append(float(target))
            meta.append({
                "seq_idx": si,
                "pos": pos,
                "token": seq[pos],
                "prev_token": seq[pos - 1] if pos > 0 else None,
                "seq": " ".join(seq),
            })
    return np.vstack(X_rows), np.asarray(y_rows, dtype=float), list(map(str, labels)), meta


def exact_label_indices(labels: Sequence[str], wanted: Sequence[str]) -> List[int]:
    wanted_set = set(wanted)
    return [i for i, lab in enumerate(labels) if str(lab) in wanted_set]


def startswith_indices(labels: Sequence[str], prefixes: Sequence[str]) -> List[int]:
    return [i for i, lab in enumerate(labels) if any(str(lab).startswith(p) for p in prefixes)]


def contains_indices(labels: Sequence[str], patterns: Sequence[str]) -> List[int]:
    return [i for i, lab in enumerate(labels) if any(p in str(lab) for p in patterns)]


def group_indices(labels: Sequence[str], program: str) -> Dict[str, List[int]]:
    labels = list(map(str, labels))
    groups: Dict[str, List[int]] = {}
    groups["all_nonpad"] = [i for i, l in enumerate(labels) if "compiler_pad" not in l]
    groups["tokens_nonbos"] = [i for i, l in enumerate(labels) if l.startswith("tokens:") and l not in {"tokens:BOS", "tokens:compiler_pad"}]
    groups["indices"] = startswith_indices(labels, ["indices:"])
    groups["prev_or_shift"] = [i for i, l in enumerate(labels) if "prev_token" in l or "shift_by" in l]
    groups["explicit_detector"] = []
    groups["true_detector"] = []
    groups["false_detector"] = []
    groups["reconstruct_from"] = []

    if program == "first_order_is_A":
        groups["true_detector"] = exact_label_indices(labels, ["is_A_1"])
        # Treat this as the explicit group for plotting and interventions.
        groups["explicit_detector"] = list(groups["true_detector"])
        groups["token_A"] = exact_label_indices(labels, ["tokens:A"])
        groups["reconstruct_from"] = list(groups["token_A"])
    elif program == "paircat_AB_to_numeric":
        groups["explicit_detector"] = exact_label_indices(labels, ["detect_AB_num_2", "paircat_AB_3:AB", "paircat_AB_3:OTHER"])
        groups["true_detector"] = exact_label_indices(labels, ["detect_AB_num_2", "paircat_AB_3:AB"])
        groups["false_detector"] = exact_label_indices(labels, ["paircat_AB_3:OTHER"])
        groups["prev_A"] = exact_label_indices(labels, ["prev_token_4:A"])
        groups["token_B"] = exact_label_indices(labels, ["tokens:B"])
        groups["reconstruct_from"] = sorted(set(groups["prev_or_shift"] + groups["tokens_nonbos"]))
    elif program == "detect_pattern_AB_raw":
        groups["explicit_detector"] = exact_label_indices(labels, [
            "detect_pattern_AB_raw_6:False", "detect_pattern_AB_raw_6:True",
            "map_10:False", "map_10:True", "map_8:False", "map_8:True",
        ])
        groups["true_detector"] = exact_label_indices(labels, ["detect_pattern_AB_raw_6:True", "map_10:True", "map_8:True"])
        groups["false_detector"] = exact_label_indices(labels, ["detect_pattern_AB_raw_6:False", "map_10:False", "map_8:False"])
        groups["shift_true"] = exact_label_indices(labels, ["shift_by(1)_7:True"])
        groups["shift_false"] = exact_label_indices(labels, ["shift_by(1)_7:False"])
        groups["token_B"] = exact_label_indices(labels, ["tokens:B"])
        groups["reconstruct_from"] = sorted(set(groups["prev_or_shift"] + groups["tokens_nonbos"]))
    else:
        groups["explicit_detector"] = [i for i, l in enumerate(labels) if ("detect" in l or "paircat" in l or "map_" in l) and "compiler_pad" not in l]
        groups["reconstruct_from"] = sorted(set(groups["prev_or_shift"] + groups["tokens_nonbos"]))
    return groups


def predict(X: np.ndarray, beta: np.ndarray, intercept: float) -> np.ndarray:
    return np.asarray(X, dtype=float) @ beta + intercept


def intervene_scale(X: np.ndarray, idxs: Sequence[int], scale: float) -> np.ndarray:
    X2 = np.asarray(X, dtype=float).copy()
    if idxs:
        X2[:, list(idxs)] *= float(scale)
    return X2


def lifted_features(X: np.ndarray, labels: Sequence[str], *, include_first_order: bool = True) -> Tuple[np.ndarray, List[str]]:
    X = np.asarray(X, dtype=float)
    feats: List[np.ndarray] = []
    names: List[str] = []
    if include_first_order:
        for i, lab in enumerate(labels):
            feats.append(X[:, i])
            names.append(str(lab))
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            feats.append(X[:, i] * X[:, j])
            names.append(f"{labels[i]} * {labels[j]}")
    if not feats:
        return np.zeros((X.shape[0], 0)), []
    return np.column_stack(feats), names


def fit_group_reconstructor(X: np.ndarray, groups: Dict[str, List[int]], labels: Sequence[str], train_idx: np.ndarray, method: str, ridge: float) -> Dict[str, Any]:
    det = groups.get("explicit_detector", [])
    src = groups.get("reconstruct_from", [])
    if not det or not src:
        return {"status": "not_applicable", "method": method, "detector_indices": det, "source_indices": src}
    Xsrc = X[:, src]
    src_labels = [labels[i] for i in src]
    if method == "first_order":
        Phi, names = Xsrc, src_labels
    elif method == "lifted":
        Phi, names = lifted_features(Xsrc, src_labels, include_first_order=True)
    else:
        raise ValueError(method)
    Y = X[:, det]
    beta, intercept = ridge_fit_multi(Phi[train_idx], Y[train_idx], ridge=ridge)
    return {
        "status": "ok",
        "method": method,
        "detector_indices": det,
        "detector_labels": [labels[i] for i in det],
        "source_indices": src,
        "source_labels": src_labels,
        "feature_names": names,
        "beta": beta,
        "intercept": intercept,
    }


def apply_group_reconstructor(X: np.ndarray, recon: Dict[str, Any], *, clip: bool) -> Tuple[np.ndarray, np.ndarray]:
    X2 = np.asarray(X, dtype=float).copy()
    if recon.get("status") != "ok":
        return X2, np.zeros((X.shape[0], 0))
    det = list(map(int, recon["detector_indices"]))
    src = list(map(int, recon["source_indices"]))
    Xsrc = X[:, src]
    if recon["method"] == "first_order":
        Phi = Xsrc
    else:
        Phi, _ = lifted_features(Xsrc, recon["source_labels"], include_first_order=True)
    Yhat = Phi @ recon["beta"] + recon["intercept"]
    if clip:
        Yhat = np.clip(Yhat, 0.0, 1.0)
    if det:
        X2[:, det] = Yhat
    return X2, Yhat


def semantic_group_reconstruction(X: np.ndarray, labels: Sequence[str], program: str, groups: Dict[str, List[int]], *, ablate_detector: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Hard-coded semantic reconstruction for sanity checks on simple Tracr AB circuits."""
    X2 = np.asarray(X, dtype=float).copy()
    det = groups.get("explicit_detector", [])
    if ablate_detector and det:
        X2[:, det] = 0.0
    info: Dict[str, Any] = {"status": "not_applicable"}

    def set_if_present(label: str, values: np.ndarray) -> None:
        idx = exact_label_indices(labels, [label])
        if idx:
            X2[:, idx[0]] = values

    if program == "first_order_is_A":
        tokA = exact_label_indices(labels, ["tokens:A"])
        if tokA:
            values = X[:, tokA[0]]
            set_if_present("is_A_1", values)
            info = {"status": "ok", "write": "is_A_1 <- tokens:A"}
        return X2, info

    if program == "paircat_AB_to_numeric":
        prevA = exact_label_indices(labels, ["prev_token_4:A"])
        tokB = exact_label_indices(labels, ["tokens:B"])
        if prevA and tokB:
            prod = X[:, prevA[0]] * X[:, tokB[0]]
            set_if_present("detect_AB_num_2", prod)
            set_if_present("paircat_AB_3:AB", prod)
            set_if_present("paircat_AB_3:OTHER", 1.0 - prod)
            info = {"status": "ok", "write": "explicit paircat/detect group <- prev_token_4:A * tokens:B"}
        return X2, info

    if program == "detect_pattern_AB_raw":
        shiftT = exact_label_indices(labels, ["shift_by(1)_7:True"])
        tokB = exact_label_indices(labels, ["tokens:B"])
        if shiftT and tokB:
            prod = X[:, shiftT[0]] * X[:, tokB[0]]
            for lab in ["detect_pattern_AB_raw_6:True", "map_10:True", "map_8:True"]:
                set_if_present(lab, prod)
            for lab in ["detect_pattern_AB_raw_6:False", "map_10:False", "map_8:False"]:
                set_if_present(lab, 1.0 - prod)
            info = {"status": "ok", "write": "explicit bool detector/map group <- shift_by(1)_7:True * tokens:B"}
        return X2, info

    return X2, info


def eval_condition(program: str, condition: str, X: np.ndarray, y: np.ndarray, beta: np.ndarray, intercept: float, idxs: Optional[Sequence[int]] = None, scale: Optional[float] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    yhat = predict(X, beta, intercept)
    row = {
        "program": program,
        "condition": condition,
        "n_examples": int(len(y)),
        "r2": r2_score(y, yhat),
        "mse": mse(y, yhat),
        "mae": mae(y, yhat),
    }
    if idxs is not None:
        row["n_labels_touched"] = int(len(idxs))
        row["label_indices"] = ";".join(map(str, idxs))
    if scale is not None:
        row["scale"] = float(scale)
    if extra:
        row.update(extra)
    return row


def reconstruction_quality_rows(program: str, method: str, recon: Dict[str, Any], X: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, labels: Sequence[str], clip: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if recon.get("status") != "ok":
        return rows
    det = list(map(int, recon["detector_indices"]))
    _, Yhat_all = apply_group_reconstructor(X, recon, clip=clip)
    Y = X[:, det]
    for local_j, det_idx in enumerate(det):
        for split, idx in [("train", train_idx), ("test", test_idx)]:
            rows.append({
                "program": program,
                "method": method,
                "split": split,
                "detector_label": labels[det_idx],
                "r2": r2_score(Y[idx, local_j], Yhat_all[idx, local_j]),
                "mse": mse(Y[idx, local_j], Yhat_all[idx, local_j]),
                "mae": mae(Y[idx, local_j], Yhat_all[idx, local_j]),
            })
    # Aggregate Frobenius R2 over the whole group, centered by column means.
    for split, idx in [("train", train_idx), ("test", test_idx)]:
        YY = Y[idx]
        PP = Yhat_all[idx]
        denom = float(np.sum((YY - YY.mean(axis=0, keepdims=True)) ** 2))
        group_r2 = float("nan") if denom < 1e-12 else 1.0 - float(np.sum((YY - PP) ** 2)) / denom
        rows.append({
            "program": program,
            "method": method,
            "split": split,
            "detector_label": "__GROUP__",
            "r2": group_r2,
            "mse": mse(YY, PP),
            "mae": mae(YY, PP),
        })
    return rows


def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    compiling, lib, rasp, err = try_import_tracr()
    if err is not None:
        raise RuntimeError(f"Could not import tracr: {err!r}")

    all_results: List[Dict[str, Any]] = []
    label_rows: List[Dict[str, Any]] = []
    recon_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "args": vars(args),
        "programs": [],
        "interpretation": "Final-residual GROUP causal readout interventions over Tracr residual labels. This tests causal readout effects of explicit detector groups and whether restricted/lifted observers can reconstruct them. It is not continuation from an intermediate residual stage.",
    }

    for program_name in args.programs:
        program, vocab, max_seq_len, expected_fn = build_rasp_program(program_name, lib, rasp)
        model = compiling.compile_rasp_to_model(program, vocab=vocab, max_seq_len=max_seq_len, compiler_bos="BOS")
        seqs = make_sequences(sorted(vocab), max_seq_len=max_seq_len, exact_len=args.seq_len, max_examples=args.max_sequences, seed=args.seed)
        X, y, labels, meta = collect_final_residual_dataset(model, seqs, expected_fn)
        tr_idx, te_idx = train_test_split(len(y), args.test_frac, args.seed)
        beta, intercept = ridge_fit(X[tr_idx], y[tr_idx], ridge=args.ridge)
        base_test = eval_condition(program_name, "baseline_readout", X[te_idx], y[te_idx], beta, intercept)

        groups = group_indices(labels, program_name)
        det_group = groups.get("explicit_detector", [])
        prog_summary = {
            "program": program_name,
            "n_sequences": len(seqs),
            "n_examples": int(len(y)),
            "n_labels": len(labels),
            "labels": list(map(str, labels)),
            "groups": {k: [str(labels[i]) for i in v] for k, v in groups.items()},
            "baseline_test_r2": base_test["r2"],
            "baseline_test_mse": base_test["mse"],
        }
        summary["programs"].append(prog_summary)
        all_results.append(base_test)

        # Group ablations/scaling on the final residual.
        for gname, idxs in groups.items():
            if not idxs:
                continue
            for scale in args.scales:
                Xmod = intervene_scale(X[te_idx], idxs, scale)
                row = eval_condition(
                    program_name,
                    f"scale_group:{gname}",
                    Xmod,
                    y[te_idx],
                    beta,
                    intercept,
                    idxs=idxs,
                    scale=scale,
                    extra={"group": gname, "labels": ";".join(str(labels[i]) for i in idxs)},
                )
                row["delta_mse_vs_base"] = row["mse"] - base_test["mse"]
                row["delta_r2_vs_base"] = row["r2"] - base_test["r2"]
                all_results.append(row)

        # Per-label ablation scan.
        for i, lab in enumerate(labels):
            if "compiler_pad" in str(lab):
                continue
            Xmod = intervene_scale(X[te_idx], [i], 0.0)
            yhat = predict(Xmod, beta, intercept)
            row = {
                "program": program_name,
                "label_index": i,
                "label": str(lab),
                "r2_after_ablation": r2_score(y[te_idx], yhat),
                "mse_after_ablation": mse(y[te_idx], yhat),
                "delta_mse_vs_base": mse(y[te_idx], yhat) - base_test["mse"],
                "abs_readout_weight": float(abs(beta[i])),
                "readout_weight": float(beta[i]),
                "activation_mean_test": float(np.mean(X[te_idx, i])),
                "activation_std_test": float(np.std(X[te_idx, i])),
            }
            label_rows.append(row)

        # Ablate the full explicit detector group.
        X_ablate_group = intervene_scale(X[te_idx], det_group, 0.0)
        row = eval_condition(
            program_name,
            "ablate_explicit_detector_group",
            X_ablate_group,
            y[te_idx],
            beta,
            intercept,
            idxs=det_group,
            scale=0.0,
            extra={"group": "explicit_detector", "labels": ";".join(str(labels[i]) for i in det_group)},
        )
        row["delta_mse_vs_base"] = row["mse"] - base_test["mse"]
        row["delta_r2_vs_base"] = row["r2"] - base_test["r2"]
        all_results.append(row)

        # Learned reconstructions: first-order restricted basis and lifted restricted basis.
        for method in ["first_order", "lifted"]:
            recon = fit_group_reconstructor(X, groups, labels, tr_idx, method=method, ridge=args.reconstruct_ridge)
            recon_rows.extend(reconstruction_quality_rows(program_name, method, recon, X, tr_idx, te_idx, labels, clip=args.clip_reconstruction))
            Xrec, _ = apply_group_reconstructor(X[te_idx], recon, clip=args.clip_reconstruction)
            # Important: apply_group_reconstructor writes detector values but does not explicitly zero first.
            # Since it overwrites the detector group, this is equivalent to ablate+write for the group.
            row = eval_condition(
                program_name,
                f"ablate_explicit_group_then_reconstruct_{method}",
                Xrec,
                y[te_idx],
                beta,
                intercept,
                idxs=det_group,
                extra={
                    "group": "explicit_detector",
                    "reconstruction_status": recon.get("status"),
                    "reconstruction_method": method,
                    "n_reconstruction_features": len(recon.get("feature_names", [])) if recon.get("status") == "ok" else 0,
                    "source_labels": ";".join(recon.get("source_labels", [])) if recon.get("status") == "ok" else "",
                },
            )
            row["delta_mse_vs_base"] = row["mse"] - base_test["mse"]
            row["delta_r2_vs_base"] = row["r2"] - base_test["r2"]
            all_results.append(row)

        # Semantic hard-coded lifted reconstruction for sanity check.
        Xsem, info = semantic_group_reconstruction(X[te_idx], labels, program_name, groups, ablate_detector=True)
        row = eval_condition(
            program_name,
            "ablate_explicit_group_then_reconstruct_semantic_lifted",
            Xsem,
            y[te_idx],
            beta,
            intercept,
            idxs=det_group,
            extra={"group": "explicit_detector", "reconstruction_status": info.get("status"), "reconstruction_write": info.get("write", "")},
        )
        row["delta_mse_vs_base"] = row["mse"] - base_test["mse"]
        row["delta_r2_vs_base"] = row["r2"] - base_test["r2"]
        all_results.append(row)

    results_df = pd.DataFrame(all_results)
    labels_df = pd.DataFrame(label_rows)
    recon_df = pd.DataFrame(recon_rows)
    results_df.to_csv(outdir / "tracr_residual_group_intervention_results.csv", index=False)
    labels_df.to_csv(outdir / "tracr_residual_label_ablation.csv", index=False)
    recon_df.to_csv(outdir / "tracr_residual_group_reconstruction_quality.csv", index=False)
    with (outdir / "tracr_residual_group_intervention_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    if plt is not None and len(results_df):
        keep_conditions = [
            "baseline_readout",
            "ablate_explicit_detector_group",
            "ablate_explicit_group_then_reconstruct_first_order",
            "ablate_explicit_group_then_reconstruct_lifted",
            "ablate_explicit_group_then_reconstruct_semantic_lifted",
        ]
        plot_df = results_df[results_df["condition"].isin(keep_conditions)].copy()
        if len(plot_df):
            fig, ax = plt.subplots(figsize=(12, 5.2))
            programs = list(dict.fromkeys(plot_df["program"].tolist()))
            x = np.arange(len(programs))
            width = 0.16
            offsets = {c: (j - (len(keep_conditions) - 1) / 2.0) * width for j, c in enumerate(keep_conditions)}
            for c in keep_conditions:
                vals = []
                for p in programs:
                    sub = plot_df[(plot_df["program"] == p) & (plot_df["condition"] == c)]
                    vals.append(float(sub["r2"].iloc[0]) if len(sub) else np.nan)
                ax.bar(x + offsets[c], vals, width=width, label=c.replace("_", " "))
            ax.set_xticks(x)
            ax.set_xticklabels(programs, rotation=20, ha="right")
            ax.set_ylabel("held-out position-level R²")
            ax.set_title("Tracr final-residual explicit-detector GROUP interventions")
            ax.axhline(0.0, linestyle="--", linewidth=1)
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(outdir / "tracr_residual_group_intervention_core.png", dpi=160)
            plt.close(fig)

        if len(recon_df):
            plot_recon = recon_df[(recon_df["split"] == "test") & (recon_df["detector_label"] == "__GROUP__")]
            if len(plot_recon):
                fig, ax = plt.subplots(figsize=(9, 4.5))
                programs = list(dict.fromkeys(plot_recon["program"].tolist()))
                methods = ["first_order", "lifted"]
                x = np.arange(len(programs))
                width = 0.32
                for j, method in enumerate(methods):
                    vals = []
                    for p in programs:
                        sub = plot_recon[(plot_recon["program"] == p) & (plot_recon["method"] == method)]
                        vals.append(float(sub["r2"].iloc[0]) if len(sub) else np.nan)
                    ax.bar(x + (j - 0.5) * width, vals, width=width, label=method)
                ax.set_xticks(x)
                ax.set_xticklabels(programs, rotation=20, ha="right")
                ax.set_ylabel("detector-group reconstruction R²")
                ax.set_title("Can restricted features reconstruct the explicit detector group?")
                ax.axhline(0.0, linestyle="--", linewidth=1)
                ax.legend()
                fig.tight_layout()
                fig.savefig(outdir / "tracr_detector_reconstruction_quality.png", dpi=160)
                plt.close(fig)

        if len(labels_df):
            top = labels_df.sort_values("delta_mse_vs_base", ascending=False).groupby("program").head(8)
            fig, axes = plt.subplots(max(1, len(top["program"].unique())), 1, figsize=(10, 3.2 * max(1, len(top["program"].unique()))), squeeze=False)
            for ax, (prog, sub) in zip(axes[:, 0], top.groupby("program")):
                ss = sub.sort_values("delta_mse_vs_base", ascending=True)
                ax.barh(ss["label"], ss["delta_mse_vs_base"])
                ax.set_title(f"{prog}: top single-label ablation effects")
                ax.set_xlabel("ΔMSE vs baseline readout")
            fig.tight_layout()
            fig.savefig(outdir / "tracr_residual_label_ablation_top.png", dpi=160)
            plt.close(fig)

    print("Wrote:")
    print(f"  {outdir / 'tracr_residual_group_intervention_results.csv'}")
    print(f"  {outdir / 'tracr_residual_group_reconstruction_quality.csv'}")
    print(f"  {outdir / 'tracr_residual_label_ablation.csv'}")
    print(f"  {outdir / 'tracr_residual_group_intervention_summary.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--programs", nargs="+", default=["first_order_is_A", "paircat_AB_to_numeric", "detect_pattern_AB_raw"])
    ap.add_argument("--seq-len", type=int, default=6)
    ap.add_argument("--max-sequences", type=int, default=None)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=1e-8)
    ap.add_argument("--reconstruct-ridge", type=float, default=1e-8)
    ap.add_argument("--clip-reconstruction", action="store_true", default=True)
    ap.add_argument("--no-clip-reconstruction", dest="clip_reconstruction", action="store_false")
    ap.add_argument("--scales", nargs="+", type=float, default=[0.0, 0.5, 1.5])
    ap.add_argument("--outdir", type=str, default="runs/tracr_residual_group_intervention_v8")
    run(ap.parse_args())
