#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
IOI Stage 2b: head-level subset prediction.

This script is the statistically powered version of the IOI Stage-2 task.
Instead of eight whole-group subsets, it samples many random subsets of
individual heads from published IOI groups and asks whether interaction-aware
observers predict held-out interventions better than singleton-additive observers.

Effects are positive drops in IOI logit difference:
    drop(S) = LD_clean - LD_ablate(S)

Primary models:
    additive_head:
        drop(S) = beta0 + sum_i beta_i 1[i in S]

    pb_group_interaction:
        additive_head + gamma_PB 1[S intersects P]1[S intersects B]

    group_interaction:
        additive_head + gamma_PB P_B + gamma_PE P_E + gamma_BE B_E

The group interaction terms are deliberately group-level, not all 78 head pairs.
This keeps the model identifiable with ~100-200 subset measurements while still
testing the self-repair hypothesis.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

try:
    from transformer_lens import HookedTransformer, utils
except Exception as e:  # pragma: no cover
    HookedTransformer = None
    utils = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None

DEFAULT_PRIMARY_HEADS = "9.9,9.6,10.0"
DEFAULT_BACKUP_HEADS = "9.0,9.7,10.1,10.2,10.6,10.10,11.2,11.9"
DEFAULT_EXTRA_HEADS = "10.7,11.10"  # canonical Negative Name Movers in Wang et al.
DEFAULT_TEMPLATE = "When {io} and {s} went to the store, {s} gave a bottle of milk to"
DEFAULT_NAMES = [
    "John", "Mary", "Bob", "Alice", "Tom", "Sarah", "James", "Emily",
    "Michael", "Jessica", "David", "Laura", "Daniel", "Emma", "Robert",
    "Anna", "Jacob", "Scott", "Kevin", "Lisa", "Mark", "Susan", "Steven",
    "Rachel", "Brian", "Karen", "Jason", "Nancy", "Eric", "Helen",
]


@dataclass(frozen=True)
class PromptRecord:
    prompt: str
    io_name: str
    s_name: str
    io_token: int
    s_token: int


@dataclass(frozen=True)
class HeadRecord:
    head_idx: int
    group: str
    layer: int
    head: int
    label: str


def parse_heads(spec: str) -> List[Tuple[int, int]]:
    if not spec.strip():
        return []
    out: List[Tuple[int, int]] = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if '.' not in item:
            raise ValueError(f"Head '{item}' must use layer.head notation")
        l, h = item.split('.', 1)
        out.append((int(l), int(h)))
    return out


def format_heads(heads: Sequence[Tuple[int, int]]) -> str:
    return ','.join(f"{l}.{h}" for l, h in heads)


def choose_device(device: str) -> str:
    if device != 'auto':
        return device
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_model(model_name: str, device: str):
    if HookedTransformer is None:
        raise ImportError(
            "Could not import transformer_lens. Install with: pip install transformer-lens\n"
            f"Original import error: {_IMPORT_ERR!r}"
        )
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    return model


def single_token_names(model, names: Sequence[str]):
    good, toks = [], {}
    for name in names:
        try:
            tok = int(model.to_single_token(' ' + name))
        except Exception:
            continue
        good.append(name)
        toks[name] = tok
    if len(good) < 4:
        raise RuntimeError("Too few single-token names. Pass --names with GPT-2 single-token names.")
    return good, toks


def make_prompts(model, n: int, seed: int, template: str, names: Optional[Sequence[str]] = None) -> List[PromptRecord]:
    rng = random.Random(seed)
    names = list(names or DEFAULT_NAMES)
    names, toks = single_token_names(model, names)
    out: List[PromptRecord] = []
    seen = set()
    attempts = 0
    while len(out) < n and attempts < n * 100:
        attempts += 1
        io, s = rng.sample(names, 2)
        prompt = template.format(io=io, s=s)
        key = (prompt, io, s)
        if key in seen:
            continue
        seen.add(key)
        out.append(PromptRecord(prompt, io, s, toks[io], toks[s]))
    while len(out) < n:
        io, s = rng.sample(names, 2)
        prompt = template.format(io=io, s=s)
        out.append(PromptRecord(prompt, io, s, toks[io], toks[s]))
    return out


def tokens_for_records(model, records: Sequence[PromptRecord], device: str):
    return model.to_tokens([r.prompt for r in records], prepend_bos=False).to(device)


def answer_token_tensors(records: Sequence[PromptRecord], device: str):
    return (
        torch.tensor([r.io_token for r in records], dtype=torch.long, device=device),
        torch.tensor([r.s_token for r in records], dtype=torch.long, device=device),
    )


def chunk_indices(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


def compute_mean_z(model, tokens: torch.Tensor, batch_size: int) -> Dict[int, torch.Tensor]:
    sums: Dict[int, torch.Tensor] = {}
    count = 0
    names_filter = lambda name: name.endswith('hook_z')
    for a, b in tqdm(list(chunk_indices(tokens.shape[0], batch_size)), desc='Computing reference head-output means'):
        batch = tokens[a:b]
        with torch.no_grad():
            _, cache = model.run_with_cache(batch, names_filter=names_filter, return_type='logits')
        bs = batch.shape[0]
        count += bs
        for layer in range(model.cfg.n_layers):
            name = utils.get_act_name('z', layer)
            z = cache[name].detach().float().cpu()
            sums[layer] = z.sum(dim=0) if layer not in sums else sums[layer] + z.sum(dim=0)
        del cache
    return {layer: s / float(count) for layer, s in sums.items()}


def make_ablation_hooks(model, heads: Sequence[Tuple[int, int]], ablation: str, mean_z, position_mode: str):
    by_layer: Dict[int, List[int]] = {}
    for layer, head in heads:
        if layer < 0 or layer >= model.cfg.n_layers:
            raise ValueError(f"Layer {layer} out of range")
        if head < 0 or head >= model.cfg.n_heads:
            raise ValueError(f"Head {head} out of range")
        by_layer.setdefault(layer, []).append(head)
    hooks = []
    for layer, head_list in by_layer.items():
        hook_name = utils.get_act_name('z', layer)

        def hook_fn(z, hook, layer=layer, head_list=head_list):
            z = z.clone()
            pos_len = z.shape[1]
            for h in head_list:
                if ablation == 'zero':
                    if position_mode == 'end':
                        z[:, -1, h, :] = 0.0
                    elif position_mode == 'all':
                        z[:, :, h, :] = 0.0
                    else:
                        raise ValueError(position_mode)
                elif ablation == 'mean':
                    if mean_z is None:
                        raise ValueError('mean_z required for mean ablation')
                    ref = mean_z[layer].to(z.device, dtype=z.dtype)
                    if ref.shape[0] < pos_len:
                        raise RuntimeError(f"Reference mean pos length {ref.shape[0]} < batch length {pos_len}")
                    if position_mode == 'end':
                        z[:, -1, h, :] = ref[pos_len - 1, h, :]
                    elif position_mode == 'all':
                        z[:, :, h, :] = ref[:pos_len, h, :].unsqueeze(0)
                    else:
                        raise ValueError(position_mode)
                else:
                    raise ValueError(ablation)
            return z

        hooks.append((hook_name, hook_fn))
    return hooks


def compute_logit_diffs(model, tokens, io_tokens, s_tokens, batch_size: int, heads=None, ablation='none', mean_z=None, position_mode='end') -> np.ndarray:
    vals = []
    hooks = []
    if heads:
        hooks = make_ablation_hooks(model, heads, ablation=ablation, mean_z=mean_z, position_mode=position_mode)
    label = f"{ablation}:{format_heads(heads or []) or 'clean'}"
    for a, b in tqdm(list(chunk_indices(tokens.shape[0], batch_size)), desc=f"Run {label}"):
        batch = tokens[a:b]
        with torch.no_grad():
            logits = model.run_with_hooks(batch, return_type='logits', fwd_hooks=hooks) if hooks else model(batch)
            final = logits[:, -1, :]
            diff = final.gather(1, io_tokens[a:b, None]).squeeze(1) - final.gather(1, s_tokens[a:b, None]).squeeze(1)
            vals.append(diff.detach().float().cpu().numpy())
    return np.concatenate(vals, axis=0)


def build_head_records(primary: Sequence[Tuple[int, int]], backup: Sequence[Tuple[int, int]], extra: Sequence[Tuple[int, int]]) -> List[HeadRecord]:
    records: List[HeadRecord] = []
    seen = set()
    idx = 0
    for group, heads in [('P', primary), ('B', backup), ('E', extra)]:
        for layer, head in heads:
            if (layer, head) in seen:
                raise ValueError(f"Head {layer}.{head} appears in multiple groups")
            seen.add((layer, head))
            records.append(HeadRecord(idx, group, layer, head, f"{group}:{layer}.{head}"))
            idx += 1
    return records


def mask_to_heads(mask: np.ndarray, head_records: Sequence[HeadRecord]) -> List[Tuple[int, int]]:
    return [(h.layer, h.head) for bit, h in zip(mask, head_records) if int(bit) == 1]


def mask_name(mask: np.ndarray, head_records: Sequence[HeadRecord]) -> str:
    if int(mask.sum()) == 0:
        return 'clean'
    parts = [h.label for bit, h in zip(mask, head_records) if int(bit) == 1]
    return '|'.join(parts)


def group_counts(mask: np.ndarray, head_records: Sequence[HeadRecord]) -> Dict[str, int]:
    out = {'P': 0, 'B': 0, 'E': 0}
    for bit, h in zip(mask, head_records):
        if int(bit) == 1:
            out[h.group] += 1
    return out


def sample_subset_masks(head_records: Sequence[HeadRecord], n_subsets: int, seed: int, p_include: float = 0.35,
                        include_all_singletons: bool = True, include_group_subsets: bool = True) -> np.ndarray:
    """Return unique binary masks over heads.

    We include clean, all singletons, and whole-group subsets as anchors, then
    fill the remaining budget with random masks.  Random masks are sampled with
    variable density and stratified enough to include mixed-group subsets.
    """
    rng = np.random.default_rng(seed)
    n = len(head_records)
    masks: List[Tuple[int, ...]] = []

    def add(m):
        masks.append(tuple(int(x) for x in m))

    add(np.zeros(n, dtype=int))  # clean

    if include_all_singletons:
        for i in range(n):
            m = np.zeros(n, dtype=int); m[i] = 1; add(m)

    if include_group_subsets:
        groups = ['P', 'B', 'E']
        for P in [0, 1]:
            for B in [0, 1]:
                for E in [0, 1]:
                    if P == B == E == 0:
                        continue
                    m = np.zeros(n, dtype=int)
                    for i, h in enumerate(head_records):
                        if (h.group == 'P' and P) or (h.group == 'B' and B) or (h.group == 'E' and E):
                            m[i] = 1
                    add(m)

    # Fill with random masks.  Use a mix of densities and force non-clean.
    max_attempts = max(5000, n_subsets * 200)
    attempts = 0
    unique = set(masks)
    while len(unique) < n_subsets and attempts < max_attempts:
        attempts += 1
        # Draw a density, not fixed, so train/test covers small and large subsets.
        density = float(np.clip(rng.beta(1.5, 2.5), 0.08, 0.85))
        # Mix in the requested inclusion probability.
        density = 0.5 * density + 0.5 * p_include
        m = rng.binomial(1, density, size=n).astype(int)
        if m.sum() == 0:
            continue
        unique.add(tuple(int(x) for x in m))
    if len(unique) < n_subsets:
        print(f"Warning: only sampled {len(unique)} unique masks out of requested {n_subsets}")
    # Keep deterministic ordering: anchors first, then sorted randoms.
    anchors = []
    seen = set()
    for m in masks:
        if m not in seen:
            anchors.append(m); seen.add(m)
    rest = sorted([m for m in unique if m not in seen])
    final = (anchors + rest)[:n_subsets]
    return np.asarray(final, dtype=int)


def build_design(mask_matrix: np.ndarray, head_records: Sequence[HeadRecord], model_name: str, ridge_pairs: bool = False):
    n = mask_matrix.shape[1]
    rows = []
    cols = ['intercept'] + [h.label for h in head_records]
    pair_terms: List[Tuple[int, int, str]] = []
    if model_name == 'additive_head':
        pass
    elif model_name == 'pb_group_interaction':
        cols += ['P_B']
    elif model_name == 'group_interaction':
        cols += ['P_B', 'P_E', 'B_E']
    elif model_name == 'head_pair_sparse_ridge':
        # Diagnostic only.  Uses all head pair terms with ridge, not the primary benchmark model.
        for i in range(n):
            for j in range(i + 1, n):
                pair_terms.append((i, j, f"{head_records[i].label}*{head_records[j].label}"))
        cols += [x[2] for x in pair_terms]
    else:
        raise ValueError(model_name)

    for mask in mask_matrix:
        g = group_counts(mask, head_records)
        feats = {'intercept': 1.0}
        for bit, h in zip(mask, head_records):
            feats[h.label] = float(bit)
        feats['P_B'] = float(g['P'] > 0 and g['B'] > 0)
        feats['P_E'] = float(g['P'] > 0 and g['E'] > 0)
        feats['B_E'] = float(g['B'] > 0 and g['E'] > 0)
        for i, j, name in pair_terms:
            feats[name] = float(mask[i] and mask[j])
        rows.append([feats[c] for c in cols])
    return np.asarray(rows, dtype=float), cols


def ridge_fit(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    reg = ridge * np.eye(X.shape[1], dtype=float)
    reg[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


def make_kfold_indices(n_rows: int, k: int, seed: int, protect_clean: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_rows)
    if protect_clean:
        nonclean = indices[1:]
        rng.shuffle(nonclean)
        folds_nonclean = np.array_split(nonclean, k)
        folds = []
        for fold in folds_nonclean:
            test = fold
            train = np.setdiff1d(indices, test)
            folds.append((train, test))
        return folds
    rng.shuffle(indices)
    chunks = np.array_split(indices, k)
    folds = []
    for chunk in chunks:
        test = chunk
        train = np.setdiff1d(np.arange(n_rows), test)
        folds.append((train, test))
    return folds


def kfold_predict(mask_matrix: np.ndarray, y: np.ndarray, head_records: Sequence[HeadRecord], model_name: str, ridge: float,
                  k_folds: int, seed: int, eval_nonclean: bool):
    X, cols = build_design(mask_matrix, head_records, model_name)
    folds = make_kfold_indices(len(y), k_folds, seed=seed, protect_clean=True)
    preds = np.full_like(y, np.nan, dtype=float)
    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        beta = ridge_fit(X[train_idx], y[train_idx], ridge)
        pred = X[test_idx] @ beta
        preds[test_idx] = pred
        for idx, p in zip(test_idx, pred):
            rows.append({'model': model_name, 'fold': fold_idx, 'subset_idx': int(idx), 'observed': float(y[idx]), 'predicted': float(p), 'error': float(p - y[idx])})
    mask = np.ones(len(y), dtype=bool)
    if eval_nonclean:
        mask[0] = False
    valid = mask & np.isfinite(preds)
    yy = y[valid]
    pp = preds[valid]
    mae = float(np.mean(np.abs(pp - yy)))
    rmse = float(np.sqrt(np.mean((pp - yy) ** 2)))
    denom = float(np.sum((yy - yy.mean()) ** 2))
    r2 = float(1.0 - np.sum((pp - yy) ** 2) / denom) if denom > 1e-12 else float('nan')
    return pd.DataFrame(rows), {'model': model_name, 'n_rows': int(valid.sum()), 'mae': mae, 'rmse': rmse, 'r2': r2, 'columns': ','.join(cols), 'n_params': len(cols)}


def fit_all(mask_matrix, y, head_records, model_name: str, ridge: float):
    X, cols = build_design(mask_matrix, head_records, model_name)
    beta = ridge_fit(X, y, ridge)
    return pd.DataFrame({'model': model_name, 'term': cols, 'coef': beta})


def annotate_prediction_rows(pred_df: pd.DataFrame, subset_df: pd.DataFrame):
    meta_cols = ['subset_idx', 'subset_name', 'n_heads', 'n_P', 'n_B', 'n_E', 'has_P', 'has_B', 'has_E', 'P_B', 'P_E', 'B_E']
    return pred_df.merge(subset_df[meta_cols], on='subset_idx', how='left')


def summarize_by_group_occupancy(pred_df: pd.DataFrame):
    rows = []
    for model, gm in pred_df.groupby('model'):
        for key in ['P_B', 'P_E', 'B_E']:
            for val, g in gm.groupby(key):
                rows.append({
                    'model': model,
                    'slice': key,
                    'value': int(val),
                    'n': len(g),
                    'mae': float(np.mean(np.abs(g['error']))),
                    'rmse': float(np.sqrt(np.mean(g['error'] ** 2))),
                })
    return pd.DataFrame(rows)


def bootstrap_summary(drops_per_prompt: np.ndarray, mask_matrix: np.ndarray, head_records: Sequence[HeadRecord], models: Sequence[str],
                      repeats: int, k_folds: int, ridge: float, seed: int, eval_nonclean: bool):
    rng = np.random.default_rng(seed)
    n_prompts = drops_per_prompt.shape[0]
    rows = []
    for b in range(repeats):
        idx = rng.integers(0, n_prompts, size=n_prompts)
        yb = drops_per_prompt[idx].mean(axis=0)
        for m in models:
            _, metrics = kfold_predict(mask_matrix, yb, head_records, m, ridge=ridge, k_folds=k_folds, seed=seed + 17, eval_nonclean=eval_nonclean)
            metrics['bootstrap'] = b
            rows.append(metrics)
    df = pd.DataFrame(rows)
    out = []
    for model, g in df.groupby('model'):
        row = {'model': model, 'n_bootstrap': len(g)}
        for col in ['mae', 'rmse', 'r2']:
            vals = g[col].to_numpy(float)
            row[f'{col}_mean'] = float(np.nanmean(vals))
            row[f'{col}_median'] = float(np.nanmedian(vals))
            row[f'{col}_std'] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f'{col}_q05'] = float(np.nanquantile(vals, 0.05))
            row[f'{col}_q95'] = float(np.nanquantile(vals, 0.95))
        out.append(row)
    return df, pd.DataFrame(out)


def plot_prediction_scatter(pred_df: pd.DataFrame, outpath: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    markers = {'additive_head': 'o', 'pb_group_interaction': 's', 'group_interaction': '^', 'head_pair_sparse_ridge': 'x'}
    for model, g in pred_df.groupby('model'):
        ax.scatter(g['observed'], g['predicted'], s=35, marker=markers.get(model, 'o'), alpha=0.75, label=model)
    lo = min(pred_df['observed'].min(), pred_df['predicted'].min())
    hi = max(pred_df['observed'].max(), pred_df['predicted'].max())
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle='--', linewidth=1)
    ax.set_xlabel('observed held-out drop in IOI logit diff')
    ax.set_ylabel('predicted held-out drop')
    ax.set_title('IOI Stage 2b: head-level subset prediction')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_mae_bar(summary: pd.DataFrame, outpath: Path):
    import matplotlib.pyplot as plt
    df = summary.copy().sort_values('mae')
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(df['model'], df['mae'])
    ax.set_ylabel('held-out MAE')
    ax.set_title('Head-level subset prediction error')
    ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_bootstrap_mae(boot_summary: pd.DataFrame, outpath: Path):
    import matplotlib.pyplot as plt
    df = boot_summary.copy().sort_values('mae_median')
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    y = df['mae_median'].to_numpy(float)
    yerr = np.vstack([y - df['mae_q05'].to_numpy(float), df['mae_q95'].to_numpy(float) - y])
    ax.bar(x, y, yerr=yerr, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], rotation=25, ha='right')
    ax.set_ylabel('bootstrap held-out MAE')
    ax.set_title('Bootstrap error across prompt resamples')
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description='IOI Stage 2b: head-level subset-prediction benchmark')
    parser.add_argument('--outdir', type=str, default='runs/ioi_stage2b_head_subset')
    parser.add_argument('--model', type=str, default='gpt2-small')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--n-prompts', type=int, default=256)
    parser.add_argument('--n-reference', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--template', type=str, default=DEFAULT_TEMPLATE)
    parser.add_argument('--names', type=str, default='')
    parser.add_argument('--primary-heads', type=str, default=DEFAULT_PRIMARY_HEADS)
    parser.add_argument('--backup-heads', type=str, default=DEFAULT_BACKUP_HEADS)
    parser.add_argument('--extra-heads', type=str, default=DEFAULT_EXTRA_HEADS)
    parser.add_argument('--ablation', type=str, default='mean', choices=['mean', 'zero'])
    parser.add_argument('--positions', type=str, default='end', choices=['end', 'all'])
    parser.add_argument('--n-subsets', type=int, default=160)
    parser.add_argument('--p-include', type=float, default=0.35)
    parser.add_argument('--k-folds', type=int, default=5)
    parser.add_argument('--ridge', type=float, default=1e-6)
    parser.add_argument('--bootstrap-repeats', type=int, default=200)
    parser.add_argument('--eval-nonclean-only', action='store_true', default=True)
    parser.add_argument('--include-clean-in-eval', action='store_true', help='Include clean row in R2/MAE evaluation')
    parser.add_argument('--include-head-pairs', action='store_true', help='Diagnostic model with all head-pair terms and ridge')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args(argv)

    if args.quick:
        args.n_prompts = min(args.n_prompts, 64)
        args.n_reference = min(args.n_reference, 128)
        args.n_subsets = min(args.n_subsets, 48)
        args.bootstrap_repeats = min(args.bootstrap_repeats, 20)

    if args.include_clean_in_eval:
        args.eval_nonclean_only = False

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    torch.set_grad_enabled(False)

    print(f'Loading {args.model} on {device}...')
    model = load_model(args.model, device)

    primary = parse_heads(args.primary_heads)
    backup = parse_heads(args.backup_heads)
    extra = parse_heads(args.extra_heads)
    head_records = build_head_records(primary, backup, extra)
    pd.DataFrame([asdict(h) for h in head_records]).to_csv(outdir / 'ioi_stage2b_head_records.csv', index=False)

    names = [x.strip() for x in args.names.split(',') if x.strip()] if args.names else None
    records = make_prompts(model, args.n_prompts, args.seed, args.template, names=names)
    ref_records = make_prompts(model, args.n_reference, args.seed + 10_000, args.template, names=names)
    tokens = tokens_for_records(model, records, device)
    ref_tokens = tokens_for_records(model, ref_records, device)
    if tokens.shape[1] != ref_tokens.shape[1]:
        raise RuntimeError(f'Eval token length {tokens.shape[1]} != reference token length {ref_tokens.shape[1]}')
    io_tokens, s_tokens = answer_token_tensors(records, device)
    pd.DataFrame([asdict(r) for r in records]).to_csv(outdir / 'ioi_stage2b_prompts.csv', index=False)

    mean_z = compute_mean_z(model, ref_tokens, args.batch_size) if args.ablation == 'mean' else None

    mask_matrix = sample_subset_masks(head_records, n_subsets=args.n_subsets, seed=args.seed + 333, p_include=args.p_include)
    subset_rows = []
    for idx, mask in enumerate(mask_matrix):
        g = group_counts(mask, head_records)
        subset_rows.append({
            'subset_idx': idx,
            'subset_name': mask_name(mask, head_records),
            'mask_bits': ''.join(str(int(x)) for x in mask),
            'heads': format_heads(mask_to_heads(mask, head_records)),
            'n_heads': int(mask.sum()),
            'n_P': g['P'], 'n_B': g['B'], 'n_E': g['E'],
            'has_P': int(g['P'] > 0), 'has_B': int(g['B'] > 0), 'has_E': int(g['E'] > 0),
            'P_B': int(g['P'] > 0 and g['B'] > 0),
            'P_E': int(g['P'] > 0 and g['E'] > 0),
            'B_E': int(g['B'] > 0 and g['E'] > 0),
        })
    subset_df = pd.DataFrame(subset_rows)
    subset_df.to_csv(outdir / 'ioi_stage2b_subset_design.csv', index=False)

    # Measure clean and all subset interventions.
    logitdiffs = []
    for idx, mask in enumerate(mask_matrix):
        heads = mask_to_heads(mask, head_records)
        ld = compute_logit_diffs(
            model, tokens, io_tokens, s_tokens, args.batch_size,
            heads=heads, ablation=args.ablation if heads else 'none', mean_z=mean_z, position_mode=args.positions,
        )
        logitdiffs.append(ld)
    logitdiffs = np.stack(logitdiffs, axis=1)
    clean_idx = int(np.where(mask_matrix.sum(axis=1) == 0)[0][0])
    clean = logitdiffs[:, clean_idx]
    drops = clean[:, None] - logitdiffs
    y = drops.mean(axis=0)

    meas_rows = []
    for idx in range(mask_matrix.shape[0]):
        row = subset_df.loc[subset_df.subset_idx == idx].iloc[0].to_dict()
        row.update({
            'ablation': args.ablation,
            'position_mode': args.positions,
            'logit_diff_mean': float(logitdiffs[:, idx].mean()),
            'logit_diff_std': float(logitdiffs[:, idx].std(ddof=1)),
            'drop_from_clean_mean': float(drops[:, idx].mean()),
            'drop_from_clean_std': float(drops[:, idx].std(ddof=1)),
            'n_prompts': int(drops.shape[0]),
        })
        meas_rows.append(row)
    meas_df = pd.DataFrame(meas_rows)
    meas_df.to_csv(outdir / 'ioi_stage2b_subset_measurements.csv', index=False)

    # Also write per-prompt drops in long form for optional downstream analyses.
    long_rows = []
    for prompt_idx, rec in enumerate(records):
        for subset_idx in range(mask_matrix.shape[0]):
            long_rows.append({
                'prompt_idx': prompt_idx,
                'subset_idx': subset_idx,
                'subset_name': subset_df.loc[subset_idx, 'subset_name'],
                'logit_diff': float(logitdiffs[prompt_idx, subset_idx]),
                'drop_from_clean': float(drops[prompt_idx, subset_idx]),
            })
    pd.DataFrame(long_rows).to_csv(outdir / 'ioi_stage2b_per_prompt_drops.csv', index=False)

    models = ['additive_head', 'pb_group_interaction', 'group_interaction']
    if args.include_head_pairs:
        models.append('head_pair_sparse_ridge')

    pred_parts = []
    metric_rows = []
    coef_parts = []
    for m in models:
        p_df, metrics = kfold_predict(
            mask_matrix, y, head_records, m, ridge=args.ridge, k_folds=args.k_folds,
            seed=args.seed + 777, eval_nonclean=args.eval_nonclean_only,
        )
        p_df = annotate_prediction_rows(p_df, subset_df)
        pred_parts.append(p_df)
        metric_rows.append(metrics)
        coef_parts.append(fit_all(mask_matrix, y, head_records, m, ridge=args.ridge))
    pred_df = pd.concat(pred_parts, ignore_index=True)
    fit_summary = pd.DataFrame(metric_rows)
    coef_df = pd.concat(coef_parts, ignore_index=True)
    occupancy_summary = summarize_by_group_occupancy(pred_df)

    pred_df.to_csv(outdir / 'ioi_stage2b_kfold_predictions.csv', index=False)
    fit_summary.to_csv(outdir / 'ioi_stage2b_fit_summary.csv', index=False)
    coef_df.to_csv(outdir / 'ioi_stage2b_coefficients.csv', index=False)
    occupancy_summary.to_csv(outdir / 'ioi_stage2b_group_occupancy_errors.csv', index=False)

    boot_df, boot_summary = bootstrap_summary(
        drops, mask_matrix, head_records, models, repeats=args.bootstrap_repeats,
        k_folds=args.k_folds, ridge=args.ridge, seed=args.seed + 999,
        eval_nonclean=args.eval_nonclean_only,
    )
    boot_df.to_csv(outdir / 'ioi_stage2b_bootstrap_metrics.csv', index=False)
    boot_summary.to_csv(outdir / 'ioi_stage2b_bootstrap_summary.csv', index=False)

    # Diagnostics focused on Stage-1-compatible group effects if whole-group masks are present.
    def find_group_mask(P=False, B=False, E=False):
        target = []
        for h in head_records:
            target.append(int((h.group == 'P' and P) or (h.group == 'B' and B) or (h.group == 'E' and E)))
        t = np.asarray(target, dtype=int)
        hits = np.where((mask_matrix == t).all(axis=1))[0]
        return int(hits[0]) if len(hits) else None

    diag = {
        'ablation': args.ablation,
        'position_mode': args.positions,
        'n_subsets': int(mask_matrix.shape[0]),
        'n_heads': int(mask_matrix.shape[1]),
        'n_prompts': int(drops.shape[0]),
        'eval_nonclean_only': bool(args.eval_nonclean_only),
        'clean_logit_diff': float(clean.mean()),
        'models': models,
    }
    idx_P = find_group_mask(P=True)
    idx_B = find_group_mask(B=True)
    idx_E = find_group_mask(E=True)
    idx_PB = find_group_mask(P=True, B=True)
    idx_PE = find_group_mask(P=True, E=True)
    idx_BE = find_group_mask(B=True, E=True)
    idx_PBE = find_group_mask(P=True, B=True, E=True)
    for label, idx in [('P', idx_P), ('B', idx_B), ('E', idx_E), ('P+B', idx_PB), ('P+E', idx_PE), ('B+E', idx_BE), ('P+B+E', idx_PBE)]:
        if idx is not None:
            diag[f'drop_{label}'] = float(y[idx])
    if idx_P is not None and idx_B is not None and idx_PB is not None:
        interaction = float(y[idx_PB] - y[idx_P] - y[idx_B])
        diag['pb_interaction_group_mask'] = interaction
        diag['pb_interaction_fraction_of_joint_group_mask'] = interaction / max(abs(float(y[idx_PB])), 1e-12)
        diag['backup_effect_primary_intact_group_mask'] = float(y[idx_B])
        diag['backup_effect_primary_ablated_group_mask'] = float(y[idx_PB] - y[idx_P])
        diag['backup_conditional_amplification_group_mask'] = float((y[idx_PB] - y[idx_P]) / max(abs(y[idx_B]), 1e-12))

    # Primary comparison summaries.
    def get_mae(m):
        return float(fit_summary.loc[fit_summary.model == m, 'mae'].iloc[0])
    diag['additive_head_mae'] = get_mae('additive_head')
    diag['group_interaction_mae'] = get_mae('group_interaction')
    diag['pb_group_interaction_mae'] = get_mae('pb_group_interaction')
    diag['group_interaction_improves_over_additive'] = bool(diag['group_interaction_mae'] < diag['additive_head_mae'])
    diag['pb_group_interaction_improves_over_additive'] = bool(diag['pb_group_interaction_mae'] < diag['additive_head_mae'])
    diag['mae_reduction_group_vs_additive'] = float((diag['additive_head_mae'] - diag['group_interaction_mae']) / max(diag['additive_head_mae'], 1e-12))

    with open(outdir / 'ioi_stage2b_diagnostics.json', 'w') as f:
        json.dump(diag, f, indent=2)

    metadata = {
        'args': vars(args),
        'device': device,
        'model': args.model,
        'primary_heads': primary,
        'backup_heads': backup,
        'extra_heads': extra,
        'sign_convention': 'effects are positive drops: clean_logit_diff - ablated_logit_diff',
        'primary_comparison': 'per-head additive observer vs per-head-plus-group-interaction observer on held-out random head subsets',
        'note': 'This is the powered Stage-2 design. The earlier 8-condition group-subset task is only a diagnostic panel.',
    }
    with open(outdir / 'ioi_stage2b_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    plot_prediction_scatter(pred_df, outdir / 'ioi_stage2b_prediction_scatter.png')
    plot_mae_bar(fit_summary, outdir / 'ioi_stage2b_mae_bar.png')
    plot_bootstrap_mae(boot_summary, outdir / 'ioi_stage2b_bootstrap_mae.png')

    # Markdown report.
    report = []
    report.append('# IOI Stage 2b head-level subset-prediction report\n')
    report.append('This is the powered Stage-2 design: random subsets of individual heads, not only 8 group-level subsets.\n')
    report.append('## Setup\n')
    setup_df = pd.DataFrame([{
        'n_subsets': mask_matrix.shape[0], 'n_heads': mask_matrix.shape[1], 'n_prompts': drops.shape[0],
        'ablation': args.ablation, 'position_mode': args.positions, 'k_folds': args.k_folds,
        'bootstrap_repeats': args.bootstrap_repeats,
    }])
    report.append(setup_df.to_markdown(index=False))
    report.append('\n\n## K-fold held-out fit summary\n')
    report.append(fit_summary.to_markdown(index=False))
    report.append('\n\n## Bootstrap summary\n')
    report.append(boot_summary.to_markdown(index=False))
    report.append('\n\n## Group-mask diagnostics\n')
    diag_keys = [k for k in diag.keys() if k.startswith('drop_') or k.endswith('group_mask') or k in ['pb_interaction_group_mask','pb_interaction_fraction_of_joint_group_mask','backup_conditional_amplification_group_mask','mae_reduction_group_vs_additive']]
    report.append(pd.DataFrame([{k: diag[k] for k in diag_keys if k in diag}]).to_markdown(index=False))
    report.append('\n\n## Occupancy-sliced errors\n')
    report.append(occupancy_summary.to_markdown(index=False))
    (outdir / 'ioi_stage2b_report.md').write_text('\n'.join(report))

    print('\n=== IOI Stage 2b fit summary ===')
    print(fit_summary.to_string(index=False))
    print('\n=== Bootstrap summary ===')
    print(boot_summary.to_string(index=False))
    print('\n=== Diagnostics ===')
    print(json.dumps(diag, indent=2))
    print('\nWrote outputs to', outdir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
