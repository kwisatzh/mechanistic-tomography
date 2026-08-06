# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""
IOI Stage 2d: per-pair decomposition with count-additive control.

This script post-processes a Stage 2c primary-stratified IOI run. It does not
rerun GPT-2. It loads per-prompt drops for many head-level subsets and fits:

    additive_head
    count_additive
    count_additive + P×B count
    count_additive + P×E count
    count_additive + B×E count
    count_additive + all three pair count terms

The goal is to attribute the Stage 2c bundled group-count win to the specific
interaction terms that earn it, while controlling for nonlinear main effects of
within-group count.

Effects are positive drops in IOI logit difference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_mask_bits(bits, n_heads: int) -> np.ndarray:
    s = str(bits)
    if s.endswith('.0'):
        s = s[:-2]
    # Pandas may read the all-zero mask as "0" and may drop leading zeros.
    s = s.zfill(n_heads)
    if len(s) > n_heads:
        s = s[-n_heads:]
    return np.asarray([int(ch) for ch in s], dtype=int)


def load_stage2c(input_run: Path):
    subset_path = input_run / 'ioi_stage2c_subset_design.csv'
    heads_path = input_run / 'ioi_stage2c_head_records.csv'
    drops_path = input_run / 'ioi_stage2c_per_prompt_drops.csv'
    missing = [p for p in [subset_path, heads_path, drops_path] if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing required files: ' + ', '.join(str(p) for p in missing))
    heads = pd.read_csv(heads_path)
    n_heads = len(heads)
    subset = pd.read_csv(subset_path, dtype={'mask_bits': str})
    subset = subset.sort_values('subset_idx').reset_index(drop=True)
    if subset['subset_idx'].tolist() != list(range(len(subset))):
        raise ValueError('subset_idx must be contiguous from 0')
    masks = np.stack([parse_mask_bits(x, n_heads) for x in subset['mask_bits']], axis=0)
    # Fill/verify counts in case the stage-2c file changed.
    groups = heads['group'].tolist()
    for g in ['P','B','E']:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        subset[f'n_{g}'] = masks[:, idx].sum(axis=1)
        subset[f'has_{g}'] = (subset[f'n_{g}'] > 0).astype(int)
    nPtot = max(1, int((heads.group == 'P').sum()))
    nBtot = max(1, int((heads.group == 'B').sum()))
    nEtot = max(1, int((heads.group == 'E').sum()))
    subset['P_B_count'] = (subset['n_P'] / nPtot) * (subset['n_B'] / nBtot)
    subset['P_E_count'] = (subset['n_P'] / nPtot) * (subset['n_E'] / nEtot)
    subset['B_E_count'] = (subset['n_B'] / nBtot) * (subset['n_E'] / nEtot)
    subset['P_B'] = ((subset['n_P'] > 0) & (subset['n_B'] > 0)).astype(int)
    subset['P_E'] = ((subset['n_P'] > 0) & (subset['n_E'] > 0)).astype(int)
    subset['B_E'] = ((subset['n_B'] > 0) & (subset['n_E'] > 0)).astype(int)

    long = pd.read_csv(drops_path)
    mat = long.pivot_table(index='prompt_idx', columns='subset_idx', values='drop_from_clean', aggfunc='first')
    mat = mat.reindex(columns=list(range(len(subset))))
    if mat.isnull().any().any():
        raise ValueError('per-prompt drop matrix contains NaNs after pivot')
    drops = mat.to_numpy(float)
    y = drops.mean(axis=0)
    return heads, subset, masks, drops, y


def count_bin_features(nP: int, nB: int, nE: int) -> Dict[str, float]:
    """Nonlinear within-group count controls, no cross terms."""
    feats = {}
    # P has 3 heads; include exact nonzero bins.  The all-zero bin is baseline.
    for k in [1,2,3]:
        feats[f'P_count_eq_{k}'] = float(nP == k)
    # B has 8 heads; bins match the Stage-2c stratification roughly.
    b_bins = {
        'B_count_eq_1': (nB == 1),
        'B_count_2_3': (2 <= nB <= 3),
        'B_count_4_5': (4 <= nB <= 5),
        'B_count_6_8': (6 <= nB <= 8),
    }
    feats.update({k: float(v) for k, v in b_bins.items()})
    # E has 2 heads.
    for k in [1,2]:
        feats[f'E_count_eq_{k}'] = float(nE == k)
    return feats


def build_design(masks: np.ndarray, heads: pd.DataFrame, subset: pd.DataFrame, model: str):
    head_labels = heads['label'].tolist()
    cols: List[str] = ['intercept'] + head_labels
    count_cols = ['P_count_eq_1','P_count_eq_2','P_count_eq_3','B_count_eq_1','B_count_2_3','B_count_4_5','B_count_6_8','E_count_eq_1','E_count_eq_2']
    pair_cols = []

    if model == 'additive_head':
        pass
    elif model == 'count_additive':
        cols += count_cols
    elif model == 'count_plus_PB_count':
        cols += count_cols + ['P_B_count']
    elif model == 'count_plus_PE_count':
        cols += count_cols + ['P_E_count']
    elif model == 'count_plus_BE_count':
        cols += count_cols + ['B_E_count']
    elif model == 'count_plus_all_pairs':
        cols += count_cols + ['P_B_count','P_E_count','B_E_count']
    else:
        raise ValueError(model)

    rows = []
    for row_idx, mask in enumerate(masks):
        nP = int(subset.loc[row_idx, 'n_P'])
        nB = int(subset.loc[row_idx, 'n_B'])
        nE = int(subset.loc[row_idx, 'n_E'])
        feats: Dict[str, float] = {'intercept': 1.0}
        for bit, label in zip(mask, head_labels):
            feats[label] = float(bit)
        feats.update(count_bin_features(nP, nB, nE))
        feats['P_B_count'] = float(subset.loc[row_idx, 'P_B_count'])
        feats['P_E_count'] = float(subset.loc[row_idx, 'P_E_count'])
        feats['B_E_count'] = float(subset.loc[row_idx, 'B_E_count'])
        rows.append([feats.get(c, 0.0) for c in cols])
    return np.asarray(rows, dtype=float), cols


def ridge_fit(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    reg = ridge * np.eye(X.shape[1], dtype=float)
    reg[0,0] = 0.0
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


def kfold_indices(n: int, k: int, seed: int, protect_clean: bool = True):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if protect_clean:
        nonclean = idx[1:]
        rng.shuffle(nonclean)
        folds = np.array_split(nonclean, k)
        out = []
        for test in folds:
            train = np.setdiff1d(idx, test)
            out.append((train, test))
        return out
    rng.shuffle(idx)
    chunks = np.array_split(idx, k)
    return [(np.setdiff1d(np.arange(n), test), test) for test in chunks]


def eval_predictions(y, preds, eval_nonclean=True):
    mask = np.isfinite(preds)
    if eval_nonclean:
        mask[0] = False
    yy = y[mask]
    pp = preds[mask]
    mae = float(np.mean(np.abs(pp-yy)))
    rmse = float(np.sqrt(np.mean((pp-yy)**2)))
    denom = float(np.sum((yy-yy.mean())**2))
    r2 = float(1.0 - np.sum((pp-yy)**2)/denom) if denom > 1e-12 else float('nan')
    return mae, rmse, r2


def kfold_predict(masks, heads, subset, y, model, ridge=1e-6, k_folds=5, seed=0, eval_nonclean=True):
    X, cols = build_design(masks, heads, subset, model)
    preds = np.full(len(y), np.nan, dtype=float)
    rows = []
    for fold, (train, test) in enumerate(kfold_indices(len(y), k_folds, seed=seed, protect_clean=True)):
        beta = ridge_fit(X[train], y[train], ridge)
        p = X[test] @ beta
        preds[test] = p
        for idx, pred in zip(test, p):
            rows.append({'model': model, 'fold': fold, 'subset_idx': int(idx), 'observed': float(y[idx]), 'predicted': float(pred), 'error': float(pred-y[idx])})
    mae, rmse, r2 = eval_predictions(y, preds, eval_nonclean=eval_nonclean)
    return pd.DataFrame(rows), {'model': model, 'n_rows': int(np.isfinite(preds[1:] if eval_nonclean else preds).sum()), 'mae': mae, 'rmse': rmse, 'r2': r2, 'columns': ','.join(cols), 'n_params': len(cols)}


def fit_all(masks, heads, subset, y, model, ridge=1e-6):
    X, cols = build_design(masks, heads, subset, model)
    beta = ridge_fit(X, y, ridge)
    return pd.DataFrame({'model': model, 'term': cols, 'coef': beta})


def annotate(pred, subset):
    meta = subset[['subset_idx','subset_name','n_heads','n_P','n_B','n_E','has_P','has_B','has_E','P_B','P_E','B_E','P_B_count','P_E_count','B_E_count']].copy()
    return pred.merge(meta, on='subset_idx', how='left')


def bootstrap(drops, masks, heads, subset, models, repeats, k_folds, ridge, seed, eval_nonclean=True):
    rng = np.random.default_rng(seed)
    n_prompts = drops.shape[0]
    rows = []
    for b in range(repeats):
        idx = rng.integers(0, n_prompts, size=n_prompts)
        yb = drops[idx].mean(axis=0)
        for m in models:
            _, metrics = kfold_predict(masks, heads, subset, yb, m, ridge=ridge, k_folds=k_folds, seed=seed+17, eval_nonclean=eval_nonclean)
            metrics['bootstrap'] = b
            rows.append(metrics)
    boot = pd.DataFrame(rows)
    summary_rows = []
    for model, g in boot.groupby('model'):
        row = {'model': model, 'n_bootstrap': len(g)}
        for col in ['mae','rmse','r2']:
            vals = g[col].to_numpy(float)
            row[f'{col}_mean'] = float(np.nanmean(vals))
            row[f'{col}_median'] = float(np.nanmedian(vals))
            row[f'{col}_std'] = float(np.nanstd(vals, ddof=1)) if len(vals)>1 else 0.0
            row[f'{col}_q05'] = float(np.nanquantile(vals, 0.05))
            row[f'{col}_q95'] = float(np.nanquantile(vals, 0.95))
        summary_rows.append(row)
    return boot, pd.DataFrame(summary_rows)


def paired_delta(boot, baseline):
    piv = boot.pivot_table(index='bootstrap', columns='model', values='mae', aggfunc='first')
    if baseline not in piv.columns:
        return pd.DataFrame()
    rows = []
    for m in piv.columns:
        if m == baseline:
            continue
        d = (piv[baseline] - piv[m]).dropna().to_numpy(float)
        rows.append({
            'baseline': baseline, 'model': m, 'n_bootstrap': int(len(d)),
            'delta_mae_mean': float(np.mean(d)),
            'delta_mae_median': float(np.median(d)),
            'delta_mae_std': float(np.std(d, ddof=1)) if len(d)>1 else 0.0,
            'delta_mae_q05': float(np.quantile(d,0.05)),
            'delta_mae_q95': float(np.quantile(d,0.95)),
            'p_delta_gt_0': float(np.mean(d>0)),
            'strict_success_q05_gt_0': bool(np.quantile(d,0.05) > 0),
            'weak_success_p_gt_0_ge_0.95': bool(np.mean(d>0) >= 0.95),
        })
    return pd.DataFrame(rows)


def plot_bar(summary, outpath, y_col='mae', title='Held-out MAE'):
    import matplotlib.pyplot as plt
    df = summary.copy().sort_values(y_col)
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.bar(df['model'], df[y_col])
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_bootstrap_mae(boot_summary, outpath):
    import matplotlib.pyplot as plt
    df = boot_summary.copy().sort_values('mae_median')
    x = np.arange(len(df))
    y = df['mae_median'].to_numpy(float)
    yerr = np.vstack([y-df['mae_q05'].to_numpy(float), df['mae_q95'].to_numpy(float)-y])
    fig, ax = plt.subplots(figsize=(8,4.6))
    ax.bar(x, y, yerr=yerr, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], rotation=25, ha='right')
    ax.set_ylabel('bootstrap held-out MAE')
    ax.set_title('Bootstrap error across prompt resamples')
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_delta(delta, outpath, title):
    import matplotlib.pyplot as plt
    if delta.empty: return
    df = delta.copy().sort_values('delta_mae_median', ascending=False)
    x = np.arange(len(df))
    y = df['delta_mae_median'].to_numpy(float)
    yerr = np.vstack([y-df['delta_mae_q05'].to_numpy(float), df['delta_mae_q95'].to_numpy(float)-y])
    fig, ax = plt.subplots(figsize=(8,4.6))
    ax.axhline(0, linestyle='--', linewidth=1)
    ax.bar(x, y, yerr=yerr, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], rotation=25, ha='right')
    ax.set_ylabel('paired Delta MAE (positive = better)')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_scatter(pred, outpath):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5,5.2))
    markers = {
        'additive_head':'o', 'count_additive':'s', 'count_plus_PB_count':'^',
        'count_plus_PE_count':'v', 'count_plus_BE_count':'P', 'count_plus_all_pairs':'X'
    }
    for model, g in pred.groupby('model'):
        ax.scatter(g['observed'], g['predicted'], s=28, alpha=0.65, marker=markers.get(model,'o'), label=model)
    lo = min(pred['observed'].min(), pred['predicted'].min())
    hi = max(pred['observed'].max(), pred['predicted'].max())
    pad = 0.05*(hi-lo+1e-9)
    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], linestyle='--', linewidth=1)
    ax.set_xlabel('observed held-out drop')
    ax.set_ylabel('predicted held-out drop')
    ax.set_title('IOI Stage 2d per-pair decomposition')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description='IOI Stage 2d per-pair decomposition from a Stage 2c run')
    ap.add_argument('--input-run', type=str, required=True, help='Directory containing Stage 2c outputs')
    ap.add_argument('--outdir', type=str, default='runs/ioi_stage2d_per_pair')
    ap.add_argument('--k-folds', type=int, default=5)
    ap.add_argument('--bootstrap-repeats', type=int, default=300)
    ap.add_argument('--ridge', type=float, default=1e-6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-nonclean-only', action='store_true', default=True)
    args = ap.parse_args(argv)

    input_run = Path(args.input_run)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    heads, subset, masks, drops, y = load_stage2c(input_run)
    heads.to_csv(outdir/'ioi_stage2d_head_records.csv', index=False)
    subset.to_csv(outdir/'ioi_stage2d_subset_design.csv', index=False)

    models = [
        'additive_head',
        'count_additive',
        'count_plus_PB_count',
        'count_plus_PE_count',
        'count_plus_BE_count',
        'count_plus_all_pairs',
    ]

    pred_parts=[]; metrics=[]; coef_parts=[]
    for m in models:
        p, met = kfold_predict(masks, heads, subset, y, m, ridge=args.ridge, k_folds=args.k_folds, seed=args.seed+777, eval_nonclean=args.eval_nonclean_only)
        pred_parts.append(annotate(p, subset))
        metrics.append(met)
        coef_parts.append(fit_all(masks, heads, subset, y, m, ridge=args.ridge))
    pred = pd.concat(pred_parts, ignore_index=True)
    fit_summary = pd.DataFrame(metrics)
    coefs = pd.concat(coef_parts, ignore_index=True)

    boot, boot_summary = bootstrap(drops, masks, heads, subset, models, repeats=args.bootstrap_repeats, k_folds=args.k_folds, ridge=args.ridge, seed=args.seed+999, eval_nonclean=args.eval_nonclean_only)
    delta_vs_count = paired_delta(boot, baseline='count_additive')
    delta_vs_add = paired_delta(boot, baseline='additive_head')

    pred.to_csv(outdir/'ioi_stage2d_kfold_predictions.csv', index=False)
    fit_summary.to_csv(outdir/'ioi_stage2d_fit_summary.csv', index=False)
    coefs.to_csv(outdir/'ioi_stage2d_coefficients.csv', index=False)
    boot.to_csv(outdir/'ioi_stage2d_bootstrap_metrics.csv', index=False)
    boot_summary.to_csv(outdir/'ioi_stage2d_bootstrap_summary.csv', index=False)
    delta_vs_count.to_csv(outdir/'ioi_stage2d_paired_delta_vs_count_additive.csv', index=False)
    delta_vs_add.to_csv(outdir/'ioi_stage2d_paired_delta_vs_additive.csv', index=False)

    diag = {
        'input_run': str(input_run),
        'n_subsets': int(len(subset)),
        'n_heads': int(len(heads)),
        'n_prompts': int(drops.shape[0]),
        'models': models,
        'baseline_primary': 'count_additive',
        'primary_success_rule': 'strict_success_q05_gt_0 for paired Delta MAE vs count_additive',
    }
    for _, row in delta_vs_count.iterrows():
        m = row['model']
        diag[f'{m}_delta_mae_mean_vs_count_additive'] = float(row['delta_mae_mean'])
        diag[f'{m}_delta_mae_q05_vs_count_additive'] = float(row['delta_mae_q05'])
        diag[f'{m}_p_delta_gt_0_vs_count_additive'] = float(row['p_delta_gt_0'])
        diag[f'{m}_strict_success_vs_count_additive'] = bool(row['strict_success_q05_gt_0'])
    # Pair coefficient summary in the all-pairs model.
    for term in ['P_B_count','P_E_count','B_E_count']:
        hit = coefs[(coefs.model=='count_plus_all_pairs') & (coefs.term==term)]
        if not hit.empty:
            diag[f'coef_all_pairs_{term}'] = float(hit.coef.iloc[0])
    with open(outdir/'ioi_stage2d_diagnostics.json','w') as f:
        json.dump(diag, f, indent=2)

    plot_bar(fit_summary, outdir/'ioi_stage2d_mae_bar.png', title='K-fold held-out MAE')
    plot_bootstrap_mae(boot_summary, outdir/'ioi_stage2d_bootstrap_mae.png')
    plot_delta(delta_vs_count, outdir/'ioi_stage2d_paired_delta_vs_count_additive.png', 'Paired improvement over count-additive control')
    plot_delta(delta_vs_add, outdir/'ioi_stage2d_paired_delta_vs_additive.png', 'Paired improvement over additive-head baseline')
    plot_scatter(pred, outdir/'ioi_stage2d_prediction_scatter.png')

    report=[]
    report.append('# IOI Stage 2d per-pair decomposition\n')
    report.append('This is a post-processing analysis of a Stage 2c primary-stratified run. It adds a count-additive control and decomposes the bundled group-count interaction into single-pair terms.\n')
    report.append('## Setup\n')
    report.append(pd.DataFrame([{'input_run': str(input_run), 'n_subsets': len(subset), 'n_heads': len(heads), 'n_prompts': drops.shape[0], 'k_folds': args.k_folds, 'bootstrap_repeats': args.bootstrap_repeats}]).to_markdown(index=False))
    report.append('\n\n## K-fold held-out fit summary\n')
    report.append(fit_summary.to_markdown(index=False))
    report.append('\n\n## Bootstrap summary\n')
    report.append(boot_summary.to_markdown(index=False))
    report.append('\n\n## Paired bootstrap improvement vs count-additive control\n')
    report.append(delta_vs_count.to_markdown(index=False))
    report.append('\n\n## Paired bootstrap improvement vs additive-head baseline\n')
    report.append(delta_vs_add.to_markdown(index=False))
    report.append('\n\n## Coefficients for pair terms\n')
    pair_terms = coefs[coefs.term.isin(['P_B_count','P_E_count','B_E_count'])]
    report.append(pair_terms.to_markdown(index=False))
    report.append('\n\n## Reading guide\n')
    report.append('- If `count_additive` is close to `count_plus_all_pairs`, the Stage 2c win was mostly within-group count curvature, not cross-group interaction.\n')
    report.append('- If a single-pair model has positive paired Delta MAE with q05 > 0 versus count-additive, that pair earns a robust predictive gain.\n')
    report.append('- The self-repair-specific term is `count_plus_PB_count`. The name-mover/negative-name-mover cancellation term is `count_plus_PE_count`.\n')
    (outdir/'ioi_stage2d_report.md').write_text('\n'.join(report))

    print('\n=== Fit summary ===')
    print(fit_summary.to_string(index=False))
    print('\n=== Bootstrap summary ===')
    print(boot_summary.to_string(index=False))
    print('\n=== Paired Delta MAE vs count_additive ===')
    print(delta_vs_count.to_string(index=False))
    print('\n=== Diagnostics ===')
    print(json.dumps(diag, indent=2))
    print('\nWrote outputs to', outdir)


if __name__ == '__main__':
    main()
