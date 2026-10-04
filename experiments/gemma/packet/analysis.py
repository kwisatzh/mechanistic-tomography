# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Frozen fitting and evaluation. Fitting accepts no held-out responses."""
import ast
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np

# Load the exact retained estimators without importing their plotting/reporting stack.
_source=Path(__file__).parent/'vendor/sparse_tomography_posthoc.py'
_names={'fit_ridge_centered','fit_omp_centered'}
_nodes=[n for n in ast.parse(_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in _names]
if {n.name for n in _nodes}!=_names:raise ValueError('Retained estimators missing')
exec(compile(ast.Module(body=_nodes,type_ignores=[]),str(_source),'exec'),globals())


def select(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or values[0] != 0:
        raise ValueError('Selection requires zero then finite menu predictions')
    return int(np.argmax(values))


def choose_fit(kind, x, y, xv, yv, config):
    candidates = []
    grid = config['ridge_grid'] if kind == 'ridge' else sorted(set(min(k, len(y)-1) for k in config['omp_support_grid']))
    for p in grid:
        b, intercept, info = (fit_ridge_centered(x, y, p) if kind == 'ridge' else fit_omp_centered(x, y, p))
        mse = float(np.mean((xv@b+intercept-yv)**2))
        candidates.append((mse, -p if kind == 'ridge' else p, b, intercept, info))
    best = min(candidates, key=lambda r: r[:2])
    return best[2], best[3], dict(best[4], validation_mse=best[0])


def commit(config, designs, fit):
    scale = config['primary_scale']; menu = designs['menu']*scale
    xv = designs['validation']*scale; yv = fit['validation'].mean(axis=1)
    rows = []
    def append(method, seed, budget, beta=None, intercept=0., info=None, values=None, charged=None):
        pred = np.r_[0., menu@beta+intercept] if beta is not None else None
        choice = select(pred if pred is not None else values)
        rows.append({'method':method, 'seed':seed, 'budget':budget, 'measurement_cost':charged,
                     'beta':None if beta is None else beta.tolist(), 'intercept':float(intercept),
                     'validation':info, 'predictions':None if pred is None else pred.tolist(), 'selected':choice})
    coordinate_lookup = {tuple(a):i for i,a in enumerate(designs['coordinate_'+str(config['design_seeds'][0])])}
    for seed in config['design_seeds']:
        order = [coordinate_lookup[tuple(a)] for a in designs[f'coordinate_{seed}']]
        yc = fit['coordinate'][order].mean(axis=1)
        ya = fit[f'aggregate_{seed}'].mean(axis=1)
        for budget in config['budgets']:
            xc = designs[f'coordinate_{seed}'][:budget]*scale
            xa = designs[f'aggregate_{seed}'][:budget]*scale
            for method,x,y,kind in [('coordinate_ridge',xc,yc[:budget],'ridge'),('aggregate_ridge',xa,ya[:budget],'ridge'),('aggregate_omp',xa,ya[:budget],'omp')]:
                b,intercept,info = choose_fit(kind,x,y,xv,yv,config)
                append(method,seed,budget,b,intercept,info,charged=budget+len(xv))
            b = np.zeros(menu.shape[1])
            for j in range(0,budget,2):
                b += (yc[j]-yc[j+1])/(2*scale)*designs[f'coordinate_{seed}'][j]
            append('coordinate_central',seed,budget,b,charged=budget)
            search = designs[f'search_order_{seed}']
            for extra in [0,len(xv)]:
                count = min(budget+extra,len(menu))
                # Unmeasured candidates are excluded, not assigned their actual response.
                indices = search[:count]
                values = np.r_[0.,fit['search'][indices].mean(axis=1)]
                best_value = float(values.max())
                chosen = 0 if best_value <= 0 else 1+int(min(indices[values[1:]==best_value]))
                rows.append({'method':'direct_search' if extra==0 else 'direct_search_cost_matched',
                             'seed':seed,'budget':budget,'measurement_cost':count,'beta':None,'intercept':None,
                             'validation':None,'predictions':None,'selected':chosen})
            for method,chosen in [('zero',0),('random',1+int(search[0]))]:
                rows.append({'method':method,'seed':seed,'budget':budget,'measurement_cost':0,
                             'beta':None,'intercept':None,'validation':None,'predictions':None,'selected':chosen})
    return {'schema':1,'scale':scale,'primary_budget':config['primary_budget'],'rows':rows,
            'information_boundary':'Fitting and validation responses only; no held-out or collateral values'}


def metrics(row, target, benign, kl):
    chosen = row['selected']; mean = target.mean(axis=1)
    out = {'target_effect':float(mean[chosen]),'regret':float(mean.max()-mean[chosen]),
           'benign_signed':float(benign[chosen].mean()),'benign_absolute':float(np.abs(benign[chosen]).mean()),
           'benign_kl':float(kl[chosen].mean())}
    if row['predictions'] is not None:
        pred = np.asarray(row['predictions'])[1:]
        out.update(mse=float(np.mean((pred-mean[1:])**2)),mae=float(np.mean(np.abs(pred-mean[1:]))),
                   per_prompt_mse=float(np.mean((pred[:,None]-target[1:])**2)),
                   per_prompt_mae=float(np.mean(np.abs(pred[:,None]-target[1:]))))
    return out


def family_draws(families, rng, repetitions=2000):
    unique = sorted(set(families))
    if len(unique)<2: return None
    groups = [np.flatnonzero(np.asarray(families)==f) for f in unique]
    return [np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]) for _ in range(repetitions)]


def evaluate(commitment, target, benign, kl, target_families, benign_families):
    if not all(np.isfinite(a).all() for a in [target,benign,kl]): raise ValueError('Nonfinite outcomes')
    rows=commitment['rows']; rng=np.random.default_rng(10330)
    draws=family_draws(target_families,rng); bdraws=family_draws(benign_families,rng)
    result=[]
    for row in rows:
        point=metrics(row,target,benign,kl)
        intervals={}
        if draws is not None:
            values=[metrics(row,target[:,d],benign,kl) for d in draws]
            for key in ['target_effect','regret','mse','mae','per_prompt_mse','per_prompt_mae']:
                if key in point: intervals[key]=np.quantile([v[key] for v in values],[.025,.975]).tolist()
        if bdraws is not None:
            values=[metrics(row,target,benign[:,d],kl[:,d]) for d in bdraws]
            for key in ['benign_signed','benign_absolute','benign_kl']:
                intervals[key]=np.quantile([v[key] for v in values],[.025,.975]).tolist()
        result.append(dict(row,metrics=point,family_intervals=intervals))
    primary=[r for r in rows if r['budget']==commitment['primary_budget'] and r['method'] in ['aggregate_ridge','coordinate_ridge']]
    def contrast(indices):
        mean=target[1:,indices].mean(axis=1)
        vals={m:[] for m in ['aggregate_ridge','coordinate_ridge']}
        for r in primary: vals[r['method']].append(np.mean((np.asarray(r['predictions'])[1:]-mean)**2))
        return float(np.mean(vals['aggregate_ridge'])-np.mean(vals['coordinate_ridge']))
    point=contrast(np.arange(target.shape[1]))
    ci=None if draws is None else np.quantile([contrast(d) for d in draws],[.025,.975]).tolist()
    verdict='neutral' if ci is None or ci[0]<=0<=ci[1] else ('positive' if ci[1]<0 else 'negative')
    return {'rows':result,'primary':{'contrast':'aggregate ridge minus coordinate ridge MSE','point':point,'interval':ci,'judgment':verdict},
            'family_counts':{'target':len(set(target_families)),'benign':len(set(benign_families))},
            'scope':'Conditional forward-only measurement comparison; no internal-specific or safety claim',
            'bootstrap_repetitions':2000,'collateral_interval_available':bdraws is not None}
