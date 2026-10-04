# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Recompute the frozen comparison and post-hoc R2, without model inference."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'packet'))
from state import Store
from collect import jobs, make_commitment, results, matrix
from replay_inputs import load_inputs

def compare(a, b):
    if isinstance(b, dict):
        assert a.keys() == b.keys()
        for key in b: compare(a[key], b[key])
    elif isinstance(b, list):
        assert len(a) == len(b)
        for x,y in zip(a,b): compare(x,y)
    elif isinstance(b, (int,float)) and not isinstance(b,bool):
        np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-11)
    else: assert a == b, (a,b)

def main():
    binding = 'be0d5ae8d740612c1279cadb4093efdeffe0a788b2185555f7c7485933b45771'
    store = Store(ROOT / 'checkpoints', binding)
    config, designs, groups = load_inputs()
    plan = jobs(config, designs, groups)
    assert [e['job'] for e in store.index['entries']] == [j['id'] for j in plan]
    assert len(plan) == 2618
    saved, committed = store.get('commitment')
    commitment = json.loads(str(saved['document']))
    compare(make_commitment(store, config, designs), commitment)
    for job in plan:
        arrays, meta = store.get(job['id'])
        assert meta['specification'] == job
        assert meta['direction_sha256'] == (None if job['id']=='directions' else store.entries['directions']['sha256'])
        assert all(np.isfinite(a).all() for a in arrays.values() if a.dtype.kind in 'fc')
        if job['phase']=='heldout': assert meta['saved_unix'] > committed['saved_unix']
    result = results(store, groups)
    original = json.loads((ROOT/'analysis/results.json').read_text())
    compare(result, original)
    target = matrix(store,'test-menu',64,96)-matrix(store,'test-clean',1,96)
    variance = np.var(target.mean(axis=1))
    posthoc = json.loads((ROOT/'analysis/posthoc_descriptives_2026_10_04.json').read_text())
    r2 = {(r['method'],r['seed'],r['budget']):r['heldout_mean_response_r2'] for r in posthoc['rows']}
    for row in result['rows']:
        if row['predictions'] is not None:
            compare(1-row['metrics']['mse']/variance, r2[row['method'],row['seed'],row['budget']])
    summary = {method:float(np.mean([v for (m,s,k),v in r2.items() if m==method and k==32]))
               for method in ['coordinate_ridge','aggregate_ridge','aggregate_omp']}
    print(json.dumps({'checkpoints':len(plan),'committed_fits_and_choices_reproduced':True,
                      'commitment_precedes_all_heldout_blocks':True,'full_frozen_analysis_reproduced':True,
                      'posthoc_r2_rows':len(r2),'mean_r2_at_32_plus_8':summary,
                      'primary':result['primary'],'new_model_calls':0},indent=2))

if __name__=='__main__': main()
