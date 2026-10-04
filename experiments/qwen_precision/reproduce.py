# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Recompute fixture displacement errors; never load a model or contact a service."""
import hashlib
import json
from pathlib import Path
import numpy as np
from audit import summarize

ROOT = Path(__file__).resolve().parent

def read(name): return json.loads((ROOT/name).read_text())

def main():
    report=read('retained/report.json')
    actions=read('actions.json')
    index=read('retained/index.json')['entries']
    config=read('config.json')
    with np.load(ROOT/'directions.npz',allow_pickle=False) as z: directions=z['directions'].copy()
    errors=[]
    assert len(index)==len(actions)==len(report['records'])==29
    for i,(entry,action,record) in enumerate(zip(index,actions,report['records'])):
        assert entry['file']==f'checkpoints/action-{i:03d}.npz'
        p=ROOT/'retained'/entry['file']
        assert p.stat().st_size==entry['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
        with np.load(p,allow_pickle=False) as z:
            before,after,intended=(z[k].copy() for k in ['before','after','intended'])
            assert before.shape==after.shape==(8,16,directions.shape[1])
            np.testing.assert_array_equal(z['layers'],config['actuator']['layers'])
            np.testing.assert_array_equal(z['action'],action['action'])
            assert z['positions'].shape==(16,) and np.all((z['positions']>=0)&(z['positions']<384))
        np.testing.assert_array_equal(intended,directions[:,None,:]*np.asarray(action['action'])[:,None,None])
        actual=summarize(before,after,intended)
        for key,value in actual.items(): np.testing.assert_allclose(value,record[key],rtol=1e-12,atol=1e-15)
        assert all(record[k]==v for k,v in action.items())
        assert record['off_position_changes']==0 and set(record['activation_dtypes'])=={'torch.bfloat16'}
        errors.extend(actual['relative_error'])
    assert len(errors)==1216
    maximum,median=float(np.max(errors)),float(np.median(errors))
    assert maximum==report['max_relative_error'] and median==report['median_relative_error']
    assert report['passed']==(maximum<=report['tolerance'])==False
    assert report['all_position_checks_pass'] and report['no_leaked_hooks']
    print(json.dumps({'archives':29,'nonzero_edits':len(errors),'median_relative_error':median,
        'max_relative_error':maximum,'tolerance':report['tolerance'],'edit_fidelity_passed':False,
        'reproduction_passed':True,'scope':'Prospective fixtures; historical activations were not retained. Off-position and cleanup checks are worker receipts, not replays of full tensors.',
        'new_model_calls':0},indent=2))

if __name__=='__main__': main()
