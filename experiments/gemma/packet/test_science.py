# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""CPU qualification of checkpoint, analysis, and phase boundaries; no model calls."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from state import Store, acknowledge, atomic_json, sha
from analysis import commit, select, metrics, family_draws, choose_fit
from collect import load_inputs, jobs, action_for, divergence, make_commitment, distribution
from design import designs, costs


class Checkpoints(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.store=Store(self.root,'frozen')
    def tearDown(self):self.tmp.cleanup()
    def test_roundtrip_and_resume(self):
        self.store.put('first',{'margin':np.arange(8.)},{'lease':'one'})
        resumed=Store(self.root,'frozen');self.assertTrue(resumed.has('first'))
        np.testing.assert_array_equal(resumed.get('first')[0]['margin'],np.arange(8.))
    def test_immutable(self):
        self.store.put('one',{'x':[1.]},{})
        with self.assertRaises(ValueError):self.store.put('one',{'x':[2.]},{})
    def test_corruption(self):
        self.store.put('one',{'x':[1.]},{});(self.root/'one.npz').write_bytes(b'broken')
        with self.assertRaises(ValueError):Store(self.root,'frozen')
    def test_wrong_freeze(self):
        self.store.put('one',{'x':[1.]},{})
        with self.assertRaises(ValueError):Store(self.root,'other')
    def test_unindexed_output_not_overwritten(self):
        (self.root/'one.npz').write_bytes(b'partial')
        with self.assertRaises(ValueError):self.store.put('one',{'x':[1.]},{})
        self.assertEqual((self.root/'one.npz').read_bytes(),b'partial')
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):self.store.put('one',{'x':[np.nan]},{})
        self.assertFalse(self.store.has('one'))
    def test_path_rejected(self):
        with self.assertRaises(ValueError):self.store.put('../one',{'x':[1.]},{})
    def test_ack_after_host_hash(self):
        entry=self.store.put('one',{'x':[1.]},{})
        atomic_json(self.root/'ack.json',{'job':'one','sha256':entry['sha256']})
        acknowledge(self.store,'one',float('inf'))
    def test_bad_ack_does_not_advance(self):
        self.store.put('one',{'x':[1.]},{})
        atomic_json(self.root/'ack.json',{'job':'one','sha256':'wrong'})
        with patch('state.time.time',side_effect=[0,1,301]),patch('state.time.sleep'):
            with self.assertRaises(TimeoutError):acknowledge(self.store,'one',999)


class Science(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.config,cls.d,cls.groups=load_inputs()
    def test_fixed_prompts_and_designs(self):
        d=designs(self.config)
        for key in d:np.testing.assert_array_equal(d[key],self.d[key])
        self.assertEqual(sha(Path(__file__).parent/'data/prompts.jsonl'),'4c505354d60341f0d24103b75d88c29731fee4ec24d68cb9931402148b893dbf')
    def test_counts_and_cost(self):
        plan=jobs(self.config,self.d,self.groups)
        self.assertEqual(len(plan),2618);self.assertEqual(len({j['id'] for j in plan}),len(plan))
        self.assertEqual(costs(self.config)['total'],150816)
        self.assertEqual(sum(8 for j in plan if j['kind'] in ['kl','kl_clean']),2080)
    def test_information_boundary(self):
        plan=jobs(self.config,self.d,self.groups);c=next(i for i,j in enumerate(plan) if j['kind']=='commitment')
        self.assertTrue(all(j['phase']=='fit' for j in plan[:c+1]))
        self.assertTrue(all(j['phase']=='heldout' for j in plan[c+1:]))
        self.assertEqual(list(inspect.signature(commit).parameters),['config','designs','fit'])
    def test_single_primary_scale(self):
        self.assertEqual(self.config['scales'],[.5]);self.assertEqual(self.config['runtime']['dtype'],'float32')
        for job in jobs(self.config,self.d,self.groups):
            if job.get('pool'):self.assertAlmostEqual(np.linalg.norm(action_for(job,self.config,self.d)),.5)
    def test_ties_and_bad_selection(self):
        self.assertEqual(select([0,0,0]),0);self.assertEqual(select([0,1,1]),1)
        with self.assertRaises(ValueError):select([0,np.nan])
    def test_synthetic_fitting_and_budget(self):
        beta=np.zeros(32);beta[0]=1;beta[4]=-.2
        pools={'coordinate':self.d['coordinate_10301'],'validation':self.d['validation'],'search':self.d['menu'],
               **{f'aggregate_{s}':self.d[f'aggregate_{s}'] for s in self.config['design_seeds']}}
        fit={k:np.repeat((a*.5@beta)[:,None],32,axis=1) for k,a in pools.items()}
        result=commit(self.config,self.d,fit);self.assertEqual(len(result['rows']),96)
        for row in result['rows']:
            self.assertTrue(0<=row['selected']<=64)
            if row['method']=='coordinate_central' and row['budget']==64:
                np.testing.assert_allclose(row['beta'],beta,atol=1e-12)
            if row['method']=='direct_search_cost_matched':
                self.assertEqual(row['measurement_cost'],min(row['budget']+8,64))
                self.assertIn(row['selected'],np.r_[0,self.d[f"search_order_{row['seed']}"][:row['measurement_cost']]+1])
        again=commit(self.config,self.d,fit);self.assertEqual(result,again)
    def test_no_collateral_interval_one_family(self):
        self.assertIsNone(family_draws(['one']*32,np.random.default_rng(1)))
    def test_family_pairing(self):
        a=family_draws(['a','a','b','b'],np.random.default_rng(1),10)
        self.assertEqual(len(a),10)
        for d in a:self.assertEqual(len(d),4);self.assertEqual(np.sum(d==0),np.sum(d==1))
    def test_kl_identity_and_known_value(self):
        p=np.log([[.25,.75]]);q=np.log([[.5,.5]])
        np.testing.assert_allclose(divergence(p,p),[0])
        np.testing.assert_allclose(divergence(p,q),[.25*np.log(.5)+.75*np.log(1.5)])
    def test_regret_and_collateral(self):
        row={'selected':1,'predictions':None}
        value=metrics(row,np.array([[0,0],[1,3],[0,1]]),np.array([[0,0],[1,-1],[0,0]]),np.array([[0,0],[.1,.2],[0,0]]))
        self.assertEqual(value['target_effect'],2);self.assertEqual(value['regret'],0)
        self.assertEqual(value['benign_signed'],0);self.assertEqual(value['benign_absolute'],1)


if __name__=='__main__':unittest.main()
