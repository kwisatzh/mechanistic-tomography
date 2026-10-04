# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from audit import summarize, observe
from qwen_refusal import QwenRefusalPlant, DirectionBundle


class AuditTests(unittest.TestCase):
    def test_metric_exact_and_bad(self):
        before=np.ones((2,3,4)); intended=np.full((2,1,4),0.1)
        self.assertLess(max(summarize(before,before+intended,intended)['relative_error']),1e-12)
        self.assertEqual(summarize(before,before,np.zeros((2,1,4)))['count'],0)
        self.assertTrue(all(x==1 for x in summarize(before,before,intended)['relative_error']))

    def test_inactive_mutation_and_nonfinite(self):
        for after in [np.ones((1,2,3)),np.full((1,2,3),np.nan)]:
            with self.assertRaises(ValueError):summarize(np.zeros_like(after),after,np.zeros((1,1,3)))

    def test_original_hook_and_cleanup(self):
        class Block(torch.nn.Module):
            def forward(self,x):return x
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.layers=torch.nn.ModuleList([Block(),Block()])
            def forward(self,input_ids,**kwargs):
                h=torch.ones((*input_ids.shape,4),dtype=torch.bfloat16)
                for layer in self.layers:h=layer(h)
                return h
        plant=QwenRefusalPlant.__new__(QwenRefusalPlant)
        plant.torch=torch;plant.layers=(0,1);plant.model=Model();plant.blocks=plant.model.layers
        plant._positions=torch.tensor([1,3]);plant.device=torch.device('cpu')
        d=np.full((2,4),0.025)
        bundle=DirectionBundle((0,1),d,np.ones(2),0.05,'fixture','fixture')
        b,a,dtypes=observe(plant,bundle,np.asarray([1.,0.]),torch.zeros((2,5),dtype=torch.long),None)
        self.assertEqual(b.shape,(2,2,4)); self.assertTrue(np.array_equal(a[1],b[1]))
        self.assertEqual(set(dtypes),{'torch.bfloat16'})
        self.assertGreater(max(summarize(b,a,d[:,None,:]*np.asarray([1.,0.])[:,None,None])['relative_error']),0)
        self.assertTrue(all(not x._forward_hooks for x in plant.blocks))


if __name__=='__main__':unittest.main()
