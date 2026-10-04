# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Run original CPU tests with text-free prompt metadata for the input check."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'packet'))
import test_science
from replay_inputs import load_inputs
from design import designs

class ReleasedScience(test_science.Science):
    @classmethod
    def setUpClass(cls): cls.config,cls.d,cls.groups=load_inputs()

    def test_fixed_prompts_and_designs(self):
        for key,value in designs(self.config).items():
            np.testing.assert_array_equal(value,self.d[key])
        self.assertEqual([len(v) for v in self.groups.values()],[32,32,96,32])
        families=[{r.family for r in v} for v in self.groups.values()]
        for i,a in enumerate(families):
            for b in families[i+1:]: self.assertFalse(a & b)

if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(test_science.Checkpoints),
                             unittest.defaultTestLoader.loadTestsFromTestCase(ReleasedScience)])
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
