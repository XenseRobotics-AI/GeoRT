import unittest
import numpy as np
from geort.stage_metrics import summarize


class StageMetricsTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(0)
        self.q=rng.uniform(.1,.3,(20,20));self.tips=rng.uniform(0,.1,(20,5,3));self.human=rng.uniform(0,.1,(20,21,3))

    def test_paired_identity_preserves_behavior(self):
        r=summarize(self.q,self.tips,self.q,self.tips,self.human)
        self.assertEqual(r['tip_shift_mean_mm'],0)
        self.assertTrue(r['passes_observed_gates'])

    def test_error_transfer_is_not_accepted(self):
        parent=self.q.copy();parent[:,-1]=-1
        candidate=self.q.copy();candidate[:,-4]=-1
        r=summarize(candidate,self.tips,parent,self.tips,self.human)
        self.assertTrue(r['gates']['backbend_reduction_80pct'])
        self.assertFalse(r['gates']['no_mcp_increase_5pp'])
        self.assertFalse(r['passes_observed_gates'])

    def test_frozen_output_cannot_pass(self):
        tips=np.repeat(self.tips[:1],len(self.tips),axis=0)
        r=summarize(self.q,tips,self.q,self.tips,self.human)
        self.assertFalse(r['gates']['motion_amplitude_retained'])

if __name__=='__main__':unittest.main()
