import unittest
import numpy as np
from geort.branch_selection import geometric_neighbors, select_branches
from geort.prepare_consistent_targets import quality_mask, posture_eligibility


class BranchSelectionTests(unittest.TestCase):
    def test_consistent_branch_selection_reduces_energy_without_averaging(self):
        q=np.array([[[0.],[1.]],[[0.],[1.]],[[0.],[1.]]])
        unary=np.array([[0.,.01],[.01,0.],[0.,.01]])
        graph=[[(1,1.)],[(0,1.),(2,1.)],[(1,1.)]]
        labels,r=select_branches(q,unary,graph,strength=.1)
        self.assertEqual(len(set(labels)),1)
        self.assertLess(r['final_energy'],r['initial_energy'])
        self.assertTrue(all(a>=b-1e-9 for a,b in zip(r['energy_trace'],r['energy_trace'][1:])))
        unary[1,0]=np.inf
        labels,_=select_branches(q,unary,graph,strength=.1)
        self.assertEqual(labels[1],1)

    def test_float32_inputs_preserve_monotonic_energy(self):
        rng=np.random.default_rng(42);q=rng.normal(size=(30,5,4)).astype('float32')
        unary=rng.random((30,5)).astype('float32')
        graph=[[(j,.1) for j in range(30) if j!=i] for i in range(30)]
        _,r=select_branches(q,unary,graph)
        self.assertLessEqual(r['final_energy'],r['initial_energy'])

    def test_same_tips_different_directions_are_not_connected(self):
        tips=np.zeros((3,3));axes=np.tile([[[1.,0.,0.]]*3],(3,1,1));axes[2]*=-1
        valid=np.ones((3,3),dtype=bool)
        graph=geometric_neighbors(tips,axes,valid)
        self.assertEqual([j for j,w in graph[0]],[1]);self.assertFalse(graph[2])
        valid[1,0]=False
        self.assertFalse(any(geometric_neighbors(tips,axes,valid)))

    def test_quality_checks_preserve_position_direction_and_mcp(self):
        result=quality_mask(np.array([0.,.003,0.,0.,0.]), np.deg2rad([10,10,40,10,180]),
                            np.deg2rad([15]*5),np.array([1,1,1,1,0],dtype=bool),
                            np.deg2rad([0,0,0,-30,0]),np.deg2rad([-15]*5))
        np.testing.assert_array_equal(result,[True,False,False,False,True])

    def test_posture_supervision_excludes_unresolved_backbend(self):
        ref=np.zeros((2,20));ref[:,16+2]=-1.;ref[:,4]=-.8
        q=ref.copy();q[1,16+2]=.2
        mask=posture_eligibility(q,ref)
        self.assertFalse(mask[0,3]);self.assertTrue(mask[1,3]);self.assertTrue(mask[:,0].all())

    def test_no_valid_candidate_fails(self):
        with self.assertRaises(ValueError):select_branches(np.zeros((1,2,4)),np.array([[np.inf,np.inf]]),[[]])


if __name__=='__main__':unittest.main()
