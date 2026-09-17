import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from geort.coordination_loss import baseline_from_outputs, baseline_objective
from geort.evaluate_losses import load_probes, sha
from scripts.cache_wuji_probes import restore_history


class LossComparisonTests(unittest.TestCase):
    def test_affine_response_has_zero_flatness_and_best_direction(self):
        torch.manual_seed(42)
        tips=torch.randn(8,5,3)*.02;offset=torch.randn_like(tips)*.002;delta=torch.randn_like(tips)*.002
        center=2*tips+.01
        weights={'coverage':80.,'response_direction':1.,'flatness':1.,'pinch':1000.}
        loss,r=baseline_from_outputs(tips,center,center+2*offset,center-2*offset,center+2*delta,delta,center,weights)
        self.assertAlmostEqual(r['response_direction']['raw'],-5.,places=5)
        self.assertLess(r['flatness']['raw'],1e-12)
        self.assertAlmostEqual(float(loss),sum(v['weighted'] for v in r.values()),places=5)

    def test_online_and_cached_formulas_agree(self):
        ids=[4,8,12,16,20];points=torch.randn(8,21,3)*.02
        model=lambda sample:sample['keypoints'][:,ids]
        fk=lambda x:2*x+.01
        weights={'coverage':80.,'response_direction':1.,'flatness':1.,'pinch':1000.}
        tips=points[:,ids];cloud=torch.randn(32,5,3)*.02
        torch.manual_seed(42)
        loss,_,records=baseline_objective(model,{'keypoints':points},fk,cloud,ids,weights)
        torch.manual_seed(42);offset=torch.randn_like(tips)*.002;delta=torch.randn_like(tips)*.002
        cached,r=baseline_from_outputs(tips,fk(tips),fk(tips+offset),fk(tips-offset),fk(tips+delta),delta,cloud,weights)
        torch.testing.assert_close(loss,cached,rtol=0,atol=0)
        self.assertEqual(records,r)

    def test_history_is_copied_and_restored_without_advancing(self):
        r=SimpleNamespace(optimizer=SimpleNamespace(last_qpos=None),lp_filter=SimpleNamespace(y=None,is_init=False))
        raw=np.array([.1,.2],dtype=np.float32);smooth=np.array([.3,.4],dtype=np.float32)
        restore_history(r,raw,smooth);r.optimizer.last_qpos[:]=9;r.lp_filter.y[:]=7
        restore_history(r,raw,smooth)
        np.testing.assert_array_equal(r.optimizer.last_qpos,raw);np.testing.assert_array_equal(r.lp_filter.y,smooth)
        restore_history(r,None,None);self.assertIsNone(r.optimizer.last_qpos);self.assertFalse(r.lp_filter.is_init)

    def test_probe_center_and_provenance_are_checked(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data.npy';np.save(data,np.zeros((2,21,3)))
            cache={'cache_sha256':'cache','metadata':{'source_commit':'commit'},'filtered':np.zeros((2,2)),'unfiltered':np.zeros((2,2))}
            meta={'baseline_cache_sha256':'cache','data_sha256':sha(data),'source_commit':'commit'}
            values={'metadata':json.dumps(meta),'source_frame':np.arange(2),'joint_names':['b','a'],'offsets':np.zeros((2,5,3)),'deltas':np.zeros((2,5,3))}
            for mode in ('filtered','unfiltered'):
                for probe in ('center','plus','minus','delta'):values[f'{mode}_{probe}']=np.zeros((2,2))
            path=root/'probes.npz';np.savez(path,**values)
            load_probes(path,cache,data,np.arange(2),['a','b'])
            values['filtered_center'][0,0]=1;np.savez(path,**values)
            with self.assertRaisesRegex(ValueError,'center'):load_probes(path,cache,data,np.arange(2),['a','b'])
            cache['cache_sha256']='changed'
            with self.assertRaisesRegex(ValueError,'provenance'):load_probes(path,cache,data,np.arange(2),['a','b'])


if __name__=='__main__':unittest.main()
