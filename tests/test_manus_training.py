import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch

from geort.manus_sessions import load_session_split, sample_balanced
from geort.mocap.prepare_manus_dataset import prepare
from geort.train_manus import make_config
from geort.model import build_ik_model
from geort.coordination_loss import ExactHand, task_objective
from geort.export import GeoRTRetargetingModel
from tests.test_manus_dataset import ManusDatasetTests
from tests.test_coordination import robot_config


class ManusTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        ManusDatasetTests.setUp(self)
        self.bundle=self.root/'bundle'
        prepare(self.sessions,self.bundle)
        self.manifest=self.bundle/'manifest.json'

    def test_training_loader_does_not_open_test_and_sampling_respects_ranges(self):
        (self.bundle/'test.npy').unlink()
        train=load_session_split(self.manifest,'train')
        val=load_session_split(self.manifest,'validation')
        self.assertEqual(len(val['keypoints']),6)
        ids=sample_balanced(train,1000,np.random.default_rng(42))
        self.assertTrue(np.all((ids>=0)&(ids<6)))
        self.assertTrue(400<np.sum(ids<3)<600)
        with self.assertRaises(FileNotFoundError):load_session_split(self.manifest,'test')

    def test_modified_boundaries_and_cross_session_identity_rejected(self):
        manifest=json.loads(self.manifest.read_text())
        bad=copy.deepcopy(manifest);bad['splits']['train']['clips'][1]['range'][0]=2
        self.manifest.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError,'boundary'):load_session_split(self.manifest,'train')
        bad=copy.deepcopy(manifest);bad['splits']['test']['session_id']=bad['splits']['train']['session_id']
        self.manifest.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError,'cross'):load_session_split(self.manifest,'train')

    def test_equal_initialization_information_and_export(self):
        train=load_session_split(self.manifest,'train')
        models=[]
        for mode in ['tip_control','skeleton']:
            cfg=make_config(robot_config(),train,mode)
            torch.manual_seed(42);model=build_ik_model(cfg).eval();models.append(model)
        for key,value in models[0].state_dict().items():
            torch.testing.assert_close(value,models[1].state_dict()[key],rtol=0,atol=0)
        points=torch.tensor(train['keypoints']);changed=points.clone();changed[:,19,0]+=.003
        torch.testing.assert_close(models[0](points),models[0](changed),rtol=0,atol=0)
        self.assertGreater(float((models[1](points)-models[1](changed)).abs().max().detach()),1e-8)
        self.assertLessEqual(float(models[1](points).abs().max().detach()),1.)
        exact=ExactHand(cfg);sample={'keypoints':points}
        loss,_,_=task_objective(models[1](sample),sample,exact,cfg,cfg['objectives']);loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in models[1].parameters()))
        self.assertGreater(float(models[1].nets[0][0].weight.grad.abs().sum()),0)
        (self.root/'config.json').write_text(json.dumps(cfg));torch.save(models[1].state_dict(),self.root/'best.pth')
        exported=GeoRTRetargetingModel(self.root/'best.pth',self.root/'config.json',device='cpu')
        with torch.inference_mode():expected=exact(models[1](points[:1]))['q'][0].numpy()
        np.testing.assert_allclose(exported.forward(points[0].numpy()),expected,atol=1e-6)


if __name__=='__main__':unittest.main()
