import unittest
import numpy as np
import torch
import copy
from geort.model import build_ik_model
from geort.train_refinement import paired_ids, response_loss
from geort.collision_surrogate import CollisionDepthModel


class RefinementTests(unittest.TestCase):
    def test_pairs_do_not_cross_clip(self):
        recording={'entry':{'clips':[{'range':[0,5]},{'range':[5,9]}]},'clip_id':np.array([0]*5+[1]*4)}
        a=np.arange(9);b=paired_ids(recording,a,np.random.default_rng(42))
        np.testing.assert_array_equal(recording['clip_id'][a],recording['clip_id'][b])
        self.assertTrue((b>=a).all())

    def test_response_gradient_and_mask(self):
        x=torch.zeros(3,21,3);y=x.clone();y[0,:,0]=.001;y[1,:,0]=.02
        rt=torch.zeros(3,5,3,requires_grad=True);other=torch.zeros_like(rt,requires_grad=True)
        loss,record=response_loss(x,y,rt,other)
        self.assertEqual(record['valid_count'],5)
        loss.backward();self.assertGreater(float(other.grad[0].abs().sum()),0)
        self.assertEqual(float(other.grad[1:].abs().sum()),0)
        perfect,_=response_loss(x,y,rt.detach(),y[:,[4,8,12,16,20]])
        self.assertEqual(float(perfect),0)
        empty,_=response_loss(x,x,rt,other);self.assertEqual(float(empty.detach()),0)

    def test_frozen_collision_keeps_input_gradient(self):
        torch.manual_seed(42);model=CollisionDepthModel().requires_grad_(False)
        x=torch.zeros(8,20,requires_grad=True);depth,_=model(x)
        depth.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(float(x.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_context_identity_gradient_and_other_finger_information(self):
        from geort.train_manus import make_config
        from tests.test_coordination import robot_config
        from tests.test_manus_dataset import ManusDatasetTests
        # Reuse the synthetic, valid-bone data fixture, never a recorded test session.
        fixture=ManusDatasetTests();fixture.setUp()
        try:
            from geort.mocap.prepare_manus_dataset import prepare
            from geort.manus_sessions import load_session_split
            bundle=fixture.root/'bundle';prepare(fixture.sessions,bundle)
            data=load_session_split(bundle/'manifest.json','train')
            cfg=make_config(robot_config(),data,'skeleton');torch.manual_seed(42)
            base=build_ik_model(cfg);cfg=copy.deepcopy(cfg);cfg['manus_features']['context']=True
            new=build_ik_model(cfg);state=base.state_dict()
            new.load_state_dict({**state,**{k:v for k,v in new.state_dict().items() if k.startswith('context_net.')}})
            x=torch.tensor(data['keypoints']);torch.testing.assert_close(base(x),new(x),rtol=0,atol=0)
            optim=torch.optim.SGD(new.parameters(),lr=.01);new(x).sum().backward()
            self.assertGreater(float(new.context_net[-1].weight.grad.abs().sum()),0)
            optim.step();optim.zero_grad();new(x).sum().backward()
            self.assertGreater(float(new.context_net[0].weight.grad.abs().sum()),0)
            changed=x.clone();changed[:,17:21,0]+=.005
            self.assertGreater(float((new(changed)[:,:4]-new(x)[:,:4]).abs().max().detach()),0)
            self.assertLessEqual(float(new(x).abs().max().detach()),1.)
        finally:
            fixture.doCleanups()


if __name__=='__main__':unittest.main()
