"""Synthetic geometry tests verify contracts, not retargeting quality."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from geort.coordination import adapt_points, initialize_coordination, skeleton_features
from geort.coordination_data import load_recording, split_indices
from geort.coordination_diagnostics import response_metrics, time_metrics
from geort.coordination_loss import ExactHand, masked_term, task_objective
from geort.export import GeoRTRetargetingModel
from geort.kinematics import FingerKinematics
from geort.model import build_ik_model, ik_input
from geort.train_coordination import validate_experiment
from geort.utils.config_utils import get_config, parse_config_keypoint_info


def robot_config(name='wuji_hand2_beta1_right'):
    cfg = get_config(name)
    lower = np.zeros(len(cfg['joint_order'])); upper = lower.copy()
    for f in cfg['fingertip_link']:
        chain = FingerKinematics(cfg, f['name'])
        lower[chain.indices] = chain.lower; upper[chain.indices] = chain.upper
    cfg['joint'] = {'lower': lower.tolist(), 'upper': upper.tolist()}
    return cfg


class CoordinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg = robot_config()
        cls.points = torch.tensor(np.load('data/human_alex.npy')[:8], dtype=torch.float32)
        cls.ids = [4, 8, 12, 16, 20]

    def models(self, mode='skeleton', context=True):
        torch.manual_seed(3)
        parent = build_ik_model(self.cfg).eval()
        cfg, model = initialize_coordination(self.cfg, parent, mode, context)
        return cfg, model, parent

    def test_all_off_and_initialization_match_original(self):
        for mode in ('tip_only', 'skeleton'):
            for context in (False, True):
                _, model, parent = self.models(mode, context)
                torch.testing.assert_close(model(self.points), parent(self.points[:, self.ids]), rtol=0, atol=0)

    def test_original_allegro_both_hands_compatible(self):
        for name in ('allegro_left', 'allegro_right'):
            cfg = robot_config(name)
            parent = build_ik_model(cfg).eval()
            cfg2, model = initialize_coordination(cfg, parent, 'tip_only', False)
            torch.testing.assert_close(model(self.points), parent(ik_input(cfg, self.points)), rtol=0, atol=0)
            with self.assertRaises(ValueError):
                initialize_coordination(cfg, parent, 'skeleton', True)

    def test_zero_output_head_can_learn(self):
        _, model, _ = self.models()
        opt = torch.optim.SGD(model.parameters(), lr=.01)
        for step in range(2):
            opt.zero_grad()
            model(self.points).sum().backward()
            self.assertGreater(float(model.context[-1].weight.grad.abs().sum()), 0)
            if step:
                self.assertGreater(float(model.context[0].weight.grad.abs().sum()), 0)
            opt.step()

    def test_same_tips_different_axes(self):
        changed = self.points.clone(); changed[:, 19, 0] += .005
        a = skeleton_features(self.points, self.ids); b = skeleton_features(changed, self.ids)
        torch.testing.assert_close(a['tips'], b['tips'])
        self.assertFalse(torch.equal(a['axes'], b['axes']))
        _, tip_model, _ = self.models('tip_only', True)
        torch.testing.assert_close(tip_model(self.points), tip_model(changed), rtol=0, atol=0)

    def test_head_information_ablation_and_frozen_base(self):
        _, _, parent = self.models()
        models = []
        for policy in ('tip_control', 'skeleton'):
            torch.manual_seed(42)
            cfg, model = initialize_coordination(self.cfg, parent, 'skeleton', False, policy)
            model.base.requires_grad_(False)
            before = {k:v.clone() for k,v in model.base.state_dict().items()}
            opt = torch.optim.Adam(model.local.parameters(), lr=.001)
            for _ in range(3):
                opt.zero_grad(); model(self.points).sum().backward(); opt.step()
            for k,v in before.items(): torch.testing.assert_close(v, model.base.state_dict()[k], rtol=0, atol=0)
            self.assertTrue(any(float(p.grad.abs().sum()) > 0 for p in model.local.parameters()))
            restored=build_ik_model(cfg).eval();restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(restored(self.points),model(self.points),rtol=0,atol=0)
            models.append(model)
        self.assertEqual(sum(p.numel() for p in models[0].parameters()),sum(p.numel() for p in models[1].parameters()))
        changed=self.points.clone();changed[:,19,0]+=.005
        torch.testing.assert_close(models[0](changed),models[0](self.points),rtol=0,atol=0)
        self.assertGreater(float(abs(models[1](changed)-models[1](self.points)).max()),1e-8)
        bad=copy.deepcopy(cfg);bad['features']['version']=1
        with self.assertRaises(ValueError):build_ik_model(bad)

    def test_context_can_express_cross_finger_change(self):
        _, model, _ = self.models('tip_only', True)
        with torch.no_grad():
            model.context[-1].weight.fill_(.02)
        changed = self.points.clone(); changed[:, 4, 0] += .02
        # Small finger's own input is identical; context has access to thumb.
        self.assertGreater(float(abs(model(changed)[:, -4:] - model(self.points)[:, -4:]).max()), 1e-8)
        _, independent, _ = self.models('tip_only', False)
        torch.testing.assert_close(independent(changed)[:, -4:], independent(self.points)[:, -4:], rtol=0, atol=0)

    def test_bounds_and_small_perturbations(self):
        _, model, _ = self.models()
        with torch.no_grad():
            model.context[-1].weight.fill_(1000.)
        q = model(self.points)
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue((q.abs() <= 1).all())
        self.assertLess(float(abs(model(self.points + 1e-6) - q).max()), .01)

    def test_masks_nonfinite_and_degenerate(self):
        p = self.points.clone(); p[:, 19] = float('nan')
        f = skeleton_features(p, self.ids)
        self.assertFalse(f['bone_valid'][:, -1, -1].any())
        self.assertTrue(torch.isfinite(f['axes']).all())
        confidence = torch.ones(8, 21); confidence[:, 2] = .1
        f = skeleton_features({'keypoints': self.points, 'confidence': confidence}, self.ids)
        self.assertFalse(f['bone_valid'][:, 0, 0].any())
        p = self.points.clone(); p[:, 19] = p[:, 20]
        self.assertFalse(skeleton_features(p, self.ids)['bone_valid'][:, -1, -1].any())
        p[:, 20] = float('inf')
        with self.assertRaises(ValueError):
            skeleton_features(p, self.ids)
        with self.assertRaises(ValueError):
            skeleton_features({'keypoints': self.points, 'valid': torch.ones(8, 21)}, self.ids)

    def test_relative_geometry_and_unit_calibration(self):
        rot = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        p = adapt_points(self.points * 1000, unit='mm', rotation=rot, translation_m=[.1, -.2, .3])
        f = skeleton_features(self.points, self.ids); g = skeleton_features(p, self.ids)
        torch.testing.assert_close(g['axes'], f['axes'] @ rot.T, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close((g['axes'][:,:,1:] * g['axes'][:,:,:-1]).sum(-1),
                                   (f['axes'][:,:,1:] * f['axes'][:,:,:-1]).sum(-1), atol=2e-6, rtol=2e-6)
        with self.assertRaises(ValueError):
            adapt_points(self.points, rotation=np.diag([-1., 1., 1.]))
        with self.assertRaises(ValueError):
            adapt_points(self.points, unit='unknown')

    def test_strict_metadata_and_state_dict(self):
        cfg, model, _ = self.models()
        restored = build_ik_model(cfg).eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(self.points), model(self.points))
        for key, value in [('handedness', 'left'), ('length_unit', 'mm'), ('mode', 'pose')]:
            bad = copy.deepcopy(cfg); bad['features'][key] = value
            with self.assertRaises(ValueError):
                build_ik_model(bad)
        bad = copy.deepcopy(cfg); bad['joint']['lower'][0] -= .1
        with self.assertRaises(ValueError):
            build_ik_model(bad)
        state = dict(model.state_dict()); state.pop(next(iter(state)))
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(state)

    def test_export_legacy_and_enhanced_masks(self):
        cfg, model, parent = self.models()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            for name, conf, net in [('old', self.cfg, parent), ('new', cfg, model)]:
                (path / f'{name}.json').write_text(json.dumps(conf))
                torch.save(net.state_dict(), path / f'{name}.pth')
            old = GeoRTRetargetingModel(path / 'old.pth', path / 'old.json', device='cpu')
            new = GeoRTRetargetingModel(path / 'new.pth', path / 'new.json', device='cpu')
            np.testing.assert_array_equal(old.forward(self.points[0]), new.forward(self.points[0]))
            p = self.points[0].clone(); p[19] = float('nan')
            self.assertTrue(np.isfinite(new.forward(p)).all())
            with self.assertRaises(ValueError):
                old.forward(p)

    def test_masked_loss_all_invalid_and_gradient(self):
        value = torch.tensor([float('nan'), 3.], requires_grad=True)
        loss, log = masked_term(value, torch.tensor([False, True]), 3., 2.)
        self.assertEqual(float(loss.detach()), 2.)
        loss.backward(); self.assertEqual(float(value.grad[0]), 0.)
        zero, log = masked_term(value, torch.tensor([False, False]), 1., 1.)
        self.assertEqual(float(zero.detach()), 0.)
        self.assertEqual(log['valid_count'], 0)

    def test_task_losses_and_network_gradients(self):
        cfg, model, _ = self.models()
        options = json.loads(Path('geort/experiments/B5.json').read_text())['task']
        options['relation_scale'] = [1.] * 4
        exact = ExactHand(cfg)
        loss, records, _ = task_objective(model(self.points), self.points, exact, cfg, options)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.context[-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.base.nets[4][0].weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_axis_sign_and_ideal_geometry(self):
        cfg, _, _ = self.models()
        exact = ExactHand(cfg)
        q = torch.zeros(2, 20, requires_grad=True)
        robot = exact(q)
        human = torch.zeros(2, 21, 3)
        for i, end in enumerate(self.ids):
            tip = robot['tips'][:, i].detach()
            bones = (robot['axes'][:, i] * robot['lengths'][:, i, :, None]).detach()
            human[:, end] = tip
            for j in range(2, -1, -1):
                human[:, end - 3 + j] = human[:, end - 2 + j] - bones[:, j]
        opt = json.loads(Path('geort/experiments/B5.json').read_text())['task']; opt['relation_scale'] = [1.] * 4
        _, records, _ = task_objective(q, human, exact, cfg, opt)
        self.assertLess(abs(records['axis']['raw']), 1e-6)
        self.assertLess(records['shape']['raw'], 1e-8)
        self.assertLess(records['relation']['raw'], 1e-10)
        flipped = human.clone()
        for end in self.ids:
            flipped[:, end - 1] = 2 * human[:, end] - human[:, end - 1]
        _, bad, _ = task_objective(q, flipped, exact, cfg, opt)
        self.assertGreater(bad['axis']['raw'], 1.99)

    def test_diagnostics_report_dead_responses(self):
        x = np.ones((4, 3)) * .001
        record = response_metrics(x, np.zeros_like(x))
        self.assertEqual(record['dead_zone_fraction'], 1.)
        self.assertIsNone(record['response_direction_error_deg'])
        self.assertEqual(record['gain']['mean'], 0.)
        no_motion = response_metrics(np.zeros_like(x), np.zeros_like(x))
        self.assertIsNone(no_motion['dead_zone_fraction'])

    def test_time_units_and_sequence_boundaries(self):
        tips = np.zeros((4, 1, 3)); tips[:,0,0] = [0, .1, 30, 30.1]
        result = time_metrics(tips, tips, np.array([0., .1, 0., .1]), np.array([0,0,1,1]))
        self.assertAlmostEqual(result['robot_speed_m_s']['mean'], 1.)
        self.assertIsNone(result['robot_acceleration_m_s2'])
        self.assertFalse(time_metrics(tips, tips, None, None)['available'])

    def test_sequence_adapter_and_split_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'clip.npz'
            np.savez(p, keypoints=self.points.numpy(), timestamps=np.arange(8)*.1, sequence_id=np.zeros(8))
            data = load_recording(p)
            with self.assertRaises(ValueError):
                split_indices(data, {'train':[0,2], 'validation':[3,5], 'test':[6,8]})
            np.savez(p, keypoints=self.points.numpy(), timestamps=np.array([0,1,2,1,4,5,6,7]), sequence_id=np.zeros(8))
            with self.assertRaises(ValueError):
                load_recording(p)

    def test_skeleton_missing_and_pose_schema_errors(self):
        _, model, _ = self.models()
        mask = torch.zeros(8, 21, dtype=torch.bool); mask[:, self.ids] = True
        with self.assertRaisesRegex(ValueError, 'no valid bones'):
            model({'keypoints': self.points, 'valid': mask})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'pose.npz'
            np.savez(path, keypoints=self.points.numpy(), quaternions=np.ones((8,21,4)))
            with self.assertRaisesRegex(ValueError, 'Unsupported recording fields'):
                load_recording(path)

    def test_relation_mask_excludes_contact_without_geometry(self):
        cfg, model, _ = self.models()
        opt = json.loads(Path('geort/experiments/B5.json').read_text())['task']; opt['relation_scale'] = [1.] * 4
        points = self.points.clone()
        points[:, self.ids] = points[:, 4, None]
        _, records, _ = task_objective(model(points), points, ExactHand(cfg), cfg, opt)
        self.assertEqual(records['relation']['valid_count'], 0)
        self.assertEqual(records['relation']['weighted'], 0.)

    def test_exact_anchor_and_invalid_reference_gradients(self):
        cfg, model, _ = self.models()
        exact = ExactHand(cfg)
        opt = json.loads(Path('geort/experiments/B5.json').read_text())['task']
        opt.update(relation_scale=[1.]*4, axis=0., shape=0., posture=0., relation=0., anchor=1., anchor_tolerance_m=.005)
        norm = model(self.points)
        ref = exact(norm)['tips'].detach().clone()
        zero, log, _ = task_objective(norm, self.points, exact, cfg, opt, ref)
        self.assertEqual(log['anchor']['raw'], 0.)
        ref[:,0] = float('nan'); ref[:,1:,0] += .005
        loss, log, _ = task_objective(norm, self.points, exact, cfg, opt, ref)
        self.assertAlmostEqual(log['anchor']['normalized'], 1., places=5)
        self.assertEqual(log['anchor']['valid_count'], len(self.points)*4)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaises(ValueError):
            task_objective(norm, self.points, exact, cfg, opt)

    def test_offline_labels_reject_distant_and_nonfinite_targets(self):
        from geort.prepare_posture_targets import filter_targets
        cfg, model, _ = self.models()
        exact = ExactHand(cfg)
        original = model(self.points).detach()
        target = exact(original)['tips'].detach()
        proposed = original.clone()
        proposed[:, 4:8] = 1.
        proposed[:, 8:12] = float('nan')
        safe, accepted, _ = filter_targets(proposed, original, target, exact)
        self.assertFalse(accepted[:, 1:3].any())
        self.assertTrue(accepted[:, [0,3,4]].all())
        self.assertTrue(torch.equal(safe, original))
        self.assertTrue(torch.isfinite(safe).all())

    def test_feasible_posture_mask_and_reference_extension(self):
        cfg, _, _ = self.models();exact=ExactHand(cfg)
        opt=json.loads(Path('geort/experiments/B5.json').read_text())['task']
        opt.update(axis=0.,shape=0.,relation=0.,posture=1.,relation_scale=[1.]*4,posture_reference_mcp=True)
        q=torch.zeros((len(self.points),20));q[:,4]=-.6;q[:,18]=-1.
        norm=(2*(q-exact.lower)/(exact.upper-exact.lower)-1).requires_grad_(True)
        eligible=torch.ones((len(q),4),dtype=torch.bool);eligible[:,3]=False
        loss,log,_=task_objective(norm,self.points,exact,cfg,opt,reference_q=q,posture_mask=eligible)
        self.assertAlmostEqual(float(loss),0.,places=6)
        self.assertEqual(log['posture']['valid_count'],len(q)*9)
        loss.backward();self.assertTrue(torch.isfinite(norm.grad).all())
        with self.assertRaisesRegex(ValueError,'Reference MCP'):
            task_objective(norm,self.points,exact,cfg,opt,posture_mask=eligible)

    def test_ablation_configs_and_no_double_pinch(self):
        for p in Path('geort/experiments').glob('B*.json'):
            validate_experiment(json.loads(p.read_text()))
        bad = json.loads(Path('geort/experiments/B5.json').read_text())
        bad['baseline']['pinch'] = 1000
        with self.assertRaises(ValueError):
            validate_experiment(bad)


if __name__ == '__main__':
    unittest.main()
