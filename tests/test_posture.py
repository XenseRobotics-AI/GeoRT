import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import numpy as np

from geort.posture import FingerKinematics, WujiPostureCorrector
from geort.utils.config_utils import get_config


class PostureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = get_config('wuji_hand2_beta1_right')
        cls.kin = FingerKinematics(cls.config)

    def command(self, finger):
        q = np.zeros(20)
        q[self.kin.indices] = finger
        return q

    def test_exact_jacobians(self):
        rng = np.random.default_rng(5)
        for q in rng.uniform(self.kin.lower, self.kin.upper, (10, 4)):
            _, direction, jac, djac = self.kin.forward(q)
            self.assertAlmostEqual(np.linalg.norm(direction), 1)
            for j in range(4):
                d = np.eye(4)[j] * 1e-6
                plus, minus = self.kin.forward(q + d), self.kin.forward(q - d)
                np.testing.assert_allclose(jac[:, j], (plus[0] - minus[0]) / 2e-6, atol=1e-8)
                np.testing.assert_allclose(djac[:, j], (plus[1] - minus[1]) / 2e-6, atol=1e-8)

    def test_matches_simulator_fk(self):
        from geort.env.hand import HandKinematicModel
        hand = HandKinematicModel.build_from_config(self.config)
        hand.initialize_keypoint(['r_pinky_tip', 'r_pinky_distal'], [[0, 0, 0]] * 2)
        for q in np.random.default_rng(1).uniform(self.kin.lower, self.kin.upper, (12, 4)):
            points = hand.keypoint_from_qpos(self.command(q), ret_vec=True)
            tip, direction, _, _ = self.kin.forward(q)
            np.testing.assert_allclose(tip, points[0], atol=1e-7)
            d = points[0] - points[1]
            np.testing.assert_allclose(direction, d / np.linalg.norm(d), atol=2e-6)

    def test_budget_and_other_fingers(self):
        corrector = WujiPostureCorrector(self.config, direction_weight=0, tip_budget_mm=2, max_joint_step_deg=0)
        improved = 0
        for finger in ([.9, 0, -.5, -.3], [.3, .1, -.4, .7], [.7, -.1, -.7, .2]):
            q = self.command(finger)
            result = corrector.correct(q)
            keep = np.ones(20, bool); keep[self.kin.indices] = False
            np.testing.assert_array_equal(result.qpos[keep], q[keep])
            after = result.qpos[self.kin.indices]
            self.assertTrue(np.all(after >= self.kin.lower))
            self.assertTrue(np.all(after <= self.kin.upper))
            self.assertGreaterEqual(after[0], min(finger[0], -np.deg2rad(15)) - 1e-10)
            shift = np.linalg.norm(self.kin.forward(after)[0] - self.kin.forward(finger)[0])
            self.assertLessEqual(shift, .002)
            self.assertLessEqual(result.after_violation_rad, result.before_violation_rad + 1e-10)
            improved += result.after_violation_rad < result.before_violation_rad - .05
        self.assertGreaterEqual(improved, 2)

    def test_normal_pose_is_untouched(self):
        corrector = WujiPostureCorrector(self.config, direction_weight=0)
        q = self.command([.5, .1, .4, .2])
        result = corrector.correct(q)
        np.testing.assert_array_equal(q, result.qpos)
        self.assertEqual(result.status, 'unchanged')
        corrector.reset()
        self.assertIsNone(corrector.previous)

    def test_failed_or_infeasible_solver_falls_back(self):
        corrector = WujiPostureCorrector(self.config, direction_weight=0)
        q = self.command([.8, 0, -.7, -.5])
        for fake in (SimpleNamespace(success=False, x=np.zeros(4)),
                     SimpleNamespace(success=True, x=np.full(4, np.nan)),
                     SimpleNamespace(success=True, x=np.full(4, 99)),
                     SimpleNamespace(success=True, x=np.array([0, 0, 1, 1]))):
            with patch('geort.posture.minimize', return_value=fake):
                result = corrector.correct(q)
            np.testing.assert_array_equal(q, result.qpos)
            self.assertEqual(result.status, 'fallback')

    def test_replay_routes_raw_and_corrected_commands(self):
        import contextlib
        import io
        import sapien.core as sapien
        from geort.mocap import replay_evaluation as replay
        q = self.command([.9, 0, -.5, -.3])
        points = np.zeros((21, 3)); points[20] = [0, 0, .02]
        for mode in ('--compare-posture', '--compare-pd'):
            hand = MagicMock()
            viewer = hand.get_viewer_env.return_value.viewer
            viewer.closed = False
            viewer.window.key_press.return_value = False
            viewer.window.camera_mode = "orthographic"
            hand.hand.get_root_pose.return_value = sapien.Pose()
            hand.convert_user_order_to_sim_order.side_effect = lambda x: x
            direct = MagicMock()
            mocap = MagicMock(human_points=points[None], T=1)
            mocap.get.side_effect = [{'status': 'recording', 'result': points}, {'status': 'quit'}]
            model = SimpleNamespace(config=self.config, forward=lambda x: q.copy())
            argv = ['replay', '-hand', self.config['name'], mode, '--posture-correction']
            with contextlib.ExitStack() as stack:
                for name, value in [('load_model', model), ('ReplayMocap', mocap),
                                    ('build_direct_comparison', direct)]:
                    stack.enter_context(patch.object(replay, name, return_value=value))
                stack.enter_context(patch.object(replay.HandKinematicModel, 'build_from_config', return_value=hand))
                stack.enter_context(patch.object(replay, 'HumanHandViewer'))
                stack.enter_context(patch.object(replay, 'reset_camera'))
                controller = stack.enter_context(patch.object(replay, 'ReplayController'))
                stack.enter_context(patch.object(replay.time, 'sleep'))
                stack.enter_context(patch('sys.argv', argv))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                replay.main()
            np.testing.assert_array_equal(direct.set_qpos.call_args.args[0], q)
            if mode == '--compare-posture':
                controller.assert_not_called()
                corrected = hand.hand.set_qpos.call_args.args[0]
            else:
                corrected = controller.return_value.advance.call_args.args[0]
            self.assertLess(np.linalg.norm(np.minimum(corrected[-2:], 0)),
                            np.linalg.norm(np.minimum(q[-2:], 0)))
            np.testing.assert_array_equal(corrected[:-4], q[:-4])
            viewer.close.assert_called_once()

    def test_continuity_survives_normal_frames_and_solver_failure(self):
        corrector = WujiPostureCorrector(self.config, direction_weight=0, max_joint_step_deg=8)
        first = self.command([.9, 0, -.5, -.3])
        previous = corrector.correct(first).qpos.copy()
        # A normal raw pose and a large target change cannot bypass the limiter.
        for finger in ([0, 0, 0, 0], [1.3, .3, 1.8, 1.3], [-.5, -.3, -.8, -.8]):
            q = self.command(finger)
            result = corrector.correct(q)
            self.assertLessEqual(np.rad2deg(abs(result.qpos - previous)).max(), 8 + 1e-8)
            self.assertEqual(result.tip_budget_satisfied, result.tip_shift_m <= .002)
            np.testing.assert_array_equal(result.qpos[:-4], q[:-4])
            previous = result.qpos.copy()
        with patch('geort.posture.minimize', return_value=SimpleNamespace(success=False, x=np.full(4, np.nan))):
            result = corrector.correct(self.command([1.5, .5, 1.9, 1.5]))
        self.assertLessEqual(np.rad2deg(abs(result.qpos - previous)).max(), 8 + 1e-8)
        self.assertEqual(result.status, 'tip_budget_relaxed')
        self.assertFalse(result.tip_budget_satisfied)
        corrector.reset()
        self.assertIsNone(corrector.previous)

    def test_mcp_conflict_is_reported_and_reset_removes_history(self):
        c = WujiPostureCorrector(self.config, direction_weight=0, max_joint_step_deg=8)
        first = self.command([-1., 0, .4, .3])
        np.testing.assert_array_equal(c.correct(first).qpos, first)
        target = self.command([1., 0, .4, .3])
        result = c.correct(target)
        self.assertFalse(result.mcp_guard_satisfied)
        self.assertFalse(result.tip_budget_satisfied)
        self.assertLessEqual(abs(result.qpos[16] - first[16]), np.deg2rad(8) + 1e-10)
        c.reset()
        np.testing.assert_array_equal(c.correct(target).qpos, target)

    def test_continuity_validation(self):
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                WujiPostureCorrector(self.config, max_joint_step_deg=value)

    def test_validation_and_degenerate_direction(self):
        for kwargs in ({'tip_budget_mm': 0}, {'tip_budget_mm': float('nan')},
                       {'direction_weight': -1}, {'extension_tolerance_deg': 60},
                       {'starts': 2}, {'max_iterations': 0}):
            with self.assertRaises(ValueError):
                WujiPostureCorrector(self.config, **kwargs)
        with self.assertRaises(ValueError):
            WujiPostureCorrector(get_config('allegro_right'))
        corrector = WujiPostureCorrector(self.config)
        q = self.command([.8, 0, -.5, -.2])
        for bad in (np.full(20, np.nan), np.zeros(19), self.command([99, 0, 0, 0])):
            with self.assertRaises(ValueError):
                corrector.correct(bad, np.zeros((21, 3)))
        for points in (None, np.zeros((5, 3)), np.full((21, 3), np.nan)):
            with self.assertRaises(ValueError):
                corrector.correct(q, points)
        # Missing/zero-length distal segment disables only the direction term.
        result = corrector.correct(q, np.zeros((21, 3)))
        self.assertTrue(np.isfinite(result.qpos).all())


if __name__ == '__main__':
    unittest.main()
