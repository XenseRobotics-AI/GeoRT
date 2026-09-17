"""Regression for the real RenderWindow's first-render projection reset."""
import unittest
from types import SimpleNamespace
from geort.mocap.replay_evaluation import render_frame


class WindowLifecycle:
    def __init__(self, resets=()):
        self.closed = False
        self.calls = 0
        self.restores = 0
        self.updates = 0
        self.pose = object()
        self.resets = set(resets)
        self.window = SimpleNamespace(camera_mode='orthographic',
            set_camera_orthographic_parameters=self.restore)

    def render(self):
        self.calls += 1
        if self.calls in self.resets:
            self.window.camera_mode = 'perspective'

    def restore(self, near, far, scale):
        self.restores += 1
        self.window.camera_mode = 'orthographic'

    def notify_render_update(self):
        self.updates += 1


class ComparisonCameraTests(unittest.TestCase):
    def test_initialization_and_later_resize_restore_without_changing_pose(self):
        viewer = WindowLifecycle(resets=[1,4])
        pose = viewer.pose
        render_frame(viewer, 1.05)  # first draw resets; corrective draw follows
        self.assertEqual(viewer.calls, 2)
        self.assertEqual(viewer.window.camera_mode, 'orthographic')
        render_frame(viewer, 1.05)  # stable frame: no extra draw
        self.assertEqual(viewer.calls, 3)
        render_frame(viewer, 1.05)  # simulate camera recreation on resize
        self.assertEqual(viewer.calls, 5)
        self.assertEqual(viewer.restores, 2)
        self.assertEqual(viewer.updates, 2)
        self.assertIs(viewer.pose, pose)

    def test_single_view_and_closed_window_are_not_reconfigured(self):
        viewer = WindowLifecycle(resets=[1])
        render_frame(viewer)
        self.assertEqual(viewer.restores, 0)
        viewer.closed = True
        render_frame(viewer, 1.05)
        self.assertEqual(viewer.restores, 0)

    def test_persistent_projection_reset_is_not_silently_accepted(self):
        viewer = WindowLifecycle(resets=[1,2])
        with self.assertRaisesRegex(RuntimeError, 'projection'):
            render_frame(viewer, 1.05)


if __name__ == '__main__':
    unittest.main()
