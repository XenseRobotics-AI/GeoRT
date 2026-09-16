import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from geort.trainer import TrainingLog, save_checkpoint, update_latest
from geort.export import load_model
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from geort import export
from geort.dataset import upsample_array
from geort.loss import pinch_distance_loss
from geort.env.hand import HandKinematicModel


class RetargetingFixTest(unittest.TestCase):
    def test_collision_exclusions_preserve_other_pairs(self):
        class Shape:
            def __init__(self): self.groups = [1, 1, 0, 0]
            def get_collision_groups(self): return self.groups.copy()
            def set_collision_groups(self, *args):
                self.groups = list(args[0]) if len(args) == 1 else list(args)
        shapes = {name: Shape() for name in ('wrist', 'ring', 'little', 'tip')}
        links = [SimpleNamespace(name=name, get_collision_shapes=lambda shape=shape: [shape])
                 for name, shape in shapes.items()]
        articulation = SimpleNamespace(get_links=lambda: links)
        hand = SimpleNamespace(hand=articulation, scene=SimpleNamespace(
            get_all_actors=lambda: [], get_all_articulations=lambda: [articulation]))
        HandKinematicModel.exclude_collision_pairs(hand, [('wrist', 'ring'), ('wrist', 'little')])
        self.assertTrue(shapes['wrist'].groups[2] & shapes['ring'].groups[2])
        self.assertTrue(shapes['wrist'].groups[2] & shapes['little'].groups[2])
        self.assertFalse(shapes['ring'].groups[2] & shapes['little'].groups[2])
        self.assertEqual(shapes['tip'].groups, [1, 1, 0, 0])
        for shape in shapes.values(): self.assertEqual(shape.groups[:2], [1, 1])
        with self.assertRaises(ValueError):
            HandKinematicModel.exclude_collision_pairs(hand, [('wrist', 'missing')])

    def test_drive_targets_stay_in_user_order(self):
        targets = np.zeros(4)
        joints = [SimpleNamespace(set_drive_target=lambda v, i=i: targets.__setitem__(i, v))
                  for i in range(4)]
        order = [2, 0, 3, 1]
        hand = SimpleNamespace(joint_lower_limit=np.zeros(4), joint_upper_limit=np.ones(4),
                               all_joints=joints, convert_user_order_to_sim_order=lambda q: q[order])
        HandKinematicModel.set_qpos_target(hand, np.array([-.1, .3, 1.2, .5]))
        expected = np.array([.001, .3, .999, .5])
        np.testing.assert_allclose(targets, expected)
        np.testing.assert_allclose(hand.qpos_target, expected[order])

    def test_checkpoint_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(export, 'get_checkpoint_root', return_value=root):
                for name in ['hand_old_tag', 'hand_new_tag', 'hand_last']:
                    path = root / name
                    path.mkdir()
                    (path / 'config.json').write_text('{}')
                self.assertEqual(export.resolve_checkpoint('hand_last').name, 'hand_last')
                self.assertEqual(export.resolve_checkpoint('old_tag').name, 'hand_old_tag')
                with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                    export.resolve_checkpoint('tag')
                with self.assertRaises(FileNotFoundError):
                    export.resolve_checkpoint('missing')

    def test_pinch_mask_and_gradient(self):
        human = torch.tensor([[[0., 0., 0.], [.01, 0., 0.]],
                              [[0., 0., 0.], [.1, 0., 0.]]])
        robot = torch.tensor([[[0., 0., 0.], [.02, 0., 0.]],
                              [[0., 0., 0.], [.5, 0., 0.]]], requires_grad=True)
        loss = pinch_distance_loss(human, robot)
        self.assertAlmostEqual(loss.item(), .02**2)
        loss.backward()
        self.assertGreater(robot.grad[0, 1, 0], 0)
        self.assertEqual(torch.count_nonzero(robot.grad[1]), 0)
        self.assertEqual(pinch_distance_loss(human[1:], robot[1:]).item(), 0)

    def test_paper_pinch_expectation_and_ordered_pairs(self):
        human = torch.tensor([[[0., 0., 0.], [.01, 0., 0.]],
                              [[0., 0., 0.], [.1, 0., 0.]],
                              [[0., 0., 0.], [.1, 0., 0.]]])
        robot = torch.tensor([[[0., 0., 0.], [.03, 0., 0.]]] * 3, requires_grad=True)
        loss = pinch_distance_loss(human, robot, paper=True)
        self.assertAlmostEqual(loss.item(), 2 * .03**2 / 3)
        loss.backward()
        self.assertEqual(torch.count_nonzero(robot.grad[1:]), 0)
        self.assertAlmostEqual(robot.grad[0, 1, 0].item(), 4 * .03 / 3)
        self.assertEqual(pinch_distance_loss(human[1:], robot[1:], paper=True).item(), 0)

    def test_resampling_singleton_and_last_point(self):
        np.random.seed(0)
        np.testing.assert_array_equal(upsample_array(np.array([[1, 2, 3]]), 2), [[1, 2, 3], [1, 2, 3]])
        self.assertEqual(set(upsample_array(np.array([0, 1]), 100)), {0, 1})
        with self.assertRaisesRegex(ValueError, 'empty'):
            upsample_array(np.empty((0, 3)))


class CheckpointTest(unittest.TestCase):
    def test_best_last_and_periodic_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / 'run'
            save_checkpoint({'x': torch.tensor(1)}, run, 0, is_best=True)
            save_checkpoint({'x': torch.tensor(2)}, run, 1, save_every=2)
            save_checkpoint({'x': torch.tensor(3)}, run, 2)
            self.assertEqual(torch.load(run / 'best.pth')['x'], 1)
            self.assertEqual(torch.load(run / 'epoch_1.pth')['x'], 2)
            self.assertEqual(torch.load(run / 'last.pth')['x'], 3)
            self.assertEqual({p.name for p in run.iterdir()}, {'best.pth', 'last.pth', 'epoch_1.pth'})

    def test_failed_save_preserves_previous_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            save_checkpoint({'x': torch.tensor(1)}, run, 0, is_best=True)
            with patch('geort.trainer.torch.save', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    save_checkpoint({}, run, 1, is_best=True)
            self.assertEqual(torch.load(run / 'last.pth')['x'], 1)
            self.assertEqual(torch.load(run / 'best.pth')['x'], 1)
            self.assertEqual(len(list(run.iterdir())), 2)

    def test_alias_and_best_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ['run1', 'run2']:
                run = root / name
                save_checkpoint({'x': torch.tensor(1)}, run, 0, is_best=True)
                (run / 'config.json').write_text('{}')
            alias = root / 'hand_last'
            update_latest(root / 'run1', alias)
            update_latest(root / 'run2', alias)
            self.assertEqual(alias.resolve(), root / 'run2')
            with patch('geort.export.get_checkpoint_root', return_value=root), patch('geort.export.GeoRTRetargetingModel') as loader:
                load_model('hand_last', weights='best')
                self.assertEqual(loader.call_args.kwargs['model_path'], alias / 'best.pth')
            legacy = root / 'legacy'
            legacy.mkdir()
            with self.assertRaises(FileExistsError):
                update_latest(root / 'run1', legacy)


class TrainingLogTest(unittest.TestCase):
    def test_csv_and_tensorboard_match_even_after_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                with TrainingLog(directory) as log:
                    log.write('ik', 0, {'raw/direction': -0.5, 'loss/total': -0.25})
                    log.write('ik', 1, {'raw/direction': -0.75, 'loss/total': -0.5})
                    raise RuntimeError('interrupted training')
            with (Path(directory) / 'ik.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            events = EventAccumulator(directory).Reload()
            values = events.Scalars('ik/loss/total')
            self.assertEqual([value.step for value in values], [0, 1])
            self.assertEqual([float(row['loss/total']) for row in rows],
                             [value.value for value in values])
