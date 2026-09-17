import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from geort.mocap.manus_capture_data import CaptureWriter, export_static, mp_mapping
from geort.mocap.prepare_manus_dataset import prepare
from geort.coordination_data import load_recording
from tests.test_manus_capture import nodes


class ManusDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sessions = {}
        points = np.load('data/human_alex.npy', allow_pickle=False)
        for index, split in enumerate(('train', 'validation', 'test')):
            root = self.root / split
            root.mkdir()
            topology = nodes()
            rows = mp_mapping(topology)
            meta = dict(session_id=split+'01', split=split, operator='fixture', hand='right',
                        source_type='manus_glove_raw', calibration_note='synthetic unit test fixture')
            tasks = ['neutral', 'little_branch']
            for task in tasks:
                writer = CaptureWriter(root/task, topology, rows, dict(meta, task=task), chunk_size=2)
                for frame in range(3):
                    raw = np.zeros((21, 10), dtype=np.float32)
                    raw[:, :3] = points[index*10+frame]
                    raw[:, 3] = 1
                    raw[:, 7:] = 1
                    writer.add(raw, 1+frame*.02, 100+frame*.02)
                writer.finish()
                export_static(root/task)
            session = dict(meta, status='complete', plan=[{'id': t} for t in tasks],
                           takes=[dict(id=t, status='complete', poll_count=3) for t in tasks],
                           node_info=topology, mp_rows=rows.tolist())
            (root/'session.json').write_text(json.dumps(session))
            self.sessions[split] = root

    def test_preserves_splits_boundaries_poll_indices_and_no_fake_time(self):
        output = self.root/'bundle'
        manifest = prepare(self.sessions, output)
        for split, data in manifest['splits'].items():
            self.assertEqual([c['range'] for c in data['clips']], [[0,3],[3,6]])
            record = load_recording(output/data['keypoints'])
            self.assertEqual(set(record), {'keypoints'})
            self.assertEqual(len(record['keypoints']), 6)
            with np.load(output/data['provenance'], allow_pickle=False) as source:
                np.testing.assert_array_equal(source['clip_id'], [0,0,0,1,1,1])
                np.testing.assert_array_equal(source['source_poll_index'], [0,1,2,0,1,2])
        with self.assertRaises(FileExistsError):
            prepare(self.sessions, output)

    def test_rejects_wrong_split_before_creating_output(self):
        self.sessions['test'] = self.sessions['train']
        with self.assertRaisesRegex(ValueError, 'requested split'):
            prepare(self.sessions, self.root/'bundle')
        self.assertFalse((self.root/'bundle').exists())

    def test_rejects_corrupted_export_and_missing_clip(self):
        path = self.sessions['validation']/'neutral/keypoints.npy'
        values = np.load(path, allow_pickle=False)
        values[0,1,0] += .001
        np.save(path, values, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, 'Export does not match'):
            prepare(self.sessions, self.root/'bundle')
        path = self.sessions['train']/'neutral/metadata.json'
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or unexpected'):
            prepare(self.sessions, self.root/'bundle')

    def test_rejects_relabelled_copied_recording(self):
        import shutil
        source = self.sessions['train']
        target = self.sessions['validation']
        for task in ['neutral', 'little_branch']:
            for file in (source/task).glob('*.npz'):
                shutil.copyfile(file, target/task/file.name)
            for name in ('keypoints.npy', 'source_poll_index.npy'):
                shutil.copyfile(source/task/name, target/task/name)
            export = json.loads((target/task/'export.json').read_text())
            export['keypoints_sha256'] = json.loads((source/task/'export.json').read_text())['keypoints_sha256']
            (target/task/'export.json').write_text(json.dumps(export))
        with self.assertRaisesRegex(ValueError, 'Identical clip reused'):
            prepare(self.sessions, self.root/'bundle')


if __name__ == '__main__':
    unittest.main()
