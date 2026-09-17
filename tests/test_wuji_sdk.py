"""Integration checks for the optional pinned native SDK, without devices."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from geort.mocap.live_comparison import ROOT, WujiWorker

PYTHON = ROOT/'.venv-wuji-sdk/bin/python'
CONFIG = ROOT/'checkpoint/stage4_M1_seed42/config.json'
SOURCE = ROOT/'data/manus/val01/micro/keypoints.npy'


@unittest.skipUnless(PYTHON.exists() and CONFIG.exists() and SOURCE.exists(), 'Optional SDK / validation artifacts unavailable')
class WujiSDKTests(unittest.TestCase):
    def test_sequential_batch_reset_and_firmware_name_reorder(self):
        points = np.load(SOURCE,allow_pickle=False)[:32]
        worker = WujiWorker(PYTHON,None,None,CONFIG,sdk=True)
        try:
            self.assertEqual(worker.metadata['version'],'2026.8.31')
            batch = np.asarray(worker.solve(0,points,True)['q'])
            frames = np.array([worker.solve(i+1,p,i==0)['q'] for i,p in enumerate(points)])
            np.testing.assert_array_equal(batch,frames)
            worker.solve(34,points[-1])
            np.testing.assert_array_equal(worker.solve(35,points[0],True)['q'],batch[0])
            cfg = json.loads(CONFIG.read_text());cfg['joint_order'].reverse()
            for key in ('lower','upper'):cfg['joint'][key].reverse()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'config.json';path.write_text(json.dumps(cfg))
                reverse = WujiWorker(PYTHON,None,None,path,sdk=True)
                try: np.testing.assert_array_equal(reverse.solve(0,points,True)['q'],batch[:,::-1])
                finally: reverse.close()
        finally: worker.close()

    def test_invalid_input_surfaces_error_without_zero_pose(self):
        worker = WujiWorker(PYTHON,None,None,CONFIG,sdk=True)
        try:
            with self.assertRaisesRegex(RuntimeError,'Expected finite'):
                worker.solve(0,np.zeros((20,3)),True)
        finally: worker.close()


if __name__ == '__main__': unittest.main()
