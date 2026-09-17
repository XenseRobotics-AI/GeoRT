import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from geort.wuji_baseline import load_wuji_cache, SPEC


class WujiCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'input.npy'
        np.save(self.data, np.zeros((3, 21, 3)))
        self.meta = dict(json.loads(SPEC.read_text()), data_sha256=hashlib.sha256(self.data.read_bytes()).hexdigest())
        self.values = dict(metadata=json.dumps(self.meta), joint_names=['b','a'], source_frame=np.arange(3),
                           qpos_filtered=np.array([[1.,2.]]*3), qpos_unfiltered=np.array([[3.,4.]]*3), nlopt_code=[1,1,1])
        self.cache = self.root / 'cache.npz'

    def load(self):
        np.savez(self.cache, **self.values)
        return load_wuji_cache(self.cache, self.data, ['a','b'])

    def test_mapping_and_filter_variants(self):
        result = self.load()
        np.testing.assert_array_equal(result['filtered'], [[2,1]]*3)
        np.testing.assert_array_equal(result['unfiltered'], [[4,3]]*3)

    def test_recording_hash_and_config_rejection(self):
        self.meta['data_sha256'] = 'wrong'
        self.values['metadata'] = json.dumps(self.meta)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): self.load()
        self.meta['config_sha256'] = 'wrong'
        self.values['metadata'] = json.dumps(self.meta)
        with self.assertRaisesRegex(ValueError, 'specification'): self.load()

    def test_missing_duplicate_and_nonfinite_rejection(self):
        for names in [['a','c'], ['a','a']]:
            self.values['joint_names'] = names
            with self.assertRaisesRegex(ValueError, 'joint names'): self.load()
        self.values['joint_names'] = ['a','b']
        self.values['qpos_filtered'][0,0] = np.nan
        with self.assertRaisesRegex(ValueError, 'values'): self.load()

    def test_frame_alignment(self):
        self.values['source_frame'] = [0,2,1]
        with self.assertRaisesRegex(ValueError, 'frame'): self.load()


if __name__ == '__main__': unittest.main()
