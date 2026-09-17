import threading
import time
from types import SimpleNamespace
import unittest
import numpy as np
from geort.mocap.live_comparison import ManusSource, Pipeline, validate_q, target_gap
from geort.mocap.manus_capture_data import canonicalize, mp_mapping
from tests.test_manus_capture import nodes


class LiveComparisonTests(unittest.TestCase):
    def source(self):
        points=np.load('data/human_alex.npy',allow_pickle=False)[0]
        raw=np.zeros((21,10));raw[:,:3]=points;raw[:,3]=1;raw[:,7:]=1
        node_info=nodes()
        glove=SimpleNamespace(is_connected=lambda:True,get_glove_id=lambda side:42,
            get_raw_skeleton=lambda side:raw,get_node_info=lambda side:[SimpleNamespace(**n) for n in node_info])
        source=ManusSource.__new__(ManusSource)
        source.glove=glove;source.glove_id=42;source.rows=mp_mapping(node_info)
        source.nodes=node_info;source.last_topology=time.monotonic()
        return source,raw

    def test_live_mapping_matches_capture_and_invalid_does_not_return_zero(self):
        source,raw=self.source()
        np.testing.assert_array_equal(source.read(),canonicalize(raw[:,:3])[0])
        raw[0,3]=0
        with self.assertRaises(ValueError):source.read()
        raw[0,3]=1;source.glove.is_connected=lambda:False
        with self.assertRaisesRegex(ValueError,'disconnected'):source.read()

    def test_identity_or_topology_change_is_fatal(self):
        source,_=self.source();source.glove.get_glove_id=lambda side:43
        with self.assertRaisesRegex(RuntimeError,'identity'):source.read()
        source.glove.get_glove_id=lambda side:42;source.last_topology=0
        source.glove.get_node_info=lambda side:[]
        with self.assertRaisesRegex(RuntimeError,'topology'):source.read()

    def test_outputs_reject_nan_limits_and_shape(self):
        cfg={'joint':{'lower':[-1.,-2.],'upper':[1.,2.]}}
        np.testing.assert_array_equal(validate_q([.2,.4],cfg),[.2,.4])
        for q in [[3,0],[np.nan,0],[1]]:
            with self.assertRaises(ValueError):validate_q(q,cfg)

    def test_queue_replaces_pending_input_without_mixing_completed_frames(self):
        entered=threading.Event();release=threading.Event();done=threading.Event()
        class Engine:
            def compute(self,seq,*args):
                if seq==0:entered.set();release.wait(2)
                if seq==2:done.set()
                return {'seq':seq,'points':np.full((21,3),seq),'q':{'M1':seq,'Wuji':seq}}
            def close(self):pass
        p=Pipeline(Engine())
        try:
            p.submit((0,));self.assertTrue(entered.wait(1))
            p.submit((1,));p.submit((2,));release.set();self.assertTrue(done.wait(1))
            p.stop.set();p.thread.join(1)
            result,error=p.snapshot();self.assertIsNone(error)
            self.assertEqual(result['seq'],2);self.assertEqual(result['q']['Wuji'],2)
            self.assertEqual(p.dropped,1)
        finally:release.set();p.close()

    def test_dropped_pending_frame_preserves_filter_reset(self):
        entered=threading.Event();release=threading.Event();done=threading.Event();received=[]
        class Engine:
            def compute(self,seq,stamp,points,reset):
                if seq==0:entered.set();release.wait(2)
                received.append((seq,reset))
                if seq==2:done.set()
                return {'seq':seq}
            def close(self):pass
        p=Pipeline(Engine())
        try:
            p.submit((0,0,None,False));self.assertTrue(entered.wait(1))
            p.submit((1,1,None,True));p.submit((2,2,None,False));release.set()
            self.assertTrue(done.wait(1));p.stop.set();p.thread.join(1)
            self.assertEqual(received,[(0,False),(2,True)])
        finally:release.set();p.close()

    def test_pipeline_surfaces_solver_error_and_target_is_bounded(self):
        event=threading.Event()
        class Engine:
            def compute(self,*args):event.set();raise ValueError('solver failed')
            def close(self):pass
        p=Pipeline(Engine())
        try:
            p.submit((1,));self.assertTrue(event.wait(1));p.thread.join(1)
            result,error=p.snapshot();self.assertIsNone(result);self.assertIn('solver failed',error)
        finally:p.close()
        values=target_gap(np.linspace(0,12,1000))
        self.assertTrue(np.all((values>=.01)&(values<=.04)))
