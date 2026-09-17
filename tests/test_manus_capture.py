import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from geort.mocap.manus_capture_data import (CaptureWriter,FINGER_SPEC,canonicalize,export_static,load_capture,mp_mapping)
from geort.mocap.inspect_manus_capture import summarize_capture
from geort.mocap.record_manus import record_clip
from geort.coordination_data import load_recording
from geort.mocap.replay_mocap import ReplayMocap


def nodes():
    result=[{'node_id':0,'parent_id':0,'chain_type':13,'finger_joint_type':0,'side':2}]
    result += [{'node_id':mp,'parent_id':0,'chain_type':chain,'finger_joint_type':joint,'side':2} for mp,chain,joint in FINGER_SPEC]
    return result


class ManusCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.points=np.load('data/human_alex.npy')[:10]
        self.nodes=nodes();self.rows=mp_mapping(self.nodes)
        self.meta={'session_id':'test_fixture','operator':'test','split':'pilot','task':'little_branch','source_type':'test_fixture_not_live_manus'}

    def raw(self,i=0):
        raw=np.zeros((21,10),dtype=np.float32);raw[:,:3]=self.points[i];raw[:,3]=1;raw[:,7:]=1
        return raw

    def test_mapping_matches_production_and_rejects_ambiguity(self):
        np.testing.assert_array_equal(self.rows,np.arange(21))
        reordered=list(reversed(self.nodes));rows=mp_mapping(reordered)
        self.assertEqual([reordered[i]['node_id'] for i in rows],list(range(21)))
        with self.assertRaises(ValueError):mp_mapping(self.nodes[:-1])
        with self.assertRaises(ValueError):mp_mapping(self.nodes+[self.nodes[1]])

    def test_canonical_rigid_invariance_and_reconstruction(self):
        rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        original,t=canonicalize(self.points[0]);moved,t2=canonicalize(self.points[0]@rotation.T+[.4,-.1,.2])
        np.testing.assert_allclose(original,moved,atol=1e-7)
        np.testing.assert_allclose(original@t[:3,:3].T+t[:3,3],self.points[0],atol=1e-7)
        np.testing.assert_allclose(t2[:3,:3].T@t2[:3,:3],np.eye(3),atol=1e-7)
        with self.assertRaises(ValueError):canonicalize(np.zeros((21,3)))
        with self.assertRaises(ValueError):canonicalize(self.points[0]*1000)

    def test_chunks_invalid_duplicates_and_static_export(self):
        path=self.root/'take';w=CaptureWriter(path,self.nodes,self.rows,self.meta,chunk_size=2)
        self.assertTrue(w.add(self.raw(),1.,10.)[0]);self.assertTrue(w.add(self.raw(),1.02,10.02)[1])
        bad=self.raw();bad[19,0]=np.nan
        self.assertFalse(w.add(bad,1.04,10.04)[0]);w.add(self.raw(1),1.06,10.06);w.finish()
        meta,data=load_capture(path);self.assertEqual(meta['completed_chunks'],2)
        np.testing.assert_array_equal(data['valid_frame'],[True,True,False,True])
        self.assertTrue(np.isnan(data['raw_skeleton'][2,19,0]))
        output=export_static(path);record=load_recording(output)
        self.assertEqual(record['keypoints'].shape,(3,21,3));self.assertNotIn('timestamps',record)
        replay=ReplayMocap(str(output));np.testing.assert_array_equal(replay.get()['result'],record['keypoints'][0])
        self.assertEqual(replay.data_path,output.resolve())
        np.testing.assert_array_equal(np.load(path/'source_poll_index.npy'),[0,1,3])
        self.assertFalse(json.loads((path/'export.json').read_text())['temporal_training_allowed'])
        with self.assertRaises(FileExistsError):export_static(path)
        with self.assertRaises(FileExistsError):CaptureWriter(path,self.nodes,self.rows,self.meta)

    def test_failed_shapes_and_clocks_are_not_silently_accepted(self):
        w=CaptureWriter(self.root/'take',self.nodes,self.rows,self.meta)
        self.assertFalse(w.add(np.zeros((0,10)),1.,1.)[0])
        with self.assertRaises(ValueError):w.add(self.raw(),1.,2.)
        self.assertFalse(w.add(self.raw(),2.,2.,source_connected=False)[0])
        raw=self.raw();raw[:,3:7]=0
        self.assertFalse(w.add(raw,3.,3.)[0]);w.finish('error')
        self.assertTrue((w.path/'rejected_000000.npz').exists())
        self.assertIsNone(export_static(w.path))

    def test_information_diagnostic_detects_actual_axis_difference(self):
        w=CaptureWriter(self.root/'take',self.nodes,self.rows,self.meta)
        a=self.raw();b=a.copy();tip=b[20,:3].copy();length=np.linalg.norm(tip-b[19,:3])
        axis=(tip-b[19,:3])/length
        alternative=np.cross(axis,[1.,0.,0.]);alternative/=np.linalg.norm(alternative)
        b[19,:3]=tip-alternative*length
        self.assertTrue(w.add(a,1.,1.)[0]);self.assertTrue(w.add(b,2.,2.)[0]);w.finish()
        report=summarize_capture(w.path)
        self.assertEqual(report['information_pairs']['little']['tip_2mm_axis_20deg_pairs'],1)
        self.assertFalse(report['temporal_training_allowed'])

    def test_interrupt_flushes_before_native_shutdown(self):
        raw=self.raw()
        class FakeGlove:
            def __init__(self):self.count=0
            def is_connected(self):return True
            def get_raw_skeleton(self,side):
                self.count+=1
                if self.count==4:raise KeyboardInterrupt()
                return raw.copy()
        path=self.root/'interrupted'
        with self.assertRaises(KeyboardInterrupt):record_clip(FakeGlove(),path,self.nodes,self.rows,self.meta,1.,240.)
        meta,data=load_capture(path);self.assertEqual(meta['status'],'interrupted');self.assertEqual(len(data['keypoints']),3)
        self.assertTrue((path/'keypoints.npy').exists())


if __name__=='__main__':unittest.main()
