"""Evaluate the pinned SDK on validation only, with per-clip state reset."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from geort.coordination_loss import ExactHand
from geort.evaluate_differences import sha, distribution
from geort.evaluate_manus import metrics
from geort.manus_sessions import load_session_split
from geort.mocap.live_comparison import ROOT, WujiWorker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'data/manus_stage4_v1/manifest.json')
    p.add_argument('--config',type=Path,default=ROOT/'checkpoint/stage4_M1_seed42/config.json')
    p.add_argument('--sdk-python',type=Path,default=ROOT/'.venv-wuji-sdk/bin/python')
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(1)
    recording = load_session_split(args.manifest,'validation')
    exact = ExactHand(json.loads(args.config.read_text()))
    worker = WujiWorker(args.sdk_python,None,None,args.config,sdk=True)
    report = {'split':'validation','test_model_inference':False,'seed':42,
              'manifest_sha256':sha(args.manifest),'config_sha256':sha(args.config),
              'frames':len(recording['keypoints']),'sdk':worker.metadata,
              'scope':'Common GeoRT FK; SDK builtin asset/config opaque; internal LP retained, alpha unknown; no hardware/contact truth',
              'sequence_policy':'reset at each clip; all recorded host polls in original order',
              'sources':{},'by_task':{}}
    arrays = {}; durations = []
    try:
        for seq,clip in enumerate(recording['entry']['clips']):
            task = clip['task']; a,b = clip['range']; points = recording['keypoints'][a:b]
            source = Path(clip['source'])/'keypoints.npy'
            if not np.array_equal(points,np.load(source,allow_pickle=False)):
                raise ValueError('Recording/bundle mismatch')
            reply = worker.solve(seq,points,True,timeout=120)
            q = np.asarray(reply['q'],dtype=np.float32)
            if q.shape != (len(points),20): raise ValueError('SDK frame count mismatch')
            arrays[task+'__SDK'] = q
            durations.extend(reply['step_ms'])
            report['sources'][task] = {'sha256':sha(source),'frames':len(points)}
            report['by_task'][task] = metrics(q,exact,points)
            print(task,report['by_task'][task]['operational']['distal_axis_error_deg']['mean'],flush=True)
    finally: worker.close()
    report['native_step_ms'] = distribution(durations)
    report['timing_scope'] = 'sequential step calls; includes internal solver/filter, excludes IPC/glove/render; not end-to-end latency'
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/'outputs.npz',**arrays)
    report['outputs_sha256'] = sha(args.output/'outputs.npz')
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('Saved',args.output,flush=True)


if __name__ == '__main__': main()
