"""Frozen full-session test comparison: every read and every collision pose."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from geort.coordination import skeleton_features, robot_signature
from geort.coordination_loss import ExactHand
from geort.evaluate_differences import sha, distribution, lp_filter, response, angle, collision_summary
from geort.manus_sessions import load_session_split
from geort.mocap.live_comparison import ROOT, ORIGINAL, WujiWorker
from geort.model import build_ik_model, ik_input
from geort.wuji_baseline import load_wuji_cache

PRIMARY=('R2_lp03','Wuji','SDK')
ALL=(*PRIMARY,'R2_raw','Wuji_raw','Original')


def response_summary(h,r):
    h=h.reshape(-1,3);r=r.reshape(-1,3);magnitude=np.linalg.norm(h,axis=-1)
    return {'moving_ge_point1mm':response(h[magnitude>=.0001],r[magnitude>=.0001]),
            'fine_point5_to10mm':response(h[(magnitude>=.0005)&(magnitude<.01)],r[(magnitude>=.0005)&(magnitude<.01)]),
            'stationary_proxy_mm':distribution(np.linalg.norm(r[magnitude<.0001],axis=-1)*1000),
            'all_transition_finger_count':len(h)}


def freeze(output,manifest):
    config=ROOT/'checkpoint/stage5_R2_seed42/config.json';weights=config.parent/'best.pth'
    decision=json.loads((ROOT/'reports/stage5/final_decision.json').read_text())
    if sha(weights)!=decision['final_weights']['stage5_R2_seed42']:raise ValueError('R2 changed since selection')
    original=ROOT/'checkpoint'/ORIGINAL/'last.pth'
    sdk=ROOT/'reports/stage4/wuji_sdk_20260831_validation/metrics.json'
    inputs=[manifest,config,weights,original,original.parent/'config.json',ROOT/'geort/baselines/wuji_manus_right.json',
            Path(__file__),ROOT/'geort/manus_model.py',ROOT/'geort/coordination_loss.py',ROOT/'geort/kinematics.py',
            ROOT/'geort/env/hand.py',ROOT/'scripts/cache_wuji_baseline.py',ROOT/'scripts/wuji_sdk_worker.py']
    return {'split':'test','seed':42,'checkpoint':'stage5_R2_seed42','primary':PRIMARY,'additional':ALL[3:],
        'robot_signature':robot_signature(json.loads(config.read_text())),
        'scope':'all test01 clips and all retained reads in original order; no frame or task selection; full collision evaluation',
        'state':'reset at every clip; no state carried between clips; R2 and Wuji LP=0.3; SDK internal state unchanged',
        'metrics':'pooled all-read and equal-task axis/opening; all-read joint limits and full collision; within-clip adjacent displacement, moving>=0.1mm and fine 0.5-10mm; no acquisition-time claims',
        'opening':'train-derived fixed relation scales; >=15mm human gap for opening proxy; smaller gaps separately described as site distance, not contact truth',
        'policy':'one frozen test run, no retraining or selection from these results',
        'files_sha256':{str(p.resolve()):sha(p) for p in inputs},
        'sdk_expected':json.loads(sdk.read_text())['sdk']}


def infer(args,recording,protocol,cfg):
    out=args.output;exact=ExactHand(cfg)
    models={}
    for name,path,weight in [('R2_raw',ROOT/'checkpoint/stage5_R2_seed42','best'),('Original',ROOT/'checkpoint'/ORIGINAL,'last')]:
        c=json.loads((path/'config.json').read_text());m=build_ik_model(c).eval()
        m.load_state_dict(torch.load(path/f'{weight}.pth',weights_only=True,map_location='cpu'));models[name]=(m,c)
    worker=WujiWorker(ROOT/'.venv-wuji-sdk/bin/python',None,None,ROOT/'checkpoint/stage5_R2_seed42/config.json',sdk=True)
    if worker.metadata!=protocol['sdk_expected']:
        worker.close();raise ValueError('SDK metadata differs from frozen validation version')
    arrays={};provenance={};durations=[]
    try:
        for seq,clip in enumerate(recording['entry']['clips']):
            task=clip['task'];a,b=clip['range'];points=recording['keypoints'][a:b]
            source=Path(clip['source'])/'keypoints.npy'
            if not np.array_equal(points,np.load(source,allow_pickle=False)):raise ValueError('Source/bundle mismatch')
            cache=out/'wuji'/f'{task}.npz'
            with (out/f'wuji_{task}.log').open('x') as log:
                subprocess.run([str(ROOT/'.venv-wuji-baseline/bin/python'),str(ROOT/'scripts/cache_wuji_baseline.py'),
                    '--source',str(Path.home()/'lerobot-xensehand/third_party/wuji-retargeting'),
                    '--config',str(Path.home()/'lerobot-xensehand/configs/manus_wuji/adaptive_analytical_manus_wuji_hand_2_right.yaml'),
                    '--data',str(source),'--output',str(cache)],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
            cached=load_wuji_cache(cache,source,cfg['joint_order'])
            reply=worker.solve(seq,points,True,timeout=120);sdk=np.asarray(reply['q'],dtype=np.float32)
            durations.extend(reply['step_ms'])
            qs={'Wuji':cached['filtered'],'Wuji_raw':cached['unfiltered'],'SDK':sdk}
            for name,(model,c) in models.items():
                with torch.inference_mode():qs[name]=exact(model(ik_input(c,torch.tensor(points))))['q'].numpy()
            qs['R2_lp03']=lp_filter(qs['R2_raw'])
            for name,q in qs.items():
                if q.shape!=(len(points),20) or not np.isfinite(q).all():raise ValueError('Output shape/value failure')
                arrays[task+'__'+name]=q
            provenance[task]={'frames':len(points),'source_sha256':sha(source),'wuji_cache_sha256':sha(cache),
                'wuji_solver_failures':len(cached['metadata']['optimization_failures'])}
            print('Completed full inference',task,len(points),flush=True)
    finally:worker.close()
    np.savez_compressed(out/'outputs.npz',**arrays)
    (out/'inference.json').write_text(json.dumps({'split':'test','frames':len(recording['keypoints']),
        'sources':provenance,'sdk':worker.metadata,'sdk_native_step_ms':distribution(durations),
        'outputs_sha256':sha(out/'outputs.npz')},indent=2))


def measure(args,recording,cfg):
    out=args.output;exact=ExactHand(cfg);archive=np.load(out/'outputs.npz',allow_pickle=False)
    meta=json.loads((out/'inference.json').read_text())
    if meta['outputs_sha256']!=sha(out/'outputs.npz'):raise ValueError('Output cache changed')
    fields=('axis','opening','q','human_delta','robot_delta','near_gap','human_gap')
    pools={n:{key:[] for key in fields} for n in ALL};by_task={}
    for clip in recording['entry']['clips']:
        task=clip['task'];a,b=clip['range'];points=recording['keypoints'][a:b]
        with torch.inference_mode():human=skeleton_features(torch.tensor(points),[4,8,12,16,20])
        if not human['bone_valid'].all():raise ValueError('Unexpected invalid test bones: report before changing masks')
        ht=human['tips'].numpy();ha=human['axes'][:,:,-1].numpy();hg=np.linalg.norm(ht[:,1:]-ht[:,:1],axis=-1)
        target=hg*np.asarray(cfg['objectives']['relation_scale']);mask=hg>=.015;rows={}
        for name in ALL:
            q=archive[task+'__'+name];norm=2*(torch.tensor(q)-exact.lower)/(exact.upper-exact.lower)-1
            with torch.inference_mode():robot=exact(norm)
            rt=robot['tips'].numpy();ra=robot['axes'][:,:,-1].numpy();rg=np.linalg.norm(rt[:,1:]-rt[:,:1],axis=-1)
            ae=angle(ha,ra);opening=abs(rg-target)*1000;hd=np.diff(ht,axis=0);rd=np.diff(rt,axis=0)
            rows[name]={'axis_deg':distribution(ae),'opening_mm':distribution(opening[mask]),
                'response':response_summary(hd,rd),'joint_limit_violations':int((norm.abs()>1.00001).sum())}
            for key,value in [('axis',ae),('opening',opening[mask]),('q',q),('human_delta',hd),('robot_delta',rd),
                              ('near_gap',rg*1000),('human_gap',hg*1000)]:pools[name][key].append(value)
        by_task[task]={'frames':b-a,'models':rows}
    aggregate={}
    for name,pool in pools.items():
        v={k:np.concatenate(rows) for k,rows in pool.items() if rows}
        hg=v['human_gap'];near=v['near_gap'];q=v['q']
        aggregate[name]={'frames':len(q),'axis_deg':distribution(v['axis']),
            'axis_by_finger':{f:distribution(v['axis'][:,i]) for i,f in enumerate(['thumb','index','middle','ring','little'])},
            'opening_mm':distribution(v['opening']),
            'equal_task_axis_deg':float(np.mean([r['models'][name]['axis_deg']['mean'] for r in by_task.values()])),
            'equal_task_opening_mm':float(np.mean([r['models'][name]['opening_mm']['mean'] for r in by_task.values()])),
            'response':response_summary(v['human_delta'],v['robot_delta']),
            'joint_limit_violations':sum(r['models'][name]['joint_limit_violations'] for r in by_task.values()),
            'severe_pip_dip_fraction':float(np.any(q[:,[6,7,10,11,14,15,18,19]]<-np.deg2rad(15),axis=1).mean()),
            'near_pinch_site_mm':{f:distribution(near[:,i][hg[:,i]<15]) for i,f in enumerate(['index','middle','ring','little'])}}
    # Every read is measured. Save chunks so expensive completed geometry is recoverable.
    from geort.env.hand import HandKinematicModel
    with contextlib.redirect_stdout(io.StringIO()):hand=HandKinematicModel.build_from_config(cfg,render=False)
    collision={};depths={};(out/'collision_chunks').mkdir(exist_ok=True)
    for name in (*PRIMARY,'Original'):
        q=np.concatenate(pools[name]['q']);chunks=[]
        for start in range(0,len(q),1000):
            path=out/'collision_chunks'/f'{name}_{start:06}.npy'
            if path.exists():d=np.load(path,allow_pickle=False)
            else:
                with contextlib.redirect_stderr(io.StringIO()):d=hand.self_collision_depth(q[start:start+1000])*1000
                np.save(path,d)
            if d.shape!=(min(1000,len(q)-start),) or not np.isfinite(d).all():raise ValueError('Invalid collision chunk')
            chunks.append(d);print('Collision full',name,min(start+1000,len(q)),'/',len(q),flush=True)
        depth=np.concatenate(chunks);depths[name]=depth;collision[name]=collision_summary(depth)
        for clip in recording['entry']['clips']:
            a,b=clip['range'];by_task[clip['task']]['models'][name]['collision']=collision_summary(depth[a:b])
        aggregate[name]['collision']=collision[name]
    np.savez_compressed(out/'collision_depths.npz',**depths)
    report={'split':'test','session':recording['entry']['session_id'],'frames':len(recording['keypoints']),
        'all_clips':[c['task'] for c in recording['entry']['clips']], 'primary':PRIMARY,
        'aggregate':aggregate,'by_task':by_task,'provenance':meta,'protocol_sha256':sha(out/'protocol.json'),
        'scope':'all retained test reads; all primary collision poses; primary pooled read-weighted metrics; equal-task means also provided; no cherry-picked clips',
        'limitations':'same operator/calibration, one held-out session; no sensor timestamps/contact truth; fixed train geometry proxies; no training or model selection after test'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('COMPLETE',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'data/manus_stage4_v1/manifest.json')
    p.add_argument('--measure-only',action='store_true',help='Resume only frozen cached evaluation; no new model inference')
    args=p.parse_args();torch.set_num_threads(1);torch.manual_seed(42)
    if args.measure_only:
        protocol=json.loads((args.output/'protocol.json').read_text())
        for path,digest in protocol['files_sha256'].items():
            if sha(path)!=digest:raise ValueError('Frozen evaluation dependency changed: '+path)
        if (args.output/'metrics.json').exists():raise FileExistsError('Completed results already exist')
    else:
        protocol=freeze(args.output,args.manifest);args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
        (args.output/'evaluator_source.py').write_bytes(Path(__file__).read_bytes())
    cfg=json.loads((ROOT/'checkpoint/stage5_R2_seed42/config.json').read_text())
    recording=load_session_split(args.manifest,'test')
    if not args.measure_only:infer(args,recording,protocol,cfg)
    measure(args,recording,cfg)


if __name__=='__main__':main()
