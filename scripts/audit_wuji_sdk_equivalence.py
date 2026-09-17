"""Separate controllable SDK comparison conditions from opaque native settings.

Run with the GeoRT Python. The legacy solver runs in its existing isolated env.
All poses come from validation; no devices, training, or test-set access.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_asset(binary):
    candidates = []
    for match in re.finditer(rb'<robot\s[^>]*>',binary):
        end = binary.find(b'</robot>',match.end())
        if end < 0: continue
        raw = binary[match.start():end+8]
        try: robot = ET.fromstring(raw)
        except ET.ParseError: continue
        if robot.find("joint[@name='r_pinky_mcp_flex']") is not None:
            candidates.append(raw)
    if len(candidates) != 1:
        raise ValueError('Embedded anatomical right Hand2 asset is missing or ambiguous')
    return candidates[0]


def legacy(args):
    import yaml
    source = args.source.resolve()
    spec = json.loads((ROOT/'geort/baselines/wuji_manus_right.json').read_text())
    commit = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
    if dirty or commit != spec['source_commit'] or sha(args.production_config) != spec['config_sha256']:
        raise ValueError('Confirmed production source/config changed')
    sys.path.insert(0,str(source))
    from wuji_retargeting import Retargeter
    cfg = yaml.safe_load(args.production_config.read_text())
    cfg['__yaml_dir'] = str(args.production_config.resolve().parent)
    native = (args.production_config.parent/cfg['optimizer']['urdf_path']).resolve()
    if sha(native) != spec['native_urdf_sha256']: raise ValueError('Production asset changed')
    variants = {'production':cfg}
    for name, asset, scale in [('sdk_asset',True,False),('unit_segments',False,True),('sdk_asset_unit_segments',True,True)]:
        c = copy.deepcopy(cfg)
        if asset: c['optimizer']['urdf_path'] = str((args.output/'embedded_hand2_right.urdf').resolve())
        if scale: c['retarget']['segment_scaling'] = {f:[1.,1.,1.] for f in ('thumb','index','middle','ring','pinky')}
        variants[name] = c
    inputs = np.load(args.output/'inputs.npz',allow_pickle=False)['points']
    joint_order = json.loads(args.joint_config.read_text())['joint_order']
    results = {}; records = {}
    for name,c in variants.items():
        ret = Retargeter.from_config(c,hand_side='right')
        order = [list(ret.optimizer.robot.dof_joint_names).index(n) for n in joint_order]
        holds = []
        for p in inputs:
            ret.reset()
            holds.append([np.asarray(ret.retarget(p))[order].copy() for _ in range(args.hold_steps)])
        results[name] = np.asarray(holds)
        records[name] = c
        print('Finished legacy control',name,flush=True)
    np.savez_compressed(args.output/'legacy_outputs.npz',**results)
    (args.output/'legacy_configs.json').write_text(json.dumps(records,indent=2))


def main(args):
    from geort.manus_sessions import load_session_split
    from geort.mocap.live_comparison import WujiWorker
    if args.output.exists(): raise FileExistsError(args.output)
    if args.hold_steps < 20: raise ValueError('Hold requires at least 20 steps')
    recording = load_session_split(args.manifest,'validation')
    points = []; provenance = []
    for clip in recording['entry']['clips']:
        a,b = clip['range']
        for index in np.linspace(a,b-1,8,dtype=int):
            points.append(recording['keypoints'][index])
            provenance.append({'task':clip['task'],'local_frame':int(index-a)})
    points = np.asarray(points)
    args.output.mkdir(parents=True,exist_ok=False)
    binaries = list(args.sdk_python.absolute().parent.parent.glob('lib/python*/site-packages/wuji_sdk/*.so'))
    if len(binaries) != 1: raise ValueError('SDK binary identification ambiguous')
    asset = extract_asset(binaries[0].read_bytes())
    (args.output/'embedded_hand2_right.urdf').write_bytes(asset)
    np.savez_compressed(args.output/'inputs.npz',points=points)
    worker = WujiWorker(args.sdk_python,None,None,args.joint_config,sdk=True)
    report = {'schema':1,'seed':42,'split':'validation','test_accessed':False,'samples':provenance,
              'hold_steps':args.hold_steps,'manifest_sha256':sha(args.manifest),
              'common_config_sha256':sha(args.joint_config),'sdk':worker.metadata,
              'script_sha256':sha(__file__),'embedded_asset_sha256':hashlib.sha256(asset).hexdigest(),
              'embedded_asset_scope':'Unique embedded right Hand2 joint set; runtime use not exposed/attested by public API',
              'strict_internal_equivalence':False,
              'unresolved':['SDK objective/scales','SDK solver settings','SDK internal LP and raw output','SDK runtime config/asset export'],
              'protocol':'72 validation poses; same finite meters/right/MediaPipe inputs; reset each pose and hold 80 calls by default; average final 10; name-mapped radians; common evaluation FK; no extra filter or clipping',
              'controls':'Changing old solver asset/scales diagnoses those factors; does not reproduce unknown SDK objectives'}
    sdk = []; rigid = []
    rot = np.array([[0,-1,0],[1,0,0],[0,0,1]],dtype=np.float32)
    try:
        for i,p in enumerate(points):
            sdk.append(worker.solve(2*i,np.repeat(p[None],args.hold_steps,axis=0),True,timeout=30)['q'])
            changed = p@rot.T+np.array([.1,.2,.3],dtype=np.float32)
            rigid.append(worker.solve(2*i+1,np.repeat(changed[None],args.hold_steps,axis=0),True,timeout=30)['q'])
    finally: worker.close()
    sdk = np.asarray(sdk);rigid = np.asarray(rigid)
    np.savez_compressed(args.output/'sdk_outputs.npz',q=sdk,rigid=rigid)
    command=[str(args.legacy_python),str(Path(__file__).resolve()),'--legacy-phase','--output',str(args.output),
             '--source',str(args.source),'--production-config',str(args.production_config),
             '--joint-config',str(args.joint_config),'--hold-steps',str(args.hold_steps)]
    subprocess.run(command,check=True)
    old = np.load(args.output/'legacy_outputs.npz',allow_pickle=False)
    steady = sdk[:,-10:].mean(1)
    report['rigid_transform_max_joint_delta_deg'] = float(np.degrees(abs(steady-rigid[:,-10:].mean(1))).max())
    report['methods'] = {}
    for name,q in [('SDK',sdk), *[(n,old[n]) for n in old.files]]:
        delta = np.degrees(abs(q[:,-10:].mean(1)-steady))
        report['methods'][name] = {'sdk_joint_mae_deg':float(delta.mean()),
            'sdk_joint_mae_by_finger_deg':delta.reshape(-1,5,4).mean((0,2)).tolist(),
            'max_last10_joint_step_deg':float(np.degrees(abs(np.diff(q[:,-10:],axis=1))).max())}
    from geort.coordination_loss import ExactHand
    import torch
    torch.set_num_threads(1)
    exact=ExactHand(json.loads(args.joint_config.read_text()))
    with torch.inference_mode():
        sdk_tips=exact(2*(torch.tensor(steady,dtype=torch.float32)-exact.lower)/(exact.upper-exact.lower)-1)['tips']
        for name in old.files:
            q=torch.tensor(old[name][:,-10:].mean(1),dtype=torch.float32)
            tips=exact(2*(q-exact.lower)/(exact.upper-exact.lower)-1)['tips']
            report['methods'][name]['sdk_common_fk_tip_mean_distance_mm']=float((tips-sdk_tips).norm(dim=-1).mean()*1000)
    report['artifacts']={p.name:sha(p) for p in args.output.iterdir() if p.is_file()}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report['methods'],indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'data/manus_stage4_v1/manifest.json')
    p.add_argument('--joint-config',type=Path,default=ROOT/'checkpoint/stage4_M1_seed42/config.json')
    p.add_argument('--sdk-python',type=Path,default=ROOT/'.venv-wuji-sdk/bin/python')
    p.add_argument('--legacy-python',type=Path,default=ROOT/'.venv-wuji-baseline/bin/python')
    p.add_argument('--source',type=Path,default=Path.home()/'lerobot-xensehand/third_party/wuji-retargeting')
    p.add_argument('--production-config',type=Path,default=Path.home()/'lerobot-xensehand/configs/manus_wuji/adaptive_analytical_manus_wuji_hand_2_right.yaml')
    p.add_argument('--hold-steps',type=int,default=80)
    p.add_argument('--legacy-phase',action='store_true')
    args=p.parse_args()
    if args.legacy_phase:legacy(args)
    else:main(args)
