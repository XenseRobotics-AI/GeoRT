"""Full validation clips with independent Wuji state and common exact geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from geort.coordination import skeleton_features
from geort.coordination_loss import ExactHand
from geort.export import resolve_checkpoint
from geort.manus_sessions import load_session_split
from geort.model import build_ik_model, ik_input
from geort.wuji_baseline import load_wuji_cache, operational_metrics


def metrics(q, exact, points):
    with torch.inference_mode():
        norm=2*(torch.tensor(q)-exact.lower)/(exact.upper-exact.lower)-1
        robot=exact(norm)
        human=skeleton_features(torch.tensor(points),[4,8,12,16,20])
        angles=torch.rad2deg(torch.acos((robot['axes'][:,:,-1]*human['axes'][:,:,-1]).sum(-1).clamp(-1,1))).numpy()
    names=['thumb','index','middle','ring','little']
    return {'axis_mean_deg_by_finger':dict(zip(names,angles.mean(0).astype(float).tolist())),
        'severe_pip_dip_fraction':{n:float(np.any(q[:,4*i+2:4*i+4]<-np.deg2rad(15),axis=1).mean()) for i,n in enumerate(names) if i},
        'mcp_below_minus15_fraction':{n:float((q[:,4*i]<-np.deg2rad(15)).mean()) for i,n in enumerate(names) if i},
        'saturated_joint_fraction':float((norm.abs()>.95).float().mean()),
        'operational':operational_metrics(q,exact,points)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--checkpoints',nargs='+',default=['stage4_M0_seed42','stage4_M1_seed42'])
    parser.add_argument('--wuji-cache-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    recording=load_session_split(args.manifest,'validation')
    models={};provenance={}
    tags={'original':('wuji_hand2_beta1_right_2026-09-15_18-01-42_std_collision_seed0_w0','last'),
          'previous_H1':('stage3_H1_seed42','best'),**{tag:(tag,'best') for tag in args.checkpoints}}
    for name,(tag,weight) in tags.items():
        path=resolve_checkpoint(tag);cfg=json.loads((path/'config.json').read_text())
        if name in args.checkpoints:
            meta=json.loads((path/'experiment.json').read_text())
            if meta['manifest_sha256']!=hashlib.sha256(args.manifest.read_bytes()).hexdigest():raise ValueError('Training manifest mismatch')
        model=build_ik_model(cfg).eval()
        weights=path/f'{weight}.pth';model.load_state_dict(torch.load(weights,weights_only=True,map_location='cpu'))
        models[name]=(model,cfg)
        provenance[name]={'weights':str(weights),'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest()}
    exact=ExactHand(models[args.checkpoints[-1]][1])
    results={};arrays={};caches={}
    for clip in recording['entry']['clips']:
        task=clip['task'];a,b=clip['range'];points=recording['keypoints'][a:b]
        source=Path(clip['source'])/'keypoints.npy'
        if not np.array_equal(np.load(source,allow_pickle=False),points):raise ValueError('Source differs from bundle')
        cache=load_wuji_cache(args.wuji_cache_dir/(task+'.npz'),source,exact.config['joint_order'])
        caches[task]={'sha256':cache['cache_sha256'],'failures':len(cache['metadata']['optimization_failures']),
                      'state_policy':cache['metadata']['sequence_policy']}
        sample=torch.tensor(points);outputs={}
        for name,(model,cfg) in models.items():
            with torch.inference_mode():outputs[name]=exact(model(ik_input(cfg,sample)))['q'].numpy()
        outputs['wuji_filtered']=cache['filtered'];outputs['wuji_unfiltered']=cache['unfiltered']
        results[task]={name:metrics(q,exact,points) for name,q in outputs.items()}
        for name,q in outputs.items():arrays[task+'__'+name]=q
        print(task,{name:round(r['operational']['distal_axis_error_deg']['mean'],2) for name,r in results[task].items()},flush=True)
    # Equal clip means; movement distributions stay within clips, never concatenate differences.
    aggregate={}
    for name in next(iter(results.values())):
        rows=[r[name] for r in results.values()]
        aggregate[name]={
            'equal_task_axis_mean_deg':float(np.mean([r['operational']['distal_axis_error_deg']['mean'] for r in rows])),
            'equal_task_opening_mean_mm':float(np.mean([r['operational']['opening_error_mm_open_poses']['mean'] for r in rows if r['operational']['opening_error_mm_open_poses'] is not None])),
            'equal_task_saturated_joint_fraction':float(np.mean([r['saturated_joint_fraction'] for r in rows])),
            'equal_task_severe_pip_dip_fraction':{f:float(np.mean([r['severe_pip_dip_fraction'][f] for r in rows])) for f in ['index','middle','ring','little']}}
    args.output.mkdir(parents=True,exist_ok=False)
    report={'split':'validation','session':recording['entry']['session_id'],'frames':len(recording['keypoints']),
        'test_model_inference':False,'scope':'common exact FK; individual clips; host poll ordering, no sensor timing; no contact truth',
        'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'models':provenance,'wuji_caches':caches,'aggregate':aggregate,'by_task':results}
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    with (args.output/'outputs.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
