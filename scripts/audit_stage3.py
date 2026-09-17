"""Audit frozen-base identity, head capacity and real information-ambiguity pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from geort.coordination import skeleton_features
from geort.coordination_loss import ExactHand
from geort.model import build_ik_model
from geort.stage_metrics import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    root=Path('checkpoint/stage2_T2_seed42');cfg=json.loads((root/'config.json').read_text())
    frozen=build_ik_model(cfg).eval();frozen.load_state_dict(torch.load(root/'best.pth',map_location='cpu',weights_only=True))
    meta=json.loads((root/'experiment.json').read_text());points=torch.tensor(np.load(meta['data']),dtype=torch.float32)
    with np.load('reports/stage2/train_targets_seed42.npz',allow_pickle=False) as f:teacher=torch.tensor(f['normalized'],dtype=torch.float32)
    exact=ExactHand(cfg);ids=[4,8,12,16,20]
    output={'models':{},'information_pairs':{},'scope':'Validation snapshots only; same partition real pairs, at least 30 source rows apart, tip distance <=2mm, distal axis difference >=20deg. No timestamp or independent-session claim.'}
    pcfg=json.loads((root/'baseline_config.json').read_text());parent=build_ik_model(pcfg).eval();parent.load_state_dict(torch.load(root/'baseline.pth',map_location='cpu',weights_only=True))
    with torch.inference_mode():
        ref=exact(parent(points[2100:2448,ids]));base=frozen(points[:2000]);lo=exact(torch.tanh(torch.atanh(base)-.1))['q'];hi=exact(torch.tanh(torch.atanh(base)+.1))['q'];target=exact(teacher)['q']
        outside=((target<lo-np.deg2rad(.5))|(target>hi+np.deg2rad(.5))).reshape(-1,5,4).any(-1)
        output['teacher_outside_fixed_residual_fraction_per_finger']=outside.float().mean(0).tolist()
        output['frozen_base_train_teacher_normalized_mse']=float((base-teacher).square().mean())
        bq=exact(base)['q']
        output['max_allowed_joint_change_deg']=float(torch.rad2deg(torch.maximum(hi-bq,bq-lo)).max())
        models={'frozen_T2':frozen}
        for arm in ('H0','H1'):
            path=Path(f'checkpoint/stage3_{arm}_seed42');config=json.loads((path/'config.json').read_text());model=build_ik_model(config).eval()
            record={'snapshots':[]}
            for weights in sorted(path.glob('step_*.pth'),key=lambda p:int(p.stem[5:])):
                model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
                if any(not torch.equal(v,model.base.state_dict()[k]) for k,v in frozen.base.state_dict().items()):raise RuntimeError('Frozen base changed')
                r=exact(model(points[2100:2448]));s=summarize(r['q'].numpy(),r['tips'].numpy(),ref['q'].numpy(),ref['tips'].numpy(),points[2100:2448].numpy())
                record['snapshots'].append({'step':int(weights.stem[5:]),'gates':s['gates'],'passes_observed_gates':s['passes_observed_gates']})
                if weights.stem=='step_1000':record['warmup_train_teacher_normalized_mse']=float((model(points[:2000])-teacher).square().mean())
            model.load_state_dict(torch.load(path/'best.pth',map_location='cpu',weights_only=True))
            for split,(a,b) in {'train':(0,2000),'validation':(2100,2448)}.items():
                features=skeleton_features(points[a:b],ids);extra=torch.cat((features['axes'].flatten(2),features['bone_valid'].float()),-1)
                if config['features']['local_features']=='tip_control':extra=torch.zeros_like(extra)
                x=torch.cat((features['tips']/.1,extra),-1)
                residual=torch.stack([torch.tanh(head(x[:,i])) for i,head in enumerate(model.local)],1)
                q=exact(model(points[a:b]))['q'];bq=exact(model.base(points[a:b,ids]))['q'];change=torch.rad2deg(abs(q-bq)).numpy()
                record[split]={'head_joint_change_deg_mean':float(change.mean()),'head_joint_change_deg_p95':float(np.percentile(change,95)),
                               'head_joint_change_deg_max':float(change.max()),'head_saturation_fraction':float((residual.abs()>.95).float().mean())}
            record['best_sha256']=hashlib.sha256((path/'best.pth').read_bytes()).hexdigest();record['frozen_base_verified_all_snapshots']=True
            output['models'][arm]=record;models[arm]=model
        for split,(a,b) in {'train':(0,2000),'validation':(2100,2448)}.items():
            x=points[a:b];h=skeleton_features(x,ids);tips=h['tips'].numpy();axes=h['axes'][:,:,-1].numpy();split_report={}
            predictions={name:exact(model(x))['axes'][:,:,-1].numpy() for name,model in models.items()}
            for finger,name in enumerate(('thumb','index','middle','ring','little')):
                nearby=np.array(sorted(cKDTree(tips[:,finger]).query_pairs(.005)),dtype=int).reshape(-1,2)
                nearby=nearby[(nearby[:,1]-nearby[:,0])>=30]
                near_angles=np.rad2deg(np.arccos(np.clip((axes[nearby[:,0],finger]*axes[nearby[:,1],finger]).sum(-1),-1,1)))
                pairs=nearby[np.linalg.norm(tips[nearby[:,0],finger]-tips[nearby[:,1],finger],axis=-1)<=.002]
                candidate_count=len(pairs);max_angle=None
                if len(pairs):
                    angle=np.rad2deg(np.arccos(np.clip((axes[pairs[:,0],finger]*axes[pairs[:,1],finger]).sum(-1),-1,1)))
                    max_angle=float(angle.max())
                    valid=h['bone_valid'][:,finger,-1].numpy();pairs=pairs[(angle>=20)&valid[pairs[:,0]]&valid[pairs[:,1]]]
                unique=np.unique(pairs)
                entry={'pair_count':len(pairs),'distinct_frame_count':len(unique),'near_tip_pair_count_before_axis_filter':candidate_count,
                       'near_tip_max_axis_difference_deg':max_angle,'relaxed_5mm_20deg_pair_count':int((near_angles>=20).sum()),'models':{}}
                for model,ra in predictions.items():
                    error=np.rad2deg(np.arccos(np.clip((ra[unique,finger]*axes[unique,finger]).sum(-1),-1,1)))
                    entry['models'][model]={'axis_error_deg_mean':float(error.mean()) if len(error) else None}
                split_report[name]=entry
            output['information_pairs'][split]=split_report
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in output.items() if k!='models'},indent=2))


if __name__=='__main__':main()
