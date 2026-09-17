"""Common GeoRT loss evaluation of networks and production Wuji probe outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from geort.coordination import robot_signature
from geort.coordination_data import load_recording, batch, split_indices
from geort.coordination_loss import ExactHand, baseline_from_outputs, shift_chains, task_objective
from geort.export import resolve_checkpoint
from geort.model import build_ik_model, ik_input, FKModel
from geort.utils.config_utils import parse_config_keypoint_info
from geort.wuji_baseline import load_wuji_cache


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_probes(path, cache, data_path, ids, joint_order):
    with np.load(path,allow_pickle=False) as f:
        meta=json.loads(str(f['metadata']))
        if meta['baseline_cache_sha256']!=cache['cache_sha256'] or meta['data_sha256']!=sha(data_path):
            raise ValueError('Probe provenance mismatch')
        if meta['source_commit']!=cache['metadata']['source_commit'] or not np.array_equal(f['source_frame'],ids):
            raise ValueError('Probe source/frame mismatch')
        names=f['joint_names'].tolist()
        if len(names)!=len(set(names)) or set(names)!=set(joint_order):raise ValueError('Probe joint mismatch')
        order=[names.index(n) for n in joint_order];result={'metadata':meta}
        for key in ('offsets','deltas'):
            result[key]=torch.as_tensor(f[key].copy(),dtype=torch.float32)
            if result[key].shape!=(len(ids),5,3) or not torch.isfinite(result[key]).all():raise ValueError('Invalid probe offsets')
        for mode in ('filtered','unfiltered'):
            result[mode]={}
            for probe in ('center','plus','minus','delta'):
                q=f[f'{mode}_{probe}']
                if q.shape!=(len(ids),len(names)) or not np.isfinite(q).all():raise ValueError('Invalid probe outputs')
                result[mode][probe]=torch.as_tensor(q[:,order].copy(),dtype=torch.float32)
            if not np.allclose(result[mode]['center'].numpy(),cache[mode][ids],rtol=0,atol=1e-5):raise ValueError('Probe center changed')
    return result


def loss_report(outputs, sample, exact, neural_fk, robot_cloud, weights, options, reference, offsets, deltas):
    """Outputs are normalized joint tensors for center/plus/minus/delta."""
    if any(q.shape!=(len(sample['keypoints']),len(exact.lower)) or not torch.isfinite(q).all() or (q.abs()>1.00001).any() for q in outputs.values()):
        raise ValueError('Invalid or out-of-limit loss outputs; no silent clipping')
    ids=parse_config_keypoint_info(exact.config)['human_id'];tips=sample['keypoints'][:,ids]
    _,task,_=task_objective(outputs['center'],sample,exact,exact.config,options,reference['tips'],reference_q=reference['q'])
    task_total=sum(r['weighted'] for r in task.values());result={}
    for mode in ('neural_fk_training','exact_fk_audit'):
        embedded={k:neural_fk(q) if mode=='neural_fk_training' else exact(q)['tips'] for k,q in outputs.items()}
        total,records=baseline_from_outputs(tips,embedded['center'],embedded['plus'],embedded['minus'],embedded['delta'],deltas,robot_cloud,weights)
        result[mode]={'baseline':records,'task':task,'weighted_baseline_total':float(total),
                      'weighted_task_total':task_total,'weighted_total':float(total)+task_total,
                      'weighted_total_without_anchor':float(total)+task_total-task.get('anchor',{}).get('weighted',0.)}
    a=neural_fk(outputs['center']);b=exact(outputs['center'])['tips'];error=(a-b).norm(dim=-1).numpy()*1000
    result['neural_fk_error_mm']={'mean':float(error.mean()),'p95':float(np.percentile(error,95)),'max':float(error.max())}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--profile-checkpoint',default='stage2_T2_seed42')
    p.add_argument('--wuji-cache',type=Path,default=Path('reports/baselines/wuji_manus_right_human_alex.npz'))
    p.add_argument('--wuji-probes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    if args.output.exists():raise FileExistsError(args.output)
    path=resolve_checkpoint(args.profile_checkpoint);cfg=json.loads((path/'config.json').read_text());meta=json.loads((path/'experiment.json').read_text())
    data_path=Path(meta['data']);recording=load_recording(data_path);ids=split_indices(recording,meta['experiment']['split'])['validation']
    expected=[value for key,value in meta['hashes'].items() if Path(key).resolve()==data_path.resolve()]
    if expected!=[sha(data_path)]:raise ValueError('Profile data changed')
    sample=batch(recording,ids);exact=ExactHand(cfg);info=parse_config_keypoint_info(cfg)
    cache=load_wuji_cache(args.wuji_cache,data_path,cfg['joint_order'])
    probes=load_probes(args.wuji_probes,cache,data_path,ids,cfg['joint_order'])
    pcfg=json.loads((path/'baseline_config.json').read_text());parent=build_ik_model(pcfg).eval()
    parent.load_state_dict(torch.load(path/'baseline.pth',map_location='cpu',weights_only=True))
    fk_path=Path(f"checkpoint/fk_model_{cfg['name']}.pth")
    expected=[v for k,v in meta['hashes'].items() if Path(k).resolve()==fk_path.resolve()]
    if expected!=[sha(fk_path)]:raise ValueError('Neural FK changed since profile training')
    fk=FKModel(info['joint']).eval();fk.load_state_dict(torch.load(fk_path,map_location='cpu',weights_only=True))
    offsets,deltas=probes['offsets'],probes['deltas']
    variants={'center':sample,'plus':{**sample,'keypoints':shift_chains(sample['keypoints'],offsets,info['human_id'])},
              'minus':{**sample,'keypoints':shift_chains(sample['keypoints'],-offsets,info['human_id'])},
              'delta':{**sample,'keypoints':shift_chains(sample['keypoints'],deltas,info['human_id'])}}
    provenance={'profile_config_sha256':sha(path/'config.json'),'profile_experiment_sha256':sha(path/'experiment.json'),
                'parent_sha256':sha(path/'baseline.pth'),'neural_fk_sha256':sha(fk_path),
                'data_sha256':sha(data_path),'probes_sha256':sha(args.wuji_probes),'cache_sha256':sha(args.wuji_cache),
                'sources':{f:sha(f) for f in ('geort/evaluate_losses.py','geort/coordination_loss.py')}}
    report={'profile':args.profile_checkpoint,'split':'validation','frames':len(ids),'seed':probes['metadata']['seed'],
            'weights':meta['experiment']['baseline'],'options':cfg['objectives'],'provenance':provenance,
            'probe_metadata':probes['metadata'],'models':{},
            'scope':'Common full validation batch, fixed 256-point coverage cloud and paired 2mm probes. Task terms always exact FK. Conditional Wuji responses restore identical past state; not temporal training or latency.',
            'interpretation':'Total is a fixed training objective, not a universal quality score. Anchor and reference MCP favor original GeoRT; relation weight zero. Coverage is sampled geometric Chamfer, not feasible operational coverage. No collision term.'}
    with torch.inference_mode():
        reference=exact(parent(ik_input(pcfg,sample['keypoints'])))
        generator=torch.Generator().manual_seed(meta['experiment']['seed']+2)
        cloud=exact(torch.rand((256,len(cfg['joint_order'])),generator=generator)*2-1)['tips']
        def evaluate(name,outputs):report['models'][name]=loss_report(outputs,sample,exact,fk,cloud,meta['experiment']['baseline'],cfg['objectives'],reference,offsets,deltas)
        evaluate('original_geort',{k:parent(ik_input(pcfg,v['keypoints'])) for k,v in variants.items()})
        for tag in args.checkpoints:
            cp=resolve_checkpoint(tag);conf=json.loads((cp/'config.json').read_text())
            if robot_signature(conf)!=robot_signature(cfg):raise ValueError('Candidate geometry differs')
            model=build_ik_model(conf).eval();model.load_state_dict(torch.load(cp/'best.pth',map_location='cpu',weights_only=True))
            evaluate(tag,{k:model(ik_input(conf,v['keypoints'])) for k,v in variants.items()})
            provenance[tag]={'config_sha256':sha(cp/'config.json'),'weights_sha256':sha(cp/'best.pth')}
        for mode in ('filtered','unfiltered'):
            evaluate('wuji_'+mode,{k:2*(q-exact.lower)/(exact.upper-exact.lower)-1 for k,q in probes[mode].items()})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print('Common exact-FK weighted losses: name | axis | posture | anchor | total without anchor')
    for name,r in report['models'].items():
        row=r['exact_fk_audit'];terms=row['task']
        print(f"{name} | {terms['axis']['weighted']:.6f} | {terms['posture']['weighted']:.6f} | {terms.get('anchor',{}).get('weighted',0):.6f} | {row['weighted_total_without_anchor']:.6f}")
    print(report['interpretation'])


if __name__=='__main__':main()
