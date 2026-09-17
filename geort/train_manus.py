"""Bounded, session-isolated information experiment; no teacher or test inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from geort.coordination import robot_signature, skeleton_features
from geort.coordination_loss import ExactHand, task_objective
from geort.export import GeoRTRetargetingModel, resolve_checkpoint
from geort.manus_sessions import load_session_split, sample_balanced
from geort.model import build_ik_model


OPTIONS = dict(axis=1., shape=.2, posture=.2, relation=.1, axis_tolerance_deg=15.,
               shape_tolerance_cos=.2, posture_tolerance_deg=15., relation_tolerance_m=.01,
               relation_min_opening_m=.015, anchor=0.)


def make_config(parent, training, mode):
    config = copy.deepcopy(parent)
    tips = torch.tensor(training['keypoints'][:,[4,8,12,16,20]])
    config['model_type'] = 'manus_v1'
    config['manus_features'] = dict(version=1, mode=mode, frame='geort_canonical', unit='m',
        robot_signature=robot_signature(config), tip_mean=tips.mean(0).tolist(),
        tip_scale=tips.std(0,unbiased=False).clamp_min(.01).tolist(),
        calibration='train session only; no validation/test statistics', context=False)
    exact = ExactHand(config)
    human = skeleton_features(torch.tensor(training['keypoints']),[4,8,12,16,20])
    with torch.inference_mode():
        robot_lengths = exact(torch.zeros(1,20))['lengths'].sum(-1)[0]
        human_lengths = human['lengths'].sum(-1)
    ratios = [float((robot_lengths[0]+robot_lengths[i])/(human_lengths[:,0]+human_lengths[:,i]).median()) for i in range(1,5)]
    config['objectives'] = dict(OPTIONS, relation_scale=ratios, relation_valid=[True]*4)
    return config


def validation_loss(model, recording, exact, config):
    # Fixed uniform subset per clip, equal weight per task, no boundary differences.
    with torch.inference_mode():
        values = []
        for clip in recording['entry']['clips']:
            a,b = clip['range']
            ids = np.linspace(a,b-1,min(256,b-a),dtype=int)
            sample = {'keypoints':torch.tensor(recording['keypoints'][ids])}
            value,_,_ = task_objective(model(sample),sample,exact,config,config['objectives'])
            values.append(float(value))
    return float(np.mean(values)), values


def run(args):
    if args.steps < 2 or args.batch_size < 2 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('Invalid training budget or learning rate')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    training = load_session_split(args.manifest,'train')
    validation = load_session_split(args.manifest,'validation')
    parent = resolve_checkpoint(args.parent)
    config = make_config(json.loads((parent/'config.json').read_text()),training,args.mode)
    torch.manual_seed(42)
    model = build_ik_model(config).eval()
    exact = ExactHand(config)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    rng = np.random.default_rng(42)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    sources = [Path(__file__),Path('geort/manus_model.py'),Path('geort/manus_sessions.py'),Path('geort/coordination_loss.py'),Path('geort/kinematics.py')]
    metadata = dict(seed=42, mode=args.mode, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        manifest=str(args.manifest.resolve()),manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        data_hashes={k:r['entry']['keypoints_sha256'] for k,r in [('train',training),('validation',validation)]},
        parent_config=str(parent/'config.json'),parent_weights_used=False,
        test_loaded=False, teacher_used=False, context=False, trainable='all parameters',
        sampling='uniform task then uniform valid poll within task; duplicates retained',
        selection='minimum equal-task validation geometric objective; <=256 fixed uniform polls per clip',
        objective_scope='exact FK axis/shape/posture/relation; no old coverage/pinch/anchor objective; not an isolated loss-vs-old ablation',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output/'experiment.json').write_text(json.dumps(metadata,indent=2))
    source_dir=args.output/'training_sources';source_dir.mkdir()
    for path in sources:(source_dir/path.name).write_bytes(path.read_bytes())
    torch.save(model.state_dict(),args.output/'initial.pth')
    best=float('inf');logs=[]
    for step in range(1,args.steps+1):
        ids=sample_balanced(training,args.batch_size,rng)
        sample={'keypoints':torch.tensor(training['keypoints'][ids])}
        optimizer.zero_grad()
        total,_,_=task_objective(model(sample),sample,exact,config,config['objectives'])
        if not torch.isfinite(total):raise RuntimeError('Nonfinite loss')
        total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite gradient')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        optimizer.step()
        if step%100==0 or step==args.steps:
            value,per_clip=validation_loss(model,validation,exact,config)
            row=dict(step=step,training_loss=float(total.detach()),gradient_norm=float(norm),validation_loss=value,validation_by_clip=per_clip)
            logs.append(row);print(json.dumps(row),flush=True)
            if value<best:
                best=value;torch.save(model.state_dict(),args.output/'best.pth')
                (args.output/'selection.json').write_text(json.dumps(row,indent=2))
            if step%500==0:torch.save(model.state_dict(),args.output/f'step_{step}.pth')
    torch.save(model.state_dict(),args.output/'last.pth')
    (args.output/'training.json').write_text(json.dumps(logs,indent=2))
    model.load_state_dict(torch.load(args.output/'best.pth',weights_only=True,map_location='cpu'))
    exported=GeoRTRetargetingModel(args.output/'best.pth',args.output/'config.json',device='cpu')
    points=validation['keypoints'][0]
    with torch.inference_mode():expected=exact(model({'keypoints':torch.tensor(points)[None]}))['q'][0].numpy()
    error=float(abs(exported.forward(points)-expected).max())
    if error>1e-5:raise RuntimeError('Export mismatch')
    (args.output/'export_check.json').write_text(json.dumps(dict(max_error_rad=error)))
    print(f'Completed {args.mode}: {args.output}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--parent',default='wuji_hand2_beta1_right_2026-09-15_18-01-42_std_collision_seed0_w0')
    parser.add_argument('--mode',choices=['tip_control','skeleton'],required=True)
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':main()
