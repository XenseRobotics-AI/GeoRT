"""Two bounded refinements, using training-only collision labels and real local pairs."""
import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from geort.collision_surrogate import CollisionDepthModel, sha
from geort.coordination import robot_signature
from geort.coordination_loss import ExactHand, task_objective, masked_term
from geort.export import GeoRTRetargetingModel, resolve_checkpoint
from geort.manus_sessions import load_session_split, sample_balanced
from geort.model import build_ik_model
from geort.train_manus import validation_loss


def paired_ids(recording, ids, rng):
    """Real poses within one clip; poll offsets are not elapsed times."""
    ends = np.array([c['range'][1] for c in recording['entry']['clips']])
    lag = rng.choice([1, 3, 6, 12], size=len(ids))
    return np.minimum(ids + lag, ends[recording['clip_id'][ids]] - 1)


def response_loss(points, other, tips, other_tips, weight=.2):
    hd = other[:, [4,8,12,16,20]] - points[:, [4,8,12,16,20]]
    rd = other_tips - tips
    magnitude = hd.norm(dim=-1)
    mask = (magnitude >= .0005) & (magnitude < .010)
    return masked_term((rd-hd).square().sum(-1), mask, .003**2, weight)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('data/manus_stage4_v1/manifest.json'))
    p.add_argument('--parent',default='stage4_M1_seed42')
    p.add_argument('--surrogate',type=Path,required=True)
    p.add_argument('--context',action='store_true')
    p.add_argument('--collision-weight',type=float,default=1.)
    p.add_argument('--response-weight',type=float,default=.2)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not np.isfinite(args.collision_weight) or args.collision_weight<=0:raise ValueError('Invalid weight')
    if not np.isfinite(args.response_weight) or args.response_weight<=0:raise ValueError('Invalid response weight')
    torch.set_num_threads(1);torch.manual_seed(42);rng=np.random.default_rng(42)
    train=load_session_split(args.manifest,'train');val=load_session_split(args.manifest,'validation')
    parent=resolve_checkpoint(args.parent);cfg=json.loads((parent/'config.json').read_text())
    if args.context:cfg['manus_features']['context']=True
    model=build_ik_model(cfg).eval()
    state=torch.load(parent/'best.pth',weights_only=True,map_location='cpu')
    if args.context:
        # Only the explicitly added branch may be absent; never silently skip old weights.
        added={k:v for k,v in model.state_dict().items() if k.startswith('context_net.')}
        if any(k in state for k in added):raise ValueError('Parent already has context')
        state={**state,**added}
    model.load_state_dict(state,strict=True)
    collision=CollisionDepthModel().eval()
    meta=json.loads((args.surrogate/'metadata.json').read_text())
    if meta['human_split']!='train' or meta['manifest_sha256']!=sha(args.manifest) or meta['robot_signature']!=robot_signature(cfg):
        raise ValueError('Collision supervision provenance mismatch')
    collision.load_state_dict(torch.load(args.surrogate/'best.pth',weights_only=True,map_location='cpu'))
    collision.requires_grad_(False)
    exact=ExactHand(cfg)
    from geort.env.hand import HandKinematicModel
    with contextlib.redirect_stdout(io.StringIO()):hand=HandKinematicModel.build_from_config(cfg,render=False)
    vi=np.concatenate([np.linspace(a,b-1,32,dtype=int) for a,b in [c['range'] for c in val['entry']['clips']]])
    vx=torch.tensor(val['keypoints'][vi]);vj=paired_ids(val,vi,np.random.default_rng(42));vy=torch.tensor(val['keypoints'][vj])
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'config.json').write_text(json.dumps(cfg,indent=2))
    sources=[Path(__file__),Path('geort/manus_model.py'),Path('geort/collision_surrogate.py'),Path('geort/coordination_loss.py'),Path('geort/kinematics.py')]
    metadata=dict(seed=42,steps=2000,batch_size=128,lr=.0003,parent=args.parent,
        parent_weights_sha256=sha(parent/'best.pth'),manifest_sha256=sha(args.manifest),
        collision_weights_sha256=sha(args.surrogate/'best.pth'),collision_metadata=meta,
        test_loaded=False,teacher_used=False,context=args.context,trainable='all IK parameters; frozen collision network',
        objective=dict(base=cfg['objectives'],collision_weight=args.collision_weight,
            collision='mean(relu(depth_mm-0.5)^2)/4',response_weight=args.response_weight,response_tolerance_m=.003,
            response='robot tip displacement vs human displacement, fixed gain 1, real within-clip pairs, 0.5 <= input <10mm'),
        selection='minimum fixed geometric validation + local response +10*actual over2mm fraction+20*actual over5mm fraction; step0 and each500; 32 uniform collision polls/task',
        selection_source_indices=vi.tolist(),source_hashes={str(s):sha(s) for s in sources})
    (args.output/'experiment.json').write_text(json.dumps(metadata,indent=2))
    snapshot=args.output/'training_sources';snapshot.mkdir()
    for s in sources:(snapshot/s.name).write_bytes(s.read_bytes())
    torch.save(model.state_dict(),args.output/'initial.pth')
    optim=torch.optim.Adam(model.parameters(),lr=.0003);best=float('inf');logs=[]
    def select(step):
        nonlocal best
        geom,per_clip=validation_loss(model,val,exact,cfg)
        with torch.inference_mode():
            norm=model(vx);robot=exact(norm);other=exact(model(vy))
            response,record=response_loss(vx,vy,robot['tips'],other['tips'])
            predicted,_=collision(norm)
        with contextlib.redirect_stderr(io.StringIO()):depth=hand.self_collision_depth(robot['q'].numpy())*1000
        over2=float((depth>2).mean());over5=float((depth>5).mean())
        score=geom+float(response)+10*over2+20*over5
        row=dict(step=step,score=score,geometry=geom,response=record,over2_fraction=over2,over5_fraction=over5,
                 surrogate_mae_mm=float(abs(predicted.numpy()-depth).mean()),geometry_by_task=per_clip)
        logs.append(row);print(json.dumps(row),flush=True)
        if score<best:
            best=score;torch.save(model.state_dict(),args.output/'best.pth')
            (args.output/'selection.json').write_text(json.dumps(row,indent=2))
        (args.output/'training.json').write_text(json.dumps(logs,indent=2))
    select(0)
    for step in range(1,2001):
        ids=sample_balanced(train,128,rng);other=paired_ids(train,ids,rng)
        x=torch.tensor(train['keypoints'][ids]);y=torch.tensor(train['keypoints'][other])
        norm=model(x);base,records,robot=task_objective(norm,x,exact,cfg,cfg['objectives'])
        local,record=response_loss(x,y,robot['tips'],exact(model(y))['tips'],args.response_weight)
        depth,_=collision(norm);penalty=(torch.relu(depth-.5)/2).square().mean()
        loss=base+local+args.collision_weight*penalty
        optim.zero_grad();loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite training')
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optim.step()
        if step%100==0:
            print(json.dumps(dict(step=step,loss=float(loss.detach()),base=records,response=record,collision=float(penalty.detach()))),flush=True)
        if step%500==0:
            torch.save(model.state_dict(),args.output/f'step_{step}.pth');select(step)
    torch.save(model.state_dict(),args.output/'last.pth')
    model.load_state_dict(torch.load(args.output/'best.pth',weights_only=True,map_location='cpu'))
    exported=GeoRTRetargetingModel(args.output/'best.pth',args.output/'config.json',device='cpu')
    with torch.inference_mode():expected=exact(model(vx[:8]))['q'].numpy()
    error=float(abs(np.stack([exported.forward(x) for x in vx[:8].numpy()])-expected).max())
    if error>1e-5:raise RuntimeError('Export mismatch')
    (args.output/'export_check.json').write_text(json.dumps(dict(max_error_rad=error)))
    print('Completed refinement:',args.output,flush=True)


if __name__=='__main__':main()
