"""Generate offline robot-feasible branch targets on TRAIN frames only.

These are synthetic robot IK solutions anchored to parent tips, not human joint
labels, not collision-certified, and never used as online correction.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from geort.export import resolve_checkpoint
from geort.model import build_ik_model
from geort.coordination import skeleton_features,robot_signature
from geort.coordination_loss import ExactHand


def filter_targets(normalized, reference_normalized, target_tips, exact, tolerance_m=.002):
    """Reject individual finger labels exceeding the Cartesian budget."""
    with torch.no_grad():
        shift = (exact(normalized)['tips'] - target_tips).norm(dim=-1)
        accepted = torch.isfinite(shift) & (shift <= tolerance_m)
        safe = torch.where(accepted.repeat_interleave(4, -1), normalized, reference_normalized)
    return safe, accepted, shift


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True);p.add_argument('--data',type=Path,default=Path('data/human_alex.npy'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--train-end',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=1000);p.add_argument('--starts',type=int,default=3)
    args=p.parse_args();torch.set_num_threads(1)
    if args.output.exists():raise FileExistsError(args.output)
    data=np.load(args.data,allow_pickle=False)
    if not 0<args.train_end<=len(data) or args.steps<1 or args.starts<1:raise ValueError('Invalid optimization budget')
    path=resolve_checkpoint(args.parent).resolve();config=json.loads((path/'config.json').read_text())
    if config['name']!='wuji_hand2_beta1_right' or config.get('model_type','fingertip_v1')!='fingertip_v1':raise ValueError('Original Wuji baseline required')
    parent=build_ik_model(config).eval();parent.load_state_dict(torch.load(path/'last.pth',map_location='cpu',weights_only=True))
    fk=ExactHand(config);x=torch.tensor(data[:args.train_end],dtype=torch.float32);features=skeleton_features(x,[4,8,12,16,20])
    with torch.no_grad():reference_normalized=parent(features['tips']);reference=fk(reference_normalized);ref=reference['q'];target=reference['tips']
    low=fk.lower[None].expand(len(x),-1).clone();high=fk.upper[None].expand(len(x),-1)
    for f in range(1,5):
        low[:,4*f]=torch.minimum(ref[:,4*f],ref.new_tensor(-np.deg2rad(15))).clamp_min(fk.lower[4*f])
        low[:,4*f+2:4*f+4]=max(float(fk.lower[4*f+2]),float(-np.deg2rad(5)))
    best_score=torch.full((len(x),5),float('inf'));best_q=ref.clone()
    valid=features['bone_valid'][:,:,-1]
    for seed in range(args.starts):
        torch.manual_seed(args.seed + seed)
        initial=ref if seed==0 else low+(high-low)*torch.rand_like(low)
        fractions=((initial-low)/(high-low)).clamp(.01,.99)
        latent=torch.logit(fractions).detach().requires_grad_(True)
        optimizer=torch.optim.Adam([latent],lr=.03)
        for step in range(args.steps):
            optimizer.zero_grad();q=low+(high-low)*torch.sigmoid(latent);q=torch.cat((ref[:,:4],q[:,4:]),1)
            result=fk(2*(q-fk.lower)/(fk.upper-fk.lower)-1)
            error=((result['tips']-target)/.002).square().sum(-1)
            axis_error=1-(result['axes'][:,:,-1]*features['axes'][:,:,-1]).sum(-1)
            # A fixed weak canonical preference and observed axis select the redundant DOF.
            nominal=q.new_tensor([.2,0.,.5,.3]).repeat(5)
            preference=((q-nominal).reshape(len(q),5,4)/np.pi).square().mean(-1)
            score=error+.02*torch.where(valid,axis_error,torch.zeros_like(axis_error))+.001*preference
            score[:,1:].mean().backward();optimizer.step()
            with torch.no_grad():
                improve=score<best_score
                for i in range(5):best_q[:,4*i:4*i+4]=torch.where(improve[:,i,None],q[:,4*i:4*i+4],best_q[:,4*i:4*i+4])
                best_score=torch.minimum(best_score,score)
        print(f'completed start {seed+1}/{args.starts}',flush=True)
    normalized=2*(best_q-fk.lower)/(fk.upper-fk.lower)-1
    normalized, accepted, attempted_shift = filter_targets(normalized, reference_normalized, target, fk)
    with torch.no_grad():result=fk(normalized)
    shift=(result['tips']-target).norm(dim=-1).numpy()*1000
    metadata={'scope':'offline IK-generated robot targets; train only; no human joint ground truth; no collision constraints',
              'schema_version':2,'max_tip_shift_m':.002,'rejected_label_policy':'retain original parent joint target per finger',
              'mcp_policy':'no additional extension beyond parent or -15 degrees',
              'accepted_fraction_per_finger':accepted.float().mean(0).tolist(),
              'attempted_tip_p95_mm_per_finger':np.percentile(attempted_shift.numpy()*1000,95,axis=0).tolist(),
              'seed':args.seed,'train_end':args.train_end,'steps':args.steps,'starts':args.starts,
              'parent_sha256':hashlib.sha256((path/'last.pth').read_bytes()).hexdigest(),
              'data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),'robot_signature':robot_signature(config),
              'tip_p95_mm_per_finger':np.percentile(shift,95,axis=0).tolist(),
              'tip_max_mm_per_finger':shift.max(0).tolist(),
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as out:
        np.savez_compressed(out,normalized=normalized.numpy(),accepted=accepted.numpy(),source_frame=np.arange(len(x)),metadata=json.dumps(metadata))
    print(json.dumps(metadata),flush=True)

if __name__=='__main__':main()
