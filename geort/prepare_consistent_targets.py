"""Generate exact-FK candidate targets and select coherent training-only branches.

Labels are synthetic robot solutions, not human joint ground truth. All graph
edges come from observed skeleton geometry; no time intervals are fabricated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from geort.branch_selection import geometric_neighbors, select_branches
from geort.coordination import skeleton_features, robot_signature
from geort.coordination_loss import ExactHand
from geort.export import resolve_checkpoint
from geort.model import build_ik_model


def quality_mask(tip_shift, axis_error, reference_axis_error, axis_valid, mcp, mcp_floor):
    return ((tip_shift <= .002) & ((~axis_valid) | (axis_error <= reference_axis_error + np.deg2rad(5)))
            & (mcp >= mcp_floor-np.deg2rad(1)))


def posture_eligibility(q, reference_q):
    """Allow preference where parent was natural or an improved solution exists."""
    floor = np.tile(np.deg2rad([-15.,-5.,-5.]),(len(q),4,1))
    ids = np.array([[4*i,4*i+2,4*i+3] for i in range(1,5)])
    floor[:,:,0] = np.minimum(floor[:,:,0],reference_q[:,ids[:,0]])
    old = np.maximum(floor-reference_q[:,ids],0).max(-1)
    new = np.maximum(floor-q[:,ids],0).max(-1)
    return (old <= np.deg2rad(.5)) | (new < old-np.deg2rad(.5))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True)
    p.add_argument('--data',type=Path,default=Path('data/human_alex.npy'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-end',type=int,default=2000)
    p.add_argument('--steps',type=int,default=700)
    p.add_argument('--starts',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--graph-strength',type=float,default=.05)
    args=p.parse_args();torch.set_num_threads(1)
    if args.output.exists():raise FileExistsError(args.output)
    data=np.load(args.data,allow_pickle=False)
    if not 1<args.train_end<=len(data) or args.steps<3 or args.starts<1:raise ValueError('Invalid budget')
    path=resolve_checkpoint(args.parent).resolve();cfg=json.loads((path/'config.json').read_text())
    if cfg['name']!='wuji_hand2_beta1_right' or cfg.get('model_type','fingertip_v1')!='fingertip_v1':
        raise ValueError('Original Wuji right baseline required')
    model=build_ik_model(cfg).eval();model.load_state_dict(torch.load(path/'last.pth',map_location='cpu',weights_only=True))
    x=torch.tensor(data[:args.train_end],dtype=torch.float32);h=skeleton_features(x,[4,8,12,16,20]);exact=ExactHand(cfg)
    with torch.no_grad():ref_norm=model(h['tips']);ref=exact(ref_norm)
    n=len(x);lower,upper=exact.lower,exact.upper
    floor=torch.tensor(np.deg2rad([-15.,-5.,-5.]),dtype=torch.float32).expand(n,4,3).clone()
    floor[:,:,0]=torch.minimum(floor[:,:,0],ref['q'].reshape(n,5,4)[:,1:,0])
    human_cos=(h['axes'][:,:,1:]*h['axes'][:,:,:-1]).sum(-1)
    shape_valid=h['bone_valid'][:,:,1:] & h['bone_valid'][:,:,:-1]
    shape_valid[:,0]=False
    candidates=[ref['q'].detach().clone()]
    def features(q):
        r=exact(2*(q-lower)/(upper-lower)-1)
        axis=(1-(r['axes'][:,:,-1]*h['axes'][:,:,-1]).sum(-1).clamp(-1,1))
        axis=torch.where(h['bone_valid'][:,:,-1],axis,0.)
        rc=(r['axes'][:,:,1:]*r['axes'][:,:,:-1]).sum(-1)
        shape=torch.where(shape_valid,(rc-human_cos).square(),0.).mean(-1)
        bends=q.reshape(n,5,4)[:,1:,[0,2,3]]
        posture=torch.relu(floor-bends).square().mean(-1)/np.deg2rad(15)**2
        posture=torch.cat((posture.new_zeros(n,1),posture),dim=1)
        distance=(r['tips']-ref['tips']).norm(dim=-1)
        unary=.2*axis/(1-np.cos(np.deg2rad(15)))+.05*shape/.2**2+posture+.1*(distance/.002)**2
        return r,axis,distance,unary
    for start in range(args.starts):
        torch.manual_seed(args.seed+start)
        if start==0:q=ref['q'].clone()
        elif start==1:
            q=ref['q'].clone()
            for f in range(1,5):q[:,f*4+2:f*4+4]=.5
        else:q=lower+(upper-lower)*torch.rand_like(ref['q'])
        latent=torch.logit(((q-lower)/(upper-lower)).clamp(.005,.995)).detach().requires_grad_(True)
        optimizer=torch.optim.Adam([latent],lr=.03)
        save_steps={max(1,args.steps//3),max(1,2*args.steps//3),args.steps}
        for step in range(1,args.steps+1):
            optimizer.zero_grad();q=lower+(upper-lower)*torch.sigmoid(latent)
            q=torch.cat((ref['q'][:,:4],q[:,4:]),dim=1)
            _,_,distance,unary=features(q)
            loss=(unary[:,1:]+.9*(distance[:,1:]/.002)**2).mean()
            loss.backward();optimizer.step()
            if step in save_steps:candidates.append(q.detach().clone())
        print(f'candidate start {start+1}/{args.starts} complete',flush=True)
    values=torch.stack(candidates).numpy().reshape(len(candidates),n,5,4).transpose(1,2,0,3)
    costs=[];masks=[];axis_errors=[];tip_shifts=[]
    with torch.no_grad():
        _,base_axis,_,_=features(ref['q'])
        base_error=torch.acos((1-base_axis).clamp(-1,1)).numpy()
        for q in candidates:
            _,axis,shift,unary=features(q)
            error=torch.acos((1-axis).clamp(-1,1)).numpy()
            mcp=q.reshape(n,5,4)[:,:,0].numpy()
            mcp_floor=np.concatenate((mcp[:,:1]-1, floor[:,:,0].numpy()),axis=1)
            mask=quality_mask(shift.numpy(),error,base_error,h['bone_valid'][:,:,-1].numpy(),mcp,mcp_floor)
            costs.append(unary.numpy());masks.append(mask);axis_errors.append(error);tip_shifts.append(shift.numpy())
    unary=np.stack(costs,axis=2);valid=np.stack(masks,axis=2)
    # Parent is a precise fallback and must always satisfy its own relative constraints.
    if not valid[:,:,0].all():raise RuntimeError('Parent candidate unexpectedly failed quality checks')
    unary=np.where(valid,unary,np.inf)
    independent=np.argmin(unary,axis=2);selected=independent.copy();graphs=[]
    for f in range(1,5):
        graph=geometric_neighbors(h['tips'][:,f].numpy(),h['axes'][:,f].numpy(),h['bone_valid'][:,f].numpy())
        selected[:,f],record=select_branches(values[:,f],unary[:,f],graph,args.graph_strength,seed=args.seed)
        graphs.append(record)
    def extract(labels):return values[np.arange(n)[:,None],np.arange(5)[None,:],labels].reshape(n,20)
    independent_q=extract(independent);selected_q=extract(selected)
    reports={}
    for name,q,labels in [('independent',independent_q,independent),('consistent',selected_q,selected)]:
        shifts=np.stack(tip_shifts,axis=2)[np.arange(n)[:,None],np.arange(5)[None,:],labels]
        reports[name]={'tip_shift_max_mm':float(shifts.max()*1000),
                       'tip_shift_p95_mm_per_finger':np.percentile(shifts*1000,95,axis=0).tolist(),
                       'parent_fallback_fraction':(labels==0).mean(0).tolist(),
                       'posture_eligible_fraction':posture_eligibility(q,ref['q'].numpy()).mean(0).tolist(),
                       'max_joint_step_deg':float(np.rad2deg(abs(np.diff(q,axis=0))).max())}
    meta={'schema_version':3,'max_tip_shift_m':.002,'seed':args.seed,'train_end':args.train_end,
          'steps':args.steps,'starts':args.starts,'candidate_count':len(candidates),'graph_strength':args.graph_strength,
          'graph_features':'training-only tip positions + observed bone axes, no timestamps or temporal training',
          'scope':'synthetic robot IK targets; not human joint labels; not collision-certified',
          'parent_sha256':hashlib.sha256((path/'last.pth').read_bytes()).hexdigest(),
          'data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),'robot_signature':robot_signature(cfg),
          'source_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path('geort/branch_selection.py')]},
          'graphs':graphs,'quality':reports}
    normalize=lambda q:2*(q-lower.numpy())/(upper-lower).numpy()-1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f:
        np.savez_compressed(f,normalized=normalize(selected_q),independent_normalized=normalize(independent_q),
             posture_mask=posture_eligibility(selected_q,ref['q'].numpy()),
             independent_posture_mask=posture_eligibility(independent_q,ref['q'].numpy()),
             source_frame=np.arange(n),metadata=json.dumps(meta))
    print(json.dumps(meta),flush=True)


if __name__=='__main__':main()
