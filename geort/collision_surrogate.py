"""Fit a local penetration-depth surrogate from simulator geometry, not hand labels."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from geort.coordination import robot_signature
from geort.coordination_loss import ExactHand
from geort.export import resolve_checkpoint
from geort.manus_sessions import load_session_split, sample_balanced
from geort.model import build_ik_model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CollisionDepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(20,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),
                               nn.Linear(256,128),nn.SiLU(),nn.Linear(128,2))

    def forward(self,q):
        x=self.net(q)
        return F.softplus(x[:,0]),x[:,1]


def diagnostics(model,x,depth):
    with torch.no_grad():
        pred,logit=model(x)
        positive=depth>2;free=depth<=.1
        return {'mae_mm':float((pred-depth).abs().mean()),
                'over2_recall':float((pred[positive]>1.).float().mean()) if positive.any() else None,
                'free_specificity':float((pred[free]<=1.).float().mean()) if free.any() else None,
                'positive_count':int(positive.sum()),'free_count':int(free.sum()),
                'classifier_over2_recall':float((logit[positive]>0).float().mean()) if positive.any() else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('data/manus_stage4_v1/manifest.json'))
    p.add_argument('--parent',default='stage4_M1_seed42')
    p.add_argument('--previous-data',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(42);rng=np.random.default_rng(42)
    parent=resolve_checkpoint(args.parent);cfg=json.loads((parent/'config.json').read_text())
    source=load_session_split(args.manifest,'train')
    model=build_ik_model(cfg).eval();model.load_state_dict(torch.load(parent/'best.pth',map_location='cpu',weights_only=True))
    ids=sample_balanced(source,2048,rng)
    with torch.inference_mode():base=model(torch.tensor(source['keypoints'][ids])).numpy()
    x=np.concatenate([base,np.clip(base+rng.normal(0,.08,base.shape),-1,1),
                      np.clip(base+rng.normal(0,.20,base.shape),-1,1),rng.uniform(-1,1,base.shape)]).astype(np.float32)
    # All perturbations of the same source read remain on the same side of the split.
    unique=np.unique(ids);hold=set(rng.choice(unique,size=max(1,len(unique)//5),replace=False).tolist())
    local_hold=np.array([int(i) in hold for i in ids]);uniform_hold=rng.random(len(base))<.2
    holdout=np.concatenate([local_hold]*3+[uniform_hold])
    exact=ExactHand(cfg)
    q=(exact.lower.numpy()+(x+1)/2*(exact.upper-exact.lower).numpy())
    from geort.env.hand import HandKinematicModel
    with contextlib.redirect_stdout(io.StringIO()):hand=HandKinematicModel.build_from_config(cfg,render=False)
    depth=hand.self_collision_depth(q)*1000
    if args.previous_data:
        prior=np.load(args.previous_data,allow_pickle=False)
        x=np.concatenate([x,prior['normalized_q']]);depth=np.concatenate([depth,prior['depth_mm']])
        # Preserve the previous geometry holdout. Local pose clouds may overlap;
        # this is surrogate interpolation QA, never a human-session test claim.
        holdout=np.concatenate([holdout,prior['holdout']])
    np.savez_compressed(args.output/'geometry.npz',normalized_q=x,depth_mm=depth,holdout=holdout,source_ids=ids)
    metadata={'seed':42,'human_split':'train','test_accessed':False,'parent':args.parent,
              'parent_sha256':sha(parent/'best.pth'),'manifest_sha256':sha(args.manifest),
              'robot_signature':robot_signature(cfg),'script_sha256':sha(__file__),
              'label':'SAPIEN common asset maximum self-penetration in mm',
              'sampling':'2048 balanced training reads; original and sigma .08/.20 normalized jitter plus 2048 uniform poses',
              'previous_data_sha256':sha(args.previous_data) if args.previous_data else None,
              'selection':'minimum held geometry Huber(depth)+0.5*BCE(over2mm), 2000 surrogate updates',
              'scope':'local differentiable approximation; final quality always measured with simulator'}
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    tensor=torch.tensor(x);targets=torch.tensor(depth)
    train=np.flatnonzero(~holdout);val=torch.tensor(x[holdout]);truth=torch.tensor(depth[holdout])
    network=CollisionDepthModel();optim=torch.optim.Adam(network.parameters(),lr=.001)
    best=float('inf');logs=[]
    for step in range(1,2001):
        selected=rng.choice(train,256,replace=True);pred,logit=network(tensor[selected]);y=targets[selected]
        loss=F.huber_loss(pred,y,delta=1.)+.5*F.binary_cross_entropy_with_logits(logit,(y>2).float())
        optim.zero_grad();loss.backward();optim.step()
        if step%100==0:
            with torch.no_grad():
                a,b=network(val);score=float(F.huber_loss(a,truth,delta=1.)+.5*F.binary_cross_entropy_with_logits(b,(truth>2).float()))
            metrics={'step':step,'score':score,**diagnostics(network,val,truth)};logs.append(metrics)
            print(json.dumps(metrics),flush=True)
            if score<best:
                best=score;torch.save(network.state_dict(),args.output/'best.pth')
                (args.output/'selection.json').write_text(json.dumps(metrics,indent=2))
    (args.output/'training.json').write_text(json.dumps(logs,indent=2))
    print('Collision supervision ready:',args.output,flush=True)


if __name__=='__main__':main()
