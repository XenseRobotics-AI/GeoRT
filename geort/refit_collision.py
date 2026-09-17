"""Repair rare-collision supervision with training geometry only, before round two."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from geort.collision_surrogate import CollisionDepthModel, diagnostics, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    data=np.load(a.source/'geometry.npz',allow_pickle=False)
    x=torch.tensor(data['normalized_q']);y=torch.tensor(data['depth_mm']);hold=data['holdout']
    train=np.flatnonzero(~hold);local=np.flatnonzero((np.arange(len(x))<2048)&(~hold))
    hard=local[data['depth_mm'][local]>2]
    vh=np.flatnonzero(hold);vl=np.flatnonzero((np.arange(len(x))<2048)&hold)
    if not len(hard) or not len(vl):raise ValueError('Missing local geometry samples')
    torch.set_num_threads(1);torch.manual_seed(42);rng=np.random.default_rng(42)
    model=CollisionDepthModel();optim=torch.optim.Adam(model.parameters(),lr=.001)
    a.output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((a.source/'metadata.json').read_text())
    meta.update(geometry_archive=str((a.source/'geometry.npz').resolve()),geometry_sha256=sha(a.source/'geometry.npz'),
        source_sha256=sha(__file__),sampling='64 local positive +64 local +128 all TRAIN geometry, preserved holdout groups',
        selection='mean global and local held geometry asymmetric Huber+BCE; 2000 preparation updates',
        purpose='repair rare R1 residual penetration under-detection before primary round2')
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2));(a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    def loss(ids):
        pred,logit=model(x[ids]);truth=y[ids]
        weight=torch.where((truth>2)&(pred<truth),4.,1.)
        return (F.huber_loss(pred,truth,reduction='none',delta=1.)*weight).mean()+.5*F.binary_cross_entropy_with_logits(logit,(truth>2).float())
    best=float('inf');logs=[]
    for step in range(1,2001):
        ids=np.concatenate([rng.choice(hard,64),rng.choice(local,64),rng.choice(train,128)])
        value=loss(ids);optim.zero_grad();value.backward();optim.step()
        if step%100==0:
            with torch.no_grad():score=float((loss(vh)+loss(vl))*.5)
            row=dict(step=step,score=score,global_geometry=diagnostics(model,x[vh],y[vh]),local_geometry=diagnostics(model,x[vl],y[vl]))
            logs.append(row);print(json.dumps(row),flush=True)
            if score<best:
                best=score;torch.save(model.state_dict(),a.output/'best.pth');(a.output/'selection.json').write_text(json.dumps(row,indent=2))
    (a.output/'training.json').write_text(json.dumps(logs,indent=2))


if __name__=='__main__':main()
