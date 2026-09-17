"""Freeze main-branch GeoRT with the historical README default Wuji checkpoint."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from geort.evaluate_differences import sha
from geort.manus_sessions import load_session_split
from geort.mocap.live_comparison import ROOT
from geort.utils.config_utils import parse_config_keypoint_info


def module_from_file(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'reports/final_main_reference')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(['git','rev-parse','main'],cwd=ROOT,text=True).strip()
    for source,name in [('geort/model.py','main_model.py'),('geort/formatter.py','main_formatter.py'),('README.md','main_README.md')]:
        (args.output/name).write_bytes(subprocess.check_output(['git','show',f'{commit}:{source}'],cwd=ROOT))
    # Pin the historical main README default. This is not selected using test performance.
    checkpoint=args.checkpoint.resolve() if args.checkpoint else ROOT/'checkpoint/wuji_hand2_beta1_right_2026-09-14_17-31-39_wuji_paper'
    config_path=checkpoint/'config.json';weights=checkpoint/'best.pth'
    cfg=json.loads(config_path.read_text());info=parse_config_keypoint_info(cfg)
    r2=json.loads((ROOT/'checkpoint/stage5_R2_seed42/config.json').read_text())
    if cfg['joint']!=r2['joint'] or cfg['joint_order']!=r2['joint_order']:raise ValueError('Different joint convention')
    manifest=ROOT/'data/manus_stage4_v1/manifest.json';data=load_session_split(manifest,'test')
    model=module_from_file('main_geort_model',args.output/'main_model.py').IKModel(info['joint']).eval()
    model.load_state_dict(torch.load(weights,weights_only=True,map_location='cpu'));torch.set_num_threads(2)
    formatter=module_from_file('main_geort_formatter',args.output/'main_formatter.py').HandFormatter(cfg['joint']['lower'],cfg['joint']['upper'])
    arrays={};batch_error=0.
    with torch.inference_mode():
        for clip in data['entry']['clips']:
            a,b=clip['range'];points=data['keypoints'][a:b][:,info['human_id']].astype(np.float32)
            normalized=model(torch.from_numpy(points));q=formatter.unnormalize(normalized.numpy()).astype(np.float32)
            single=formatter.unnormalize(model(torch.from_numpy(points[:1])).numpy()).astype(np.float32)
            batch_error=max(batch_error,float(np.max(abs(q[:1]-single))))
            assert q.shape==(b-a,20) and np.isfinite(q).all()
            arrays[clip['task']+'__Original']=q
    np.savez_compressed(args.output/'outputs.npz',**arrays)
    metadata={'main_commit':commit,'checkpoint':str(weights),'checkpoint_sha256':sha(weights),'config':str(config_path),'config_sha256':sha(config_path),
        'selection':('user-requested main GeoRT retraining on train01, seed42, best by training loss; no test selection' if args.checkpoint else 'main README Wuji preview: wuji_hand2_beta1_right_last --weights best; historical alias pinned to 2026-09-14 wuji_paper; not chosen by test metrics'),
        'manifest_sha256':sha(manifest),'frames':len(data['keypoints']),'outputs_sha256':sha(args.output/'outputs.npz'),
        'source_sha256':{f:sha(args.output/f) for f in ('main_model.py','main_formatter.py','main_README.md')},
        'inference':'Unmodified main IKModel.eval, select configured fingertip IDs, main HandFormatter. CPU device; no training, BN recalibration, posture correction or filtering.',
        'single_vs_batch_max_abs_rad':batch_error,'other_methods':'Unchanged stage6/test_full frozen R2 LP0.3, Wuji LP0.3 and SDK outputs'}
    (args.output/'protocol.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
