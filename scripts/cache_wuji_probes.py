"""Recompute independent Wuji perturbations with identical per-frame history.

Run in .venv-wuji-baseline. Source/config/URDF and history come from the verified
production cache. This is conditional response, not a temporal/latency test.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore_history(retargeter, raw_previous, filtered_previous):
    retargeter.optimizer.last_qpos = None if raw_previous is None else raw_previous.astype(np.float64).copy()
    retargeter.lp_filter.y = None if filtered_previous is None else filtered_previous.copy()
    retargeter.lp_filter.is_init = filtered_previous is not None


def shift(frame, offsets):
    result=frame.copy()
    for i,end in enumerate([4,8,12,16,20]):result[end-3:end+1]+=offsets[i]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('reports/baselines/wuji_manus_right_human_alex.npz'))
    p.add_argument('--data',type=Path,default=Path('data/human_alex.npy'))
    p.add_argument('--start',type=int,default=2100);p.add_argument('--end',type=int,default=2448)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    with np.load(args.cache,allow_pickle=False) as c:
        meta=json.loads(str(c['metadata']));raw=c['qpos_unfiltered'].copy();filtered=c['qpos_filtered'].copy()
        names=c['joint_names'].copy();frames=c['source_frame'].copy()
    spec=json.loads((Path(__file__).resolve().parents[1]/'geort/baselines/wuji_manus_right.json').read_text())
    for key in ('source_commit','config_sha256','native_urdf_sha256','hand_side','lp_alpha'):
        if meta[key]!=spec[key]:raise ValueError(f'Baseline specification mismatch: {key}')
    source=Path(meta['source_path']);config=Path(meta['config_path'])
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=spec['source_commit']:
        raise ValueError('Source commit changed')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():raise ValueError('Dirty baseline source')
    if sha(config)!=spec['config_sha256'] or sha(meta['native_urdf_path'])!=spec['native_urdf_sha256'] or sha(args.data)!=meta['data_sha256']:
        raise ValueError('Baseline files changed')
    for file,digest in meta['source_files'].items():
        if sha(source/file)!=digest:raise ValueError(f'Baseline source file changed: {file}')
    points=np.load(args.data,allow_pickle=False)
    if not 0<=args.start<args.end<=len(points) or not np.array_equal(frames,np.arange(len(points))):raise ValueError('Invalid frame range')
    sys.path.insert(0,str(source))
    from wuji_retargeting import Retargeter
    import wuji_retargeting
    if not Path(wuji_retargeting.__file__).resolve().is_relative_to(source):raise ValueError('Wrong import')
    np.random.seed(args.seed)
    retargeter=Retargeter.from_yaml(str(config),hand_side=spec['hand_side'])
    if list(retargeter.optimizer.robot.dof_joint_names)!=names.tolist():raise ValueError('Joint order changed')
    ids=np.arange(args.start,args.end);rng=np.random.default_rng(args.seed)
    offsets=(rng.standard_normal((len(ids),5,3))*.002).astype(np.float32)
    deltas=(rng.standard_normal((len(ids),5,3))*.002).astype(np.float32)
    output={f'{mode}_{probe}':[] for mode in ('filtered','unfiltered') for probe in ('center','plus','minus','delta')}
    failures=[];center_error=0.;order_error=0.
    def compute(frame,i,probe):
        restore_history(retargeter,raw[i-1] if i else None,filtered[i-1] if i else None)
        console=io.StringIO()
        with contextlib.redirect_stdout(console):
            q=np.asarray(retargeter.retarget(frame,apply_filter=False)).copy()
            smooth=np.asarray(retargeter.lp_filter.next(q.copy())).copy()
        code=retargeter.optimizer.opt.last_optimize_result()
        if code<0 or 'Optimization failed:' in console.getvalue():failures.append({'frame':int(i),'probe':probe,'code':int(code),'message':console.getvalue()})
        if not np.isfinite([q,smooth]).all():raise ValueError('Nonfinite probe output')
        return q,smooth
    for row,i in enumerate(ids):
        probes={'center':points[i], 'plus':shift(points[i],offsets[row]),'minus':shift(points[i],-offsets[row]),'delta':shift(points[i],deltas[row])}
        for probe,frame in probes.items():
            q,smooth=compute(frame,i,probe)
            output['unfiltered_'+probe].append(q);output['filtered_'+probe].append(smooth)
            if probe=='center':center_error=max(center_error,float(abs(q-raw[i]).max()),float(abs(smooth-filtered[i]).max()))
        if row<8:
            for probe in reversed(probes):
                q,smooth=compute(probes[probe],i,'order_check_'+probe)
                order_error=max(order_error,float(abs(q-output['unfiltered_'+probe][-1]).max()),float(abs(smooth-output['filtered_'+probe][-1]).max()))
        if (row+1)%100==0:print(f'{row+1}/{len(ids)}',flush=True)
    if max(center_error,order_error)>1e-5:raise ValueError(f'History reconstruction failed: center={center_error}, order={order_error}')
    metadata={'seed':args.seed,'baseline_cache_sha256':sha(args.cache),'data_sha256':sha(args.data),
              'source_commit':spec['source_commit'],'generator_sha256':sha(__file__),
              'center_reproduction_max_rad':center_error,'probe_order_check_max_rad':order_error,'failures':failures,
              'state_policy':'For every probe restore preceding cached raw optimizer qpos and filtered LP state; probes never advance recording history',
              'perturbation':'Gaussian 2 mm whole finger-chain translations, as GeoRT geometric objective; native preprocessing retained'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f:np.savez_compressed(f,**output,offsets=offsets,deltas=deltas,source_frame=ids,joint_names=names,metadata=json.dumps(metadata))
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
