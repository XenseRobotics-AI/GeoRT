"""Isolated stdin/stdout retargeting worker. No robot or glove device APIs."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(message):
    print(json.dumps(message, allow_nan=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--joint-config', type=Path, required=True)
    p.add_argument('--spec', type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    commit = subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(args.source),'status','--porcelain'], text=True).strip()
    if dirty or commit != spec['source_commit'] or sha(args.config) != spec['config_sha256']:
        raise ValueError('Production Wuji source/config differs from confirmed baseline')
    cfg = yaml.safe_load(args.config.read_text())
    urdf = (args.config.parent/cfg['optimizer']['urdf_path']).resolve()
    if sha(urdf) != spec['native_urdf_sha256']:
        raise ValueError('Production Wuji asset changed')
    sys.path.insert(0, str(args.source.resolve()))
    with contextlib.redirect_stdout(sys.stderr):
        from wuji_retargeting import Retargeter
        import wuji_retargeting
        if not Path(wuji_retargeting.__file__).resolve().is_relative_to(args.source.resolve()):
            raise ValueError('Unexpected Wuji import')
        np.random.seed(42)
        retargeter = Retargeter.from_yaml(str(args.config), hand_side='right')
        retargeter.reset()
    joint_cfg = json.loads(args.joint_config.read_text())
    names = list(retargeter.optimizer.robot.dof_joint_names)
    target = joint_cfg['joint_order']
    if len(set(names)) != len(names) or set(names) != set(target):
        raise ValueError('Joint sets do not match')
    order = [names.index(n) for n in target]
    emit({'ready': True, 'source_commit': commit, 'lp_alpha': spec['lp_alpha']})
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('quit'): break
        points = np.asarray(request['points'], dtype=np.float64)
        if points.shape != (21,3) or not np.isfinite(points).all():
            raise ValueError('Expected finite (21,3) skeleton')
        with contextlib.redirect_stdout(sys.stderr):
            if request.get('reset'): retargeter.reset()
            raw = np.asarray(retargeter.retarget(points, apply_filter=False)).copy()
            filtered = np.asarray(retargeter.lp_filter.next(raw.copy())).copy()
        if not np.isfinite([raw, filtered]).all(): raise ValueError('Nonfinite Wuji output')
        emit({'seq': request['seq'], 'raw': raw[order].tolist(), 'filtered': filtered[order].tolist(),
              'solver_code': int(retargeter.optimizer.opt.last_optimize_result())})


if __name__ == '__main__':
    try: main()
    except Exception as e:
        emit({'error': f'{type(e).__name__}: {e}'})
        raise
