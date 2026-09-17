"""Run the confirmed production retargeter in its own Python environment.

No GeoRT/Torch import: Pinocchio dependencies stay isolated. Input is a single
ordered right-hand recording of 21 MediaPipe-layout points, in meters.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/human_alex.npy'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--spec', type=Path, default=Path(__file__).resolve().parents[1] / 'geort/baselines/wuji_manus_right.json')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = json.loads(args.spec.read_text())
    source, config_path = args.source.resolve(), args.config.resolve()
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    if commit != spec['source_commit'] or dirty or sha(config_path) != spec['config_sha256']:
        raise ValueError('Baseline source/config changed; explicitly register a new baseline before comparing')
    config = yaml.safe_load(config_path.read_text())
    urdf = (config_path.parent / config['optimizer']['urdf_path']).resolve()
    if sha(urdf) != spec['native_urdf_sha256']:
        raise ValueError('Native Wuji URDF differs from confirmed baseline')
    sys.path.insert(0, str(source))
    from wuji_retargeting import Retargeter
    import wuji_retargeting
    if not Path(wuji_retargeting.__file__).resolve().is_relative_to(source):
        raise ValueError('Unexpected wuji_retargeting import location')
    points = np.load(args.data, allow_pickle=False)
    if points.ndim != 3 or points.shape[1:] != (21, 3) or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('Expected a finite single-sequence (T,21,3) recording in meters')
    np.random.seed(spec['seed'])
    retargeter = Retargeter.from_yaml(str(config_path), hand_side=spec['hand_side'])
    retargeter.reset()
    names = list(retargeter.optimizer.robot.dof_joint_names)
    raw, filtered, codes, durations, failures = [], [], [], [], []
    for i, frame in enumerate(points):
        console = io.StringIO()
        start = time.perf_counter()
        with contextlib.redirect_stdout(console):
            q = np.asarray(retargeter.retarget(frame, apply_filter=False)).copy()
            smooth = np.asarray(retargeter.lp_filter.next(q.copy())).copy()
        durations.append((time.perf_counter() - start) * 1000)
        if q.shape != (len(names),) or smooth.shape != q.shape or not np.isfinite([q, smooth]).all():
            raise RuntimeError(f'Invalid baseline output at frame {i}')
        code = retargeter.optimizer.opt.last_optimize_result()
        codes.append(code)
        if code < 0 or 'Optimization failed:' in console.getvalue():
            failures.append({'frame': i, 'nlopt_code': code, 'message': console.getvalue().strip()})
        raw.append(q); filtered.append(smooth)
        if (i+1) % 500 == 0:
            print(f'{i+1}/{len(points)} frames', flush=True)
    # Evidence that canonical/world rigid transforms do not require an invented adapter.
    from wuji_retargeting.mediapipe import apply_mediapipe_transformations
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    invariance = max(float(np.max(abs(apply_mediapipe_transformations(frame, 'right') -
                    apply_mediapipe_transformations(frame @ rotation.T + [.3, -.1, .2], 'right'))))
                    for frame in points[::max(1, len(points)//20)])
    if invariance > 1e-6:
        raise ValueError('Input frame invariance check failed')
    metadata = dict(spec, data_sha256=sha(args.data), config_path=str(config_path), config=config,
                    source_path=str(source), source_files={str(f.relative_to(source)): sha(f) for f in sorted((source/'wuji_retargeting').rglob('*.py'))},
                    native_urdf_path=str(urdf), native_urdf_sha256=sha(urdf), spec_sha256=sha(args.spec),
                    generator_sha256=sha(__file__),
                    dependencies={name: importlib.metadata.version(name) for name in ['pin', 'nlopt', 'numpy', 'scipy', 'pyyaml']},
                    sequence_policy='single sequence; reset at frame 0; warm start + LP state carried in original order',
                    input='21 MediaPipe-layout points in meters; unchanged geometry; native retarget() coordinate transform',
                    rigid_transform_invariance_max_m=invariance, optimization_failures=failures,
                    timing_scope='local wall time: native preprocessing + solver + LP filter; no hardware latency',
                    runtime_ms_p50=float(np.median(durations)), runtime_ms_p95=float(np.percentile(durations, 95)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as f:
        np.savez_compressed(f, qpos_filtered=filtered, qpos_unfiltered=raw, joint_names=np.array(names),
                            source_frame=np.arange(len(points)), nlopt_code=codes, runtime_ms=durations,
                            metadata=json.dumps(metadata))
    print(json.dumps({'output': str(args.output), 'frames': len(points), 'failures': len(failures),
                      'runtime_ms_p95': metadata['runtime_ms_p95']}))


if __name__ == '__main__':
    main()
