"""Pinned, isolated SDK retargeting only. Never constructs a device manager."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np

VERSION = '2026.8.31'
# Firmware order, also used by LeRobot's WUJI_DEVICE_JOINT_BASENAMES.
JOINTS = ['r_'+n for n in (
    'thumb_cmc_flex', 'thumb_cmc_abd', 'thumb_mcp', 'thumb_ip',
    'index_finger_mcp_flex', 'index_finger_mcp_abd', 'index_finger_pip', 'index_finger_dip',
    'middle_finger_mcp_flex', 'middle_finger_mcp_abd', 'middle_finger_pip', 'middle_finger_dip',
    'ring_finger_mcp_flex', 'ring_finger_mcp_abd', 'ring_finger_pip', 'ring_finger_dip',
    'pinky_mcp_flex', 'pinky_mcp_abd', 'pinky_pip', 'pinky_dip')]


def emit(message):
    print(json.dumps(message, allow_nan=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--joint-config', type=Path, required=True)
    args = p.parse_args()
    if version('wuji-sdk') != VERSION:
        raise ValueError(f'This baseline requires wuji-sdk=={VERSION}')
    import wuji_sdk.wuji_sdk as native
    from wuji_sdk import RetargetSession, HandModel, Handedness
    cfg = json.loads(args.joint_config.read_text())
    target = cfg['joint_order']
    if len(target) != 20 or set(target) != set(JOINTS):
        raise ValueError('Firmware / evaluation joint sets differ')
    order = [JOINTS.index(n) for n in target]
    session = RetargetSession.for_hand(HandModel.WujiHand2, Handedness.Right)
    emit({'ready': True, 'version': VERSION, 'hand': 'WujiHand2', 'side': 'Right',
          'native_sha256': hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
          'numpy_version': np.__version__, 'joint_order': target,
          'configuration': 'SDK builtin; warm start and internal low pass; alpha not exposed',
          'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    for line in sys.stdin:
        req = json.loads(line)
        points = np.asarray(req['points'], dtype=np.float32)
        single = points.shape == (21, 3)
        frames = points[None] if single else points
        if frames.ndim != 3 or frames.shape[1:] != (21, 3) or not len(frames) or not np.isfinite(frames).all():
            raise ValueError('Expected finite (21,3) or nonempty (T,21,3) meter keypoints')
        if req.get('reset'): session.reset()
        out = []; durations = []
        for frame in frames:
            start = time.perf_counter()
            q = np.asarray(session.step(frame))
            durations.append((time.perf_counter()-start)*1000)
            if q.shape != (20,) or not np.isfinite(q).all():
                raise ValueError('Invalid SDK joint output')
            q = q[order]
            if np.any(q < np.array(cfg['joint']['lower'])-1e-5) or np.any(q > np.array(cfg['joint']['upper'])+1e-5):
                raise ValueError('SDK exceeds common physical limits; not clipping')
            out.append(q.tolist())
        emit({'seq': req['seq'], 'q': out[0] if single else out, 'step_ms': durations})


if __name__ == '__main__':
    try: main()
    except Exception as e:
        emit({'error': f'{type(e).__name__}: {e}'})
        raise
