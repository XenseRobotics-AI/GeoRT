"""Optional metadata adapter; legacy recordings remain unchanged on disk."""
from pathlib import Path
import numpy as np
import torch


def load_recording(path):
    path = Path(path)
    if path.suffix == '.npy':
        data = {'keypoints': np.load(path, allow_pickle=False)}
    else:
        with np.load(path, allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
    unknown = set(data) - {'keypoints', 'valid', 'confidence', 'timestamps', 'sequence_id'}
    if unknown:
        raise ValueError(f'Unsupported recording fields (no silent schema downgrade): {sorted(unknown)}')
    if 'keypoints' not in data:
        raise ValueError('Missing keypoints field')
    points = data['keypoints']
    if points.ndim != 3 or points.shape[1:] != (21, 3) or len(points) < 2:
        raise ValueError('Expected at least two (21,3) skeleton frames')
    n = len(points)
    for name in ('valid', 'confidence'):
        if name in data and data[name].shape != (n, 21):
            raise ValueError(f'{name} must have shape (T,21)')
    if 'valid' in data and data['valid'].dtype != np.bool_:
        raise ValueError('valid must be boolean')
    if ('timestamps' in data) != ('sequence_id' in data):
        raise ValueError('Time diagnostics require both timestamps (seconds) and sequence_id')
    if 'timestamps' in data:
        t, seq = data['timestamps'], data['sequence_id']
        if t.shape != (n,) or seq.shape != (n,) or not np.isfinite(t).all():
            raise ValueError('Invalid timestamps/sequence_id')
        if np.issubdtype(seq.dtype, np.number) and not np.isfinite(seq).all():
            raise ValueError('Nonfinite sequence IDs')
        for name in np.unique(seq):
            ids = np.flatnonzero(seq == name)
            if np.any(np.diff(ids) != 1) or np.any(np.diff(t[ids]) <= 0):
                raise ValueError('Sequences must be contiguous, with strictly increasing timestamps')
    return data


def batch(recording, ids):
    result = {'keypoints': torch.as_tensor(recording['keypoints'][ids], dtype=torch.float32)}
    for name in ('valid', 'confidence'):
        if name in recording:
            result[name] = torch.as_tensor(recording[name][ids])
    return result


def split_indices(recording, specification):
    n = len(recording['keypoints'])
    result = {}
    for name in ('train', 'validation', 'test'):
        start, end = specification[name]
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= n:
            raise ValueError(f'Invalid {name} range')
        result[name] = np.arange(start, end)
    if result['train'][-1] + 1 >= result['validation'][0] or result['validation'][-1] + 1 >= result['test'][0]:
        raise ValueError('Use ordered disjoint splits with a guard gap')
    if 'sequence_id' in recording:
        for name, ids in result.items():
            for sequence in np.unique(recording['sequence_id'][ids]):
                all_ids = np.flatnonzero(recording['sequence_id'] == sequence)
                if all_ids[0] < ids[0] or all_ids[-1] > ids[-1]:
                    raise ValueError(f'{name} cuts through a sequence; use complete sequences')
        sets = [set(recording['sequence_id'][result[k]].tolist()) for k in result]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError('Sequence IDs must not cross train/validation/test splits')
    return result
