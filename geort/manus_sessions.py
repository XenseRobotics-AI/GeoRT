"""Explicit split loader: never loads test geometry during development training."""
import hashlib
import json
from pathlib import Path
import numpy as np


def load_session_split(manifest_path, split):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if manifest['kind'] != 'manus_static_sessions' or manifest['schema_version'] != 1 or manifest['temporal_training_allowed']:
        raise ValueError('Expected static Manus session manifest')
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Unknown split')
    identities = [v['session_id'] for v in manifest['splits'].values()]
    if len(identities) != 3 or len(set(identities)) != 3:
        raise ValueError('Sessions must not cross splits')
    entry = manifest['splits'][split]
    files = {}
    for key in ('keypoints', 'provenance'):
        file = (path.parent / entry[key]).resolve()
        if not file.is_relative_to(path.parent):
            raise ValueError('Bundle file escapes manifest directory')
        if hashlib.sha256(file.read_bytes()).hexdigest() != entry[key+'_sha256']:
            raise ValueError('Session bundle hash mismatch')
        files[key] = file
    points = np.load(files['keypoints'], allow_pickle=False)
    if points.shape != (entry['frames'],21,3) or not np.isfinite(points).all():
        raise ValueError('Invalid session keypoints')
    with np.load(files['provenance'], allow_pickle=False) as f:
        clip_id, poll = f['clip_id'].copy(), f['source_poll_index'].copy()
    if clip_id.shape != (len(points),) or poll.shape != (len(points),):
        raise ValueError('Invalid source index shapes')
    cursor = 0
    for i, clip in enumerate(entry['clips']):
        a,b = clip['range']
        if a != cursor or not a < b <= len(points) or not np.all(clip_id[a:b] == i) or np.any(np.diff(poll[a:b]) <= 0):
            raise ValueError('Invalid clip boundary or poll ordering')
        cursor = b
    if cursor != len(points):
        raise ValueError('Incomplete clip coverage')
    return {'keypoints': points, 'clip_id': clip_id, 'source_poll_index': poll, 'entry': entry}


def sample_balanced(recording, size, rng):
    clips = recording['entry']['clips']
    chosen = rng.integers(len(clips), size=size)
    starts = np.array([c['range'][0] for c in clips])[chosen]
    lengths = np.array([c['range'][1]-c['range'][0] for c in clips])[chosen]
    return starts + (rng.random(size)*lengths).astype(np.int64)
