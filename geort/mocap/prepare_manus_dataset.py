"""Verify independent Manus sessions and package static geometry with provenance.

One array per split; clip boundaries and original poll indices remain explicit.
This does not make the arrays a single continuous recording or invent timestamps.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from geort.mocap.manus_capture_data import load_capture, mp_mapping


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_session(path, expected_split):
    path = Path(path).resolve()
    session = json.loads((path / 'session.json').read_text())
    if (session['status'] != 'complete' or session['split'] != expected_split
            or session['source_type'] != 'manus_glove_raw' or session['hand'] != 'right'):
        raise ValueError('Expected a completed right-hand raw Manus session in the requested split')
    planned = [task['id'] for task in session['plan']]
    if not planned or len(planned) != len(set(planned)) or any(Path(task).name != task or task in ('.', '..') for task in planned):
        raise ValueError('Invalid or duplicate task IDs')
    if set(planned) != {p.parent.name for p in path.glob('*/metadata.json')}:
        raise ValueError('Missing or unexpected task directories')
    if len(session['takes']) != len(planned) or {t['id'] for t in session['takes']} != set(planned):
        raise ValueError('Session take list differs from plan')
    takes = {t['id']: t for t in session['takes']}
    points, indices, clips = [], [], []
    offset = 0
    for task in planned:
        directory = path / task
        meta, data = load_capture(directory)
        if any(meta[key] != session[key] for key in ('session_id', 'split', 'source_type', 'hand', 'operator')):
            raise ValueError('Clip identity differs from session')
        if meta['task'] != task or meta['status'] != 'complete' or takes[task]['status'] != 'complete':
            raise ValueError('Incomplete or mismatched task')
        if meta['mp_rows'] != session['mp_rows'] or meta['node_info'] != session['node_info']:
            raise ValueError('Clip topology differs from session')
        if not np.array_equal(mp_mapping(meta['node_info']), meta['mp_rows']):
            raise ValueError('Invalid node mapping')
        if len(data['poll_index']) != meta['poll_count'] or len(data['poll_index']) != takes[task]['poll_count']:
            raise ValueError('Manifest poll count mismatch')
        valid = data['valid_frame']
        p = np.load(directory / 'keypoints.npy', allow_pickle=False)
        poll = np.load(directory / 'source_poll_index.npy', allow_pickle=False)
        export = json.loads((directory / 'export.json').read_text())
        if (len(p) < 2 or not np.isfinite(p).all() or not np.array_equal(p, data['keypoints'][valid])
                or not np.array_equal(poll, data['poll_index'][valid])):
            raise ValueError('Export does not match valid raw chunks')
        if any(export[key] != meta[key] for key in ('session_id', 'split', 'task', 'source_type')):
            raise ValueError('Export identity differs from clip')
        if (export['keypoints_sha256'] != sha(directory / 'keypoints.npy')
                or export['valid_frames'] != len(p) or export['temporal_training_allowed']
                or export['kind'] != 'static_geometry_only'
                or export['omitted_invalid_polls'] != int((~valid).sum())):
            raise ValueError('Invalid static export provenance')
        raw_points = data['raw_skeleton'][valid][:, meta['mp_rows'], :3]
        transform = data['canonical_to_source'][valid]
        restored = np.einsum('bik,bjk->bij', p, transform[:, :3, :3]) + transform[:, None, :3, 3]
        if not np.allclose(restored, raw_points, atol=1e-6, rtol=0):
            raise ValueError('Canonical geometry does not reconstruct raw SDK positions')
        files = [directory / name for name in ('metadata.json', 'export.json', 'keypoints.npy', 'source_poll_index.npy')]
        files += sorted(directory.glob('chunk_*.npz'))
        clips.append({'task': task, 'range': [offset, offset + len(p)], 'source': str(directory),
                      'poll_count': len(valid), 'valid_count': len(p),
                      'omitted_invalid_polls': int((~valid).sum()),
                      'hashes': {str(f.relative_to(path)): sha(f) for f in files}})
        offset += len(p)
        points.append(p)
        indices.append(poll)
    return session, np.concatenate(points), np.concatenate(indices), clips


def prepare(sessions, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if set(sessions) != {'train', 'validation', 'test'}:
        raise ValueError('Provide exactly train, validation and test sessions')
    checked = {split: inspect_session(path, split) for split, path in sessions.items()}
    if len({v[0]['session_id'] for v in checked.values()}) != 3:
        raise ValueError('Session IDs must not cross splits')
    if len({Path(p).resolve() for p in sessions.values()}) != 3:
        raise ValueError('Source sessions must be distinct')
    # A copied recording renamed as a new session must not pass the split check.
    seen = {}
    for split, (_, _, _, clips) in checked.items():
        for clip in clips:
            digest = clip['hashes'][f"{clip['task']}/keypoints.npy"]
            if digest in seen and seen[digest] != split:
                raise ValueError('Identical clip reused across splits')
            seen[digest] = split
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'kind': 'manus_static_sessions', 'seed': 42,
                'temporal_training_allowed': False,
                'policy': 'All valid polls retained, including duplicates. Split by session; never compute differences or carry filter state across clip boundaries.',
                'test_policy': 'Quality inspection only until model selection is frozen.', 'splits': {}}
    for split, (session, p, poll, clips) in checked.items():
        geometry = output / f'{split}.npy'
        with geometry.open('xb') as f:
            np.save(f, p, allow_pickle=False)
        clip_id = np.concatenate([np.full(c['range'][1]-c['range'][0], i, dtype=np.int64) for i,c in enumerate(clips)])
        provenance = output / f'{split}_source.npz'
        with provenance.open('xb') as f:
            np.savez_compressed(f, clip_id=clip_id, source_poll_index=poll)
        manifest['splits'][split] = {'session_id': session['session_id'], 'operator': session['operator'],
            'source': str(Path(sessions[split]).resolve()), 'session_sha256': sha(Path(sessions[split])/'session.json'),
            'calibration_note': session['calibration_note'], 'frames': len(p), 'clips': clips,
            'keypoints': geometry.name, 'keypoints_sha256': sha(geometry),
            'provenance': provenance.name, 'provenance_sha256': sha(provenance)}
    with (output / 'manifest.json').open('x') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, allow_nan=False)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ('train', 'validation', 'test'):
        parser.add_argument('--' + split, required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    manifest = prepare({s: getattr(args, s) for s in ('train', 'validation', 'test')}, args.output)
    print(json.dumps({s: {'session': v['session_id'], 'frames': v['frames'], 'clips': len(v['clips'])}
                      for s,v in manifest['splits'].items()}, indent=2))


if __name__ == '__main__':
    main()
