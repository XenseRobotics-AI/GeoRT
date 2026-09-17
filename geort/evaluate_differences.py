"""Frozen Stage-4 validation: five difference audits, without training or test access."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from geort.coordination import skeleton_features, robot_signature
from geort.coordination_loss import ExactHand
from geort.manus_sessions import load_session_split
from geort.mocap.manus_capture_data import load_capture
from geort.wuji_baseline import load_wuji_cache

NAMES = {'Original': 'original', 'M0': 'stage4_M0_seed42', 'M1': 'stage4_M1_seed42',
         'Wuji': 'wuji_filtered', 'Wuji_raw': 'wuji_unfiltered'}
FINGERS = ['thumb', 'index', 'middle', 'ring', 'little']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    a = np.asarray(values).reshape(-1)
    if not len(a):
        return None
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite diagnostic values')
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), maximum=float(a.max()))


def lp_filter(q, alpha=.3):
    q = np.asarray(q)
    if q.ndim != 2 or not len(q) or not np.isfinite(q).all() or not 0 < alpha <= 1:
        raise ValueError('Expected nonempty finite sequence and 0 < alpha <= 1')
    out = q.copy()
    for i in range(1, len(q)):
        out[i] = out[i-1] + alpha * (q[i]-out[i-1])
    return out


def angle(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a*b, axis=-1), -1, 1)))


def disjoint_pairs(pairs, seed=42):
    """Each read can occur once within this diagnostic cell; no independence claim."""
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    if not len(pairs):
        return pairs
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    order = np.random.default_rng(seed).permutation(len(pairs))
    used, chosen = set(), []
    for k in order:
        a, b = pairs[k]
        if a != b and a not in used and b not in used:
            chosen.append((a, b)); used.update((a, b))
    return np.asarray(chosen, dtype=int).reshape(-1, 2)


def response(h, r):
    hn, rn = np.linalg.norm(h, axis=-1), np.linalg.norm(r, axis=-1)
    if not len(hn):
        return None
    if (hn <= 0).any():
        raise ValueError('Zero input displacement requires a separate stationary diagnostic')
    responsive = rn >= .0001  # Below 0.1 mm: report low response, do not assign angle.
    angles = angle(h[responsive]/hn[responsive, None], r[responsive]/rn[responsive, None])
    return {'n': len(hn), 'gain': distribution(rn/hn),
            'gain_deviation_from_one': distribution(abs(rn/hn-1)),
            'low_response_fraction': float((~responsive).mean()),
            'gain_below_point1_fraction': float((rn/hn < .1).mean()),
            'direction_deg_responsive_only': distribution(angles),
            'vector_difference_mm': distribution(np.linalg.norm(r-h, axis=-1)*1000)}


def collision_summary(depth):
    return {'depth_mm': distribution(depth), **{f'over_{n}mm_fraction': float((depth > n).mean()) for n in (1, 2, 5)}}


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    recording = load_session_split(args.manifest, 'validation')
    previous = json.loads((args.evaluation/'metrics.json').read_text())
    if previous['manifest_sha256'] != sha(args.manifest):
        raise ValueError('Validation cache manifest mismatch')
    for name in ('original', 'stage4_M0_seed42', 'stage4_M1_seed42'):
        meta = previous['models'][name]
        if sha(meta['weights']) != meta['weights_sha256']:
            raise ValueError('Checkpoint changed since cached evaluation')
    config_path = Path(previous['models']['stage4_M1_seed42']['weights']).parent/'config.json'
    cfg = json.loads(config_path.read_text()); exact = ExactHand(cfg)
    names = dict(NAMES)
    additional = {}
    for item in getattr(args, 'additional_model', []):
        name, tag = item.split('=', 1)
        if not name or name in names or name in ('SDK', 'M1_lp03') or name.endswith('_lp03'):
            raise ValueError('Additional model name must be unique')
        meta = previous['models'][tag]
        candidate = json.loads((Path(meta['weights']).parent/'config.json').read_text())
        if (sha(meta['weights']) != meta['weights_sha256']
                or robot_signature(candidate) != robot_signature(cfg)
                or candidate['objectives']['relation_scale'] != cfg['objectives']['relation_scale']):
            raise ValueError('Additional checkpoint geometry/provenance mismatch')
        names[name] = tag; additional[name] = tag
    archive = np.load(args.evaluation/'outputs.npz', allow_pickle=False)
    sdk_archive = sdk_meta = None
    if args.sdk_evaluation:
        sdk_meta = json.loads((args.sdk_evaluation/'metrics.json').read_text())
        if (sdk_meta['split'] != 'validation' or sdk_meta['manifest_sha256'] != sha(args.manifest)
                or sdk_meta['config_sha256'] != sha(config_path)
                or sdk_meta['outputs_sha256'] != sha(args.sdk_evaluation/'outputs.npz')
                or sdk_meta['sdk']['version'] != '2026.8.31'
                or sdk_meta['sdk']['joint_order'] != cfg['joint_order']):
            raise ValueError('SDK cache provenance mismatch')
        sdk_archive = np.load(args.sdk_evaluation/'outputs.npz', allow_pickle=False)
    report = {'seed': 42, 'split': 'validation', 'frames': len(recording['keypoints']),
              'scope': 'cached frozen outputs; common exact FK; no test inference, contact truth, or sensor timing',
              'provenance': {'manifest': sha(args.manifest), 'outputs': sha(args.evaluation/'outputs.npz'),
                             'model_config': sha(config_path), 'script': sha(__file__), 'weights': previous['models']},
              'by_task': {}, 'pair_audit': [], 'micro': [], 'pinch': [], 'collision': {}, 'stationary': {}}
    if sdk_meta: report['provenance']['wuji_sdk'] = sdk_meta
    selected_q = {n: [] for n in [*names, 'M1_lp03', *[n+'_lp03' for n in additional], *(['SDK'] if sdk_meta else [])]}
    collision_ids = {}; rng = np.random.default_rng(42)
    for clip in recording['entry']['clips']:
        task = clip['task']; a, b = clip['range']; points = recording['keypoints'][a:b]
        source = Path(clip['source']); _, capture = load_capture(source)
        if not np.array_equal(points, capture['keypoints']):
            raise ValueError('Capture differs from validation bundle')
        cache = load_wuji_cache(args.wuji_cache_dir/(task+'.npz'), source/'keypoints.npy', cfg['joint_order'])
        if cache['cache_sha256'] != previous['wuji_caches'][task]['sha256']:
            raise ValueError('Wuji cache changed since evaluation')
        qs = {n: archive[task+'__'+key] for n, key in names.items()}
        for name, mode in [('Wuji', 'filtered'), ('Wuji_raw', 'unfiltered')]:
            if not np.array_equal(qs[name], cache[mode]):
                raise ValueError('Cached Wuji output mismatch')
        filter_error = float(np.max(abs(lp_filter(qs['Wuji_raw'])-qs['Wuji'])))
        if filter_error > 1e-6:
            raise ValueError('LP formula does not reproduce production cache')
        qs['M1_lp03'] = lp_filter(qs['M1'])
        for name in additional: qs[name+'_lp03'] = lp_filter(qs[name])
        if sdk_archive is not None:
            if sdk_meta['sources'][task] != {'sha256':sha(source/'keypoints.npy'),'frames':len(points)}:
                raise ValueError('SDK cache source mismatch')
            qs['SDK'] = sdk_archive[task+'__SDK']
        with torch.inference_mode():
            human = skeleton_features(torch.tensor(points), [4,8,12,16,20])
        ht = human['tips'].numpy(); ha = human['axes'][:, :, -1].numpy()
        if not human['bone_valid'].all():
            raise ValueError('This frozen-data audit requires valid bones; do not include invalid axes')
        hg = np.linalg.norm(ht[:, 1:]-ht[:, :1], axis=-1)
        target_gap = hg*np.asarray(cfg['objectives']['relation_scale'])
        robs = {}; task_report = {'filter_reproduction_max_rad': filter_error, 'models': {}}
        ids = np.sort(np.concatenate([rng.choice(block, size=min(10, len(block)), replace=False)
                                     for block in np.array_split(np.arange(len(points)), 20)]))
        collision_ids[task] = ids.tolist()
        for name, q in qs.items():
            if q.shape != (len(points), 20) or not np.isfinite(q).all():
                raise ValueError('Invalid q cache')
            norm = 2*(torch.tensor(q)-exact.lower)/(exact.upper-exact.lower)-1
            if (norm.abs() > 1.00001).any():
                raise ValueError('Physical joint limit exceeded')
            with torch.inference_mode():
                robot = exact(norm)
            rt = robot['tips'].numpy(); ra = robot['axes'][:, :, -1].numpy()
            rg = np.linalg.norm(rt[:, 1:]-rt[:, :1], axis=-1)
            ae = angle(ha, ra); ge = abs(rg-target_gap)*1000
            blocks = [{'axis_mean_deg': float(ae[idx].mean()),
                       'opening_mean_mm': distribution(ge[idx][hg[idx] >= .015])}
                      for idx in np.array_split(np.arange(len(q)), 10)]
            task_report['models'][name] = {'axis_deg': distribution(ae),
                'axis_by_finger': {f: distribution(ae[:, i]) for i, f in enumerate(FINGERS)},
                'opening_mm': distribution(ge[hg >= .015]), 'blocks': blocks,
                'severe_non_thumb_fraction': float((q[:, [4,6,7,8,10,11,12,14,15,16,18,19]] < -np.deg2rad(15)).any(1).mean()),
                'saturated_joint_fraction': float((norm.abs() > .95).float().mean())}
            robs[name] = (rt, ra, rg); selected_q[name].append(q[ids])
        report['by_task'][task] = task_report
        # Pair selection uses human inputs only, never scores a model to select samples.
        t = capture['host_mono_s']
        for f, finger in enumerate(FINGERS):
            all_pairs = cKDTree(ht[:, f]).query_pairs(.005, output_type='ndarray')
            all_pairs = all_pairs[t[all_pairs[:, 1]]-t[all_pairs[:, 0]] >= .5]
            pd = np.linalg.norm(ht[all_pairs[:, 1], f]-ht[all_pairs[:, 0], f], axis=-1)
            pa = angle(ha[all_pairs[:, 1], f], ha[all_pairs[:, 0], f])
            for radius, degrees in [(.002, 10), (.002, 20), (.005, 20)]:
                candidates = all_pairs[(pd <= radius) & (pa >= degrees)]
                pairs = disjoint_pairs(candidates)
                row = dict(task=task, finger=finger, max_tip_mm=radius*1000, min_angle_deg=degrees,
                           candidate_pairs=len(candidates), candidate_distinct_polls=len(np.unique(candidates)),
                           selected_pairs=pairs.tolist(), n=len(pairs), models={})
                if len(pairs):
                    i, j = pairs.T
                    row['human_angle_deg'] = distribution(angle(ha[i, f], ha[j, f]))
                    row['human_tip_mm'] = distribution(np.linalg.norm(ht[j, f]-ht[i, f], axis=-1)*1000)
                    for name, (rt, ra, _) in robs.items():
                        row['models'][name] = {'angle_change_deg': distribution(angle(ra[i, f], ra[j, f])),
                            'tip_drift_mm': distribution(np.linalg.norm(rt[j, f]-rt[i, f], axis=-1)*1000),
                            'axis_delta_error': distribution(np.linalg.norm((ra[j, f]-ra[i, f])-(ha[j, f]-ha[i, f]), axis=-1))}
                report['pair_audit'].append(row)
        if task == 'micro':
            small = np.linalg.norm(np.diff(ht,axis=0),axis=-1) < .0001
            report['stationary'] = {'input_tip_step_below_mm': .1, 'n': int(small.sum()),
                'scope': 'adjacent host reads; small input displacement proxy, not verified physical rest or sensor rate',
                'models': {name: distribution(np.linalg.norm(np.diff(rt,axis=0),axis=-1)[small]*1000)
                           for name,(rt,_,_) in robs.items()}}
            for lag in (1, 3, 6, 12):
                # Separate disjoint endpoint pairs per lag; never span a clip boundary.
                i = np.arange(0, len(ht)-lag, lag+1); j = i+lag
                hd = ht[j]-ht[i]; hn = np.linalg.norm(hd, axis=-1)*1000
                for lo, hi in [(.5, 2), (2, 5), (5, 10)]:
                    mask = (hn >= lo) & (hn < hi)
                    row = dict(lag_polls=lag, input_mm=[lo, hi], n=int(mask.sum()), models={})
                    for name, (rt, _, _) in robs.items():
                        row['models'][name] = response(hd[mask], (rt[j]-rt[i])[mask])
                    report['micro'].append(row)
        if task == 'pinch_axis':
            for f, finger in enumerate(FINGERS[1:]):
                for lo, hi in [(0,15), (15,30), (30,60), (60,1000)]:
                    mask = (hg[:, f]*1000 >= lo) & (hg[:, f]*1000 < hi)
                    row = dict(finger=finger, human_gap_mm=[lo, hi], n=int(mask.sum()), models={})
                    for name, (_, _, rg) in robs.items():
                        row['models'][name] = {'site_gap_mm': distribution(rg[mask, f]*1000),
                            'opening_error_mm': distribution(abs(rg[mask, f]-target_gap[mask, f])*1000) if lo >= 15 else None}
                    report['pinch'].append(row)
        print('Completed geometry/pairs:', task, flush=True)
    from geort.env.hand import HandKinematicModel
    with contextlib.redirect_stdout(io.StringIO()):
        hand = HandKinematicModel.build_from_config(cfg, render=False)
    depths = {}
    for name, rows in selected_q.items():
        with contextlib.redirect_stderr(io.StringIO()):
            depth = hand.self_collision_depth(np.concatenate(rows))*1000
        depths[name] = depth; per_task = {}; pos = 0
        for task, ids in collision_ids.items():
            per_task[task] = collision_summary(depth[pos:pos+len(ids)]); pos += len(ids)
        report['collision'][name] = {'aggregate': collision_summary(depth), 'by_task': per_task}
        print('Collision', name, report['collision'][name]['aggregate'], flush=True)
    report['collision_source_indices'] = collision_ids
    report['collision_scope'] = '1800 shared stratified polls; common GeoRT SAPIEN geometry, not actual contact or safety certification'
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(args.output/'collision_depths.npz', **depths)
    print('Saved', args.output, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('data/manus_stage4_v1/manifest.json'))
    p.add_argument('--evaluation', type=Path, default=Path('reports/stage4/validation'))
    p.add_argument('--wuji-cache-dir', type=Path, default=Path('reports/stage4/wuji_val01'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sdk-evaluation',type=Path,help='Optional pinned SDK validation cache from geort.evaluate_wuji_sdk')
    p.add_argument('--additional-model',action='append',default=[],metavar='NAME=CHECKPOINT',
                   help='Extra frozen candidate in the evaluation cache, also audited with LP=0.3')
    run(p.parse_args())


if __name__ == '__main__':
    main()
