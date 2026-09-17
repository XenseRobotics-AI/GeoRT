"""Validated offline reference cache; no Pinocchio dependency in GeoRT."""
import hashlib
import json
from pathlib import Path
import numpy as np


SPEC = Path(__file__).parent / 'baselines/wuji_manus_right.json'


def load_wuji_cache(path, data_path, joint_order):
    spec = json.loads(SPEC.read_text())
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata']))
        for key in ('source_commit', 'config_sha256', 'hand_side', 'lp_alpha', 'native_urdf_sha256'):
            if meta.get(key) != spec[key]:
                raise ValueError(f'Wuji baseline differs from confirmed specification: {key}')
        if meta.get('data_sha256') != hashlib.sha256(Path(data_path).read_bytes()).hexdigest():
            raise ValueError('Wuji cache recording hash mismatch')
        points = np.load(data_path, allow_pickle=False)
        if points.ndim != 3 or points.shape[1:] != (21, 3):
            raise ValueError('Wuji cache requires a single (T,21,3) npy recording')
        names = archive['joint_names'].tolist()
        if len(names) != len(set(names)) or set(names) != set(joint_order) or len(joint_order) != len(set(joint_order)):
            raise ValueError('Wuji cache joint names must match exactly without duplicates')
        if not np.array_equal(archive['source_frame'], np.arange(len(points))):
            raise ValueError('Wuji cache must preserve every input frame in original order')
        order = [names.index(name) for name in joint_order]
        result = {'metadata': meta, 'cache_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for mode in ('filtered', 'unfiltered'):
            array = archive[f'qpos_{mode}']
            if array.shape != (len(points), len(names)) or not np.isfinite(array).all():
                raise ValueError('Invalid Wuji qpos cache shape or values')
            result[mode] = array[:, order].copy()
        code = archive['nlopt_code']
        if code.shape != (len(points),):
            raise ValueError('Invalid solver status length')
        result['nlopt_code'] = code.copy()
    return result


def comparison_report(cache, ids, exact, q, tips, ref_q, ref_tips, human):
    import torch
    from geort.stage_metrics import summarize
    selected_ids = set(ids.tolist())
    report = {'primary': 'filtered', 'cache_sha256': cache['cache_sha256'],
              'source_commit': cache['metadata']['source_commit'],
              'config_sha256': cache['metadata']['config_sha256'],
              'native_urdf_sha256': cache['metadata']['native_urdf_sha256'],
              'evaluation_geometry': 'GeoRT exact FK, common joint names; native Wuji URDF differs; end-to-end comparison',
              'sequence_policy': cache['metadata']['sequence_policy'],
              'wuji_runtime_full_recording_ms': {key: cache['metadata'][key] for key in ('runtime_ms_p50', 'runtime_ms_p95')},
              'timing_scope': cache['metadata']['timing_scope'] + '; full recording, not a speed ratio against batched GeoRT',
              'solver_negative_status_frames': int((cache['nlopt_code'][ids] < 0).sum()),
              'solver_fallback_frames': sum(f['frame'] in selected_ids for f in cache['metadata']['optimization_failures'])}
    report['candidate_operational'] = operational_metrics(q, exact, human)
    report['original_geort_operational'] = operational_metrics(ref_q, exact, human)
    for mode in ('filtered', 'unfiltered'):
        baseline_q = cache[mode][ids]
        tensor = torch.as_tensor(baseline_q, dtype=torch.float32)
        if ((tensor < exact.lower - 1e-5) | (tensor > exact.upper + 1e-5)).any():
            raise ValueError('Wuji output exceeds common physical joint limits; do not silently clip')
        with torch.inference_mode():
            baseline_tips = exact(2*(tensor-exact.lower)/(exact.upper-exact.lower)-1)['tips'].numpy()
        report[mode] = {
            'operational': operational_metrics(baseline_q, exact, human),
            'wuji_vs_original_geort': summarize(baseline_q, baseline_tips, ref_q, ref_tips, human),
            'candidate_vs_wuji': summarize(q, tips, baseline_q, baseline_tips, human),
        }
    report['interpretation'] = 'Position deltas are behavior differences, not human-target errors. Legacy gates are not a Wuji acceptance specification.'
    return report


def operational_metrics(q, exact, points):
    """Common observed task proxies; no claim of contact, timing, or coverage truth."""
    import torch
    from geort.coordination import skeleton_features
    from geort.coordination_diagnostics import distribution, response_metrics
    human = skeleton_features(torch.as_tensor(points, dtype=torch.float32), [4,8,12,16,20])
    tensor = torch.as_tensor(q, dtype=torch.float32)
    norm = 2*(tensor-exact.lower)/(exact.upper-exact.lower)-1
    with torch.inference_mode():
        robot = exact(norm)
    tips, axes = robot['tips'].numpy(), robot['axes'].numpy()
    human_tips, human_axes = human['tips'].numpy(), human['axes'].numpy()
    valid = human['bone_valid'].numpy()
    direction = np.rad2deg(np.arccos(np.clip((axes[:,:,-1]*human_axes[:,:,-1]).sum(-1), -1, 1)))
    shape = ((axes[:,:,1:]*axes[:,:,:-1]).sum(-1) - (human_axes[:,:,1:]*human_axes[:,:,:-1]).sum(-1))
    shape_valid = (valid[:,:,1:] & valid[:,:,:-1]).copy(); shape_valid[:,0] = False
    hi, ro = np.diff(human_tips, axis=0), np.diff(tips, axis=0)
    hv, rv = human_tips[:,1:] - human_tips[:,:1], tips[:,1:] - tips[:,:1]
    options = exact.config.get('objectives', {})
    scale = options.get('relation_scale')
    opening = None
    opening_fraction = None
    if scale is not None:
        mask = (np.linalg.norm(hv, axis=-1) >= options['relation_min_opening_m']) & np.array(options.get('relation_valid', [True]*4))[None]
        errors = abs(np.linalg.norm(rv, axis=-1)-np.linalg.norm(hv*np.asarray(scale)[None,:,None], axis=-1))*1000
        opening = distribution(errors[mask]); opening_fraction = float(mask.mean())
    near = np.any(abs(norm.numpy()) > .95, axis=-1)
    pair_near = near[1:] | near[:-1]
    return {'distal_axis_error_deg': distribution(direction[valid[:,:,-1]]),
            'distal_axis_valid_fraction': float(valid[:,:,-1].mean()),
            'adjacent_bone_cosine_error': distribution(abs(shape[shape_valid])),
            'shape_valid_fraction_non_thumb': float(shape_valid[:,1:].mean()),
            'opening_error_mm_open_poses': opening, 'opening_valid_fraction': opening_fraction,
            'local_response': response_metrics(hi, ro),
            'common_motion': response_metrics(hi.mean(1), ro.mean(1)),
            'relative_motion': response_metrics(np.diff(hv,axis=0), np.diff(rv,axis=0)),
            'near_limit_response': response_metrics(hi[pair_near], ro[pair_near]),
            'stationary_input_robot_step_mm': distribution(np.linalg.norm(ro,axis=-1)[np.linalg.norm(hi,axis=-1)<.0002]*1000),
            'joint_limit_violation_count': int(((norm.numpy() < -1.00001) | (norm.numpy() > 1.00001)).sum()),
            'temporal': {'available': False, 'reason': 'legacy recording has no acquisition timestamps'},
            'collision': 'not measured; no safety claim', 'feasible_coverage_fraction': None,
            'scope': 'observed adjacent-frame responses; includes native filter and warm-start effects; no Jacobian claim'}
