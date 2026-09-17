"""Observed response diagnostics: never infer an unmeasured feasible workspace."""
import math
import time
import numpy as np
import torch
from geort.coordination import skeleton_features
from geort.coordination_data import batch
from geort.coordination_loss import task_objective
from geort.utils.config_utils import parse_config_keypoint_info


def distribution(values):
    values = np.asarray(values)
    if not values.size:
        return None
    return {'mean': float(values.mean()), 'p05': float(np.percentile(values, 5)),
            'p95': float(np.percentile(values, 95)), 'max': float(values.max()), 'count': values.size}


def response_metrics(human, robot, active_threshold=.0002, dead_threshold=.00002):
    hi = np.linalg.norm(human, axis=-1); ro = np.linalg.norm(robot, axis=-1)
    active = hi >= active_threshold
    responding = active & (ro >= dead_threshold)
    gains = ro[active] / hi[active]
    cosine = np.sum(human[responding] * robot[responding], axis=-1) / (hi[responding] * ro[responding])
    return {'active_count': int(active.sum()), 'total_count': active.size,
            'active_fraction': float(active.mean()) if active.size else 0.,
            'dead_zone_fraction': float(np.mean(ro[active] < dead_threshold)) if active.any() else None,
            'gain': distribution(gains), 'response_direction_error_deg': distribution(np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))),
            'direction_valid_count': int(responding.sum()),
            'input_threshold_m': active_threshold, 'output_dead_threshold_m': dead_threshold}


def time_metrics(tips, human_tips, times, sequence_id):
    if times is None:
        return {'available': False, 'reason': 'No recorded timestamps/sequence IDs; replay FPS is not acquisition time'}
    same = sequence_id[1:] == sequence_id[:-1]
    dt = np.diff(times)
    if np.any(dt[same] <= 0):
        raise ValueError('Nonpositive time interval')
    velocities = np.diff(tips, axis=0)[same] / dt[same, None, None]
    human_v = np.diff(human_tips, axis=0)[same] / dt[same, None, None]
    # Triples must stay inside a single recording; midpoint spacing for irregular time.
    triples = same[1:] & same[:-1]
    v_all = np.zeros_like(np.diff(tips, axis=0))
    v_all[same] = velocities
    acc = np.diff(v_all, axis=0)[triples] / ((dt[1:] + dt[:-1])[triples, None, None] * .5)
    still = np.linalg.norm(np.diff(human_tips, axis=0)[same], axis=-1) < .0002
    return {'available': True, 'robot_speed_m_s': distribution(np.linalg.norm(velocities, axis=-1)),
            'human_speed_m_s': distribution(np.linalg.norm(human_v, axis=-1)),
            'robot_acceleration_m_s2': distribution(np.linalg.norm(acc, axis=-1)),
            'stationary_input_robot_speed_m_s': distribution(np.linalg.norm(velocities, axis=-1)[still]),
            'lag_seconds': None, 'lag_note': 'No ground-truth robot trajectory or verified task signal correspondence; latency not inferred'}


def evaluate(model, parent, recording, ids, exact, neural_fk, config, options):
    model.eval(); parent.eval()
    sample = batch(recording, ids)
    info = parse_config_keypoint_info(config)
    with torch.inference_mode():
        start = time.perf_counter()
        output = model(sample)
        seconds = time.perf_counter() - start
        human = skeleton_features(sample, info['human_id'])
        baseline = exact(parent(human['tips']))
        _, losses, robot = task_objective(output, sample, exact, config, options, baseline['tips'], reference_q=baseline['q'])
        approx = neural_fk(output)
    q, tips, human_tips = robot['q'].numpy(), robot['tips'].numpy(), human['tips'].numpy()
    axes = human['axes'].numpy(); valid = human['bone_valid'].numpy()
    cosine = (robot['axes'][:, :, -1].numpy() * axes[:, :, -1]).sum(-1)
    direction_error = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    pairs = np.diff(ids) == 1
    times, seq = None, None
    if 'timestamps' in recording:
        times = recording['timestamps'][ids]; seq = recording['sequence_id'][ids]
        pairs &= seq[1:] == seq[:-1]
    hi, ro = np.diff(human_tips, axis=0)[pairs], np.diff(tips, axis=0)[pairs]
    normalized = output.numpy()
    near_limit = np.any(np.abs(normalized) > .95, axis=1)
    pair_near = (near_limit[1:] | near_limit[:-1])[pairs]
    common = response_metrics(hi.mean(1), ro.mean(1))
    thumb = info['human_id'].index(4)
    others = [i for i in range(len(info['human_id'])) if i != thumb]
    human_relative = human_tips[:, others] - human_tips[:, thumb, None]
    robot_relative = tips[:, others] - tips[:, thumb, None]
    opening = response_metrics(np.diff(human_relative, axis=0)[pairs], np.diff(robot_relative, axis=0)[pairs])
    scales = np.asarray(options['relation_scale'])[None, :, None]
    opening_error = abs(np.linalg.norm(robot_relative, axis=-1) - np.linalg.norm(human_relative * scales, axis=-1)) * 1000
    open_valid = (np.linalg.norm(human_relative, axis=-1) >= options['relation_min_opening_m']) & np.asarray(options.get('relation_valid', [True] * len(others)))[None]
    cross = {}
    skeleton_change = np.diff(human['points'].numpy(), axis=0)[pairs]
    for f, end in enumerate(info['human_id']):
        other_points = [j for e in info['human_id'] if e != end for j in range(e - 3, e + 1)]
        only = (np.linalg.norm(hi[:, f], axis=-1) >= .0002) & (np.linalg.norm(skeleton_change[:, other_points], axis=-1).max(-1) < .0001)
        other_fingers = [j for j in range(len(info['human_id'])) if j != f]
        cross[str(end)] = {'isolated_pair_count': int(only.sum()),
                          'other_finger_response_m': distribution(np.linalg.norm(ro[only][:, other_fingers], axis=-1))}
    opening_size = np.linalg.norm(human_relative, axis=-1).mean(-1)
    pair_opening = ((opening_size[1:] + opening_size[:-1]) * .5)[pairs]
    regions = {'narrow_lt30mm': pair_opening < .03,
               'middle_30to60mm': (pair_opening >= .03) & (pair_opening < .06),
               'wide_ge60mm': pair_opening >= .06}
    human_axis = axes[:, :, -1]
    robot_axis = robot['axes'][:, :, -1].numpy()
    ha = np.arccos(np.clip((human_axis[1:] * human_axis[:-1]).sum(-1), -1, 1))[pairs]
    ra = np.arccos(np.clip((robot_axis[1:] * robot_axis[:-1]).sum(-1), -1, 1))[pairs]
    angle_valid = (valid[1:, :, -1] & valid[:-1, :, -1])[pairs] & (ha > math.radians(1))
    angle_response = {'active_count': int(angle_valid.sum()),
                      'angular_gain': distribution(ra[angle_valid] / ha[angle_valid]),
                      'axis_dead_zone_fraction': float(np.mean(ra[angle_valid] < math.radians(.1))) if angle_valid.any() else None}
    still = np.linalg.norm(hi, axis=-1) < .0002
    finger_metrics = {}
    for i, finger in enumerate(config['fingertip_link']):
        joint_ids = info['joint'][i]
        entry = {'axis_error_deg': distribution(direction_error[:, i][valid[:, i, -1]]),
                 'axis_valid_fraction': float(valid[:, i, -1].mean()),
                 'joint_range_deg': np.rad2deg(np.ptp(q[:, joint_ids], axis=0)).tolist()}
        if i != thumb:
            pinch = np.linalg.norm(human_tips[:, i] - human_tips[:, thumb], axis=-1) < .015
            entry.update(backbend_over_15deg_fraction=float(np.mean(np.any(q[:, joint_ids[2:]] < -np.deg2rad(15), axis=1))),
                         mcp_extension_over_15deg_fraction=float(np.mean(q[:, joint_ids[0]] < -np.deg2rad(15))),
                         legacy_pinch_frame_count=int(pinch.sum()),
                         legacy_pinch_site_distance_mm=distribution(np.linalg.norm(tips[pinch, i] - tips[pinch, thumb], axis=-1) * 1000))
        finger_metrics[finger['name']] = entry
    report = {'scope': 'observed held-out frames, not feasible-workspace coverage or hardware validation',
              'per_finger': finger_metrics,
              'frames': len(ids), 'features': config['features'], 'losses': losses,
              'axis_error_deg': distribution(direction_error[valid[:, :, -1]]),
              'axis_valid_fraction': float(valid[:, :, -1].mean()),
              'opening_error_mm_open_poses': distribution(opening_error[open_valid]),
              'near_contact_excluded_fraction': float((np.linalg.norm(human_relative, axis=-1) < options['relation_min_opening_m']).mean()),
              'relation_valid_fraction': float(open_valid.mean()),
              'parent_tip_delta_mm': distribution(np.linalg.norm(tips - baseline['tips'].numpy(), axis=-1) * 1000),
              'neural_fk_vs_exact_mm': distribution(np.linalg.norm(tips - approx.numpy(), axis=-1) * 1000),
              'local_response': response_metrics(hi, ro), 'common_motion': common, 'relative_motion': opening,
              'near_limit_response': response_metrics(hi[pair_near], ro[pair_near]),
              'interior_response': response_metrics(hi[~pair_near], ro[~pair_near]),
              'single_finger_cross_response': cross,
              'opening_region_response': {name: response_metrics(hi[mask], ro[mask]) for name, mask in regions.items()},
              'axis_change_response': angle_response,
              'stationary_input_robot_step_m': distribution(np.linalg.norm(ro, axis=-1)[still]),
              'joint_limit_violation_count': int(np.sum((normalized < -1) | (normalized > 1))),
              'near_limit_frame_fraction': float(near_limit.mean()),
              'joint_step_rad': distribution(abs(np.diff(q, axis=0)[pairs])),
              'human_tip_excursion_m': np.ptp(human_tips, axis=0).tolist(),
              'robot_tip_excursion_m': np.ptp(tips, axis=0).tolist(),
              'occupied_5mm_voxels_per_finger': [len(np.unique(np.floor(tips[:, i] / .005), axis=0)) for i in range(tips.shape[1])],
              'occupied_5mm_joint_tip_combinations': len(np.unique(np.floor(tips / .005).reshape(len(tips), -1), axis=0)),
              'feasible_coverage_fraction': None, 'coverage_reason': 'No independent collision-feasible reference set',
              'temporal': time_metrics(tips, human_tips, times, seq),
              'inference_batch_mean_ms': seconds * 1000 / len(ids),
              'latency_scope': 'CPU batched model only; not end-to-end',
              'runtime_corrections': 'disabled for model evaluation', 'collision': 'not evaluated'}
    return report, {'qpos': q, 'tips': tips, 'source_frame': ids}
