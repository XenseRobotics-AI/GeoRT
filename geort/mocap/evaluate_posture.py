"""CPU evaluation of Wuji models and optional diagnostic posture correction."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm

from geort.export import resolve_checkpoint
from geort.formatter import HandFormatter
from geort.model import build_ik_model, ik_input
from geort.posture import FingerKinematics, WujiPostureCorrector
from geort.utils.config_utils import get_config, parse_config_keypoint_info, parse_config_joint_limit


def distribution(values):
    values = np.asarray(values)
    return {'mean': float(np.mean(values)), 'p95': float(np.percentile(values, 95)),
            'max': float(np.max(values))}


def evaluate(qposes, human, chains, tolerance):
    tips, directions = [], []
    for q in qposes:
        fk = [chain.forward(q[chain.indices]) for chain in chains]
        tips.append([x[0] for x in fk]); directions.append(fk[-1][1])
    tips, directions = np.asarray(tips), np.asarray(directions)
    violation = np.maximum(-np.deg2rad(tolerance) - qposes[:, -2:], 0)
    human_direction = human[:, 20] - human[:, 19]
    valid = np.linalg.norm(human_direction, axis=1) > 1e-8
    human_direction = human_direction[valid] / np.linalg.norm(human_direction[valid], axis=1, keepdims=True)
    angles = np.rad2deg(np.arccos(np.clip(np.sum(directions[valid] * human_direction, axis=1), -1, 1)))
    metrics = {'backbend_frame_fraction': float(np.mean(np.any(violation > np.deg2rad(.1), axis=1))),
               'backbend_over_15deg_fraction': float(np.mean(np.any(qposes[:, -2:] < -np.deg2rad(15), axis=1))),
               'mcp_extension_over_15deg_fraction': float(np.mean(qposes[:, -4] < -np.deg2rad(15.1))),
               'backbend_excess_deg': distribution(np.rad2deg(violation.max(axis=1))),
               'distal_direction_error_deg': distribution(angles) if len(angles) else None,
               'little_joint_step_deg': distribution(np.rad2deg(np.abs(np.diff(qposes[:, -4:], axis=0)))) if len(qposes) > 1 else None,
               'little_joint_range_deg': np.rad2deg(np.ptp(qposes[:, -4:], axis=0)).tolist(),
               'little_tip_occupied_5mm_voxels': int(len(np.unique(np.floor(tips[:, -1] / .005), axis=0)))}
    for finger, name in enumerate(('index', 'middle', 'ring', 'little'), 1):
        mask = np.linalg.norm(human[:, 4] - human[:, 4 * (finger + 1)], axis=1) < .015
        metrics[f'{name}_pinch_count'] = int(mask.sum())
        metrics[f'{name}_pinch_mm'] = distribution(1000 * np.linalg.norm(tips[mask, 0] - tips[mask, finger], axis=1)) if mask.any() else None
    return metrics, tips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='wuji_hand2_beta1_right_last')
    parser.add_argument('--reference-checkpoint', help='Compare raw model output to this checkpoint')
    parser.add_argument('--reference-weights', choices=['best', 'last'], default='last')
    parser.add_argument('--weights', choices=['best', 'last'], default='best')
    parser.add_argument('--data', type=Path, default=Path('data/human_alex.npy'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--starts', type=int, choices=[1, 3], default=3)
    parser.add_argument('--continuity-comparison', action='store_true',
                        help='Compare round-one mode with 4/8/12 degree per-frame bounds')
    parser.add_argument('--raw-only', action='store_true', help='Evaluate model output without runtime correction')
    parser.add_argument('--collision', action='store_true', help='Also measure actual SAPIEN self-contact depths')
    args = parser.parse_args()
    if args.stride < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error('stride and max-frames must be positive')
    human_all = np.load(args.data)
    if human_all.ndim != 3 or human_all.shape[1:] != (21, 3) or not len(human_all) or not np.isfinite(human_all).all():
        parser.error('data must be finite, nonempty (T, 21, 3)')
    ids = np.arange(len(human_all))[::args.stride][:args.max_frames]
    human = human_all[ids]
    run = resolve_checkpoint(args.checkpoint).resolve()
    config = get_config('wuji_hand2_beta1_right')
    model_config = json.loads((run / 'config.json').read_text())
    if any(model_config.get(k) != v for k, v in config.items() if k != 'collision_exclusions'):
        parser.error('Checkpoint does not match current Wuji kinematics')
    weights = run / f'{args.weights}.pth'
    info = parse_config_keypoint_info(config)
    torch.set_num_threads(1)
    model = build_ik_model(model_config).eval()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    with torch.no_grad():
        normalized = model(ik_input(model_config, torch.tensor(human, dtype=torch.float32))).numpy()
    raw = HandFormatter(*parse_config_joint_limit(model_config)).unnormalize(normalized)
    args.output.mkdir(parents=True, exist_ok=False)
    chains = [FingerKinematics(config, item['name']) for item in config['fingertip_link']]
    report = {'checkpoint': str(run), 'weights': args.weights,
              'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
              'urdf_sha256': hashlib.sha256(Path(config['urdf_path']).read_bytes()).hexdigest(),
              'data': str(args.data.resolve()), 'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'frames': len(human), 'stride': args.stride, 'starts': args.starts,
              'mcp_extension_tolerance_deg': 15., 'extension_tolerance_deg': 5., 'backbend_metric_slack_deg': .1, 'scope': 'kinematic targets; not hardware or PD validation',
              'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                for name in ('geort/posture.py', 'geort/kinematics.py', 'geort/model.py', 'geort/mocap/evaluate_posture.py')},
              'variants': {}}
    hand = None
    if args.collision:
        from geort.env.hand import HandKinematicModel
        hand = HandKinematicModel.build_from_config(config)
    raw_metrics, raw_tips = evaluate(raw, human, chains, 5)
    if args.reference_checkpoint:
        reference_run = resolve_checkpoint(args.reference_checkpoint).resolve()
        reference_config = json.loads((reference_run / 'config.json').read_text())
        if any(reference_config.get(k) != v for k, v in config.items() if k != 'collision_exclusions'):
            parser.error('Reference checkpoint does not match current Wuji kinematics')
        reference_weights = reference_run / f'{args.reference_weights}.pth'
        reference_model = build_ik_model(reference_config).eval()
        reference_model.load_state_dict(torch.load(reference_weights, map_location='cpu', weights_only=True))
        with torch.inference_mode():
            reference_normalized = reference_model(ik_input(reference_config, torch.tensor(human, dtype=torch.float32))).numpy()
        reference_q = HandFormatter(*parse_config_joint_limit(reference_config)).unnormalize(reference_normalized)
        reference_metrics, reference_tips = evaluate(reference_q, human, chains, 5)
        report['reference'] = {'checkpoint': str(reference_run), 'weights': args.reference_weights,
                               'weights_sha256': hashlib.sha256(reference_weights.read_bytes()).hexdigest(),
                               'metrics': reference_metrics,
                               'raw_little_tip_shift_mm': distribution(1000 * np.linalg.norm(raw_tips[:, -1] - reference_tips[:, -1], axis=1)),
                               'other_fingers_unchanged': bool(np.array_equal(raw[:, :16], reference_q[:, :16]))}
    variants = [('baseline', None, 0, 0), ('penalty_2mm', 2., 0, 8),
                ('direction_1mm', 1., .05, 8), ('direction_2mm', 2., .05, 8), ('direction_5mm', 5., .05, 8)]
    if args.continuity_comparison:
        variants = [('baseline', None, 0, 0), ('round1_2mm', 2., .05, 0),
                    ('step4_2mm', 2., .05, 4), ('step8_2mm', 2., .05, 8), ('step12_2mm', 2., .05, 12)]
    if args.raw_only:
        variants = [('baseline', None, 0, 0)]
    for name, budget, weight, max_step in variants:
        times, statuses, corrected, budget_flags = [], [], [], []
        if budget is None:
            qposes = raw.copy()
        else:
            corrector = WujiPostureCorrector(config, tip_budget_mm=budget,
                                             direction_weight=weight, starts=args.starts,
                                             max_joint_step_deg=max_step)
            for q, points in tqdm(zip(raw, human), total=len(raw), desc=name):
                start = time.perf_counter()
                result = corrector.correct(q, points)
                times.append((time.perf_counter() - start) * 1000)
                statuses.append(result.status); corrected.append(result.qpos)
                budget_flags.append(result.tip_budget_satisfied)
            qposes = np.asarray(corrected)
        metrics, tips = evaluate(qposes, human, chains, 5)
        shift = np.linalg.norm(tips[:, -1] - raw_tips[:, -1], axis=1) * 1000
        metrics['tip_shift_mm'] = distribution(shift)
        metrics['tip_budget_mm'] = budget
        metrics['direction_weight'] = weight
        metrics['max_joint_step_deg'] = max_step
        metrics['tip_budget_exceeded_frames'] = int(np.sum(shift > budget + 1e-8)) if budget else 0
        metrics['correction_ms'] = distribution(times) if times else None
        metrics['status'] = dict(Counter(statuses))
        metrics['new_mcp_overextension_frames'] = int(np.sum(qposes[:, -4] < np.minimum(raw[:, -4], -np.deg2rad(15)) - 1e-8))
        metrics['other_fingers_unchanged'] = bool(np.array_equal(qposes[:, :-4], raw[:, :-4]))
        metrics['little_pair_distance_change_mm'] = distribution(np.abs(
            np.linalg.norm(tips[:, :-1] - tips[:, -1, None], axis=2)
            - np.linalg.norm(raw_tips[:, :-1] - raw_tips[:, -1, None], axis=2)) * 1000)
        if budget is not None:
            if not metrics['other_fingers_unchanged']:
                raise RuntimeError('Correction changed another finger')
            if not max_step and shift.max() > budget + 1e-8:
                raise RuntimeError('Independent-frame mode violated Cartesian budget')
            if max_step and len(qposes) > 1 and np.rad2deg(np.abs(np.diff(qposes[:, -4:], axis=0))).max() > max_step + 1e-8:
                raise RuntimeError('Continuity bound violated')
            if np.any(np.asarray(budget_flags) & (shift > budget + 1e-8)):
                raise RuntimeError('Unreported Cartesian relaxation')
        depths = np.array([])
        if hand is not None:
            depths = hand.self_collision_depth(qposes) * 1000
            metrics['self_penetration_mm'] = distribution(depths)
            metrics['penetration_over_2mm_fraction'] = float(np.mean(depths > 2))
        np.savez_compressed(args.output / f'{name}.npz', qpos=qposes, tips=tips,
                            source_frame=ids, self_penetration_mm=depths)
        report['variants'][name] = metrics
        (args.output / 'metrics.json').write_text(json.dumps(report, indent=2))
        print(name, json.dumps(metrics), flush=True)
    print(f'Report: {args.output / "metrics.json"}')


if __name__ == '__main__':
    main()
