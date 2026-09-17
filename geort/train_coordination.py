"""Bounded CPU experiments for optional whole-hand coordination (no hardware)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from geort.coordination import initialize_coordination, skeleton_features, robot_signature
from geort.coordination_data import load_recording, batch, split_indices
from geort.coordination_diagnostics import evaluate
from geort.coordination_loss import ExactHand, baseline_objective, task_objective
from geort.export import resolve_checkpoint, GeoRTRetargetingModel
from geort.model import build_ik_model, FKModel
from geort.utils.config_utils import parse_config_keypoint_info


def validate_experiment(experiment):
    required = {'name', 'mode', 'context', 'seed', 'steps', 'batch_size', 'lr', 'split', 'baseline', 'task'}
    if set(experiment) != required:
        raise ValueError('Unknown or missing experiment fields')
    if experiment['mode'] not in ('tip_only', 'skeleton') or not isinstance(experiment['context'], bool):
        raise ValueError('Only tip_only/skeleton supported; pose lacks calibrated orientations')
    for name in ('steps', 'batch_size'):
        if not isinstance(experiment[name], int) or experiment[name] < 2:
            raise ValueError(f'{name} must be an integer >=2')
    if not np.isfinite(experiment['lr']) or experiment['lr'] <= 0:
        raise ValueError('lr must be finite and positive')
    if set(experiment['baseline']) != {'coverage', 'response_direction', 'flatness', 'pinch'}:
        raise ValueError('Unsupported baseline weights')
    names = {'axis', 'shape', 'posture', 'relation', 'axis_tolerance_deg', 'shape_tolerance_cos',
             'posture_tolerance_deg', 'relation_tolerance_m', 'relation_min_opening_m'}
    optional = {'anchor', 'anchor_tolerance_m'}
    if not names <= set(experiment['task']) or set(experiment['task']) - names - optional:
        raise ValueError('Unsupported task options')
    for mapping in (experiment['baseline'], experiment['task']):
        if not all(np.isfinite(value) and value >= 0 for value in mapping.values()):
            raise ValueError('Weights/options must be finite and nonnegative')
    for key in names - {'axis', 'shape', 'posture', 'relation'}:
        if experiment['task'][key] <= 0:
            raise ValueError('Tolerances must be positive')
    if experiment['task'].get('anchor', 0) > 0 and experiment['task'].get('anchor_tolerance_m', 0) <= 0:
        raise ValueError('Anchor tolerance must be positive')
    if not 0 < experiment['task']['axis_tolerance_deg'] <= 180:
        raise ValueError('Invalid angular tolerance')
    if experiment['task']['relation'] > 0 and experiment['baseline']['pinch'] != 0:
        raise ValueError('Relation objective replaces pinch; explicitly set baseline.pinch=0')


def run(args):
    experiment = json.loads(args.config.read_text())
    for key in ('steps', 'batch_size', 'seed'):
        value = getattr(args, key)
        if value is not None:
            experiment[key] = value
    validate_experiment(experiment)
    if args.save_every < 0 or args.warmup_steps < 0:
        raise ValueError('save-every must be nonnegative')
    torch.set_num_threads(1)
    torch.manual_seed(experiment['seed'])
    recording = load_recording(args.data)
    split = split_indices(recording, experiment['split'])
    parent_path = resolve_checkpoint(args.parent).resolve()
    parent_config = json.loads((parent_path / 'config.json').read_text())
    if parent_config.get('model_type', 'fingertip_v1') != 'fingertip_v1':
        raise ValueError('Use a preserved original fingertip baseline as parent')
    if parent_config['name'] != 'wuji_hand2_beta1_right':
        raise ValueError('First experiment calibration supports Wuji right only; original robot trainers remain available')
    parent = build_ik_model(parent_config).eval()
    weights_path = parent_path / f'{args.weights}.pth'
    parent.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    config, model = initialize_coordination(parent_config, parent, experiment['mode'], experiment['context'], args.local_features)
    init_path = None
    if args.init_base:
        init_path = resolve_checkpoint(args.init_base).resolve()
        init_config = json.loads((init_path / 'config.json').read_text())
        if robot_signature(init_config) != robot_signature(config) or init_config.get('model_type') != 'coordination_v1':
            raise ValueError('Base initialization must be a compatible coordination checkpoint')
        saved_parent = torch.load(init_path / 'baseline.pth', map_location='cpu', weights_only=True)
        if set(saved_parent) != set(parent.state_dict()) or any(not torch.equal(value, parent.state_dict()[key]) for key,value in saved_parent.items()):
            raise ValueError('Base initialization uses a different original parent')
        initialized = build_ik_model(init_config).eval()
        initialized.load_state_dict(torch.load(init_path / 'best.pth', map_location='cpu', weights_only=True))
        model.base.load_state_dict(initialized.base.state_dict())
    if args.freeze_base:
        if not model.local and model.context is None:
            raise ValueError('Freezing base requires trainable heads')
        model.base.requires_grad_(False)
    if args.head_lr is not None and (not np.isfinite(args.head_lr) or args.head_lr <= 0 or (not model.local and model.context is None)):
        raise ValueError('Positive head-lr requires trainable heads')
    info = parse_config_keypoint_info(config)
    exact = ExactHand(config)
    wuji_cache = None
    if args.wuji_cache:
        from geort.wuji_baseline import load_wuji_cache
        wuji_cache = load_wuji_cache(args.wuji_cache, args.data, config['joint_order'])
    neural_fk = FKModel(info['joint']).eval().requires_grad_(False)
    fk_path = args.fk_checkpoint or Path(f"checkpoint/fk_model_{config['name']}.pth")
    if not fk_path.exists():
        raise FileNotFoundError('Baseline Neural FK is required; generate it with the original trainer or provide --fk-checkpoint')
    neural_fk.load_state_dict(torch.load(fk_path, map_location='cpu', weights_only=True))
    training = batch(recording, split['train'])
    features = skeleton_features(training, info['human_id'])
    validity = features['bone_valid']
    if (experiment['mode'] == 'skeleton' or experiment['task']['axis'] > 0 or experiment['task']['shape'] > 0) and not validity.any():
        raise ValueError('Skeleton/axis/shape explicitly requested, but all bone data are invalid')
    if experiment['task']['axis'] > 0 and not validity[:, :, -1].any():
        raise ValueError('Axis loss requires valid distal bones')
    if experiment['task']['shape'] > 0 and not (validity[:, 1:, 1:] & validity[:, 1:, :-1]).any():
        raise ValueError('Shape loss requires valid adjacent non-thumb bones')
    for name in ('validation', 'test'):
        skeleton_features(batch(recording, split[name]), info['human_id'])
    thumb = info['human_id'].index(4)
    others = [i for i in range(len(info['human_id'])) if i != thumb]
    with torch.inference_mode():
        # Training-only geometric length calibration, not paired robot labels.
        human_lengths = features['lengths'].sum(-1)
        robot_lengths = exact(torch.zeros(1, len(config['joint_order'])))['lengths'].sum(-1)[0]
        ratios, ratio_valid = [], []
        for i in others:
            mask = validity[:, i].all(-1) & validity[:, thumb].all(-1)
            ratio_valid.append(bool(mask.any()))
            if not mask.any():
                if experiment['task']['relation'] > 0:
                    raise ValueError('Missing valid bone lengths for relation calibration')
                ratios.append(1.)  # inactive relation; explicitly marked unavailable below
            else:
                ratios.append(float((robot_lengths[i] + robot_lengths[thumb]) /
                                    (human_lengths[mask, i] + human_lengths[mask, thumb]).median()))
    teacher = None
    teacher_posture_mask = None
    if args.warmup_steps:
        if args.warmup_targets is None or not 0 < args.warmup_steps < experiment['steps']:
            raise ValueError('Warmup requires targets and must be shorter than the training budget')
        with np.load(args.warmup_targets, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            expected = (hashlib.sha256(weights_path.read_bytes()).hexdigest(), hashlib.sha256(args.data.read_bytes()).hexdigest(), robot_signature(config))
            if (meta['parent_sha256'], meta['data_sha256'], meta['robot_signature']) != expected:
                raise ValueError('Warmup target provenance does not match parent/data/robot')
            if not np.array_equal(data['source_frame'], split['train']):
                raise ValueError('Warmup targets must cover exactly the training split')
            if meta.get('schema_version') not in (2, 3) or meta.get('max_tip_shift_m') != .002:
                raise ValueError('Warmup targets require quality-filtered schema v2/v3 with 2 mm budget')
            key = 'independent_normalized' if args.target_variant == 'independent' else 'normalized'
            if key not in data:
                raise ValueError('Requested target variant is unavailable')
            teacher = torch.tensor(data[key], dtype=torch.float32)
            if meta['schema_version'] == 3:
                mask_key = 'independent_posture_mask' if args.target_variant == 'independent' else 'posture_mask'
                if data[mask_key].dtype != np.bool_ or data[mask_key].shape != (len(split['train']), 4):
                    raise ValueError('Invalid training target posture mask')
                teacher_posture_mask = torch.tensor(data[mask_key], dtype=torch.bool)
        if teacher.shape != (len(split['train']), len(config['joint_order'])) or not torch.isfinite(teacher).all() or (teacher.abs() > 1.00001).any():
            raise ValueError('Invalid normalized warmup targets')
        with torch.no_grad():
            original_tips = exact(parent(features['tips']))['tips']
            shifts = (exact(teacher)['tips'] - original_tips).norm(dim=-1)
        if not torch.isfinite(shifts).all() or shifts.max() > .002001:
            raise ValueError('Warmup targets violate the exact FK 2 mm position budget')
        if teacher_posture_mask is not None:
            from geort.prepare_consistent_targets import posture_eligibility, quality_mask
            with torch.no_grad():
                original = exact(parent(features['tips']))
                prepared = exact(teacher)
                angle = lambda axes: torch.acos((axes[:,:,-1]*features['axes'][:,:,-1]).sum(-1).clamp(-1,1)).numpy()
                mcp = prepared['q'].reshape(-1,5,4)[:,1:,0].numpy()
                floor = np.minimum(original['q'].reshape(-1,5,4)[:,1:,0].numpy(), -np.deg2rad(15))
                ok = quality_mask(shifts.numpy()[:,1:], angle(prepared['axes'])[:,1:], angle(original['axes'])[:,1:],
                    features['bone_valid'].numpy()[:,1:,-1], mcp, floor)
            if not ok.all():
                raise ValueError('Training targets violate direction or MCP quality constraints')
            if not np.array_equal(teacher_posture_mask.numpy(), posture_eligibility(prepared['q'].numpy(), original['q'].numpy())):
                raise ValueError('Posture eligibility does not match training targets')
    elif args.warmup_targets is not None:
        raise ValueError('Set positive warmup-steps when providing targets')
    if args.posture_policy == 'feasible' and teacher_posture_mask is None:
        raise ValueError('Feasible posture policy requires schema v3 training-only targets')
    options = {'posture_reference_mcp': args.posture_policy == 'feasible', **experiment['task'], 'relation_scale': ratios, 'relation_valid': ratio_valid}
    config['objectives'] = options
    config['calibration'] = {'relation': 'train_median_sum_of_thumb_and_other_chain_lengths',
                             'valid_bone_fraction': float(validity.float().mean()),
                             'confidence': 'provided' if 'confidence' in recording else 'unavailable; geometric finite/length checks only',
                             'directions': 'geort canonical vectors vs robot base FK bone axes; no pad normals'}
    # Re-seed AFTER model construction: all ablations see identical samples/noise.
    torch.manual_seed(experiment['seed'] + 1)
    rng = torch.Generator().manual_seed(experiment['seed'] + 2)
    with torch.inference_mode():
        robot_cloud = exact(torch.rand((256, len(config['joint_order'])), generator=rng) * 2 - 1)['tips']
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
              (args.data, weights_path, fk_path, args.config, Path('geort/coordination.py'),
               Path('geort/coordination_loss.py'), Path('geort/train_coordination.py'))}
    metadata = {'experiment': experiment, 'parent': str(parent_path), 'parent_weights': args.weights,
                'hashes': hashes, 'data': str(args.data.resolve()),
                'split_note': 'whole sequences' if 'sequence_id' in recording else 'single legacy recording with guard gaps; no independent sessions',
                'sampling': 'same-frame uniform sampling shared across B0-B5; original trainer voxel-resampled coverage remains separate',
                'temporal_training': 'disabled: this experiment does not assume timestamps',
                'collision_training': 'original baseline defaults to 0; unchanged original optional collision trainer is available',
                'normalization': 'legacy geometric weights unchanged; new task tolerances explicit',
                'quality_status': 'smoke/development experiment; no demonstrated quality improvement',
                'wuji_baseline_cache_sha256': wuji_cache['cache_sha256'] if wuji_cache else None,
                'posture_policy': args.posture_policy, 'target_variant': args.target_variant,
                'freeze_base': args.freeze_base, 'head_lr': args.head_lr,
                'init_base': str(init_path) if init_path else None,
                'init_base_best_sha256': hashlib.sha256((init_path / 'best.pth').read_bytes()).hexdigest() if init_path else None,
                'warmup_steps': args.warmup_steps,
                'warmup_targets_sha256': hashlib.sha256(args.warmup_targets.read_bytes()).hexdigest() if args.warmup_targets else None,
                'baseline_commit': '258e3713ffecadd82da6869eacba3e1b792280d8'}
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    (output / 'experiment.json').write_text(json.dumps(metadata, indent=2))
    torch.save(parent.state_dict(), output / 'baseline.pth')
    (output / 'baseline_config.json').write_text(json.dumps(parent_config, indent=2))
    print(json.dumps({'enabled': experiment, 'calibration': config['calibration'], 'temporal_training': 'disabled',
                      'relation_replaces_pinch': options['relation'] > 0}), flush=True)
    def make_optimizer():
        base = [p for p in model.base.parameters() if p.requires_grad]
        heads = [p for name,p in model.named_parameters() if not name.startswith('base.') and p.requires_grad]
        groups = []
        if base: groups.append({'params': base, 'lr': experiment['lr']})
        if heads: groups.append({'params': heads, 'lr': args.head_lr or experiment['lr']})
        return torch.optim.Adam(groups)
    optimizer = make_optimizer()
    logs, best = [], float('inf')
    val_ids = split['validation'][np.linspace(0, len(split['validation']) - 1, len(split['validation']), dtype=int)]
    validation = batch(recording, val_ids)
    def objective(sample, posture_mask=None):
        loss, normalized, base_logs = baseline_objective(model, sample, neural_fk, robot_cloud, info['human_id'], experiment['baseline'])
        with torch.no_grad():
            reference = exact(parent(sample['keypoints'][:, info['human_id']])) if options.get('anchor', 0) or options['posture_reference_mcp'] else None
        task, task_logs, _ = task_objective(normalized, sample, exact, config, options,
            reference['tips'] if reference is not None else None,
            reference_q=reference['q'] if reference is not None else None, posture_mask=posture_mask)
        return loss + task, {'baseline': base_logs, 'task': task_logs}
    for step in range(experiment['steps']):
        indices = torch.randint(len(split['train']), (experiment['batch_size'],), generator=rng).numpy()
        sample = batch(recording, split['train'][indices])
        if step == args.warmup_steps and args.warmup_steps:
            optimizer = make_optimizer()
        optimizer.zero_grad()
        if step < args.warmup_steps:
            total = (model(sample) - teacher[indices]).square().mean()
            records = {'phase': 'offline_robot_target_warmup', 'joint_normalized_mse': float(total.detach())}
        else:
            mask = teacher_posture_mask[indices] if args.posture_policy == 'feasible' else None
            total, records = objective(sample, mask)
            records['phase'] = 'geometric_finetuning'
        if not torch.isfinite(total):
            raise RuntimeError('Nonfinite training loss')
        total.backward()
        parameters = [p for p in model.parameters() if p.grad is not None]
        if not all(torch.isfinite(p.grad).all() for p in parameters):
            raise RuntimeError('Nonfinite gradients')
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 10.)
        optimizer.step()
        row = {'step': step + 1, 'total': float(total.detach()), 'grad_norm': float(grad_norm), **records}
        if (step + 1) % max(1, min(100, max(1, experiment['steps'] // 4))) == 0 or step + 1 == experiment['steps']:
            with torch.random.fork_rng(), torch.inference_mode():
                torch.manual_seed(901)
                val, val_records = objective(validation)
            row['validation_total'] = float(val)
            row['validation_losses'] = val_records
            if args.save_every and (step + 1) % args.save_every == 0:
                torch.save(model.state_dict(), output / f'step_{step + 1}.pth')
            if step >= args.warmup_steps and float(val) < best:
                best = float(val)
                torch.save(model.state_dict(), output / 'best.pth')
                (output / 'selection.json').write_text(json.dumps({'step': step + 1, 'validation_total': best}))
            print(json.dumps(row), flush=True)
        logs.append(row)
    torch.save(model.state_dict(), output / 'last.pth')
    (output / 'training.json').write_text(json.dumps(logs, indent=2))
    model.load_state_dict(torch.load(output / 'best.pth', map_location='cpu', weights_only=True))
    report, arrays = evaluate(model, parent, recording, split[args.evaluation_split], exact, neural_fk, config, options)
    report['external_baseline_status'] = 'missing; future acceptance requires Wuji comparison'
    if wuji_cache is not None:
        from geort.wuji_baseline import comparison_report
        ids = split[args.evaluation_split]
        with torch.inference_mode():
            parent_output = exact(parent(batch(recording, ids)['keypoints'][:, info['human_id']]))
        report['wuji_comparison'] = comparison_report(wuji_cache, ids, exact, arrays['qpos'], arrays['tips'],
            parent_output['q'].numpy(), parent_output['tips'].numpy(), recording['keypoints'][ids])
        report['external_baseline_status'] = 'included'
    # Verify the actual external inference entry point, not just state_dict load.
    exported = GeoRTRetargetingModel(output / 'best.pth', output / 'config.json', device='cpu')
    index = split[args.evaluation_split][0]
    single = exported.forward(recording['keypoints'][index],
                              valid=recording['valid'][index] if 'valid' in recording else None,
                              confidence=recording['confidence'][index] if 'confidence' in recording else None)
    report['evaluation_split'] = args.evaluation_split
    report['export_reload_max_error_rad'] = float(abs(single - arrays['qpos'][0]).max())
    if report['export_reload_max_error_rad'] > 1e-3:
        raise RuntimeError('Export disagrees with batched inference')
    if args.collision:
        from geort.env.hand import HandKinematicModel
        hand = HandKinematicModel.build_from_config(parent_config)
        depth = hand.self_collision_depth(arrays['qpos']) * 1000
        report['collision'] = {'penetration_over_2mm_fraction': float(np.mean(depth > 2)),
                               'max_penetration_mm': float(depth.max()), 'source': 'SAPIEN exact geometry, no runtime correction'}
    np.savez_compressed(output / 'evaluation.npz', **arrays)
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Saved checkpoint, losses and diagnostics: {output}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('geort/experiments/B0.json'))
    p.add_argument('--parent', required=True)
    p.add_argument('--weights', choices=['best', 'last'], default='last')
    p.add_argument('--data', type=Path, default=Path('data/human_alex.npy'))
    p.add_argument('--fk-checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--warmup-targets', type=Path, help='Offline robot IK targets with matching train-only provenance')
    p.add_argument('--target-variant', choices=['consistent', 'independent'], default='consistent')
    p.add_argument('--posture-policy', choices=['uniform', 'feasible'], default='uniform')
    p.add_argument('--warmup-steps', type=int, default=0, help='Steps within the same total budget used for branch warmup')
    p.add_argument('--local-features', choices=['skeleton', 'tip_control'], help='Versioned equal-architecture input ablation; tip_control zeros only head input axes/masks')
    p.add_argument('--init-base', help='Initialize base only from a compatible coordination best checkpoint')
    p.add_argument('--freeze-base', action='store_true', help='Train only optional heads, preserving base parameters and BN buffers')
    p.add_argument('--head-lr', type=float, help='Explicit learning rate for optional local/context heads')
    p.add_argument('--evaluation-split', choices=['validation', 'test'], default='test')
    p.add_argument('--save-every', type=int, default=0, help='Optional step snapshots for validation-only selection')
    p.add_argument('--wuji-cache', type=Path, help='Confirmed Wuji reference; required for future experiment acceptance')
    p.add_argument('--collision', action='store_true', help='Also evaluate self-contact on final test targets')
    run(p.parse_args())


if __name__ == '__main__':
    main()
