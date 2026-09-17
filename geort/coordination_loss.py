"""Explicit baseline geometric objective plus masked operational objectives."""
import math
import torch
from torch.nn import functional as F
from geort.coordination import skeleton_features
from geort.kinematics import FingerKinematics, TorchFingerKinematics
from geort.loss import chamfer_distance, pinch_distance_loss
from geort.utils.config_utils import parse_config_keypoint_info


class ExactHand:
    def __init__(self, config):
        self.config = config
        self.chains = [FingerKinematics(config, f['name']) for f in config['fingertip_link']]
        self.fks = [TorchFingerKinematics(chain) for chain in self.chains]
        self.lower = torch.tensor(config['joint']['lower'], dtype=torch.float32)
        self.upper = torch.tensor(config['joint']['upper'], dtype=torch.float32)
        expected_lower = self.lower.clone(); expected_upper = self.upper.clone()
        for chain in self.chains:
            expected_lower[chain.indices] = torch.tensor(chain.lower, dtype=torch.float32)
            expected_upper[chain.indices] = torch.tensor(chain.upper, dtype=torch.float32)
        if not torch.equal(expected_lower, self.lower) or not torch.equal(expected_upper, self.upper):
            raise ValueError('Checkpoint limits do not match physical URDF limits')

    def qpos(self, normalized):
        return self.lower + (normalized + 1) * .5 * (self.upper - self.lower)

    def __call__(self, normalized):
        q = self.qpos(normalized)
        results = [fk(q[:, chain.indices], return_bones=True) for fk, chain in zip(self.fks, self.chains)]
        return {'q': q, 'tips': torch.stack([r[0] for r in results], 1),
                'axes': torch.stack([r[2] for r in results], 1),
                'lengths': torch.stack([r[3] for r in results], 1)}


def masked_term(raw, mask, scale, weight):
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(weight) or weight < 0:
        raise ValueError('Loss scale must be positive and weight nonnegative')
    if mask.shape != raw.shape or mask.dtype != torch.bool:
        raise ValueError('Loss mask must match scalar errors')
    if not torch.isfinite(raw[mask]).all():
        raise ValueError('Nonfinite valid loss')
    clean = torch.where(mask, raw, torch.zeros_like(raw))
    count = mask.sum()
    mean = clean.sum() / count.clamp_min(1)
    normalized = mean / scale
    weighted = weight * normalized
    return weighted, {'raw': float(mean.detach()), 'normalized': float(normalized.detach()),
                      'weighted': float(weighted.detach()), 'valid_count': int(count),
                      'total_count': mask.numel(), 'valid_fraction': float(count) / max(mask.numel(), 1)}


def shift_chains(points, offsets, human_ids):
    """Translate each entire finger chain: perturb tips without bending bones."""
    out = points.clone()
    for i, end in enumerate(human_ids):
        out[:, end - 3:end + 1] = out[:, end - 3:end + 1] + offsets[:, i, None]
    return out


def baseline_objective(model, sample, neural_fk, robot_cloud, human_ids, weights):
    points = sample['keypoints'] if isinstance(sample, dict) else sample
    tips = points[:, human_ids]
    normalized = model(sample)
    embedded = neural_fk(normalized)
    def shifted(offset):
        changed = shift_chains(points, offset, human_ids)
        if isinstance(sample, dict):
            changed = {**sample, 'keypoints': changed}
        return neural_fk(model(changed))
    offset = torch.randn_like(tips) * .002
    plus, minus = shifted(offset), shifted(-offset)
    delta = torch.randn_like(tips) * .002
    loss, records = baseline_from_outputs(tips, embedded, plus, minus, shifted(delta), delta, robot_cloud, weights)
    return loss, normalized, records


def baseline_from_outputs(tips, embedded, plus, minus, shifted_delta, delta, robot_cloud, weights):
    """Shared geometric formulas for neural and externally recomputed outputs."""
    flatness = (plus + minus - 2 * embedded).square().sum((1, 2)).mean()
    delta_robot = shifted_delta - embedded
    direction = -(F.normalize(delta, dim=-1, eps=1e-5) * F.normalize(delta_robot, dim=-1, eps=1e-5)).sum(-1).sum(-1).mean()
    cover = sum(chamfer_distance(embedded[:, i][None], robot_cloud[:, i][None]) for i in range(tips.shape[1]))
    pinch = pinch_distance_loss(tips, embedded, paper=True)
    raw = {'coverage': cover, 'response_direction': direction, 'flatness': flatness, 'pinch': pinch}
    loss = sum(weights[name] * value for name, value in raw.items())
    records = {name: {'raw': float(value.detach()), 'normalized': float(value.detach()),
                     'weighted': float((weights[name] * value).detach()), 'valid_count': len(tips),
                     'total_count': len(tips), 'valid_fraction': 1.} for name, value in raw.items()}
    return loss, records


def task_objective(normalized, sample, exact, config, options, reference_tips=None, reference_q=None, posture_mask=None):
    info = parse_config_keypoint_info(config)
    human = skeleton_features(sample, info['human_id'])
    robot = exact(normalized)
    records, loss = {}, normalized.sum() * 0
    def add(name, raw, mask, scale):
        nonlocal loss
        term, record = masked_term(raw, mask, scale, options[name])
        loss = loss + term
        records[name] = record
    valid = human['bone_valid'][:, :, -1]
    raw = 1 - (robot['axes'][:, :, -1] * human['axes'][:, :, -1]).sum(-1).clamp(-1, 1)
    add('axis', raw, valid, 1 - math.cos(math.radians(options['axis_tolerance_deg'])))
    # Rotation-invariant relative bone alignment; not human joint-angle labels.
    human_cos = (human['axes'][:, :, 1:] * human['axes'][:, :, :-1]).sum(-1)
    robot_cos = (robot['axes'][:, :, 1:] * robot['axes'][:, :, :-1]).sum(-1)
    shape_valid = human['bone_valid'][:, :, 1:] & human['bone_valid'][:, :, :-1]
    thumb = info['human_id'].index(4)
    shape_valid[:, thumb] = False  # thumb linkage is not homologous
    add('shape', (human_cos - robot_cos).square(), shape_valid, options['shape_tolerance_cos'] ** 2)
    nonthumb = [ids for i, ids in enumerate(info['joint']) if i != thumb]
    bending = torch.stack([robot['q'][:, [ids[0], ids[2], ids[3]]] for ids in nonthumb], dim=1)
    floor = bending.new_tensor([-math.radians(15), -math.radians(5), -math.radians(5)])
    if options.get('posture_reference_mcp', False):
        if reference_q is None or reference_q.shape != robot['q'].shape or not torch.isfinite(reference_q).all():
            raise ValueError('Reference MCP preference requires finite paired parent joints')
        floor = floor.expand_as(bending).clone()
        floor[:, :, 0] = torch.minimum(floor[:, :, 0], reference_q[:, [ids[0] for ids in nonthumb]])
    raw = F.relu(floor - bending).square()
    mask = torch.ones_like(raw, dtype=torch.bool)
    if posture_mask is not None:
        if posture_mask.dtype != torch.bool or posture_mask.shape != raw.shape[:2]:
            raise ValueError('posture_mask must be bool (B, nonthumb fingers)')
        mask = posture_mask[:, :, None].expand_as(raw)
    add('posture', raw, mask, math.radians(options['posture_tolerance_deg']) ** 2)
    others = [i for i in range(len(info['human_id'])) if i != thumb]
    hv = human['tips'][:, others] - human['tips'][:, thumb, None]
    rv = robot['tips'][:, others] - robot['tips'][:, thumb, None]
    scales = rv.new_tensor(options['relation_scale'])
    # No contact geometry: exclude near-pinch gestures rather than invent site contact.
    relation_valid = (hv.norm(dim=-1) >= options['relation_min_opening_m']) & torch.tensor(options.get('relation_valid', [True] * len(others)), dtype=torch.bool, device=rv.device)[None]
    add('relation', (rv - scales[None, :, None] * hv).square().sum(-1), relation_valid, options['relation_tolerance_m'] ** 2)
    if options.get('anchor', 0) > 0:
        if reference_tips is None or reference_tips.shape != robot['tips'].shape:
            raise ValueError('Exact-tip anchor requires paired baseline FK targets')
        mask = torch.isfinite(reference_tips).all(-1)
        safe_reference = torch.where(mask[..., None], reference_tips, robot['tips'].detach())
        raw = (robot['tips'] - safe_reference).square().sum(-1)
        add('anchor', raw, mask, options['anchor_tolerance_m'] ** 2)
    return loss, records, robot
