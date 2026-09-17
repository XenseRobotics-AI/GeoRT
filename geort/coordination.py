"""Versioned, optional skeleton/context mapping; original IKModel is unchanged."""
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from geort.model import IKModel
from geort.utils.config_utils import parse_config_keypoint_info


def robot_signature(config):
    keys = ('name', 'base_link', 'joint_order', 'fingertip_link', 'joint')
    digest = hashlib.sha256(json.dumps({k: config[k] for k in keys}, sort_keys=True).encode())
    path = Path(config['urdf_path'])
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    digest.update(path.read_bytes())
    return digest.hexdigest()


def feature_metadata(config, mode='tip_only', context=False, local_features=None):
    if local_features is not None and (mode != 'skeleton' or local_features not in ('skeleton', 'tip_control')):
        raise ValueError('Local feature ablation requires skeleton architecture and an explicit valid feature policy')
    result = {'version': 1, 'mode': mode, 'context': context,
            'frame': 'geort_canonical', 'length_unit': 'm', 'angle_unit': 'rad',
            'handedness': 'left' if config['name'].endswith('_left') else 'right',
            'human_ids': parse_config_keypoint_info(config)['human_id'],
            'bone_order': 'base_to_proximal,proximal_to_distal,distal_to_tip',
            'point_layout': 'wrist0_thumb1:5_index5:9_middle9:13_ring13:17_little17:21',
            'tip_scale_m': .1, 'residual_logit_scale': .1, 'confidence_threshold': .5,
            'robot_signature': robot_signature(config),
            'orientation': 'bone_axis_only_no_pad_normal_no_quaternion'}
    if local_features is not None:
        result.update(version=2, local_features=local_features)
    return result


def adapt_points(points, *, unit='m', rotation=None, translation_m=None):
    """Explicit rigid calibration. No automatic hand mirroring or scale inference."""
    if unit not in ('m', 'mm'):
        raise ValueError('Length unit must be m or mm')
    points = points * (1. if unit == 'm' else .001)
    if rotation is not None:
        rotation = torch.as_tensor(rotation, dtype=points.dtype, device=points.device)
        if rotation.shape != (3, 3) or not torch.isfinite(rotation).all() or not torch.allclose(rotation.T @ rotation, torch.eye(3, device=points.device, dtype=points.dtype), atol=1e-5) or not torch.isclose(torch.det(rotation), points.new_tensor(1.), atol=1e-5):
            raise ValueError('Calibration must be a proper rotation, not an implicit reflection')
        points = points @ rotation.T
    if translation_m is not None:
        t = torch.as_tensor(translation_m, dtype=points.dtype, device=points.device)
        if t.shape != (3,) or not torch.isfinite(t).all():
            raise ValueError('Translation must be a finite 3-vector in meters')
        points = points + t
    return points


def skeleton_features(sample, human_ids, confidence_threshold=.5):
    """Return actual bone axes and validity. Missing vectors are never supervised."""
    if isinstance(sample, dict):
        unknown = set(sample) - {'keypoints', 'valid', 'confidence'}
        if unknown:
            raise ValueError(f'Unsupported sample fields: {sorted(unknown)}')
        points = sample['keypoints']
        validity = sample.get('valid')
        confidence = sample.get('confidence')
    else:
        points, validity, confidence = sample, None, None
    if points.ndim != 3 or points.shape[1:] != (21, 3) or not len(points):
        raise ValueError('Expected nonempty (B,21,3) keypoints')
    finite = torch.isfinite(points).all(-1)
    valid = finite.clone()
    if validity is not None:
        if validity.shape != points.shape[:2] or validity.dtype != torch.bool:
            raise ValueError('valid must be bool (B,21)')
        valid = valid & validity
    if confidence is not None:
        if confidence.shape != points.shape[:2]:
            raise ValueError('confidence must be (B,21)')
        valid = valid & torch.isfinite(confidence) & (confidence >= confidence_threshold) & (confidence <= 1)
    if not valid[:, human_ids].all():
        raise ValueError('Required fingertip positions are invalid; refusing unsafe inference')
    clean = torch.where(finite[..., None], points, torch.zeros_like(points))
    chains = torch.stack([clean[:, i - 3:i + 1] for i in human_ids], dim=1)
    chain_valid = torch.stack([valid[:, i - 3:i + 1] for i in human_ids], dim=1)
    bones = chains[:, :, 1:] - chains[:, :, :-1]
    lengths = bones.norm(dim=-1)
    bone_valid = chain_valid[:, :, 1:] & chain_valid[:, :, :-1] & (lengths > 1e-7)
    axes = F.normalize(bones, dim=-1, eps=1e-7)
    axes = torch.where(bone_valid[..., None], axes, torch.zeros_like(axes))
    return {'points': clean, 'tips': clean[:, human_ids], 'axes': axes,
            'bone_valid': bone_valid, 'lengths': lengths}


class CoordinationIKModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        meta = config['features']
        if not isinstance(meta.get('context'), bool):
            raise ValueError('context must be a boolean')
        if meta != feature_metadata(config, meta.get('mode'), meta.get('context'), meta.get('local_features')):
            raise ValueError('Incompatible feature schema, calibration, or robot signature')
        if meta['mode'] not in ('tip_only', 'skeleton'):
            raise ValueError('pose mode unavailable: full calibrated orientations are missing')
        if meta['mode'] == 'skeleton' and config['name'] != 'wuji_hand2_beta1_right':
            raise ValueError('Skeleton calibration currently defined only for Wuji right; use legacy/tip_only for other robots')
        self.meta = meta
        info = parse_config_keypoint_info(config)
        self.human_ids, self.joints = info['human_id'], info['joint']
        self.base = IKModel(self.joints)
        self.local = nn.ModuleList()
        if meta['mode'] == 'skeleton':
            for joint in self.joints:
                self.local.append(self._head(15, len(joint)))  # tip3 + axes9 + validity3
        width = len(self.joints) * (15 if meta['mode'] == 'skeleton' else 3)
        self.context = self._head(width, len(config['joint_order'])) if meta['context'] else None

    @staticmethod
    def _head(inputs, outputs):
        head = nn.Sequential(nn.Linear(inputs, 64), nn.SiLU(), nn.Linear(64, outputs))
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    def forward(self, sample):
        features = skeleton_features(sample, self.human_ids, self.meta['confidence_threshold'])
        if self.local and not features['bone_valid'].any():
            raise ValueError('Skeleton mode requested but no valid bones are available')
        tips = features['tips']
        local = tips / self.meta['tip_scale_m']
        if self.local:
            extra = torch.cat((features['axes'].flatten(2), features['bone_valid'].to(tips.dtype)), dim=-1)
            if self.meta.get('local_features') == 'tip_control':
                # Deliberate information ablation, not missing-data supervision.
                extra = torch.zeros_like(extra)
            local = torch.cat((local, extra), dim=-1)
        context = self.context(local.flatten(1)) if self.context is not None else None
        result = tips.new_zeros((len(tips), self.base.n_total_joint))
        for i, (net, indices) in enumerate(zip(self.base.nets, self.joints)):
            # Reuse exact base logits; no inverse tanh/clamping of legacy outputs.
            logits = net[:-1](tips[:, i])
            delta = torch.zeros_like(logits)
            if self.local:
                delta = delta + self.local[i](local[:, i])
            if context is not None:
                delta = delta + context[:, indices]
            logits = logits + self.meta['residual_logit_scale'] * torch.tanh(delta)
            result[:, indices] = torch.tanh(logits)
        return result


def initialize_coordination(config, parent, mode, context, local_features=None):
    config = copy.deepcopy(config)
    config['model_type'] = 'coordination_v1'
    config['features'] = feature_metadata(config, mode, context, local_features)
    model = CoordinationIKModel(config).eval()
    model.base.load_state_dict(parent.state_dict(), strict=True)
    return config, model
