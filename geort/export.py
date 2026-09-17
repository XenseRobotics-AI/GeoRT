# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pathlib import Path
from geort.formatter import HandFormatter
from geort.model import build_ik_model, ik_input
from geort.utils.path import to_package_root, get_checkpoint_root
from geort.utils.config_utils import load_json, parse_config_keypoint_info, parse_config_joint_limit


class GeoRTRetargetingModel:
    '''
        Used by external programs.
    '''
    def __init__(self, model_path, config_path, device=None):
        config = load_json(config_path)
        self.config = config
        keypoint_info = parse_config_keypoint_info(config)
        joint_lower_limit, joint_upper_limit = parse_config_joint_limit(config)
        print(keypoint_info["joint"])
        self.human_ids = keypoint_info["human_id"]
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = build_ik_model(config).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.qpos_normalizer = HandFormatter(joint_lower_limit, joint_upper_limit) # GeoRT will do normalization.

    def forward(self, keypoints, valid=None, confidence=None):
        # keypoints: [N, 3]
        points = torch.as_tensor(keypoints, dtype=torch.float32, device=self.device).unsqueeze(0)
        enhanced = self.config.get('model_type') in ('coordination_v1', 'manus_v1')
        if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] <= max(self.human_ids) or (not enhanced and not torch.isfinite(points).all()):
            raise ValueError('Expected finite (N, 3) keypoints covering configured human IDs')
        if not enhanced and (valid is not None or confidence is not None):
            raise ValueError('Masks require a coordination_v1 checkpoint')
        self.last_input_validity = None
        inputs = ik_input(self.config, points)
        if enhanced:
            inputs = {'keypoints': points}
            if valid is not None:
                inputs['valid'] = torch.as_tensor(valid, device=self.device).unsqueeze(0)
            if confidence is not None:
                inputs['confidence'] = torch.as_tensor(confidence, dtype=torch.float32, device=self.device).unsqueeze(0)
        if enhanced:
            from geort.coordination import skeleton_features
            threshold = self.config.get('features', {}).get('confidence_threshold', .5)
            features = skeleton_features(inputs, self.human_ids, threshold)
            mask = features['bone_valid']
            self.last_input_validity = {'bone_valid_count': int(mask.sum()), 'bone_total_count': mask.numel(),
                                        'bone_valid_fraction': float(mask.float().mean())}
        with torch.inference_mode():
            joint_normalized = self.model(inputs)
        joint_raw = self.qpos_normalizer.unnormalize(joint_normalized.detach().cpu().numpy())
        return joint_raw[0]


def resolve_checkpoint(tag):
    root = Path(get_checkpoint_root())
    exact = root / tag
    if tag and exact.is_dir() and (exact / 'config.json').is_file():
        return exact
    matches = sorted(p for p in root.iterdir()
                     if p.is_dir() and tag in p.name and (p / 'config.json').is_file())
    if not matches:
        raise FileNotFoundError(f'No checkpoint matches {tag!r} in {root}')
    if len(matches) != 1:
        raise ValueError(f'Ambiguous checkpoint {tag!r}; use a full name: '
                         + ', '.join(p.name for p in matches))
    return matches[0]


def load_model(tag='', epoch=0, weights='last'):
    '''
        Loading API.
    '''
    checkpoint_root = resolve_checkpoint(tag)
    if weights not in ('last', 'best'):
        raise ValueError('weights must be last or best')
    if epoch > 0 and weights != 'last':
        raise ValueError('Choose either a numbered epoch or best weights')
    if epoch > 0:
        model_path = checkpoint_root / f"epoch_{epoch}.pth"
    else:
        model_path = checkpoint_root / f"{weights}.pth"
    if not model_path.is_file():
        raise FileNotFoundError(f'Checkpoint weights not found: {model_path}')
    
    config_path = checkpoint_root / "config.json"
    print(f'Loading checkpoint: {model_path.resolve()}')
    return GeoRTRetargetingModel(model_path=model_path, config_path=config_path)

if __name__ == '__main__':
    # load the model in one line.
    load_model(tag="allegro_last")
