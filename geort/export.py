# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pathlib import Path
from geort.formatter import HandFormatter
from geort.model import IKModel
from geort.utils.path import to_package_root, get_checkpoint_root
from geort.utils.config_utils import load_json, parse_config_keypoint_info, parse_config_joint_limit


class GeoRTRetargetingModel:
    '''
        Used by external programs.
    '''
    def __init__(self, model_path, config_path):
        config = load_json(config_path)
        self.config = config
        keypoint_info = parse_config_keypoint_info(config)
        joint_lower_limit, joint_upper_limit = parse_config_joint_limit(config)
        print(keypoint_info["joint"])
        self.human_ids = keypoint_info["human_id"]
        self.model = IKModel(keypoint_joints=keypoint_info["joint"]).cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.qpos_normalizer = HandFormatter(joint_lower_limit, joint_upper_limit) # GeoRT will do normalization.

    def forward(self, keypoints):
        # keypoints: [N, 3]
        keypoints = keypoints[self.human_ids] # extract.
        joint_normalized = self.model.forward(torch.from_numpy(keypoints).unsqueeze(0).reshape(1, -1, 3).float().cuda())
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
