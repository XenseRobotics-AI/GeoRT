# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

def get_finger_fk(n_joint=4, hidden=128):
    return nn.Sequential(
        nn.Linear(n_joint, hidden), 
        nn.LeakyReLU(), 
        nn.BatchNorm1d(hidden),
        nn.Linear(hidden, hidden), 
        nn.LeakyReLU(), 
        nn.BatchNorm1d(hidden),
        nn.Linear(hidden, 3)
    ) 

def get_finger_ik(n_joint=4, hidden=128):
    return nn.Sequential(
        nn.Linear(3, hidden), 
        nn.LeakyReLU(), 
        nn.BatchNorm1d(hidden),
        nn.Linear(hidden, hidden), 
        nn.LeakyReLU(), 
        nn.BatchNorm1d(hidden),
        nn.Linear(hidden, n_joint),
        nn.Tanh()   # Normalize.
    ) 

class FKModel(nn.Module):
    def __init__(self, keypoint_joints):
        # keypoint_joints: a list of list.
        # keypoint[i] is the indices of joints that drive the i-th keypoint.
        # Example: For allegro, [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]

        super().__init__()
        num_fingers = len(keypoint_joints)
        
        self.nets = []
        self.n_total_joint = 0

        for joint in keypoint_joints:
            net = get_finger_fk(n_joint=len(joint))
            self.nets.append(net)
            self.n_total_joint += len(joint)

        self.nets = nn.ModuleList(self.nets)

        self.keypoint_joints = keypoint_joints

    def forward(self, joint):
        # x: [B, DOF], joint values. normalized to [-1, 1]. 
        # out:   [B, N, 3], sequence of keypoint.
        keypoints = []
        for i, net in enumerate(self.nets):
            joint_ids = self.keypoint_joints[i]
            keypoint = net(joint[:, joint_ids])
            keypoints.append(keypoint)

        return torch.stack(keypoints, dim=1)

    
class IKModel(nn.Module):
    def __init__(self, keypoint_joints):
        # keypoint_joints: a list of list.
        # keypoint[i] is the indices of joints that drive the i-th keypoint.
        # Example: [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]

        super().__init__()
        self.n_total_joint = 0
        self.nets = []

        for joint in keypoint_joints:
            net = get_finger_ik(n_joint=len(joint))
            self.nets.append(net)
            self.n_total_joint += len(joint)

        self.nets = nn.ModuleList(self.nets)
        self.keypoint_joints = keypoint_joints 

    def forward(self, x):
        # x:   [B, N, 3], sequence of keypoint.
        # out: [B, DOF], joint values. normalized to [-1, 1]. 
        batch_size = x.size(0)
        n_points = x.size(1)
        out = torch.zeros((batch_size, self.n_total_joint)).to(x.device)
        for i in range(n_points):
            joint = self.nets[i](x[:, i])
            out[:, self.keypoint_joints[i]] = joint 
        return out 


class CollisionModel(nn.Sequential):
    """Predict self-collision logits; sigmoid gives the collision probability."""
    def __init__(self, n_joint):
        super().__init__(
            nn.Linear(n_joint, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 1))


class PostureIKModel(IKModel):
    """Wuji v1: retain tip-only mapping for four fingers, add little-finger bones."""
    def __init__(self, keypoint_joints, human_ids):
        super().__init__(keypoint_joints)
        if len(keypoint_joints) != 5 or human_ids != [4, 8, 12, 16, 20]:
            raise ValueError('Wuji posture v1 requires five fingers in thumb-to-little order')
        self.human_ids = human_ids
        self.nets[4][0] = nn.Linear(12, 128)

    def forward(self, points):
        if points.ndim != 3 or points.shape[1:] != (21, 3):
            raise ValueError('Posture model requires the full (B, 21, 3) skeleton')
        values = []
        for i, net in enumerate(self.nets):
            features = points[:, self.human_ids[i]]
            if i == 4:
                bones = points[:, 18:21] - points[:, 17:20]
                bones = torch.nn.functional.normalize(bones, dim=-1, eps=1e-8)
                features = torch.cat([features, .05 * bones.flatten(1)], dim=1)
            values.append(net(features))
        result = points.new_zeros((len(points), self.n_total_joint))
        for ids, value in zip(self.keypoint_joints, values):
            result[:, ids] = value
        return result


def build_ik_model(config):
    from geort.utils.config_utils import parse_config_keypoint_info
    info = parse_config_keypoint_info(config)
    kind = config.get('model_type', 'fingertip_v1')
    if kind == 'manus_v1':
        from geort.manus_model import ManusIKModel
        return ManusIKModel(config)
    if kind == 'coordination_v1':
        from geort.coordination import CoordinationIKModel
        return CoordinationIKModel(config)
    if kind == 'fingertip_v1':
        return IKModel(info['joint'])
    if kind == 'wuji_posture_v1' and config['name'] == 'wuji_hand2_beta1_right':
        return PostureIKModel(info['joint'], info['human_id'])
    raise ValueError(f'Unsupported model type: {kind}')


def ik_input(config, human):
    from geort.utils.config_utils import parse_config_keypoint_info
    if config.get('model_type', 'fingertip_v1') == 'fingertip_v1':
        return human[:, parse_config_keypoint_info(config)['human_id']]
    if config.get('model_type') in ('wuji_posture_v1', 'coordination_v1', 'manus_v1'):
        return human
    raise ValueError('Unsupported model input format')
