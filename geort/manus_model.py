"""Opt-in direct per-finger model for the independent Manus session experiment."""
import torch
from torch import nn

from geort.coordination import robot_signature, skeleton_features
from geort.utils.config_utils import parse_config_keypoint_info


class ManusIKModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        meta = config['manus_features']
        if (meta['version'] != 1 or meta['mode'] not in ('tip_control', 'skeleton')
                or meta['frame'] != 'geort_canonical' or meta['unit'] != 'm'
                or meta['robot_signature'] != robot_signature(config)):
            raise ValueError('Invalid Manus feature schema')
        mean = torch.tensor(meta['tip_mean'], dtype=torch.float32)
        scale = torch.tensor(meta['tip_scale'], dtype=torch.float32)
        if mean.shape != (5,3) or scale.shape != (5,3) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale < .01).any():
            raise ValueError('Invalid training-only feature normalization')
        info = parse_config_keypoint_info(config)
        if info['human_id'] != [4,8,12,16,20]:
            raise ValueError('Expected five MediaPipe fingertips')
        self.register_buffer('tip_mean', mean)
        self.register_buffer('tip_scale', scale)
        self.mode = meta['mode']
        self.human_ids, self.joints = info['human_id'], info['joint']
        # Identical architecture and parameter count in both information arms.
        self.nets = nn.ModuleList([nn.Sequential(nn.Linear(15,128), nn.SiLU(),
             nn.Linear(128,128), nn.SiLU(), nn.Linear(128,len(j)), nn.Tanh()) for j in self.joints])
        lower = torch.tensor(config['joint']['lower'])
        upper = torch.tensor(config['joint']['upper'])
        rest = (2 * (torch.zeros_like(lower) - lower) / (upper - lower) - 1).clamp(-.99,.99)
        with torch.no_grad():
            for net, ids in zip(self.nets, self.joints):
                net[4].weight.mul_(.05)
                net[4].bias.copy_(torch.atanh(rest[ids]))
        self.context_net = None
        if meta.get('context', False):
            self.context_net = nn.Sequential(nn.Linear(75,64), nn.SiLU(), nn.Linear(64,20))
            # Zero only the output layer: exact parent output, with a live gradient path.
            nn.init.zeros_(self.context_net[-1].weight)
            nn.init.zeros_(self.context_net[-1].bias)

    def forward(self, sample):
        features = skeleton_features(sample, self.human_ids)
        if self.mode == 'skeleton' and not features['bone_valid'].any():
            raise ValueError('Skeleton mode requires valid bones')
        tips = (features['tips'] - self.tip_mean) / self.tip_scale
        extra = torch.cat((features['axes'].flatten(2), features['bone_valid'].to(tips.dtype)), -1)
        if self.mode == 'tip_control':
            extra = torch.zeros_like(extra)  # intentional information ablation, not fake supervision
        features = torch.cat((tips, extra), -1)
        out = tips.new_zeros((len(tips), sum(map(len,self.joints))))
        for i, (net, ids) in enumerate(zip(self.nets, self.joints)):
            out[:,ids] = net(features[:,i])
        if self.context_net is not None:
            shift = torch.tanh(.25 * torch.tanh(self.context_net(features.flatten(1))))
            # tanh(atanh(out)+delta), algebraically stable even at a saturated out.
            out = (out + shift) / (1 + out * shift)
        return out
