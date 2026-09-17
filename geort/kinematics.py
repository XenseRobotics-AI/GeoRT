"""Exact serial-chain kinematics shared by training and diagnostic correction."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

class FingerKinematics:
    """Exact URDF serial-chain FK and geometric Jacobians (revolute/fixed only)."""

    def __init__(self, config, finger='little'):
        info = next(item for item in config['fingertip_link'] if item['name'] == finger)
        self.indices = np.array([config['joint_order'].index(n) for n in info['joint']])
        self.names = info['joint']
        self.offset = np.asarray(info['center_offset'], dtype=float)
        path = Path(config['urdf_path'])
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        joints = ET.parse(path).getroot().findall('joint')
        by_child = {j.find('child').get('link'): j for j in joints}
        chain, link = [], info['link']
        while link != config['base_link']:
            joint = by_child[link]
            chain.append(joint)
            link = joint.find('parent').get('link')
        self.chain = []
        limits = {}
        for joint in reversed(chain):
            kind, name = joint.get('type'), joint.get('name')
            if kind not in ('revolute', 'fixed') or joint.find('mimic') is not None:
                raise ValueError('Only independent revolute/fixed chains are supported')
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
            transform[:3, 3] = xyz
            idx, axis = None, None
            if kind == 'revolute':
                if name not in self.names:
                    raise ValueError(f'Unconfigured moving joint in chain: {name}')
                idx = self.names.index(name)
                axis_node = joint.find('axis')
                axis = np.fromstring(axis_node.get('xyz') if axis_node is not None else '1 0 0', sep=' ')
                axis = axis / np.linalg.norm(axis)
                limit = joint.find('limit')
                limits[name] = (float(limit.get('lower')), float(limit.get('upper')))
            self.chain.append((transform, idx, axis))
        if set(limits) != set(self.names):
            raise ValueError('Finger joints do not match URDF chain')
        self.lower, self.upper = np.array([limits[n] for n in self.names]).T

    def forward(self, q):
        """Return tip, distal unit direction, and their 3 x DOF Jacobians."""
        q = np.asarray(q, dtype=float)
        if q.shape != self.lower.shape or not np.isfinite(q).all():
            raise ValueError('Expected finite finger joint angles')
        transform = np.eye(4)
        origins, axes = np.empty((len(q), 3)), np.empty((len(q), 3))
        distal_origin = None
        for fixed, idx, axis in self.chain:
            transform = transform @ fixed
            if idx is not None:
                origins[idx] = transform[:3, 3].copy()
                axes[idx] = transform[:3, :3] @ axis
                distal_origin = transform[:3, 3].copy()
                transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(axis * q[idx]).as_matrix()
        tip = transform[:3, 3] + transform[:3, :3] @ self.offset
        direction = tip - distal_origin
        direction /= np.linalg.norm(direction)
        jac = np.cross(axes, tip - origins).T
        direction_jac = np.cross(axes, direction).T
        return tip, direction, jac, direction_jac


class TorchFingerKinematics:
    """Batched differentiable FK; no learned approximation or simulator state."""

    def __init__(self, kinematics, device='cpu', dtype=None):
        import torch
        dtype = dtype or torch.float32
        self.chain = []
        for fixed, idx, axis in kinematics.chain:
            skew = np.zeros((3, 3)) if axis is None else np.array(
                [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            self.chain.append((torch.tensor(fixed, device=device, dtype=dtype), idx,
                               torch.tensor(skew, device=device, dtype=dtype)))
        self.offset = torch.tensor(kinematics.offset, device=device, dtype=dtype)

    def __call__(self, q, return_bones=False):
        import torch
        rotation = torch.eye(3, device=q.device, dtype=q.dtype).expand(len(q), 3, 3)
        translation = q.new_zeros((len(q), 3))
        eye = torch.eye(3, device=q.device, dtype=q.dtype)
        distal_origin = None
        origins = {}
        for fixed, idx, skew in self.chain:
            translation = translation + rotation @ fixed[:3, 3]
            rotation = rotation @ fixed[:3, :3]
            if idx is not None:
                distal_origin = translation
                origins[idx] = translation
                angle = q[:, idx, None, None]
                rotation = rotation @ (eye + angle.sin() * skew + (1 - angle.cos()) * (skew @ skew))
        tip = translation + rotation @ self.offset
        direction = torch.nn.functional.normalize(tip - distal_origin, dim=-1)
        if return_bones:
            if len(origins) != 4:
                raise ValueError('Bone correspondence requires an explicit four-joint chain')
            nodes = torch.stack([origins[1], origins[2], origins[3], tip], dim=1)
            bones = nodes[:, 1:] - nodes[:, :-1]
            lengths = bones.norm(dim=-1)
            if torch.any(lengths < 1e-7):
                raise ValueError('Degenerate robot bone axis')
            return tip, direction, torch.nn.functional.normalize(bones, dim=-1), lengths
        return tip, direction
