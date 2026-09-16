"""Fine-tune Wuji little-finger posture with exact FK and full human bones.

Input is one contiguous, uniformly sampled recording. Validation is separated
by a frame gap; do not concatenate recordings before running this command.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from geort.export import resolve_checkpoint
from geort.kinematics import FingerKinematics, TorchFingerKinematics
from geort.model import build_ik_model, ik_input
from geort.utils.config_utils import parse_config_joint_limit


def initialize_model(config, parent):
    config = copy.deepcopy(config)
    config['model_type'] = 'wuji_posture_v1'
    model = build_ik_model(config)
    state = parent.state_dict()
    state['nets.4.0.weight'] = F.pad(state['nets.4.0.weight'], (0, 9))
    model.load_state_dict(state)
    for i, net in enumerate(model.nets):
        net.requires_grad_(i == 4)
    # Fixed BN statistics avoid train/inference mismatch on correlated frames.
    model.eval()
    return config, model


def posture_loss(q, tips, direction, target, human, posture_weight=10., temporal_weight=.1):
    tip = ((tips - target) / .005).square().sum(-1).mean()
    bend = (F.relu(-np.deg2rad(5) - q[:, 2:]) / .3).square().mean()
    mcp = (F.relu(-np.deg2rad(15) - q[:, 0]) / .3).square().mean()
    bone = human[:, 20] - human[:, 19]
    valid = bone.norm(dim=-1) > 1e-8
    alignment = (1 - (direction * F.normalize(bone, dim=-1)).sum(-1))[valid]
    alignment = alignment.mean() if valid.any() else direction.sum() * 0
    # Penalize abrupt movement, not a hard inference clamp or human angle copy.
    temporal = (torch.diff(q, dim=0) / .1).square().mean() if len(q) > 1 else q.sum() * 0
    total = tip + posture_weight * (bend + .25 * mcp) + .1 * alignment + temporal_weight * temporal
    return total, {'tip': tip, 'bend': bend, 'mcp': mcp, 'direction': alignment, 'temporal': temporal}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', required=True)
    p.add_argument('--weights', choices=['best', 'last'], default='last')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=1500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--train-end', type=int, default=2000)
    p.add_argument('--validation-start', type=int, default=2100)
    p.add_argument('--lr', type=float, default=.00001)
    p.add_argument('--posture-weight', type=float, default=10.)
    p.add_argument('--temporal-weight', type=float, default=.1)
    args = p.parse_args()
    data = np.load(args.data)
    if data.ndim != 3 or data.shape[1:] != (21, 3) or not np.isfinite(data).all():
        p.error('Expected finite (T,21,3) data')
    if not 1 < args.train_end < args.validation_start < len(data):
        p.error('Need nonempty train/validation segments and a gap')
    if args.epochs < 1 or not np.isfinite([args.lr, args.posture_weight, args.temporal_weight]).all() or args.lr <= 0 or min(args.posture_weight, args.temporal_weight) < 0:
        p.error('Invalid optimization parameters')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    parent_path = resolve_checkpoint(args.parent).resolve()
    config = json.loads((parent_path / 'config.json').read_text())
    if config['name'] != 'wuji_hand2_beta1_right' or config.get('model_type', 'fingertip_v1') != 'fingertip_v1':
        p.error('Parent must be an original Wuji fingertip model')
    parent = build_ik_model(config).eval()
    weight_path = parent_path / f'{args.weights}.pth'
    parent.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
    config, model = initialize_model(config, parent)
    human = torch.tensor(data, dtype=torch.float32)
    lower, upper = [torch.tensor(x, dtype=torch.float32) for x in parse_config_joint_limit(config)]
    def unnormalize(x):
        return lower + (x + 1) * .5 * (upper - lower)
    chain = FingerKinematics(config)
    if list(chain.indices) != [16, 17, 18, 19]:
        p.error('Unsupported little-finger joint order')
    fk = TorchFingerKinematics(chain)
    with torch.no_grad():
        original = unnormalize(parent(ik_input({**config, 'model_type': 'fingertip_v1'}, human)))
        target, _ = fk(original[:, -4:])
    optimizer = torch.optim.Adam(model.nets[4].parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {**vars(args), 'data': str(args.data.resolve()), 'output': str(args.output.resolve()),
                'parent': str(parent_path), 'parent_sha256': hashlib.sha256(weight_path.read_bytes()).hexdigest(),
                'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ('geort/train_posture.py', 'geort/model.py', 'geort/kinematics.py')},
                'model_type': config['model_type'], 'scope': 'one contiguous clip; validation may have been seen by parent; final test must be separate'}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    (args.output / 'training.json').write_text(json.dumps(metadata, indent=2))
    history, best = [], float('inf')
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        q = unnormalize(model(human[:args.train_end]))[:, -4:]
        loss, _ = posture_loss(q, *fk(q), target[:args.train_end], human[:args.train_end], args.posture_weight, args.temporal_weight)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.nets[4].parameters(), 10.)
        optimizer.step()
        if epoch % 25 == 0 or epoch == args.epochs:
            with torch.no_grad():
                q = unnormalize(model(human[args.validation_start:]))[:, -4:]
                val, terms = posture_loss(q, *fk(q), target[args.validation_start:], human[args.validation_start:], args.posture_weight, args.temporal_weight)
            row = {'epoch': epoch, 'train': float(loss.detach()), 'validation': float(val), **{k: float(v) for k, v in terms.items()}}
            history.append(row)
            if float(val) < best:
                best = float(val)
                torch.save(model.state_dict(), args.output / 'best.pth')
                (args.output / 'selection.json').write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)
    torch.save(model.state_dict(), args.output / 'last.pth')
    (args.output / 'history.json').write_text(json.dumps(history, indent=2))


if __name__ == '__main__':
    main()
