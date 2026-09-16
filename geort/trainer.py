# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sapien.core as sapien
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from geort.utils.hand_utils import get_entity_by_name, get_active_joints, get_active_joint_indices
from geort.utils.path import get_human_data 
from geort.utils.config_utils import get_config, save_json
from geort.model import FKModel, IKModel, CollisionModel
from geort.env.hand import HandKinematicModel
from geort.loss import chamfer_distance, pinch_distance_loss, collision_penalty
from geort.formatter import HandFormatter
from geort.dataset import RobotKinematicsDataset, MultiPointDataset
from datetime import datetime
from tqdm import tqdm 
import os
from pathlib import Path 
import math
import csv
import uuid
import hashlib
from importlib.metadata import version
import json
import xml.etree.ElementTree as ET
from torch.utils.tensorboard import SummaryWriter

def _link_checkpoint(source, target):
    temporary = target.with_name(f'.{uuid.uuid4().hex}.tmp')
    try:
        os.link(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(state, directory, epoch, save_every=0, is_best=False):
    if save_every < 0:
        raise ValueError('save_every must be nonnegative')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f'.{uuid.uuid4().hex}.pth.tmp'
    try:
        torch.save(state, temporary)
        temporary.replace(directory / 'last.pth')
    finally:
        temporary.unlink(missing_ok=True)
    # Later atomic replacement of last.pth leaves these snapshots intact.
    if is_best:
        _link_checkpoint(directory / 'last.pth', directory / 'best.pth')
    if save_every and (epoch + 1) % save_every == 0:
        _link_checkpoint(directory / 'last.pth', directory / f'epoch_{epoch}.pth')


def update_latest(directory, alias):
    directory, alias = Path(directory).resolve(), Path(alias)
    if alias.is_dir() and not alias.is_symlink():
        raise FileExistsError(f'{alias} is a legacy directory; rename it or use --no-update-last')
    temporary = alias.with_name(f'.{alias.name}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.symlink_to(os.path.relpath(directory, alias.parent), target_is_directory=True)
        temporary.replace(alias)
    finally:
        temporary.unlink(missing_ok=True)


class TrainingLog:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.writer = SummaryWriter(str(self.directory))

    def write(self, stage, epoch, metrics):
        path = self.directory / f'{stage}.csv'
        new_file = not path.exists()
        with path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['epoch', *metrics])
            if new_file:
                writer.writeheader()
            writer.writerow({'epoch': epoch, **metrics})
        for name, value in metrics.items():
            self.writer.add_scalar(f'{stage}/{name}', value, epoch)
        self.writer.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.writer.close()


def merge_dict_list(dl):
    keys = dl[0].keys()
    
    result = {k: [] for k in keys}
    for data in dl:
        for k in keys:
            result[k].append(data[k])
    
    result = {k: np.array(v) for k, v in result.items()}
    return result

def format_loss(value):
    return f"{value:.4e}" if math.fabs(value) < 1e-3 else f"{value:.4f}"

def get_float_list_from_np(np_vector):
    float_list = np_vector.tolist()
    float_list = [float(x) for x in float_list]
    return float_list

def generate_current_timestring():
    """
        Utility Function. Generate a current timestring in the format 'YYYY-MM-DD_HH-MM-SS'.
    """
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

class GeoRTTrainer:
    def __init__(self, config):
        self.config = config
        self.hand = HandKinematicModel.build_from_config(self.config)

    def get_collision_model(self, logger):
        # Cache geometry and simulator provenance, not just the hand name.
        urdf = Path(self.config['urdf_path'])
        digest = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode())
        digest.update(('binary-v1:n32768:seed0:penetration>0:' + version('sapien')).encode())
        digest.update(urdf.read_bytes())
        meshes = {urdf.parent / mesh.attrib['filename']
                  for mesh in ET.parse(urdf).iter('mesh')}
        for mesh in sorted(meshes):
            digest.update(mesh.read_bytes())
        fingerprint = digest.hexdigest()
        path = Path('checkpoint') / f"collision_model_{self.config['name']}.pth"
        # Surrogate fitting must not alter the IK random stream.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(0)
            model = CollisionModel(self.hand.get_n_dof()).cuda()
            if path.exists():
                cached = torch.load(path, weights_only=True)
                if cached['fingerprint'] == fingerprint:
                    model.load_state_dict(cached['model'])
                    print(f'Loaded collision surrogate: {path}; validation {cached["validation"]}')
                    logger.write('collision', 0, cached['validation'])
                    save_json({'fingerprint': fingerprint, 'validation': cached['validation']},
                              logger.directory / 'collision.json')
                    return model.eval().requires_grad_(False)

            lower, upper = self.hand.get_joint_limit()
            normalized = np.random.default_rng(0).uniform(-1, 1, (32768, len(lower))).astype(np.float32)
            poses = HandFormatter(lower, upper).unnormalize(normalized)
            depths = self.hand.self_collision_depth(poses)
            labels = (depths > 0).astype(np.float32)
            counts = np.bincount(labels.astype(int), minlength=2)
            print(f'Collision labels: {counts[0]} free, {counts[1]} colliding')
            if counts.min() < 10:
                raise ValueError('Too few examples of both collision classes; inspect collision geometry before training')
            if counts.min() / len(labels) < .01:
                print('WARNING: collision labels are extremely imbalanced; inspect geometry and per-class validation before IK training')
            points = torch.from_numpy(normalized).cuda()
            targets = torch.from_numpy(labels[:, None]).cuda()
            # Stratification keeps rare collision-free poses in the holdout.
            train_indices, valid_indices = [], []
            rng = np.random.default_rng(1)
            for label in (0, 1):
                ids = rng.permutation(np.flatnonzero(labels == label))
                split = int(len(ids) * .9)
                train_indices.extend(ids[:split])
                valid_indices.extend(ids[split:])
            train_indices = torch.tensor(train_indices, device='cuda')
            valid_indices = torch.tensor(valid_indices, device='cuda')
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            best_score, best_state, best_metrics = float('inf'), None, None
            for epoch in tqdm(range(200), desc='Collision surrogate', unit='epoch', dynamic_ncols=True):
                model.train()
                indices = train_indices[torch.randperm(len(train_indices), device='cuda')]
                for batch in indices.split(512):
                    loss = F.binary_cross_entropy_with_logits(model(points[batch]), targets[batch])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    logits = model(points[valid_indices])
                    truth = targets[valid_indices]
                    losses = F.binary_cross_entropy_with_logits(logits, truth, reduction='none')
                    positive = truth.bool()
                    # Select by both classes, not misleading >99% majority accuracy.
                    score = (losses[positive].mean() + losses[~positive].mean()).item() / 2
                    metrics = {'validation/bce': losses.mean().item(),
                               'validation/balanced_bce': score,
                               'validation/collision_recall': (logits[positive] >= 0).float().mean().item(),
                               'validation/free_recall': (logits[~positive] < 0).float().mean().item(),
                               'validation/free_count': int((~positive).sum()),
                               'validation/collision_count': int(positive.sum()),
                               'labels/collision_fraction': float(labels.mean())}
                logger.write('collision', epoch, metrics)
                if score < best_score:
                    best_score, best_metrics = score, metrics
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if best_state is None:
                raise RuntimeError('Collision surrogate did not produce finite validation loss')
            model.load_state_dict(best_state)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f'.{uuid.uuid4().hex}.tmp')
            try:
                torch.save({'model': best_state, 'fingerprint': fingerprint,
                            'validation': best_metrics}, temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            print(f'Collision surrogate validation: {best_metrics}')
            save_json({'fingerprint': fingerprint, 'validation': best_metrics},
                      logger.directory / 'collision.json')
            return model.eval().requires_grad_(False)

    def get_robot_pointcloud(self, keypoint_names):
        '''
            Utility getter function. Return the robot fingertip point cloud.
        '''
        kinematics_dataset = self.get_robot_kinematics_dataset()
        return kinematics_dataset.export_robot_pointcloud(keypoint_names)
        
    def get_robot_kinematics_dataset(self):
        '''
            Utility getter function. Return the robot kinematics dataset
        '''
        dataset_path = self.get_robot_kinematics_dataset_path(postfix=True)
        if not os.path.exists(dataset_path):
            dataset = self.generate_robot_kinematics_dataset(n_total=100000, save=True)
            dataset_path = self.get_robot_kinematics_dataset_path(postfix=True)
        
        keypoint_names = self.get_keypoint_info()["link"]

        kinematics_dataset = RobotKinematicsDataset(dataset_path, keypoint_names=keypoint_names)
        return kinematics_dataset

    def get_robot_kinematics_dataset_path(self, postfix=False):
        '''
            Utility getter function. Return the path to the robot kinematics dataset.
        '''
        data_name = self.config["name"]
        
        out = f"data/{data_name}"
        if postfix:
            out += '.npz'
        return out 

    def get_keypoint_info(self):
        keypoint_links = []
        keypoint_offsets = []
        keypoint_joints = []
        keypoint_human_ids = []

        joint_order = self.config["joint_order"]

        for info in self.config["fingertip_link"]:
            keypoint_links.append(info["link"])
            keypoint_offsets.append(info['center_offset'])
            keypoint_human_ids.append(info['human_hand_id'])
            
            keypoint_joint = []
            for joint in info["joint"]:
                keypoint_joint.append(joint_order.index(joint))

            keypoint_joints.append(keypoint_joint)

        out = {
            "link": keypoint_links,
            "offset": keypoint_offsets,
            "joint": keypoint_joints,
            "human_id": keypoint_human_ids,
        }

        return out 

    def generate_robot_kinematics_dataset(self, n_total=100000, save=True):
        '''
            This function will generate a (joint position, keypoint position) dataset. 
            - The joint order is specified by "joint_order" in configuration.
            - The keypoint order is specified by "fingertip_link" field in configuration.
        '''
        info = self.get_keypoint_info()
        
        self.hand.initialize_keypoint(keypoint_link_names=info["link"], keypoint_offsets=info["offset"])

        data = []
        joint_range_low, joint_range_high = self.hand.get_joint_limit() # joint order is based on user config specification.
        joint_range_low = np.array(joint_range_low)
        joint_range_high = np.array(joint_range_high)

        all_data_qpos = []
        all_data_keypoint = []
        
        for _ in tqdm(range(n_total)):
            qpos = np.random.uniform(0, 1, len(joint_range_low)) * (joint_range_high - joint_range_low) + joint_range_low
            keypoint = self.hand.keypoint_from_qpos(qpos)
            all_data_qpos.append(qpos)
            all_data_keypoint.append(keypoint)
            
        all_data_keypoint = merge_dict_list(all_data_keypoint)    
        
        dataset = {"qpos": all_data_qpos, "keypoint": all_data_keypoint}

        if save:
            # save data to disk for future use.
            os.makedirs("data", exist_ok=True)
            np.savez(self.get_robot_kinematics_dataset_path(), **dataset)

        return dataset

    def get_fk_checkpoint_path(self):
        name = self.config["name"]
        os.makedirs("checkpoint", exist_ok=True)
        return f"checkpoint/fk_model_{name}.pth"
    
    def get_robot_neural_fk_model(self, force_train=False, logger=None):
        '''
            This function will return a forward kinematics model.
            If the fk model does not exist, this function will train one first.
        '''

        # Normalizer.
        joint_lower_limit, joint_upper_limit = self.hand.get_joint_limit()
        qpos_normalizer = HandFormatter(joint_lower_limit, joint_upper_limit)
        
        # Model.
        print(self.get_keypoint_info()["joint"])
        fk_model = FKModel(keypoint_joints=self.get_keypoint_info()["joint"]).cuda()
        
        # If the model exists, load it.
        fk_checkpoint_path = self.get_fk_checkpoint_path()
        if os.path.exists(fk_checkpoint_path) and not force_train:
            fk_model.load_state_dict(torch.load(fk_checkpoint_path))

        else:
            # If the model does not exist, train it.
            print("Train Neural Forward Kinematics (FK) from Scratch")
        
            fk_dataset = self.get_robot_kinematics_dataset()
            fk_dataloader = DataLoader(fk_dataset, batch_size=256, shuffle=True)
            fk_optim = optim.Adam(fk_model.parameters(), lr=5e-4)

            criterion_fk = nn.MSELoss()
            with tqdm(total=200 * len(fk_dataloader), desc='FK', unit='batch', dynamic_ncols=True) as progress:
                for epoch in range(200):
                    progress.set_description(f"FK epoch {epoch + 1}/200", refresh=False)
                    all_fk_error = 0
                    n_samples = 0
                    for batch in fk_dataloader:
                        keypoint = batch["keypoint"].cuda().float()
                        qpos = batch["qpos"].cuda().float()
                        qpos = qpos_normalizer.normalize_torch(qpos)
                        predicted_keypoint = fk_model(qpos)
                        fk_optim.zero_grad()
                        loss = criterion_fk(predicted_keypoint, keypoint)
                        loss.backward()
                        fk_optim.step()

                        all_fk_error += loss.item() * qpos.size(0)
                        n_samples += qpos.size(0)
                        progress.set_postfix(mse=format_loss(all_fk_error / n_samples), refresh=False)
                        progress.update(1)
                
                    avg_fk_error = all_fk_error / n_samples
                    if logger is not None:
                        logger.write('fk', epoch, {'loss/mse': avg_fk_error,
                                                  'learning_rate': fk_optim.param_groups[0]['lr']})
            
            torch.save(fk_model.state_dict(), fk_checkpoint_path)
        
        fk_model.eval()
        fk_model.requires_grad_(False)
        return fk_model
        
    def train(self, human_data_path, **kwargs):
        kwargs.setdefault('paper_loss', True)
        kwargs.setdefault('paired_sampling', True)
        kwargs.setdefault('w_chamfer', 80.0)
        kwargs.setdefault('w_curvature', 1.0 if kwargs['paper_loss'] else 0.1)
        kwargs.setdefault('w_pinch', 1000.0 if kwargs['paper_loss'] else 1.0)
        kwargs.setdefault('w_collision', 0.0)
        if not math.isfinite(kwargs['w_collision']) or kwargs['w_collision'] < 0:
            raise ValueError('w_collision must be finite and nonnegative')

        np.random.seed(kwargs.get('seed', 0))
        torch.manual_seed(kwargs.get('seed', 0))
        run_name = f"{self.config['name']}_{generate_current_timestring()}"
        if kwargs.get('tag', ''):
            run_name += f"_{kwargs['tag']}"
        save_dir = Path('checkpoint') / run_name
        save_dir.mkdir(parents=True, exist_ok=False)
        log_dir = Path(kwargs.get('log_dir', 'runs')) / run_name
        with TrainingLog(log_dir) as logger:
            save_json({'hand': self.config, 'human_data': str(human_data_path),
                       'checkpoint_dir': str(save_dir), 'options': kwargs,
                       'aggregation': 'sample-weighted epoch mean'}, log_dir / 'training.json')
            print(f"Local training curves: {log_dir}")
            print(f"tensorboard --logdir {kwargs.get('log_dir', 'runs')} --host 127.0.0.1")
            self._train(human_data_path, save_dir, logger, **kwargs)

    def _train(self, human_data_path, save_dir, logger, **kwargs):
        '''
            This is the main trainer.
        '''

        fk_model = self.get_robot_neural_fk_model(logger=logger)
        ik_model = IKModel(keypoint_joints=self.get_keypoint_info()["joint"]).cuda()
        os.makedirs("./checkpoint", exist_ok=True)

        ik_optim = optim.AdamW(ik_model.parameters(), lr=1e-4)

        # Workspace.
        n_epoch = kwargs.get("epoch", 200)
        save_every = kwargs.get('save_every', 0)
        if n_epoch <= 0 or save_every < 0:
            raise ValueError('epoch must be positive and save_every must be nonnegative')
        best_loss, best_epoch = float('inf'), None
        hand_model_name = self.config["name"]

        paper_loss = kwargs.get('paper_loss', True)
        w_chamfer = kwargs.get("w_chamfer", 80.0)
        w_curvature = kwargs.get("w_curvature", 1.0 if paper_loss else 0.1)
        w_collision = kwargs.get("w_collision", 0.0)
        w_pinch = kwargs.get("w_pinch", 1000.0 if paper_loss else 1.0)
        paired_sampling = kwargs.get('paired_sampling', True)
        if not math.isfinite(w_collision) or w_collision < 0:
            raise ValueError('w_collision must be finite and nonnegative')
        collision_model = self.get_collision_model(logger) if w_collision > 0 else None



        last_save_dir = f"./checkpoint/{hand_model_name}_last"

        os.makedirs(save_dir, exist_ok=True)
        update_last = not kwargs.get('no_update_last', False)
        if update_last and Path(last_save_dir).is_dir() and not Path(last_save_dir).is_symlink():
            raise FileExistsError(f'{last_save_dir} is a legacy directory; rename it or use --no-update-last')

        # Save the config including robot joint info to the checkpoint directory.
        joint_lower_limit, joint_upper_limit = self.hand.get_joint_limit()

        export_config = self.config.copy()
        export_config["joint"] = {
            "lower": get_float_list_from_np(joint_lower_limit),
            "upper": get_float_list_from_np(joint_upper_limit)
        }

        save_json(export_config, Path(save_dir) / "config.json")

        # Dataset.
        robot_keypoint_names = self.get_keypoint_info()['link']
        n_keypoints = len(robot_keypoint_names)

        robot_points = self.get_robot_pointcloud(robot_keypoint_names)

        human_finger_idxes = self.get_keypoint_info()["human_id"]
        for robot_keypoint_name, human_id in zip(robot_keypoint_names, human_finger_idxes):
            print(f"Robot Keypoint {robot_keypoint_name}: Human Id: {human_id}")

        human_points = np.load(human_data_path)
        human_points = np.array([human_points[:, idx, :3] for idx in human_finger_idxes]) # [N_finger, N, 3]
        if paired_sampling:
            paired_points = torch.as_tensor(human_points.transpose(1, 0, 2), dtype=torch.float32, device='cuda')

        point_dataset_human = MultiPointDataset.from_points(human_points, n=20000)
        point_dataloader = DataLoader(point_dataset_human, batch_size=2048, shuffle=True)

        # Training / Optimization
        with tqdm(total=n_epoch * len(point_dataloader), desc='IK', unit='batch', dynamic_ncols=True) as progress:
            for epoch in range(n_epoch):
                progress.set_description(f"IK epoch {epoch + 1}/{n_epoch}", refresh=False)
                epoch_sums = {}
                n_samples = 0
                for batch in point_dataloader:
                    direction_loss = 0

                    point = batch.cuda() # [B, N, 3]
                    training = ik_model.training
                    if paper_loss:
                        # Refresh BN statistics, then use ONE deployed mapping
                        # for all losses and every perturbation.
                        if training:
                            with torch.no_grad():
                                ik_model(point)
                        ik_model.eval()
                    joint = ik_model(point) # [B, DOF]
                    embedded_point = fk_model(joint) # [B, N, 3]
                    cover_embedded = embedded_point

                    if paired_sampling:
                        # Geometric constraints share a real gesture batch;
                        # coverage continues to use independent point clouds.
                        point = paired_points[torch.randint(len(paired_points), (point.size(0),), device='cuda')]
                        joint = ik_model(point)
                        embedded_point = fk_model(joint)

                    pinch_loss = pinch_distance_loss(point, embedded_point, paper=paper_loss)

                    # [Curvature loss] -- Ensuring flatness.
                    direction = F.normalize(torch.randn_like(point), dim=-1, p=2)
                    scale = 0.002
                    delta1 = direction * scale
                    if paper_loss:
                        delta1 = torch.randn_like(point) * scale
                    point_delta_1p = point + delta1
                    point_delta_1n = point - delta1

                    joint_p = ik_model(point_delta_1p)
                    joint_n = ik_model(point_delta_1n)
                    embedded_point_p = fk_model(joint_p)
                    embedded_point_n = fk_model(joint_n)
                    curvature_loss = ((embedded_point_p + embedded_point_n - 2 * embedded_point) ** 2).mean()
                    if paper_loss:
                        # Eq. 5: squared vector norm, then expectation over x,d.
                        curvature_loss = curvature_loss * (3 * n_keypoints)
                
                    # [Chamfer loss]
                    selected_idx = np.random.randint(0, robot_points.shape[1], 2048)
                    target = torch.from_numpy(robot_points[:, selected_idx, :]).permute(1, 0, 2).float().cuda()
                
                    chamfer_loss = 0
                    for i in range(n_keypoints):
                        chamfer_loss += chamfer_distance(cover_embedded[:, i, :].unsqueeze(0), target[:, i, :].unsqueeze(0))

                    # [Direction Loss]
                    direction = F.normalize(torch.randn_like(point), dim=-1, p=2)
                    scale = 0.001 + torch.rand(point.size(0)).cuda().unsqueeze(-1).unsqueeze(-1) * 0.01
                    point_delta = point + direction * scale
                    if paper_loss:
                        point_delta = point + torch.randn_like(point) * 0.002

                    joint_delta = ik_model(point_delta)
                    embedded_point_delta = fk_model(joint_delta)

                    d1 = (point_delta - point).reshape(-1, 3)
                    d2 = (embedded_point_delta - embedded_point).reshape(-1, 3)
                    direction_loss = -(((F.normalize(d1, dim=-1, p=2, eps=1e-5) * F.normalize(d2, dim=-1, p=2, eps=1e-5)).sum(-1))).mean()
                    if paper_loss:
                        # Eq. 3 sums the expectation for each finger.
                        direction_loss = direction_loss * n_keypoints
                        ik_model.train(training)

                    # Full geometry gestures, not artificial coverage combinations.
                    collision_loss = (collision_penalty(collision_model, joint)
                                      if collision_model is not None else joint.new_zeros(()))

                    loss = direction_loss + \
                           chamfer_loss * w_chamfer + \
                           curvature_loss * w_curvature + \
                           collision_loss * w_collision + \
                           pinch_loss * w_pinch

                    ik_optim.zero_grad()
                    loss.backward()
                    ik_optim.step()

                    metrics = {
                        'raw/direction': direction_loss.item(),
                        'raw/chamfer': chamfer_loss.item(),
                        'raw/curvature': curvature_loss.item(),
                        'raw/collision': collision_loss.item(),
                        'raw/pinch': pinch_loss.item(),
                        'loss/total': loss.item(),
                    }
                    for name, weight in [('direction', 1.0), ('chamfer', w_chamfer),
                                         ('curvature', w_curvature), ('collision', w_collision),
                                         ('pinch', w_pinch)]:
                        metrics[f'weighted/{name}'] = metrics[f'raw/{name}'] * weight
                    for name, value in metrics.items():
                        epoch_sums[name] = epoch_sums.get(name, 0.0) + value * point.size(0)
                    n_samples += point.size(0)

                    progress.set_postfix({
                        'loss': format_loss(epoch_sums['loss/total'] / n_samples),
                        'dir': format_loss(epoch_sums['raw/direction'] / n_samples),
                        'chamfer': format_loss(epoch_sums['raw/chamfer'] / n_samples),
                        'pinch': format_loss(epoch_sums['raw/pinch'] / n_samples),
                        'collision': format_loss(epoch_sums['raw/collision'] / n_samples),
                    }, refresh=False)
                    progress.update(1)

                epoch_metrics = {name: value / n_samples for name, value in epoch_sums.items()}
                epoch_metrics['learning_rate'] = ik_optim.param_groups[0]['lr']
                logger.write('ik', epoch, epoch_metrics)

                is_best = epoch_metrics['loss/total'] < best_loss
                save_checkpoint(ik_model.state_dict(), save_dir, epoch, save_every, is_best)
                if is_best:
                    best_loss, best_epoch = epoch_metrics['loss/total'], epoch
                save_json({'last_epoch': epoch, 'last_loss': epoch_metrics['loss/total'],
                           'best_epoch': best_epoch, 'best_loss': best_loss,
                           'selection': 'minimum sample-weighted epoch mean of loss/total'},
                          Path(save_dir) / 'checkpoint.json')
                if update_last:
                    update_latest(save_dir, last_save_dir)

        return 


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('-hand', type=str, default='allegro')
    parser.add_argument('-human_data', type=str, default='human')
    parser.add_argument('-ckpt_tag', type=str, default='')
    parser.add_argument('--log-dir', default='runs', help='Local TensorBoard and CSV directory')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save-every', type=int, default=0,
                        help='Keep a numbered snapshot every N epochs; 0 keeps only best.pth and last.pth')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-update-last', action='store_true', help='Preserve the existing hand_last alias')
    loss_profile = parser.add_mutually_exclusive_group()
    loss_profile.add_argument('--paper-loss', dest='paper_loss', action='store_true', default=True,
                              help='Paper reductions/weights and consistent BatchNorm mapping (default); flatness=1, pinch=1000')
    loss_profile.add_argument('--no-paper-loss', dest='paper_loss', action='store_false',
                              help='Original loss reductions and training-mode BatchNorm; flatness=0.1, pinch=1')
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument('--paired-sampling', dest='paired_sampling', action='store_true', default=True,
                          help='Use real same-frame gestures for geometric losses (default)')
    sampling.add_argument('--no-paired-sampling', dest='paired_sampling', action='store_false',
                          help='Use official independent per-finger sampling for every loss')

    parser.add_argument('--w_chamfer', type=float, default=80.0)
    parser.add_argument('--w_curvature', type=float)
    parser.add_argument('--w_collision', type=float, default=0.0,
                        help='GeoRT Eq.8 collision weight (paper range 1e-4..1e-2); 0 disables')
    parser.add_argument('--w_pinch', type=float)

    args = parser.parse_args()
    if args.epochs <= 0 or args.save_every < 0:
        parser.error('--epochs must be positive and --save-every must be nonnegative')
    if not math.isfinite(args.w_collision) or args.w_collision < 0:
        parser.error('--w_collision must be finite and nonnegative')
    if args.w_curvature is None:
        args.w_curvature = 1.0 if args.paper_loss else 0.1
    if args.w_pinch is None:
        args.w_pinch = 1000.0 if args.paper_loss else 1.0

    config = get_config(args.hand)
    trainer = GeoRTTrainer(config)

    human_data_path = get_human_data(args.human_data)
    print("Training with human data:", human_data_path.as_posix())
    
    trainer.train(
        human_data_path, 
        tag=args.ckpt_tag, 
        log_dir=args.log_dir,
        epoch=args.epochs,
        save_every=args.save_every,
        seed=args.seed,
        no_update_last=args.no_update_last,
        paper_loss=args.paper_loss,
        paired_sampling=args.paired_sampling,
        w_chamfer=args.w_chamfer, 
        w_curvature=args.w_curvature, 
        w_collision=args.w_collision,
        w_pinch=args.w_pinch)
