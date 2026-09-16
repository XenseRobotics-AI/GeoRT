# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from geort.utils.config_utils import get_config, save_json
from geort.utils.hand_utils import get_entity_by_name, get_active_joints, get_active_joint_indices
from datetime import datetime
from tqdm import tqdm 
import os
from pathlib import Path 
import math

class HandKinematicModel:
    def __init__(self, 
                 scene=None, 
                 render=False, 
                 hand=None, 
                 hand_urdf='', 
                 n_hand_dof=16, 
                 base_link='base_link', 
                 joint_names=[],
                 # Ideally, these two guys (PD controller args) shouldn't be here. 
                 # -- There should be a controller class. I leave them here for code simplicity (maybe truth: or because I am lazy).
                 # If you see your hand model doing something weird (in the simulation viewer below), tune them.
                 kp=400.0, 
                 kd=10):
        
        self.engine = None
        renderer = None
        if scene is None and hasattr(sapien, 'physx'):
            sapien.physx.set_scene_config(enable_pcm=False)
            sapien.physx.set_default_material(1.0, 1.0, 0.0)
            sapien.physx.set_shape_config(contact_offset=0.02)
            sapien.physx.set_body_config(solver_position_iterations=25, solver_velocity_iterations=1)
            systems = [sapien.physx.PhysxCpuSystem()]
            if render:
                renderer = sapien.SapienRenderer()
                systems.append(sapien.render.RenderSystem())
                print("Enable Render Mode.")
            scene = sapien.Scene(systems)
        elif scene is None:
            engine = sapien.Engine()
            
            if render:
                renderer = sapien.VulkanRenderer()  
                engine.set_renderer(renderer)
                print("Enable Render Mode.")
            else:
                renderer = None 
            scene_config = sapien.SceneConfig()
            scene_config.default_dynamic_friction = 1.0
            scene_config.default_static_friction = 1.0
            scene_config.default_restitution = 0.00
            scene_config.contact_offset = 0.02
            scene_config.enable_pcm = False
            scene_config.solver_iterations = 25
            scene_config.solver_velocity_iterations = 1
            scene = engine.create_scene(scene_config)  
            self.engine = engine 

        self.scene = scene 
        self.renderer = renderer 

        if hand is not None:
            self.hand = hand

        else:
            loader = scene.create_urdf_loader()
            if hasattr(sapien, 'physx') and not render:
                builder = loader.load_file_as_articulation_builder(hand_urdf)
                # SAPIEN 3 builds render components even in a physics-only scene.
                for link_builder in builder.link_builders:
                    link_builder.visual_records.clear()
                self.hand = builder.build()
            else:
                self.hand = loader.load(hand_urdf)
            self.hand.set_root_pose(sapien.Pose([0, 0, 0.35], [0.695, 0, -0.718, 0]))

        self.pmodel = self.hand.create_pinocchio_model()

        # Setup hand base link.
        self.base_link = get_entity_by_name(self.hand.get_links(), base_link)
        self.base_link_idx = self.hand.get_links().index(self.base_link)

        # Setup hand dofs.
        self.all_joints = get_active_joints(self.hand, joint_names)
        all_limits = [joint.get_limits() for joint in self.all_joints]

        self.joint_names = joint_names
        self.user_idx_to_sim_idx = get_active_joint_indices(self.hand, joint_names)
        print("User-to-Sim Joint", self.user_idx_to_sim_idx)
        self.sim_idx_to_user_idx = [self.user_idx_to_sim_idx.index(i) for i in range(len(self.user_idx_to_sim_idx))]
        print("Sim-to-User Joint", self.sim_idx_to_user_idx)

        self.joint_lower_limit = np.array([l[0][0] for l in all_limits])  # this is in user specified "joint_name" order
        self.joint_upper_limit = np.array([l[0][1] for l in all_limits])  # this is in user specified "joint_name" order
        print(self.joint_lower_limit, self.joint_upper_limit)

        init_qpos = self.convert_user_order_to_sim_order((self.joint_lower_limit + self.joint_upper_limit) / 2)
        self.hand.set_qpos(init_qpos)
        self.hand.set_qvel(0.0 * init_qpos)
        self.qpos_target = init_qpos

        for i, joint in enumerate(self.all_joints):
            print(i, self.joint_names[i], joint, self.joint_lower_limit[i], self.joint_upper_limit[i])
            joint.set_drive_property(kp, kd, force_limit=10)

    def get_n_dof(self):
        '''
            number of dof.
        '''
        return len(self.joint_lower_limit)

    def self_collision_depth(self, qposes):
        """Measure maximum self-penetration (m) at each user-ordered pose."""
        qposes = np.asarray(qposes)
        if qposes.ndim != 2 or qposes.shape[1] != self.get_n_dof() or not np.isfinite(qposes).all():
            raise ValueError('Expected finite self-collision poses with shape (N, DOF)')
        qpos, qvel = self.hand.get_qpos().copy(), self.hand.get_qvel().copy()
        timestep = self.scene.get_timestep()
        links = self.hand.get_links()
        depths = []
        try:
            # A tiny step refreshes contacts at the requested pose, before
            # the solver can substantially move it. No ground is added here.
            self.scene.set_timestep(1e-6)
            for pose in tqdm(qposes, desc='Collision labels', unit='pose', dynamic_ncols=True):
                self.hand.set_qpos(self.convert_user_order_to_sim_order(pose))
                self.hand.set_qvel(np.zeros(self.get_n_dof()))
                self.scene.step()
                depth = 0.0
                for contact in self.scene.get_contacts():
                    bodies = contact.bodies if hasattr(contact, 'bodies') else (contact.actor0, contact.actor1)
                    if all(body in links for body in bodies):
                        for point in contact.points:
                            depth = max(depth, -float(point.separation))
                depths.append(depth)
        finally:
            self.scene.set_timestep(timestep)
            self.hand.set_qpos(qpos)
            self.hand.set_qvel(qvel)
        return np.asarray(depths, dtype=np.float32)

    def get_joint_limit(self):
        '''
            Get the hand joint limit.
        '''
        return self.joint_lower_limit, self.joint_upper_limit

    def initialize_keypoint(self, keypoint_link_names, keypoint_offsets):
        '''
            Setup keypoints to track.
        '''
        keypoint_links = [get_entity_by_name(self.hand.get_links(), link) for link in keypoint_link_names]
        print(keypoint_links)

        keypoint_links_id_dict = {link_name: (self.hand.get_links().index(keypoint_links[i]), i) for i, link_name in enumerate(keypoint_link_names)}
        self.keypoint_links = keypoint_links
        self.keypoint_links_id_dict = keypoint_links_id_dict
        self.keypoint_offsets = np.array(keypoint_offsets)

    def convert_user_order_to_sim_order(self, qpos):
        return qpos[self.sim_idx_to_user_idx]

    def keypoint_from_qpos(self, qpos, ret_vec=False):
        '''
            Get keypoints from hand qpos. qpos is specified using the user order.
        '''
        qpos = self.convert_user_order_to_sim_order(qpos)
        self.pmodel.compute_forward_kinematics(qpos)
        base_pose = self.pmodel.get_link_pose(self.base_link_idx)

        result = {} 
        vec_result = []

        for m, (link_idx, i) in self.keypoint_links_id_dict.items():
            pose = self.pmodel.get_link_pose(link_idx)
            new_pose = sapien.Pose(p=pose.p + (pose.to_transformation_matrix()[:3, :3] @ self.keypoint_offsets[i].reshape(3, 1)).reshape(-1), q=pose.q)

            x = (base_pose.inv() * new_pose).p # convert to hand base frame.
            vec_result.append(x)
            result[m] = x

        if ret_vec:
            return np.array(vec_result)
        return result

    def exclude_collision_pairs(self, pairs):
        """Apply explicit asset exclusions without disabling other self contacts."""
        if not pairs:
            return
        links = {link.name: link for link in self.hand.get_links()}
        for pair in pairs:
            if len(pair) != 2 or pair[0] == pair[1] or any(name not in links for name in pair):
                raise ValueError(f'Invalid collision exclusion: {pair}')
        # Reserve distinct ignore bits across the scene, so another hand cannot
        # accidentally inherit these pair exclusions through the same group ID.
        bodies = list(self.scene.get_all_actors())
        for articulation in self.scene.get_all_articulations():
            bodies.extend(articulation.get_links())
        used = 0
        for body in bodies:
            for shape in body.get_collision_shapes():
                used |= shape.get_collision_groups()[2]
        bits = [1 << i for i in range(32) if not used & (1 << i)]
        if len(pairs) > len(bits):
            raise ValueError('Not enough collision filter bits for explicit exclusions')
        for pair, bit in zip(pairs, bits):
            shapes = [shape for name in pair for shape in links[name].get_collision_shapes()]
            if len({shape.get_collision_groups()[3] & 0xffff for shape in shapes}) > 1:
                raise ValueError(f'Collision group IDs differ for {pair}')
            for shape in shapes:
                groups = list(shape.get_collision_groups())
                groups[2] |= bit
                if hasattr(sapien, 'physx'):
                    shape.set_collision_groups(groups)
                else:
                    shape.set_collision_groups(*groups)

    @staticmethod
    def build_from_config(config, **kwargs):
        '''
            Build a kinematic model from user config.
        '''
        render = kwargs.get("render", False)
        urdf_path = config["urdf_path"]
        n_hand_dof = len(config["joint_order"])
        base_link = config["base_link"]
        joint_order = config["joint_order"]

        model = HandKinematicModel(hand_urdf=urdf_path, render=render, n_hand_dof=n_hand_dof,base_link=base_link, joint_names=joint_order)
        model.exclude_collision_pairs(config.get('collision_exclusions', []))
        return model 

    def get_viewer_env(self):
        return HandViewerEnv(self)

    def get_scene(self):
        return self.scene

    def get_renderer(self):
        return self.renderer

    def set_qpos_target(self, qpos):
        '''
            This function is only used during visualization
        '''
        qpos = np.clip(qpos, self.joint_lower_limit + 1e-3, self.joint_upper_limit - 1e-3)
        self.qpos_target = self.convert_user_order_to_sim_order(qpos)

        # all_joints is in user order; articulation qpos_target is in simulator order.
        for joint, target in zip(self.all_joints, qpos):
            joint.set_drive_target(target)

class HandViewerEnv:
    def __init__(self, model):
        scene = model.get_scene()
        scene.set_timestep(1 / 100.0) 
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        scene.add_ground(altitude=0) 

        viewer = Viewer(model.get_renderer())
        viewer.set_scene(scene) 
        viewer.window.set_camera_position([0.1550926,-0.1623763, 0.7064089])
        viewer.window.set_camera_rotation([0.8716827, 0.3260138, 0.12817779, 0.3427167])
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
        
        self.model = model
        self.scene = scene 
        self.viewer = viewer 

    def update(self):
        self.scene.step()
        self.scene.update_render()  
        self.viewer.render()

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand', type=str, default='allegro')

    args = parser.parse_args()

    # Load Hand Model
    config = get_config(args.hand)
    model = HandKinematicModel.build_from_config(config, render=True)
    viewer_env = model.get_viewer_env()
   
    # Control Loop
    n_dof = model.get_n_dof()
    dof_lower, dof_upper = model.get_joint_limit()

    steps = 0
    while True:
        viewer_env.update()

        steps += 1
        if steps % 30 == 0:
            targets = np.random.uniform(0, 1, n_dof) * (dof_upper - dof_lower - 1e-7) + dof_lower + 1e-7
            model.set_qpos_target(targets)
