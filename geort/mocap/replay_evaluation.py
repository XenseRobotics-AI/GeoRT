# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the LICENSE file in the repository root.

"""Side-by-side human skeleton and official PD replay."""
import argparse
import numpy as np
import sapien.core as sapien

from geort import load_model, get_config
from geort.env.hand import HandKinematicModel
from geort.mocap.replay_mocap import ReplayMocap
from scipy.spatial.transform import Rotation
from geort.utils.path import to_package_root
import time


class HumanHandViewer:
    """Display the recorded 21-point hand without changing the model input."""
    def __init__(self, scene, viewer, offset):
        self.viewer = viewer
        self.offset = np.array(offset)
        self.lines = None
        colors = [[1, 0.35, 0.25], [0.3, 0.85, 0.4], [0.3, 0.65, 1],
                  [1, 0.75, 0.2], [0.8, 0.4, 1]]
        self.edges = []
        self.edge_colors = []
        self.joints = []
        for i in range(21):
            color = [0.9, 0.9, 0.9] if i == 0 else colors[(i - 1) // 4]
            builder = scene.create_actor_builder()
            builder.add_sphere_visual(radius=0.003, material=color)
            self.joints.append(builder.build_kinematic(name=f"human_keypoint_{i}"))
        # Wrist=0; thumb/index/middle/ring/little occupy four points each.
        for finger in range(5):
            chain = [0] + list(range(1 + finger * 4, 5 + finger * 4))
            for start, end in zip(chain[:-1], chain[1:]):
                self.edges.append((start, end))
                self.edge_colors.extend([colors[finger] + [1]] * 2)
        for start, end in [(5, 9), (9, 13), (13, 17)]:
            self.edges.append((start, end))
            self.edge_colors.extend([[0.7, 0.7, 0.7, 1]] * 2)

    def update(self, keypoints):
        points = np.asarray(keypoints) + self.offset
        for joint, point in zip(self.joints, points):
            joint.set_pose(sapien.Pose(point))
        if self.lines is not None:
            self.viewer.render_scene.remove_node(self.lines)
        vertices = points[np.array(self.edges)].reshape(-1).astype(np.float32)
        colors = np.array(self.edge_colors, dtype=np.float32).reshape(-1)
        self.lines = self.viewer.render_scene.add_line_set(
            self.viewer.renderer_context.create_line_set(vertices, colors))


def reset_camera(viewer, frame_pose=None):
    # SAPIEN camera axes: +X forward, +Y left, +Z up. Keep world +Z upright.
    position = np.array([0.65, 0, 0.36])
    target = np.array([0, 0, 0.20])
    forward = target - position
    forward /= np.linalg.norm(forward)
    left = np.cross([0, 0, 1], forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    quat = Rotation.from_matrix(np.column_stack([forward, left, up])).as_quat()
    pose = sapien.Pose(position, quat[[3, 0, 1, 2]])
    viewer.set_camera_pose(pose if frame_pose is None else frame_pose * pose)
    viewer.window.set_camera_parameters(near=0.01, far=10, fovy=0.8)


def build_hand(name):
    if name == 'taccap_slave':
        return HandKinematicModel(
            hand_urdf=str(to_package_root('assets/taccap_slave/urdf/taccap_slave.urdf')),
            render=True, base_link='geort_base',
            joint_names=['grip_left_joint', 'grip_right_joint'])
    return HandKinematicModel.build_from_config(get_config(name), render=True)


def preview_qpos(hand, name, position):
    lower, upper = hand.get_joint_limit()
    if name == 'taccap_slave':
        # SAPIEN exposes both joints; explicitly preserve q_right = q_left
        # when setting poses directly without stepping the mimic constraint.
        return np.full(2, position)
    # Sweep flexion from neutral; keep finger abduction neutral.
    closed = np.array([0 if 'abd' in joint else 0.6 * limit
                       for joint, limit in zip(hand.joint_names, upper)])
    return np.clip(position * closed, lower, upper)


def preview_hand(args):
    hand = build_hand(args.hand)
    viewer = hand.get_viewer_env().viewer
    hand.hand.set_root_pose(sapien.Pose([0, 0, 0.14]))
    reset_camera(viewer)
    print('Kinematic asset preview; no retargeting model or hardware control.')
    print('Press R to reset the camera. Close the window or Ctrl+C to exit.')
    start = time.monotonic()
    try:
        while not viewer.closed:
            frame_start = time.monotonic()
            position = (1 - np.cos((time.monotonic() - start) * np.pi / 2)) / 2 if args.animate else args.position
            qpos = preview_qpos(hand, args.hand, position)
            hand.hand.set_qpos(hand.convert_user_order_to_sim_order(qpos))
            if viewer.window.key_press('r'):
                reset_camera(viewer)
            hand.scene.update_render()
            viewer.render()
            time.sleep(max(0, 1 / (args.fps or 60) - (time.monotonic() - frame_start)))
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-hand', default='allegro_right')
    parser.add_argument('-ckpt_tag', default='allegro_right_last')
    parser.add_argument('-data', default='human_alex')
    parser.add_argument('--weights', choices=['last', 'best'], default='last')
    parser.add_argument('--direct-qpos', action='store_true',
                        help='Set model joint positions directly without physics/PD steps')
    parser.add_argument('--fps', type=float, help='Target data frames per second; default: unlimited replay, 60 for preview')
    parser.add_argument('--preview', action='store_true', help='Preview a TacCap/Wuji asset without a model')
    parser.add_argument('--animate', action='store_true', help='Animate asset preview')
    parser.add_argument('--position', type=float, default=0.5, help='Asset preview pose fraction, 0..1')
    args = parser.parse_args()
    if args.fps is not None and (not np.isfinite(args.fps) or args.fps <= 0):
        parser.error('--fps must be finite and positive')
    if args.preview:
        if args.hand not in ('taccap_slave', 'wuji_hand2_beta1_right') or not 0 <= args.position <= 1:
            parser.error('Preview requires TacCap/Wuji and --position in [0, 1]')
        preview_hand(args)
        return
    model = load_model(args.ckpt_tag, weights=args.weights)
    mocap = ReplayMocap(args.data)
    if mocap.human_points.shape[1:] != (21, 3) or not mocap.T:
        parser.error('Human skeleton requires nonempty (T, 21, 3) data')
    if not np.isfinite(mocap.human_points).all():
        parser.error('Replay data must contain only finite coordinates')
    config = get_config(args.hand)
    # Collision filters change physical feasibility, not model/FK joint semantics.
    if any(model.config.get(key) != value for key, value in config.items()
           if key != 'collision_exclusions'):
        parser.error('Checkpoint and -hand configuration differ')
    hand = HandKinematicModel.build_from_config(config, render=True)
    env = hand.get_viewer_env()
    viewer = env.viewer
    # Preserve the official robot pose and gravity/contact geometry. Rotate
    # the camera and skeleton into its frame instead of rotating the robot.
    robot_pose = hand.hand.get_root_pose()
    center_pose = robot_pose * sapien.Pose([0, -0.15, -0.16])
    human_pose = robot_pose * sapien.Pose([0, -0.30, 0])
    transform = human_pose.to_transformation_matrix()
    human = HumanHandViewer(hand.scene, viewer, [0, 0, 0])
    hand.scene.set_ambient_light([0.7, 0.7, 0.7])
    direction = robot_pose.to_transformation_matrix()[:3, :3] @ np.array([-1, 0, -1])
    hand.scene.add_directional_light(direction, [0.7, 0.7, 0.7], shadow=False)
    reset_camera(viewer, frame_pose=center_pose)
    if args.direct_qpos:
        print('Left: recorded human skeleton. Right: model joint positions, without physics/PD.')
    else:
        print('Left: recorded human skeleton. Right: robot under PD control; tracking may lag.')
    print('Same input frame and scale. R resets camera.')
    # Original loop advances ten physics steps before each new target.
    # Initialize visuals with frame zero so no skeleton flashes at the origin.
    human.update(mocap.human_points[0] @ transform[:3, :3].T + transform[:3, 3])
    try:
        while not viewer.closed:
            frame_start = time.monotonic()
            if not args.direct_qpos:
                for _ in range(10):
                    if viewer.closed:
                        break
                    if viewer.window.key_press('r'):
                        reset_camera(viewer, frame_pose=center_pose)
                    env.update()
            if viewer.closed:
                break
            result = mocap.get()
            if result['status'] == 'quit':
                break
            if result['status'] == 'recording' and result['result'] is not None:
                points = result['result']
                qpos = model.forward(points)
                if args.direct_qpos:
                    hand.hand.set_qpos(hand.convert_user_order_to_sim_order(qpos))
                else:
                    hand.set_qpos_target(qpos)
                human.update(points @ transform[:3, :3].T + transform[:3, 3])
            if args.direct_qpos:
                if viewer.window.key_press('r'):
                    reset_camera(viewer, frame_pose=center_pose)
                hand.scene.update_render()
                viewer.render()
            if args.fps is not None:
                time.sleep(max(0, 1 / args.fps - (time.monotonic() - frame_start)))
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == '__main__':
    main()
