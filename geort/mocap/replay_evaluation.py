# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the LICENSE file in the repository root.

"""Side-by-side human skeleton and configurable PD or direct-joint replay."""
import argparse
import numpy as np
import sapien.core as sapien

from geort import load_model, get_config
from geort.env.hand import HandKinematicModel
from geort.mocap.replay_mocap import ReplayMocap
from scipy.spatial.transform import Rotation
from geort.utils.path import to_package_root
import time


class ReplayController:
    """Advance exactly one data interval, without tying physics to rendering."""
    def __init__(self, hand, fps, control_hz, physics_hz, max_velocity=0.0, interpolate=True):
        controls, substeps = self.step_counts(fps, control_hz, physics_hz)
        if not np.isfinite(max_velocity) or max_velocity < 0:
            raise ValueError('max-joint-velocity must be finite and nonnegative')
        self.hand = hand
        self.controls, self.substeps = controls, substeps
        self.max_delta = max_velocity / control_hz if max_velocity else np.inf
        self.interpolate = interpolate
        self.command = hand.hand.get_qpos()[hand.user_idx_to_sim_idx].copy()
        self.previous_target = self.command.copy()
        hand.scene.set_timestep(1 / physics_hz)

    @staticmethod
    def step_counts(fps, control_hz, physics_hz):
        rates = (fps, control_hz, physics_hz)
        if not all(np.isfinite(rate) and rate > 0 for rate in rates):
            raise ValueError('Replay rates must be finite and positive')
        ratios = (control_hz / fps, physics_hz / control_hz)
        if any(ratio < 1 or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9) for ratio in ratios):
            raise ValueError('control-hz must be a multiple of fps; physics-hz must be a multiple of control-hz')
        return tuple(int(round(ratio)) for ratio in ratios)

    def advance(self, target):
        target = np.asarray(target)
        if target.shape != self.command.shape or not np.isfinite(target).all():
            raise ValueError('Expected one finite target per joint')
        target = np.clip(target, self.hand.joint_lower_limit + 1e-3, self.hand.joint_upper_limit - 1e-3)
        for tick in range(1, self.controls + 1):
            desired = (self.previous_target + (target - self.previous_target) * tick / self.controls
                       if self.interpolate else target)
            self.command += np.clip(desired - self.command, -self.max_delta, self.max_delta)
            self.hand.set_qpos_target(self.command)
            for _ in range(self.substeps):
                self.hand.scene.step()
        self.previous_target = target.copy()


def build_direct_comparison(hand, config, root_pose):
    """Visual comparison with no collision shapes; reset its pose after PD steps."""
    loader = hand.scene.create_urdf_loader()
    loader.fix_root_link = True
    builder = loader.load_file_as_articulation_builder(config['urdf_path'])
    for link in builder.link_builders:
        link.collision_records.clear()
    direct = builder.build()
    direct.set_root_pose(root_pose)
    # Both articulations must expose exactly the same simulator joint order.
    if [j.name for j in direct.get_active_joints()] != [j.name for j in hand.hand.get_active_joints()]:
        raise ValueError('Comparison articulation joint order differs')
    direct.set_qpos(hand.hand.get_qpos())
    return direct


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


def reset_camera(viewer, frame_pose=None, distance=0.65, orthographic=False):
    # SAPIEN camera axes: +X forward, +Y left, +Z up. Keep world +Z upright.
    position = np.array([distance, 0, 0.36])
    target = np.array([0, 0, 0.20])
    forward = target - position
    forward /= np.linalg.norm(forward)
    left = np.cross([0, 0, 1], forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    quat = Rotation.from_matrix(np.column_stack([forward, left, up])).as_quat()
    pose = sapien.Pose(position, quat[[3, 0, 1, 2]])
    viewer.set_camera_pose(pose if frame_pose is None else frame_pose * pose)
    if orthographic:
        # A shared perspective camera views side-by-side palms from different
        # directions. Parallel rays keep identical poses visually comparable.
        viewer.window.set_camera_orthographic_parameters(0.01, 10, distance * np.tan(0.8 / 2))
    else:
        viewer.window.set_camera_parameters(near=0.01, far=10, fovy=0.8)


def render_frame(viewer, orthographic_distance=None):
    """Keep comparison projection after RenderWindow initialization/resizing.

    SAPIEN can recreate its camera as perspective during the first render (and
    after resizing). An offscreen camera does not exercise this window lifecycle.
    Restore projection and redraw without resetting the user's camera pose.
    """
    viewer.render()
    if orthographic_distance is None or viewer.closed:
        return
    if viewer.window.camera_mode != 'orthographic':
        viewer.window.set_camera_orthographic_parameters(
            0.01, 10, orthographic_distance * np.tan(0.8 / 2))
        viewer.notify_render_update()
        viewer.render()
        if not viewer.closed and viewer.window.camera_mode != 'orthographic':
            raise RuntimeError('Comparison window failed to retain orthographic projection')


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
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--compare-pd', action='store_true',
                       help='Show human, raw model output and PD result side by side')
    modes.add_argument('--compare-wuji-cache', help='Show cached production Wuji output and model output on identical frames')
    parser.add_argument('--wuji-output', choices=['filtered', 'unfiltered'], default='filtered')
    modes.add_argument('--compare-checkpoint', metavar='REFERENCE_TAG',
                       help='Show human, reference checkpoint and selected checkpoint without physics')
    parser.add_argument('--reference-weights', choices=['last', 'best'], default='last')
    modes.add_argument('--compare-posture', action='store_true',
                       help='Show human, raw model output and posture-corrected output without physics')
    modes.add_argument('--direct-qpos', action='store_true',
                        help='Set model joint positions directly without physics/PD steps')
    parser.add_argument('--posture-correction', action='store_true',
                        help='Opt-in Wuji little-finger posture correction')
    parser.add_argument('--posture-budget-mm', type=float, default=2.0)
    parser.add_argument('--posture-extension-deg', type=float, default=5.0)
    parser.add_argument('--posture-mcp-extension-deg', type=float, default=15.0)
    parser.add_argument('--posture-direction-weight', type=float, default=0.05)
    parser.add_argument('--posture-max-step-deg', type=float, default=8.0,
                        help='Max little-finger command change per data frame; 0 disables')
    parser.add_argument('--posture-starts', type=int, choices=[1, 3], default=3)
    parser.add_argument('--fps', type=float, help='Data/playback Hz; default: 100 for replay, 60 for preview')
    parser.add_argument('--pd-timing', choices=['official', 'realtime'], default='official',
                        help='official: 0.1 simulated seconds per frame; realtime: 1/fps (default: official)')
    parser.add_argument('--control-hz', type=float, help='Realtime PD target update Hz (default: 500)')
    parser.add_argument('--physics-hz', type=float, help='Realtime physics Hz (default: 1000)')
    parser.add_argument('--max-joint-velocity', type=float, default=0.0,
                        help='Command slew limit in rad/s; 0 disables (default: 0)')
    parser.add_argument('--no-interpolation', action='store_true', help='Hold each data target instead of interpolating')
    parser.add_argument('--kp', type=float, default=400.0, help='Simulation position gain (default: 400)')
    parser.add_argument('--kd', type=float, default=10.0, help='Simulation damping gain (default: 10)')
    parser.add_argument('--force-limit', type=float, default=10.0, help='Simulation joint torque limit in N m (default: 10)')
    parser.add_argument('--preview', action='store_true', help='Preview a TacCap/Wuji asset without a model')
    parser.add_argument('--animate', action='store_true', help='Animate asset preview')
    parser.add_argument('--position', type=float, default=0.5, help='Asset preview pose fraction, 0..1')
    args = parser.parse_args()
    if args.fps is not None and (not np.isfinite(args.fps) or args.fps <= 0):
        parser.error('--fps must be finite and positive')
    for name in ('kp', 'kd', 'force_limit', 'max_joint_velocity'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and nonnegative')
    if (args.compare_checkpoint or args.compare_wuji_cache) and (args.preview or args.posture_correction):
        parser.error('Checkpoint comparison requires raw model replay without posture correction')
    corrector = None
    if args.posture_correction or args.compare_posture:
        if args.preview:
            parser.error('Posture correction requires model replay, not asset preview')
        from geort.posture import WujiPostureCorrector
        try:
            corrector = WujiPostureCorrector(
                get_config(args.hand), tip_budget_mm=args.posture_budget_mm,
                extension_tolerance_deg=args.posture_extension_deg,
                direction_weight=args.posture_direction_weight, starts=args.posture_starts,
                mcp_extension_tolerance_deg=args.posture_mcp_extension_deg,
                max_joint_step_deg=args.posture_max_step_deg)
        except ValueError as error:
            parser.error(str(error))
    args.direct_qpos = args.direct_qpos or args.compare_posture or bool(args.compare_checkpoint or args.compare_wuji_cache)
    compare = args.compare_pd or args.compare_posture or bool(args.compare_checkpoint or args.compare_wuji_cache)
    if args.preview:
        if args.hand not in ('taccap_slave', 'wuji_hand2_beta1_right') or not 0 <= args.position <= 1:
            parser.error('Preview requires TacCap/Wuji and --position in [0, 1]')
        preview_hand(args)
        return
    args.fps = args.fps or 100.0
    simulation_fps = args.fps
    if not args.direct_qpos:
        if args.pd_timing == 'official':
            if args.control_hz is not None or args.physics_hz is not None or args.max_joint_velocity:
                parser.error('Custom rates/velocity limit require --pd-timing realtime')
            # Official dwell time and timestep, while rendering human and robot together
            # after applying this frame's target rather than rendering the previous frame.
            simulation_fps, args.control_hz, args.physics_hz = 10.0, 10.0, 100.0
            args.no_interpolation = True
        else:
            args.control_hz = 500.0 if args.control_hz is None else args.control_hz
            args.physics_hz = 1000.0 if args.physics_hz is None else args.physics_hz
        try:
            ReplayController.step_counts(simulation_fps, args.control_hz, args.physics_hz)
        except ValueError as error:
            parser.error(str(error))
    model = load_model(args.ckpt_tag, weights=args.weights)
    reference = load_model(args.compare_checkpoint, weights=args.reference_weights) if args.compare_checkpoint else None
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
    if reference is not None:
        if any(reference.config.get(key) != value for key, value in config.items() if key != 'collision_exclusions') or reference.config['joint'] != model.config['joint']:
            parser.error('Reference checkpoint and selected hand kinematics differ')
    cached_wuji = None
    if args.compare_wuji_cache:
        from geort.wuji_baseline import load_wuji_cache
        cached_wuji = load_wuji_cache(args.compare_wuji_cache, mocap.data_path, config['joint_order'])[args.wuji_output]
    hand = HandKinematicModel.build_from_config(config, render=True)
    env = hand.get_viewer_env()
    viewer = env.viewer
    controller = None
    if not args.direct_qpos:
        for joint in hand.all_joints:
            joint.set_drive_property(args.kp, args.kd, force_limit=args.force_limit)
        controller = ReplayController(hand, simulation_fps, args.control_hz, args.physics_hz,
                                      args.max_joint_velocity, not args.no_interpolation)
    # Preserve the official robot pose and gravity/contact geometry. Rotate
    # the camera and skeleton into its frame instead of rotating the robot.
    robot_pose = hand.hand.get_root_pose()
    direct = (build_direct_comparison(hand, config, robot_pose * sapien.Pose([0, -0.30, 0]))
              if compare else None)
    center_pose = robot_pose * sapien.Pose([0, -0.30 if compare else -0.15, -0.16])
    camera_distance = 1.05 if compare else 0.65
    human_pose = robot_pose * sapien.Pose([0, -0.60 if compare else -0.30, 0])
    transform = human_pose.to_transformation_matrix()
    human = HumanHandViewer(hand.scene, viewer, [0, 0, 0])
    hand.scene.set_ambient_light([0.7, 0.7, 0.7])
    direction = robot_pose.to_transformation_matrix()[:3, :3] @ np.array([-1, 0, -1])
    hand.scene.add_directional_light(direction, [0.7, 0.7, 0.7], shadow=False)
    reset_camera(viewer, frame_pose=center_pose, distance=camera_distance, orthographic=compare)
    if args.compare_wuji_cache:
        print(f'Left: human. Center: Wuji reference ({args.wuji_output}). Right: {args.ckpt_tag}. Common GeoRT geometry, no physics.')
    elif args.compare_checkpoint:
        print(f'Left: human. Center: {args.compare_checkpoint} ({args.reference_weights}). Right: {args.ckpt_tag} ({args.weights}). No physics or correction.')
    elif args.compare_posture:
        print('Left: human skeleton. Center: raw model output. Right: posture-corrected output (no physics).')
    elif args.direct_qpos:
        print('Left: recorded human skeleton. Right: model joint positions, without physics/PD.')
    else:
        print('Left: human skeleton. Center: raw model output. Right: PD result.' if args.compare_pd else
              'Left: recorded human skeleton. Right: robot under PD control; tracking may lag.')
        print(f'PD timing: {args.pd_timing}; playback {args.fps:g} FPS, simulated frame {1 / simulation_fps:g} s; '
              f'control {args.control_hz:g} Hz, physics {args.physics_hz:g} Hz; '
              f'command limit {args.max_joint_velocity:g} rad/s (0=off).')
    if corrector is not None:
        print(f'Posture correction ON: little PIP/DIP, tip budget {args.posture_budget_mm:g} mm; '
              'display/drive command corrected; center comparison stays raw. No collision guarantee.')
    print('Same input frame and scale. R resets camera.' +
          (' Orthographic comparison: parallel palm views.' if compare else ''))
    # Initialize visuals with frame zero so no skeleton flashes at the origin.
    human.update(mocap.human_points[0] @ transform[:3, :3].T + transform[:3, 3])
    relaxed_frames = 0
    mcp_relaxed_frames = 0
    try:
        while not viewer.closed:
            frame_start = time.monotonic()
            frame_index = mocap.t
            result = mocap.get()
            if result['status'] == 'quit':
                break
            if result['status'] == 'recording' and result['result'] is not None:
                points = result['result']
                raw_qpos = model.forward(points)
                if corrector is not None:
                    correction = corrector.correct(raw_qpos, points)
                    qpos = correction.qpos
                    if not correction.tip_budget_satisfied:
                        relaxed_frames += 1
                    if not correction.mcp_guard_satisfied:
                        mcp_relaxed_frames += 1
                else:
                    qpos = raw_qpos
                if args.direct_qpos:
                    hand.hand.set_qpos(hand.convert_user_order_to_sim_order(qpos))
                else:
                    controller.advance(qpos)
                if direct is not None:
                    # Physics may move this collision-free display articulation between
                    # frames; overwrite it only after stepping, with the raw model output.
                    comparison_qpos = (cached_wuji[frame_index] if cached_wuji is not None else
                                       reference.forward(points) if reference is not None else raw_qpos)
                    direct.set_qpos(hand.convert_user_order_to_sim_order(comparison_qpos))
                    direct.set_qvel(np.zeros_like(qpos))
                human.update(points @ transform[:3, :3].T + transform[:3, 3])
            if viewer.window.key_press('r'):
                reset_camera(viewer, frame_pose=center_pose, distance=camera_distance, orthographic=compare)
            hand.scene.update_render()
            render_frame(viewer, camera_distance if compare else None)
            if args.fps is not None:
                time.sleep(max(0, 1 / args.fps - (time.monotonic() - frame_start)))
    except KeyboardInterrupt:
        pass
    finally:
        if corrector is not None:
            print(f'Posture correction: {relaxed_frames} frames exceeded the tip budget; '
                  f'{mcp_relaxed_frames} frames relaxed the MCP preference to preserve continuity.')
        viewer.close()


if __name__ == '__main__':
    main()
