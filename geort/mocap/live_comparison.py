"""Live right-Manus comparison scene; simulation visuals only, no hand driver."""
import argparse
from collections import deque
import contextlib
import io
import json
from pathlib import Path
import queue
import selectors
import subprocess
import threading
import time

import numpy as np

from geort.mocap.manus_capture_data import canonicalize, mp_mapping, node_records
from geort.mocap.record_manus import DEFAULT_MODULE, load_sdk

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = 'wuji_hand2_beta1_right_2026-09-15_18-01-42_std_collision_seed0_w0'


def validate_q(q, config):
    q = np.asarray(q, dtype=np.float64)
    low, high = np.array(config['joint']['lower']), np.array(config['joint']['upper'])
    if q.shape != low.shape or not np.isfinite(q).all():
        raise ValueError('Invalid joint output')
    if np.any(q < low-1e-5) or np.any(q > high+1e-5):
        raise ValueError('Joint output outside physical limits; no silent clipping')
    return q


class ManusSource:
    def __init__(self, module_dir, mode, timeout=20):
        self.glove = None
        try:
            module = load_sdk(module_dir)
            self.glove = module.ManusGlove()
            if not self.glove.connect(mode=mode, world_coordinates=True, timeout_seconds=10):
                raise RuntimeError('Manus connection failed')
            deadline = time.monotonic()+timeout
            while time.monotonic() < deadline:
                if self.glove.get_glove_id('right'):
                    nodes = node_records(self.glove.get_node_info('right'))
                    raw = np.asarray(self.glove.get_raw_skeleton('right'))
                    if nodes and raw.shape == (len(nodes),10):
                        self.rows, self.nodes = mp_mapping(nodes), nodes
                        self.glove_id = int(self.glove.get_glove_id('right'))
                        self.last_topology = time.monotonic()
                        try:self.read()
                        except ValueError:
                            time.sleep(.1);continue
                        return
                time.sleep(.1)
            raise RuntimeError('Right glove / valid skeleton not ready; check Manus Core')
        except BaseException:
            self.close(); raise

    def read(self):
        if not self.glove.is_connected() or not self.glove.get_glove_id('right'):
            raise ValueError('Right Manus disconnected')
        if int(self.glove.get_glove_id('right')) != self.glove_id:
            raise RuntimeError('Right glove identity changed; restart the demo')
        if time.monotonic()-self.last_topology > 2:
            if node_records(self.glove.get_node_info('right')) != self.nodes:
                raise RuntimeError('Manus node topology changed; restart the demo')
            self.last_topology = time.monotonic()
        raw = np.asarray(self.glove.get_raw_skeleton('right'))
        if raw.shape != (len(self.nodes),10) or not np.isfinite(raw).all():
            raise ValueError('Invalid raw Manus skeleton')
        if np.any(abs(np.linalg.norm(raw[:,3:7],axis=-1)-1) > .1):
            raise ValueError('Invalid Manus quaternion')
        return canonicalize(raw[self.rows,:3])[0]

    def close(self):
        if self.glove is not None:
            self.glove.disconnect(); self.glove = None


class WujiWorker:
    def __init__(self, python, source, config, joint_config, sdk=False):
        cmd = ([str(python), '-u', str(ROOT/'scripts/wuji_sdk_worker.py'), '--joint-config',str(joint_config)] if sdk else
               [str(python), '-u', str(ROOT/'scripts/wuji_live_worker.py'), '--source',str(source),
               '--config',str(config), '--joint-config',str(joint_config),
               '--spec',str(ROOT/'geort/baselines/wuji_manus_right.json')])
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1, cwd=ROOT)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            ready = self.receive(30)
            if not ready.get('ready'): raise RuntimeError('Wuji worker did not initialize')
            self.metadata = ready
        except BaseException:
            self.close(); raise

    def receive(self, timeout):
        if not self.selector.select(timeout): raise TimeoutError('Wuji worker timed out')
        line = self.process.stdout.readline()
        if not line: raise RuntimeError('Wuji worker exited')
        reply = json.loads(line)
        if 'error' in reply: raise RuntimeError(reply['error'])
        return reply

    def solve(self, seq, points, reset=False, timeout=3):
        self.process.stdin.write(json.dumps({'seq':seq,'points':points.tolist(),'reset':reset})+'\n')
        self.process.stdin.flush()
        reply = self.receive(timeout)
        if reply.get('seq') != seq: raise RuntimeError('Wuji response/input sequence mismatch')
        return reply

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=2)
            except subprocess.TimeoutExpired: self.process.kill(); self.process.wait(timeout=2)
        self.selector.close()
        self.process.stdin.close(); self.process.stdout.close()


class Inference:
    def __init__(self, args):
        import torch
        from geort.coordination import robot_signature
        from geort.export import GeoRTRetargetingModel, resolve_checkpoint
        torch.set_num_threads(1)
        self.models = {}
        model_specs = [('M1',args.checkpoint,'best')]
        if not args.sdk_reference: model_specs.append(('Reference',args.reference,args.reference_weights))
        for name, tag, weights in model_specs:
            path = resolve_checkpoint(tag)
            with contextlib.redirect_stdout(io.StringIO()):
                self.models[name] = GeoRTRetargetingModel(path/f'{weights}.pth',path/'config.json',device='cpu')
        self.config = self.models['M1'].config
        if 'Reference' in self.models and robot_signature(self.models['Reference'].config) != robot_signature(self.config):
            raise ValueError('Model robot geometries differ')
        self.worker = WujiWorker(args.wuji_python, args.wuji_source, args.wuji_config,
                                 resolve_checkpoint(args.checkpoint)/'config.json')
        self.sdk_worker = None
        try:
            if args.sdk_reference:
                self.sdk_worker = WujiWorker(args.sdk_python, None, None,
                    resolve_checkpoint(args.checkpoint)/'config.json', sdk=True)
        except BaseException:
            self.worker.close(); raise
        self.filtered = None
        self.last_time = None

    def compute(self, seq, stamp, points, reset=False):
        start = time.monotonic()
        reset = reset or self.last_time is None or stamp-self.last_time > .5
        reply = self.worker.solve(seq,points,reset)
        self.last_time = stamp
        qs = {n:validate_q(m.forward(points),self.config) for n,m in self.models.items()}
        if self.sdk_worker:
            qs['Reference'] = validate_q(self.sdk_worker.solve(seq,points,reset)['q'],self.config)
        self.filtered = qs['M1'].copy() if reset else self.filtered+.3*(qs['M1']-self.filtered)
        qs.update(M1_lp=self.filtered.copy(), Wuji=validate_q(reply['filtered'],self.config),
                  Wuji_raw=validate_q(reply['raw'],self.config))
        return dict(seq=seq, stamp=stamp, points=points.copy(), q=qs,
                    solver_code=reply['solver_code'], compute_ms=(time.monotonic()-start)*1000)

    def close(self):
        self.worker.close()
        if self.sdk_worker: self.sdk_worker.close()


class Pipeline:
    """One pending input; completed packets contain all methods for the same read."""
    def __init__(self, engine):
        self.engine = engine; self.pending = queue.Queue(maxsize=1)
        self.lock = threading.Lock(); self.latest = None; self.error = None
        self.stop = threading.Event(); self.dropped = 0
        self.thread = threading.Thread(target=self.run, daemon=True); self.thread.start()

    def submit(self, packet):
        try: self.pending.put_nowait(packet)
        except queue.Full:
            try:
                old=self.pending.get_nowait(); self.dropped += 1
                if len(old)==4 and old[3] and len(packet)==4:
                    packet=(*packet[:3],True)  # Preserve resets when a pending read is replaced.
            except queue.Empty: pass
            self.pending.put_nowait(packet)

    def run(self):
        while not self.stop.is_set():
            try: packet = self.pending.get(timeout=.1)
            except queue.Empty: continue
            try:
                result = self.engine.compute(*packet)
                with self.lock: self.latest = result
            except Exception as e:
                with self.lock: self.error = str(e)
                self.stop.set()

    def snapshot(self):
        with self.lock: return self.latest, self.error

    def close(self):
        self.stop.set(); self.thread.join(timeout=4)
        self.engine.close()
        if self.thread.is_alive(): self.thread.join(timeout=1)


def target_gap(elapsed):
    return .025+.015*np.sin(2*np.pi*elapsed/12)


class Scene:
    def __init__(self, config, gui=True, window_size=(1800,1000)):
        import sapien
        from sapien.utils import Viewer
        from geort.env.hand import HandKinematicModel
        from geort.mocap.replay_evaluation import build_direct_comparison, HumanHandViewer
        from geort.kinematics import FingerKinematics
        self.sapien = sapien; self.config = config
        self.window_size = tuple(window_size)
        with contextlib.redirect_stdout(io.StringIO()):
            self.hand = HandKinematicModel.build_from_config(config, render=True)
        self.scene = self.hand.scene
        self.base = self.hand.hand.get_root_pose()
        offsets = [-.39,-.13,.13,.39]
        self.poses = [self.base*sapien.Pose([0,y,0]) for y in offsets]
        self.hand.hand.set_root_pose(self.poses[3])
        self.robots = [build_direct_comparison(self.hand,config,self.poses[i]) for i in (1,2)] + [self.hand.hand]
        for robot in self.robots:
            robot.set_qpos(self.hand.convert_user_order_to_sim_order(np.zeros(20)))
        self.scene.set_ambient_light([.7,.7,.7])
        self.scene.add_directional_light(self.base.to_transformation_matrix()[:3,:3]@np.array([-1,0,-1]),[.8]*3,shadow=False)
        from sapien.utils.viewer.control_window import ControlWindow
        class CameraControls(ControlWindow):
            def get_ui_windows(self):return []
        self.viewer = Viewer(self.hand.renderer,resolutions=self.window_size,plugins=[CameraControls()]) if gui else None
        if self.viewer:
            self.viewer.set_scene(self.scene)
            self.human = HumanHandViewer(self.scene,self.viewer,[0,0,0])
        else:
            self.human = [self.sphere(.003,[.4,.7,.9],f'human_{i}') for i in range(21)]
        self.fks = [FingerKinematics(config,f['name']) for f in config['fingertip_link']]
        self.target = [[self.sphere(.004,[.2,.8,.4],f'target_{lane}_{i}') for i in range(2)] for lane in range(1,4)]
        self.markers = [[self.sphere(.0035,[1.,.7,.1],f'hold_{lane}_{i}') for i in range(5)] for lane in range(1,4)]
        self.hide_markers()
        self.traces = [deque(maxlen=90) for _ in range(3)]; self.trace_nodes=[]
        self.current_tips = None; self.current_q = None; self.current_points = None; self.last_packet = None
        self.frame = self.base*sapien.Pose([0,0,-.09]); self.distance=.95
        if self.viewer: self.reset_camera()

    def sphere(self, radius, color, name):
        b=self.scene.create_actor_builder();b.add_sphere_visual(radius=radius,material=color)
        return b.build_kinematic(name=name)

    def world(self, points, lane):
        t=self.poses[lane].to_transformation_matrix()
        return np.asarray(points)@t[:3,:3].T+t[:3,3]

    def reset_camera(self):
        from geort.mocap.replay_evaluation import reset_camera
        reset_camera(self.viewer,self.frame,self.distance,orthographic=True)

    def hide_markers(self):
        for row in self.markers:
            for actor in row: actor.set_pose(self.sapien.Pose([0,0,-10]))

    def freeze_reference(self):
        if self.current_tips is None: return
        for lane,row in enumerate(self.markers,1):
            for actor,tip in zip(row,self.current_tips[lane-1]):
                actor.set_pose(self.sapien.Pose(self.world(tip,lane)))

    def clear_traces(self):
        for trace in self.traces:trace.clear()
        if self.viewer:
            for node in self.trace_nodes:self.viewer.render_scene.remove_node(node)
        self.trace_nodes=[]

    def update(self, packet, filtered, traces):
        self.last_packet=packet
        points=packet['points'];q=packet['q']
        selected=[q['Reference'],q['Wuji'],q['M1_lp'] if filtered else q['M1']]
        self.current_q=selected;self.current_points=points
        for robot,pose in zip(self.robots,selected):
            robot.set_qpos(self.hand.convert_user_order_to_sim_order(pose))
        human_points=self.world(points,0)
        if self.viewer:self.human.update(human_points)
        else:
            for actor,p in zip(self.human,human_points):actor.set_pose(self.sapien.Pose(p))
        self.current_tips=np.array([[fk.forward(pose[fk.indices])[0] for fk in self.fks] for pose in selected])
        if not traces:self.clear_traces();return
        for lane,tips in enumerate(self.current_tips):self.traces[lane].append(self.world(tips,lane+1))
        if self.viewer:
            for node in self.trace_nodes:self.viewer.render_scene.remove_node(node)
            self.trace_nodes=[]
            for trace in self.traces:
                if len(trace)<2:continue
                p=np.array(trace);vertices=np.stack([p[:-1],p[1:]],axis=2).reshape(-1).astype(np.float32)
                colors=np.tile([.1,.65,.95,1.],len(vertices)//3).astype(np.float32)
                self.trace_nodes.append(self.viewer.render_scene.add_line_set(self.viewer.renderer_context.create_line_set(vertices,colors)))

    def update_target(self, gap, show=True):
        for lane,row in enumerate(self.target,1):
            for i,actor in enumerate(row):
                p=self.world([.025,(i-.5)*gap,-.025],lane) if show else [0,0,-10]
                actor.set_pose(self.sapien.Pose(p))

    def snapshot(self, path):
        from scipy.spatial.transform import Rotation
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        camera=self.scene.add_camera('live_demo_snapshot',*self.window_size,.8,.01,10)
        top=self.distance*np.tan(.4)
        right=top*self.window_size[0]/self.window_size[1]
        # RenderCamera's three-argument overload uses a square frustum, unlike
        # the viewer window. Specify aspect explicitly so side lanes stay visible.
        camera.set_orthographic_parameters(.01,10,-right,right,-top,top)
        pos=np.array([self.distance,0,.36]);target=np.array([0,0,.20]);f=target-pos;f/=np.linalg.norm(f)
        left=np.cross([0,0,1],f);left/=np.linalg.norm(left);up=np.cross(f,left)
        q=Rotation.from_matrix(np.column_stack([f,left,up])).as_quat()
        camera.set_entity_pose(self.frame*self.sapien.Pose(pos,q[[3,0,1,2]]))
        self.scene.update_render();camera.take_picture()
        from PIL import Image
        Image.fromarray((np.clip(camera.get_picture('Color')[:,:,:3],0,1)*255).astype(np.uint8)).save(path)

    def close(self):
        if self.viewer and not self.viewer.closed:self.viewer.close()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='stage4_M1_seed42')
    p.add_argument('--reference',default=ORIGINAL)
    p.add_argument('--reference-weights',choices=['best','last'],default='last')
    p.add_argument('--sdk-reference',action='store_true',help='Show SDK 2026.8.31 in reference column; retain production Wuji and M1')
    p.add_argument('--sdk-python',type=Path,default=ROOT/'.venv-wuji-sdk/bin/python')
    p.add_argument('--module-dir',type=Path,default=DEFAULT_MODULE)
    p.add_argument('--mode',choices=['integrated','local','remote'],default='integrated')
    p.add_argument('--wuji-python',type=Path,default=ROOT/'.venv-wuji-baseline/bin/python')
    p.add_argument('--wuji-source',type=Path,default=Path.home()/'lerobot-xensehand/third_party/wuji-retargeting')
    p.add_argument('--wuji-config',type=Path,default=Path.home()/'lerobot-xensehand/configs/manus_wuji/adaptive_analytical_manus_wuji_hand_2_right.yaml')
    p.add_argument('--fps',type=float,default=60)
    p.add_argument('--window-size',type=int,nargs=2,metavar=('WIDTH','HEIGHT'),default=(1800,1000),help='Viewer size in pixels (default: 1800 1000)')
    p.add_argument('--m1-filter',choices=['none','lp03'],default='none')
    p.add_argument('--replay',type=Path,help='Recorded canonical keypoints instead of live SDK input; clearly labeled')
    p.add_argument('--headless',action='store_true',help='Offscreen replay verification only')
    p.add_argument('--frames',type=int,default=0,help='Exit after N displayed input frames; 0 means interactive')
    p.add_argument('--snapshot',type=Path,help='Save an offscreen image on exit, without overwriting')
    p.add_argument('--check',action='store_true',help='Check SDK import/models/native solver with saved validation input; do not connect glove')
    return p


def main():
    args=parser().parse_args()
    if not np.isfinite(args.fps) or not 1<=args.fps<=120:raise ValueError('fps must be 1..120')
    if args.frames<0:raise ValueError('frames must be nonnegative')
    if not (1024<=args.window_size[0]<=7680 and 640<=args.window_size[1]<=4320):
        raise ValueError('Window size requires width 1024..7680 and height 640..4320')
    if args.headless and (not args.replay or not args.frames):raise ValueError('Headless requires explicit replay and positive frames')
    if args.snapshot and args.snapshot.exists():raise FileExistsError(args.snapshot)
    engine=Inference(args);source=scene=pipeline=None
    try:
        if args.check:
            module=load_sdk(args.module_dir)
            points=np.load(ROOT/'data/manus/val01/neutral/keypoints.npy',allow_pickle=False)[0]
            result=engine.compute(0,time.monotonic(),points,True)
            print(f'CHECK OK: SDK {module.__file__}; models + native Wuji solver OK; glove NOT connected; solver status={result["solver_code"]}')
            if engine.sdk_worker: print('Wuji SDK reference:',json.dumps(engine.sdk_worker.metadata))
            return
        replay=None
        if args.replay:
            replay=np.load(args.replay,allow_pickle=False)
            if replay.ndim!=3 or replay.shape[1:]!=(21,3) or not len(replay) or not np.isfinite(replay).all():
                raise ValueError('Replay must be finite canonical (T,21,3) keypoints')
        else:
            print('Connecting right Manus (existing calibration; simulation only)...',flush=True)
            source=ManusSource(args.module_dir,args.mode)
        scene=Scene(engine.config,gui=not args.headless,window_size=args.window_size)
        if args.headless:
            for i in range(args.frames):
                result=engine.compute(i,i/args.fps,replay[i%len(replay)],i%len(replay)==0)
                scene.update(result,args.m1_filter=='lp03',False);scene.update_target(target_gap(i/args.fps))
            if args.snapshot:scene.snapshot(args.snapshot)
            print(f'HEADLESS REPLAY OK: {args.frames} synchronized frames; no SDK connection; no physical hand')
            return
        from sapien.utils.viewer.plugin import Plugin
        import sapien
        ui=sapien.internal_renderer
        class DemoPanel(Plugin):
            def __init__(self):
                self.filtered=args.m1_filter=='lp03';self.traces=False;self.targets=False;self.message='Waiting for first input'
                self.metrics='';self.target_mm=25.;self.window=None
            def toggle_filter(self,_):
                self.filtered=not self.filtered;scene.clear_traces();scene.hide_markers()
                if scene.last_packet is not None:scene.update(scene.last_packet,self.filtered,False)
            def toggle_traces(self,_):self.traces=not self.traces;scene.clear_traces()
            def toggle_targets(self,_):self.targets=not self.targets
            def get_ui_windows(self):
                self.window=ui.UIWindow().Label('Manus live retargeting - SIMULATION ONLY').Pos(8,8).Size(1010,225).append(
                    ui.UIDisplayText().Text('LEFT -> RIGHT: Human | '+('Wuji SDK 2026.8.31' if args.sdk_reference else 'Original GeoRT' if args.reference==ORIGINAL else args.reference)+' | Wuji LP=0.3 | '+args.checkpoint),
                    ui.UIDisplayText().Text(self.message),
                    ui.UIDisplayText().Text('M1 filter: '+('LP=0.3' if self.filtered else 'OFF')+' | SDK acquisition freshness unavailable'),
                    ui.UIDisplayText().Text(self.metrics),
                    ui.UIDisplayText().Text(f'Opening target: {self.target_mm:.1f} mm (visual guide, no contact physics)' if self.targets else 'Direct joint display. No PD, contact simulation, or hardware hand driver.'),
                    ui.UISameLine().append(ui.UIButton().Label('F: Freeze tip reference').Callback(lambda _:scene.freeze_reference()),
                        ui.UIButton().Label('T: Tip trails').Callback(self.toggle_traces),ui.UIButton().Label('L: M1 filter').Callback(self.toggle_filter),
                        ui.UIButton().Label('G: Moving gap target').Callback(self.toggle_targets),ui.UIButton().Label('R: Reset view').Callback(lambda _:scene.reset_camera())),
                    ui.UIDisplayText().Text('SPACE: pause display | C: clear reference/trails | ESC: quit | Targets and gold reference dots are visual guides.'))
                return [self.window]
        panel=DemoPanel();panel.init(scene.viewer);scene.viewer.plugins.append(panel)
        pipeline=Pipeline(engine);last_seq=-1;seq=0;count=0;paused=False;start=time.monotonic();last_input=start;last_log=start
        latest=None;status='';display_times=deque(maxlen=60)
        print('Live simulation: LEFT Human | '+('Wuji SDK 2026.8.31' if args.sdk_reference else 'Original/reference')+' | production Wuji | candidate. F/T/L/G/R/C/SPACE/ESC. No physical hand driver.',flush=True)
        while not scene.viewer.closed:
            now=time.monotonic();window=scene.viewer.window
            if window.key_press('esc'):break
            if window.key_press('space'):paused=not paused
            if window.key_press('r'):scene.reset_camera()
            if window.key_press('f'):scene.freeze_reference()
            if window.key_press('t'):panel.toggle_traces(None)
            if window.key_press('l'):panel.toggle_filter(None)
            if window.key_press('g'):panel.toggle_targets(None)
            if window.key_press('c'):scene.clear_traces();scene.hide_markers()
            if not paused:
                try:
                    points=source.read() if source else replay[seq%len(replay)]
                    reset=now-last_input>.5 or (replay is not None and seq%len(replay)==0)
                    pipeline.submit((seq,now,points,reset));seq+=1;last_input=now;status=''
                except ValueError as e:status='HOLD: '+str(e)
                latest,error=pipeline.snapshot()
                if error:raise RuntimeError(error)
                if latest and not status and now-latest['stamp']<.5:
                    if latest['seq']!=last_seq:
                        scene.update(latest,panel.filtered,panel.traces);last_seq=latest['seq'];count+=1;display_times.append(now)
                elif not status:status='WAITING: no recent completed input'
            rate=(len(display_times)-1)/(display_times[-1]-display_times[0]) if len(display_times)>1 else 0
            age=(now-latest['stamp'])*1000 if latest else 0
            panel.message=('PAUSED' if paused else status or ('LIVE right Manus' if source else 'RECORDED REPLAY - not live'))+f' | completed {rate:.1f} Hz | host read age {age:.0f} ms | dropped {pipeline.dropped}'
            if latest and latest['solver_code']<0:panel.message+=' | WUJI SOLVER FALLBACK'
            if scene.current_tips is not None:
                gaps=np.linalg.norm(scene.current_tips[:,1]-scene.current_tips[:,0],axis=-1)*1000
                panel.metrics='Thumb-index site gap (mm): Ref %.1f | Wuji %.1f | M1 %.1f | compute %.1f ms'%(*gaps,latest['compute_ms'])
            elapsed=now-start;panel.target_mm=target_gap(elapsed)*1000
            scene.update_target(panel.target_mm/1000,panel.targets and not status and not paused)
            scene.scene.update_render()
            from geort.mocap.replay_evaluation import render_frame
            render_frame(scene.viewer,scene.distance)
            if now-last_log>5:
                print(panel.message+' | '+panel.metrics,flush=True);last_log=now
            if args.frames and count>=args.frames:break
            time.sleep(max(0,1/args.fps-(time.monotonic()-now)))
        if args.snapshot:scene.snapshot(args.snapshot)
        print(f'GUI completed: {count} synchronized displayed frames; projection={scene.viewer.window.camera_mode if not scene.viewer.closed else "closed"}',flush=True)
    except KeyboardInterrupt:pass
    finally:
        if pipeline:pipeline.close()
        else:engine.close()
        if source:source.close()
        if scene:scene.close()


if __name__=='__main__':main()
