"""Three independent PD simulations replaying the complete frozen test cache."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import time

import numpy as np

from geort.evaluate_differences import sha
from geort.manus_sessions import load_session_split
from geort.mocap.live_comparison import ROOT, Scene
from geort.mocap.replay_evaluation import ReplayController, render_frame

NAMES=('R2_lp03','Wuji','SDK')


class PDComparison:
    def __init__(self,config,fps=60,control_hz=600,physics_hz=1200,kp=400.,kd=10.,force_limit=10.,names=NAMES):
        from geort.env.hand import HandKinematicModel
        ReplayController.step_counts(fps,control_hz,physics_hz)
        for value in (kp,kd,force_limit):
            if not np.isfinite(value) or value<0:raise ValueError('PD gains and effort limit must be finite and nonnegative')
        self.names=tuple(names)
        if not self.names or len(set(self.names))!=len(self.names):raise ValueError('Method names must be unique')
        self.config=config;self.rates=(fps,control_hz,physics_hz);self.hands=[];self.controllers=[]
        for _ in self.names:
            with contextlib.redirect_stdout(io.StringIO()):hand=HandKinematicModel.build_from_config(config,render=False)
            for joint in hand.all_joints:joint.set_drive_property(kp,kd,force_limit=force_limit)
            self.hands.append(hand)
        self.reset()

    def reset(self):
        self.controllers=[]
        for hand in self.hands:
            q=np.clip(np.zeros(20),hand.joint_lower_limit+1e-3,hand.joint_upper_limit-1e-3)
            hand.hand.set_qpos(hand.convert_user_order_to_sim_order(q));hand.hand.set_qvel(np.zeros(20))
            hand.set_qpos_target(q)
            self.controllers.append(ReplayController(hand,*self.rates,interpolate=True))

    def advance(self,targets):
        actual={};depth={}
        for name,hand,controller in zip(self.names,self.hands,self.controllers):
            controller.advance(targets[name])
            q=hand.hand.get_qpos()[hand.user_idx_to_sim_idx].copy()
            if not np.isfinite(q).all():raise RuntimeError('Nonfinite PD state')
            actual[name]=q
            links=hand.hand.get_links();maximum=0.
            for contact in hand.scene.get_contacts():
                bodies=contact.bodies if hasattr(contact,'bodies') else (contact.actor0,contact.actor1)
                if all(body in links for body in bodies):
                    for point in contact.points:maximum=max(maximum,-float(point.separation)*1000)
            depth[name]=maximum
        return actual,depth


def load_frozen(cache,manifest):
    protocol=json.loads((cache/'protocol.json').read_text());meta=json.loads((cache/'inference.json').read_text())
    if protocol['split']!='test' or meta['split']!='test' or meta['outputs_sha256']!=sha(cache/'outputs.npz'):
        raise ValueError('Expected unchanged complete test cache')
    if protocol['files_sha256'][str(manifest.resolve())]!=sha(manifest):raise ValueError('Manifest changed')
    cfg_path=ROOT/'checkpoint/stage5_R2_seed42/config.json'
    if protocol['files_sha256'][str(cfg_path)]!=sha(cfg_path):raise ValueError('Robot configuration changed')
    recording=load_session_split(manifest,'test');archive=np.load(cache/'outputs.npz',allow_pickle=False)
    for clip in recording['entry']['clips']:
        a,b=clip['range']
        for name in NAMES:
            q=archive[clip['task']+'__'+name]
            if q.shape!=(b-a,20) or not np.isfinite(q).all():raise ValueError('Incomplete target cache')
    return recording,archive,json.loads(cfg_path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=ROOT/'reports/stage6/test_full')
    p.add_argument('--manifest',type=Path,default=ROOT/'data/manus_stage4_v1/manifest.json')
    p.add_argument('--fps',type=float,default=60.)
    p.add_argument('--control-hz',type=float,default=600.)
    p.add_argument('--physics-hz',type=float,default=1200.)
    p.add_argument('--kp',type=float,default=400.);p.add_argument('--kd',type=float,default=10.)
    p.add_argument('--force-limit',type=float,default=10.)
    p.add_argument('--window-size',nargs=2,type=int,default=[1800,1000])
    p.add_argument('--headless',action='store_true')
    p.add_argument('--frames',type=int,default=0,help='0: complete test once, then pause; positive: bounded verification')
    p.add_argument('--snapshot',type=Path)
    p.add_argument('--output',type=Path,help='Save every displayed PD state for reproducible offline analysis')
    args=p.parse_args()
    if args.frames<0:raise ValueError('frames must be nonnegative')
    if args.output and args.output.exists():raise FileExistsError(args.output)
    if args.snapshot and args.snapshot.exists():raise FileExistsError(args.snapshot)
    data,archive,cfg=load_frozen(args.cache,args.manifest)
    simulation=PDComparison(cfg,args.fps,args.control_hz,args.physics_hz,args.kp,args.kd,args.force_limit)
    scene=None
    if not args.headless or args.snapshot:scene=Scene(cfg,gui=not args.headless,window_size=args.window_size)
    clips=data['entry']['clips'];total=len(data['keypoints']);current=0;previous_clip=-1;count=0;paused=False;targets_view=False;done=False
    saved={n:[] for n in NAMES};depths={n:[] for n in NAMES};source_ids=[]
    last_actual=last_target=last_points=None
    print(f'PD REPLAY: all {len(clips)} test clips / {total} reads; Human | GeoRT R2 LP.3 | Wuji LP.3 | SDK',flush=True)
    print(f'Independent physics scenes; kp={args.kp}, kd={args.kd}, torque limit={args.force_limit} Nm; assumed input {args.fps}Hz, control {args.control_hz}Hz, physics {args.physics_hz}Hz. No hardware.',flush=True)
    def display(actual,points):
        if scene:
            scene.update({'points':points,'q':{'Reference':actual[NAMES[0]],'Wuji':actual[NAMES[1]],'M1':actual[NAMES[2]],'M1_lp':actual[NAMES[2]]}},False,False)
            scene.update_target(.025,False)
    if scene and scene.viewer:
        import sapien
        from sapien.utils.viewer.plugin import Plugin
        ui=sapien.internal_renderer
        class Panel(Plugin):
            status='';errors=''
            def get_ui_windows(self):
                return [ui.UIWindow().Label('FULL TEST - THREE PD REPLAYS - SIMULATION').Pos(8,8).Size(1200,175).append(
                    ui.UIDisplayText().Text('LEFT -> RIGHT: Human | GeoRT R2 LP=0.3 | Wuji LP=0.3 | SDK 2026.8.31'),
                    ui.UIDisplayText().Text(self.status),ui.UIDisplayText().Text(self.errors),
                    ui.UIDisplayText().Text(f'PD kp={args.kp:g}, kd={args.kd:g}, limit={args.force_limit:g} Nm | input/control/physics={args.fps:g}/{args.control_hz:g}/{args.physics_hz:g} Hz'),
                    ui.UIDisplayText().Text('SPACE pause | D actual PD / model targets | R camera | ESC quit. All 9 clips in order; no skipped frames.'))]
        panel=Panel();panel.init(scene.viewer);scene.viewer.plugins.append(panel)
    try:
        while True:
            start=time.monotonic()
            if scene and scene.viewer:
                if scene.viewer.closed or scene.viewer.window.key_press('esc'):break
                win=scene.viewer.window
                if win.key_press('space') and not done:paused=not paused
                if win.key_press('r'):scene.reset_camera()
                if win.key_press('d'):
                    targets_view=not targets_view
                    if last_actual is not None:display(last_target if targets_view else last_actual,last_points)
            if not paused and not done:
                clip_index=int(data['clip_id'][current]);clip=clips[clip_index];local=current-clip['range'][0]
                if clip_index!=previous_clip:
                    simulation.reset();previous_clip=clip_index
                    print('CLIP',clip_index+1,clip['task'],clip['range'],flush=True)
                targets={n:archive[clip['task']+'__'+n][local] for n in NAMES}
                actual,depth=simulation.advance(targets);points=data['keypoints'][current]
                last_actual,last_target,last_points=actual,targets,points
                display(targets if targets_view else actual,points)
                for n in NAMES:saved[n].append(actual[n]);depths[n].append(depth[n])
                source_ids.append(current);count+=1;current+=1
                if scene and scene.viewer:
                    panel.status=f'{clip_index+1}/9 {clip["task"]} | {current}/{total} | '+('MODEL TARGETS (PD still advances)' if targets_view else 'ACTUAL PD JOINT STATES')
                    panel.errors='Joint tracking MAE (deg): R2 %.2f | Wuji %.2f | SDK %.2f'%tuple(np.rad2deg(abs(actual[n]-targets[n])).mean() for n in NAMES)
                if count%500==0:print(f'PD states {count}/{total}',flush=True)
                if current==total:
                    done=True
                    if scene and scene.viewer:panel.status+=' | COMPLETE - paused'
                if args.frames and count>=args.frames:break
            if scene and scene.viewer:
                scene.scene.update_render();render_frame(scene.viewer,scene.distance)
                time.sleep(max(0,1/args.fps-(time.monotonic()-start)))
            elif done:break
        if args.snapshot:scene.snapshot(args.snapshot)
    finally:
        if scene:scene.close()
        if args.output:
            args.output.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(args.output/'actual.npz',source_ids=source_ids,**{n:np.asarray(v) for n,v in saved.items()},**{n+'__depth_mm':np.asarray(v) for n,v in depths.items()})
            (args.output/'protocol.json').write_text(json.dumps({'frames':count,'complete':count==total,'source_frames':total,
                'input_outputs_sha256':sha(args.cache/'outputs.npz'),'manifest_sha256':sha(args.manifest),'script_sha256':sha(__file__),
                'fps_assumption':args.fps,'control_hz':args.control_hz,'physics_hz':args.physics_hz,'kp':args.kp,'kd':args.kd,'force_limit_nm':args.force_limit,
                'reset':'common clipped zero joints and zero velocity at each clip; includes initial transient; no per-frame qpos teleport',
                'controller':'linear interpolation, no slew cap; existing drive clamp margin 0.001rad',
                'depth':'maximum contact penetration from final physics step per input interval',
                'scope':'independent identical physics scenes, no hardware, no acquisition-time or calibrated actuator claim'},indent=2))
        print(f'Completed {count} synchronized PD frames; no robot/glove connection.',flush=True)


if __name__=='__main__':main()
