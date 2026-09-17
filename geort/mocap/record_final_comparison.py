"""Record all frozen test clips: four methods, direct targets above actual PD."""
import argparse
import concurrent.futures
import contextlib
import io
import json
import multiprocessing
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geort.evaluate_differences import sha
from geort.mocap.live_comparison import ROOT
from geort.mocap.replay_pd_comparison import PDComparison, load_frozen

NAMES=('Original','R2_lp03','SDK','Wuji')
LABELS=('主分支 GeoRT（重训）','最终 GeoRT R2','Wuji SDK 2026.8.31','wuji-retargeting')
FILTERS=('train01 · seed42 · best · 无滤波','固定 R2 checkpoint · LP=0.3','内置状态 / 滤波参数未公开','生产配置 · LP=0.3')
TASKS={'neutral':'基础手型','little_branch':'小指构型','index_branch':'食指构型','ring_branch':'无名指构型',
       'pinch_axis':'捏合与方向','micro':'小幅动作','context':'多指上下文','open_close':'张手与握拳','wrist':'手腕动作'}
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


class FourHandCamera:
    def __init__(self,cfg):
        import sapien
        from scipy.spatial.transform import Rotation
        from geort.env.hand import HandKinematicModel
        from geort.mocap.replay_evaluation import build_direct_comparison
        with contextlib.redirect_stdout(io.StringIO()):self.hand=HandKinematicModel.build_from_config(cfg,render=True)
        self.scene=self.hand.scene;base=self.hand.hand.get_root_pose()
        poses=[base*sapien.Pose([0,y,0]) for y in (-.45,-.15,.15,.45)]
        self.hand.hand.set_root_pose(poses[0])
        self.robots=[self.hand.hand]+[build_direct_comparison(self.hand,cfg,p) for p in poses[1:]]
        self.scene.set_ambient_light([.65,.65,.65])
        rotation=base.to_transformation_matrix()[:3,:3]
        self.scene.add_directional_light(rotation@np.array([-1.,0,-1]),[.85]*3,shadow=False)
        self.camera=self.scene.add_camera('four_hand_capture',1920,430,.8,.01,10)
        right=.64;top=right*430/1920
        self.camera.set_orthographic_parameters(.01,10,-right,right,-top,top)
        pos=np.array([.95,0,.225]);target=np.array([0,0,.065]);f=target-pos;f/=np.linalg.norm(f)
        left=np.cross([0,0,1],f);left/=np.linalg.norm(left);up=np.cross(f,left)
        q=Rotation.from_matrix(np.column_stack([f,left,up])).as_quat()
        self.camera.set_entity_pose(base*sapien.Pose(pos,q[[3,0,1,2]]))

    def image(self,poses):
        for robot,name in zip(self.robots,NAMES):robot.set_qpos(self.hand.convert_user_order_to_sim_order(poses[name]))
        self.scene.update_render();self.camera.take_picture()
        return Image.fromarray((np.clip(self.camera.get_picture('Color')[:,:,:3],0,1)*255).astype(np.uint8))


def render_clip(job):
    cfg,clip,index,arrays,output,limit=job
    output=Path(output);task=clip['task'];total=clip['range'][1]-clip['range'][0];count=min(total,limit) if limit else total
    path=output/f'{index+1:02d}_{task}.mp4';partial=output/f'{index+1:02d}_{task}.partial.mp4'
    if path.exists() or partial.exists():raise FileExistsError(path)
    physics=PDComparison(cfg,names=NAMES);camera=FourHandCamera(cfg)
    font=ImageFont.truetype(FONT,25);small=ImageFont.truetype(FONT,19);subtitle=ImageFont.truetype(FONT,16);title=ImageFont.truetype(FONT,30)
    colors=['#627084','#287953','#8164ac','#b07434']
    background=Image.new('RGB',(1920,1080),'#f5f6f8');d=ImageDraw.Draw(background)
    for i,(label,fil) in enumerate(zip(LABELS,FILTERS)):
        x=i*480;d.rectangle((x,66,x+479,118),fill=colors[i]);d.text((x+16,70),label,font=font,fill='white',anchor='lt');d.text((x+16,99),fil,font=subtitle,fill='white',anchor='lt')
    d.text((16,120),'上排：模型目标关节姿态（直接输出）',font=small,fill='#172331',anchor='lt')
    d.text((16,575),'下排：PD 实际执行姿态（各方法相同控制参数）',font=small,fill='#172331',anchor='lt')
    d.text((16,1044),'完整 test01 · 60 Hz 回放假设 · PD 400 / 10，力矩上限 10 Nm · 控制 600 Hz / 物理 1200 Hz · 仿真，不代表实机参数',font=small,fill='#334255',anchor='lt')
    log=(output/f'{index+1:02d}_{task}_encode.log').open('x')
    encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','60','-i','pipe:0',
        '-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(partial)],stdin=subprocess.PIPE,stderr=log)
    actuals={n:[] for n in NAMES};depths={n:[] for n in NAMES};started=time.monotonic()
    try:
        for frame in range(count):
            target={n:arrays[n][frame] for n in NAMES};actual,depth=physics.advance(target)
            image=background.copy();image.paste(camera.image(target),(0,143));image.paste(camera.image(actual),(0,600));draw=ImageDraw.Draw(image)
            draw.text((16,10),f'{index+1:02d} / 09  {TASKS[task]}  |  {task}',font=title,fill='#172331')
            draw.text((1110,15),f'{frame+1:,} / {total:,} 帧     {frame/60:05.1f} / {total/60:05.1f} 秒',font=font,fill='#172331')
            for i,n in enumerate(NAMES):
                error=np.rad2deg(abs(actual[n]-target[n])).mean()
                draw.text((i*480+12,1005),f'关节跟踪 MAE {error:.2f}°',font=small,fill='#23364b',anchor='lt')
                actuals[n].append(actual[n]);depths[n].append(depth[n])
            encoder.stdin.write(image.tobytes())
            if frame in (0,count//2,count-1):image.save(output/f'{index+1:02d}_{task}_frame{frame:05d}.jpg',quality=90)
            if (frame+1)%300==0:print(json.dumps({'task':task,'frames':frame+1,'total':count,'wall_seconds':round(time.monotonic()-started,1)}),flush=True)
        encoder.stdin.close()
        if encoder.wait()!=0:raise RuntimeError('Video encoder failed: '+str(path))
        partial.rename(path)
    finally:
        if encoder.poll() is None:encoder.kill();encoder.wait()
        log.close()
    np.savez_compressed(output/f'{index+1:02d}_{task}_pd.npz',source_ids=np.arange(clip['range'][0],clip['range'][0]+count),
        **actuals,**{n+'__depth_mm':v for n,v in depths.items()})
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries',
        'stream=width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(path)],text=True))['streams'][0]
    if int(probe['nb_read_frames'])!=count or probe['r_frame_rate']!='60/1':raise RuntimeError('Encoded frame count/rate mismatch')
    result={'task':task,'video':path.name,'input_range':clip['range'],'frames':count,'complete':count==total,'video_sha256':sha(path),'probe':probe,
            'wall_seconds':round(time.monotonic()-started,2)}
    (output/f'{index+1:02d}_{task}.json').write_text(json.dumps(result,indent=2))
    print('VIDEO COMPLETE',path,flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=ROOT/'reports/stage6/test_full')
    p.add_argument('--main-reference',type=Path,default=ROOT/'reports/final_main_reference')
    p.add_argument('--manifest',type=Path,default=ROOT/'data/manus_stage4_v1/manifest.json')
    p.add_argument('--tasks',nargs='+',help='Omit for all 9 clips; explicit subset only for code checks')
    p.add_argument('--frames',type=int,default=0,help='0=all; positive=explicit short code check')
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if args.frames<0 or args.workers not in (1,2):raise ValueError('Invalid recording budget')
    data,archive,cfg=load_frozen(args.cache,args.manifest)
    frozen=json.loads((args.cache/'protocol.json').read_text())
    r2_weights=ROOT/'checkpoint/stage5_R2_seed42/best.pth'
    if sha(r2_weights)!=frozen['files_sha256'][str(r2_weights)]:raise ValueError('R2 checkpoint changed')
    reference=json.loads((args.main_reference/'protocol.json').read_text())
    for key,digest in [('checkpoint','checkpoint_sha256'),('config','config_sha256')]:
        if sha(reference[key])!=reference[digest]:raise ValueError('Main checkpoint/config changed')
    if reference['manifest_sha256']!=sha(args.manifest) or reference['frames']!=len(data['keypoints']):raise ValueError('Main reference dataset mismatch')
    if sha(args.main_reference/'outputs.npz')!=reference['outputs_sha256']:raise ValueError('Main reference cache changed')
    main_archive=np.load(args.main_reference/'outputs.npz',allow_pickle=False)
    if args.tasks and not set(args.tasks).issubset(TASKS):raise ValueError('Unknown task')
    args.output.mkdir(parents=True,exist_ok=False)
    metadata={'input_cache_sha256':sha(args.cache/'outputs.npz'),'manifest_sha256':sha(args.manifest),
        'main_reference':reference,'methods':dict(zip(NAMES,LABELS)),'filters':dict(zip(NAMES,FILTERS)),'test_frames':len(data['keypoints']),
        'input_fps_assumption':60,'video_fps':60,'video_resolution':[1920,1080],'pd':{'kp':400,'kd':10,'force_limit_nm':10,'control_hz':600,'physics_hz':1200},
        'reset':'all hands same clipped-zero state at clip start; no omitted startup frames',
        'layout':'columns Original/R2/SDK/Wuji; top targets, bottom independent PD actual states; identical assets and orthographic camera',
        'scope':'full static test recordings; no glove/robot connections; frozen models; no training',
        'subset_for_check':args.tasks,'frame_limit_for_check':args.frames,
        'source_sha256':{str(path):sha(path) for path in [Path(__file__),ROOT/'geort/mocap/replay_pd_comparison.py',ROOT/'geort/mocap/replay_evaluation.py',ROOT/'geort/env/hand.py']}}
    (args.output/'protocol.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False));(args.output/'recorder_source.py').write_bytes(Path(__file__).read_bytes())
    jobs=[]
    for i,clip in enumerate(data['entry']['clips']):
        if args.tasks and clip['task'] not in args.tasks:continue
        arrays={n:(main_archive if n=='Original' else archive)[clip['task']+'__'+n] for n in NAMES}
        for q in arrays.values():
            if q.shape!=(clip['range'][1]-clip['range'][0],20) or not np.isfinite(q).all():raise ValueError('Invalid targets')
        jobs.append((cfg,clip,i,arrays,str(args.output),args.frames))
    if args.workers==1:results=[render_clip(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            results=list(pool.map(render_clip,jobs))
    (args.output/'videos.json').write_text(json.dumps(results,indent=2))
    if not args.frames and not args.tasks:
        if sum(r['frames'] for r in results)!=len(data['keypoints']) or not all(r['complete'] for r in results):raise RuntimeError('Incomplete test recording')
        (args.output/'concat.txt').write_text(''.join(f"file '{r['video']}'\n" for r in results))
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','concat','-safe','0','-i',str(args.output/'concat.txt'),
            '-c','copy','-movflags','+faststart',str(args.output/'00_complete_test.mp4')],check=True)
    print('RECORDING FINISHED',args.output,flush=True)


if __name__=='__main__':main()
