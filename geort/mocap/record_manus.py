"""Guided right-Manus raw capture. Imports no robot, retargeter, or ROS module."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
import numpy as np
from geort.mocap.manus_capture_data import CaptureWriter, export_static, mp_mapping, node_records, write_json

PLAN=Path(__file__).with_name('manus_capture_plan.json')
DEFAULT_MODULE=Path.home()/'lerobot-xensehand/third_party/manussdk/build/python'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sdk(module_dir):
    module_dir=Path(module_dir).expanduser().resolve()
    if not module_dir.is_dir():raise FileNotFoundError(f'Manus module directory missing: {module_dir}')
    sys.path.insert(0,str(module_dir));module=importlib.import_module('manus_glove')
    if not Path(module.__file__).resolve().is_relative_to(module_dir):raise ValueError('Imported Manus module from unexpected directory')
    for name in ('connect','disconnect','is_connected','get_raw_skeleton','get_node_info','get_glove_id'):
        if not hasattr(module.ManusGlove,name):raise ValueError(f'Manus binding lacks {name}')
    return module


def record_clip(glove,path,nodes,rows,metadata,seconds,rate):
    writer=CaptureWriter(path,nodes,rows,metadata,chunk_size=max(1,int(rate*5)))
    start=time.monotonic();deadline=start+seconds;next_sample=start;next_log=start+5
    status='complete';invalid=repeated=0
    try:
        while time.monotonic()<deadline:
            delay=next_sample-time.monotonic()
            if delay>0:time.sleep(min(delay,max(0,deadline-time.monotonic())))
            if time.monotonic()>=deadline:break
            raw=glove.get_raw_skeleton('right')
            now=time.monotonic()
            ok,same,reason=writer.add(raw,now,time.time(),source_connected=bool(glove.is_connected()))
            invalid+=not ok;repeated+=same
            next_sample=max(next_sample+1/rate,time.monotonic())
            if now>=next_log:
                print(f"  剩余 {max(0,deadline-now):.0f}s | 读取 {writer.count} | 无效 {invalid} | 完全相同读数 {repeated}",flush=True)
                next_log=now+5
                # Node mapping must not silently change during recording.
                if node_records(glove.get_node_info('right'))!=nodes:raise RuntimeError('Manus node topology changed; reconnect in a new session')
    except KeyboardInterrupt:
        status='interrupted';raise
    except BaseException:
        status='error';raise
    finally:
        writer.finish(status)
        if writer.count:
            output=export_static(path)
            print(f'已保存：{path}；静态骨骼：{output}',flush=True)
    return {'status':status,'poll_count':writer.count,'invalid_polls':invalid,'identical_polls':repeated}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list',action='store_true',help='Print collection content without importing SDK')
    parser.add_argument('--check',action='store_true',help='Check local SDK import only; never instantiate/connect glove')
    parser.add_argument('--plan',choices=['pilot','full'],default='pilot')
    parser.add_argument('--tasks',nargs='+',help='Explicit subset of task IDs, in this order')
    parser.add_argument('--module-dir',type=Path,default=DEFAULT_MODULE)
    parser.add_argument('--mode',choices=['integrated','local','remote'],default='integrated')
    parser.add_argument('--output-root',type=Path,default=Path('data/manus'))
    parser.add_argument('--session',help='Unique recording session ID; existing directories are never overwritten')
    parser.add_argument('--operator',help='Anonymous operator ID, e.g. op01')
    parser.add_argument('--split',choices=['pilot','train','validation','test'],default='pilot')
    parser.add_argument('--calibration-note',default='Existing Core calibration; not exported or verified by recorder')
    parser.add_argument('--calibration-file',type=Path,help='Explicitly apply this .mcal to right glove before recording; omitted keeps current calibration')
    parser.add_argument('--rate',type=float,default=60.,help='Host polling Hz, not asserted sensor frequency')
    parser.add_argument('--connect-timeout',type=int,default=10)
    parser.add_argument('--ready-timeout',type=float,default=20.)
    args=parser.parse_args()
    plan=json.loads(PLAN.read_text());wanted=args.tasks or (plan['pilot'] if args.plan=='pilot' else [t['id'] for t in plan['tasks']])
    lookup={t['id']:t for t in plan['tasks']}
    if not wanted or len(wanted)!=len(set(wanted)) or any(t not in lookup for t in wanted):parser.error('Task IDs must be distinct members of --list')
    tasks=[lookup[t] for t in wanted]
    if args.list:
        for task in tasks:print(f"{task['id']:16} {task['seconds']:3}s  {task['title']}\n  {task['instruction']}")
        print(f"净采集时间：{sum(t['seconds'] for t in tasks)} 秒，不含准备/休息。")
        return
    if not np.isfinite(args.rate) or not 1<=args.rate<=240 or args.connect_timeout<=0 or not np.isfinite(args.ready_timeout) or args.ready_timeout<=0:
        parser.error('Use finite rate 1..240 Hz and positive timeouts')
    if args.calibration_file and not args.calibration_file.is_file():parser.error('Calibration file not found')
    if not args.check:
        if not args.operator or not args.session or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.session):
            parser.error('Recording requires --operator and an alphanumeric --session (underscore/dash allowed)')
        if args.plan=='pilot' and args.split!='pilot':parser.error('Pilot plan must use pilot split; use --plan full for formal sessions')
    module=load_sdk(args.module_dir)
    if args.check:
        print(f'SDK导入通过：{module.__file__}\n尚未连接手套。Python接口无SDK帧号/采集时间；仅可核验主机读取时间。')
        return
    session=args.output_root/args.session
    session.mkdir(parents=True,exist_ok=False)
    manifest={'session_id':args.session,'operator':args.operator,'split':args.split,'hand':'right',
        'source_type':'manus_glove_raw','world_coordinates':True,'coordinate_system':'SDK RIGHT_HANDED Z_UP',
        'sdk_module':str(Path(module.__file__).resolve()),'sdk_module_sha256':sha(module.__file__),
        'calibration_note':args.calibration_note,'calibration_applied':False,'host_requested_poll_hz':args.rate,
        'sdk_acquisition_timing_available':False,'created_utc':datetime.now(timezone.utc).isoformat(),
        'source_hashes':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('manus_capture_data.py'),PLAN]},
        'plan':tasks,'takes':[],'status':'connecting'}
    write_json(session/'session.json',manifest)
    glove=None
    try:
        glove=module.ManusGlove()
        if not glove.connect(mode=args.mode,world_coordinates=True,timeout_seconds=args.connect_timeout):raise RuntimeError('Manus connect returned False')
        deadline=time.monotonic()+args.ready_timeout
        while not glove.get_glove_id('right'):
            if time.monotonic()>=deadline:raise RuntimeError('Right glove not detected')
            time.sleep(.1)
        manifest['glove_id']=int(glove.get_glove_id('right'))
        if args.calibration_file:
            blob=args.calibration_file.read_bytes();code=int(glove.set_calibration('right',blob))
            if code!=1:raise RuntimeError(f'Calibration application failed: {code}')
            with (session/'Right.mcal').open('xb') as f:f.write(blob)
            manifest.update(calibration_applied=True,calibration_sha256=hashlib.sha256(blob).hexdigest())
        # Poll topology after calibration and require a stable, complete raw skeleton.
        deadline=time.monotonic()+args.ready_timeout
        while True:
            raw=np.asarray(glove.get_raw_skeleton('right'));nodes=node_records(glove.get_node_info('right'))
            if len(nodes) and raw.shape==(len(nodes),10):
                rows=mp_mapping(nodes);break
            if time.monotonic()>=deadline:raise RuntimeError('Complete raw skeleton/node mapping unavailable')
            time.sleep(.1)
        manifest.update(status='ready',node_info=nodes,mp_rows=rows.tolist())
        write_json(session/'session.json',manifest)
        print('只读取右手骨骼。请先确认Core校准、手套跟踪正常；每项按回车开始，输入q结束。',flush=True)
        for task in tasks:
            print(f"\n[{task['id']}] {task['title']}，{task['seconds']} 秒\n{task['instruction']}",flush=True)
            answer=input('准备好后按回车（q退出）：').strip().lower()
            if answer=='q':manifest['status']='stopped';break
            for i in range(3,0,-1):print(i,flush=True);time.sleep(1)
            take={**task,'directory':task['id'],'status':'recording'};manifest['takes'].append(take)
            write_json(session/'session.json',manifest)
            try:
                take.update(record_clip(glove,session/task['id'],nodes,rows,
                    {**{k:v for k,v in manifest.items() if k not in ('plan','takes','status')},
                     'task':task['id'],'instruction':task['instruction'],'requested_seconds':task['seconds']},task['seconds'],args.rate))
            except BaseException:
                take['status']='interrupted_or_error';raise
            write_json(session/'session.json',manifest)
        else:manifest['status']='complete'
    except (KeyboardInterrupt,EOFError):
        manifest['status']='interrupted';print('\n已停止，保留已录制内容。',flush=True)
    except BaseException as error:
        manifest.update(status='error',error=str(error));raise
    finally:
        write_json(session/'session.json',manifest)
        # Flush recording and manifests before native disconnect (may block).
        if glove is not None:glove.disconnect()
    print(f'会话保存于 {session}。下一步运行 inspect_manus_capture 检查；completed不等于质量合格。')


if __name__=='__main__':main()
