"""Offline capture checks and skeleton-only preview; no retargeter/hardware."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from geort.mocap.manus_capture_data import export_static, load_capture


def summarize_capture(path):
    meta,data=load_capture(path);valid=data['valid_frame'];points=data['keypoints'][valid]
    times=data['host_mono_s'];dt=np.diff(times)
    report={'path':str(Path(path).resolve()),'session':meta['session_id'],'split':meta['split'],'task':meta['task'],
        'source_type':meta['source_type'],'status':meta['status'],'poll_count':len(valid),'valid_count':int(valid.sum()),
        'valid_fraction':float(valid.mean()),'identical_raw_fraction':float(data['same_raw_as_previous'].mean()),
        'host_duration_s':float(times[-1]-times[0]),'host_interval_p95_ms':float(np.percentile(dt,95)*1000) if len(dt) else None,
        'host_interval_max_ms':float(dt.max()*1000) if len(dt) else None,
        'sdk_freshness':'unknown: getter supplies no frame ID or capture timestamp; identical raw values are not proof of tracking loss',
        'invalid_reasons':{str(k):int(v) for k,v in zip(*np.unique(data['invalid_reason'][~valid],return_counts=True))},
        'information_pairs':{},'temporal_training_allowed':False,'warnings':[]}
    if meta['status']!='complete':report['warnings'].append('录制未正常完成，检查中断原因和已保存分块')
    if valid.mean()<.99:report['warnings'].append('有效率低于99%，先排查校准、节点映射或跟踪')
    if data['same_raw_as_previous'].mean()>.8:report['warnings'].append('完全相同读数超过80%；静止与缓存停更无法自动区分，请结合动作复查')
    if len(dt) and dt.max()>.25:report['warnings'].append('主机读取存在大于250 ms的间隔；不能据此推断传感器丢帧')
    if len(points)<2:return report
    # Uniform subsampling controls quadratic storage on long static clips.
    selected=np.linspace(0,len(points)-1,min(len(points),1200),dtype=int)
    p=points[selected];host_times=times[valid][selected]
    report['pair_search_frames']=len(p);report['pair_search_policy']='at most 1200 uniformly sampled valid polls; host separation >=0.5s; no acquisition-time claim'
    for f,end in enumerate((4,8,12,16,20)):
        tip=p[:,end];bone=tip-p[:,end-1];axis=bone/np.linalg.norm(bone,axis=1)[:,None]
        pairs=cKDTree(tip).query_pairs(.005,output_type='ndarray')
        pairs=pairs[(host_times[pairs[:,1]]-host_times[pairs[:,0]])>=.5]
        distance=np.linalg.norm(tip[pairs[:,0]]-tip[pairs[:,1]],axis=1)
        angles=np.rad2deg(np.arccos(np.clip((axis[pairs[:,0]]*axis[pairs[:,1]]).sum(1),-1,1)))
        strict=(distance<=.002)&(angles>=20);relaxed=(angles>=20)
        report['information_pairs'][('thumb','index','middle','ring','little')[f]]={
            'tip_2mm_axis_20deg_pairs':int(strict.sum()),'distinct_polls_in_strict_pairs':int(len(np.unique(pairs[strict]))),
            'tip_5mm_axis_20deg_pairs':int(relaxed.sum()),
            'max_axis_difference_at_2mm_deg':float(angles[distance<=.002].max()) if (distance<=.002).any() else None,
            'tip_excursion_mm':(np.ptp(tip,axis=0)*1000).tolist()}
    return report


def preview(path,fps=30):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    meta,data=load_capture(path);indices=np.flatnonzero(data['valid_frame'])
    if not len(indices):raise ValueError('No valid skeleton to preview')
    p=data['keypoints'][indices];start=data['host_mono_s'][indices][0];t=data['host_mono_s'][indices]-start
    fig=plt.figure();ax=fig.add_subplot(projection='3d');colors=('tab:orange','tab:blue','tab:green','tab:red','tab:purple')
    lines=[ax.plot([],[],[],color=c,marker='o',markersize=3)[0] for c in colors]
    ax.set(xlim=(-.13,.13),ylim=(-.13,.13),zlim=(-.02,.24),xlabel='canonical x (m)',ylabel='y (m)',zlabel='z (m)')
    ax.set_box_aspect((1,1,1));ax.view_init(elev=15,azim=-65)
    display_time=np.arange(0,max(t[-1],1/fps),1/fps);frames=np.searchsorted(t,display_time).clip(0,len(p)-1)
    def update(i):
        for finger,line in enumerate(lines):
            chain=[0]+list(range(1+4*finger,5+4*finger));q=p[i,chain]
            line.set_data(q[:,0],q[:,1]);line.set_3d_properties(q[:,2])
        label='Raw Manus skeleton' if meta['source_type']=='manus_glove_raw' else 'TEST FIXTURE (not live Manus)'
        ax.set_title(f'{label} | host poll {indices[i]} | {t[i]:.2f}s\nHost-read timing; no robot/retargeting')
        return lines
    animation=FuncAnimation(fig,update,frames=frames,interval=1000/fps,cache_frame_data=False)
    plt.show()
    return animation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('path',type=Path,help='A session directory or one task directory')
    p.add_argument('--output',type=Path,help='New JSON report path; never overwrite')
    p.add_argument('--preview',action='store_true',help='Preview one task in a skeleton-only Matplotlib window')
    p.add_argument('--recover-export',action='store_true',help='Export static geometry from flushed chunks after interruption; never overwrite')
    args=p.parse_args()
    if args.output and args.output.exists():raise FileExistsError(args.output)
    paths=[args.path] if (args.path/'metadata.json').is_file() else sorted(x.parent for x in args.path.glob('*/metadata.json'))
    if not paths:raise ValueError('No captured task directories found')
    if args.preview and len(paths)!=1:p.error('--preview requires one task directory')
    reports=[]
    for path in paths:
        if args.recover_export:export_static(path)
        report=summarize_capture(path);reports.append(report)
        print(f"{report['task']}: 有效 {report['valid_count']}/{report['poll_count']}，相同读数 {report['identical_raw_fraction']:.1%}")
        for finger,r in report['information_pairs'].items():
            print(f"  {finger}: 2mm/20°={r['tip_2mm_axis_20deg_pairs']} 对，5mm/20°={r['tip_5mm_axis_20deg_pairs']} 对")
        for warning in report['warnings']:print('  检查：'+warning)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as f:json.dump(reports,f,indent=2,ensure_ascii=False,allow_nan=False)
    if args.preview:preview(paths[0])


if __name__=='__main__':main()
