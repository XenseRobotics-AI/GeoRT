"""Read-only glove capture geometry and recoverable local storage; no robot APIs."""
import hashlib
import json
from pathlib import Path
import numpy as np

NODE_FIELDS = ('node_id', 'parent_id', 'chain_type', 'finger_joint_type', 'side')
FINGER_SPEC = [(1,5,1),(2,5,2),(3,5,4),(4,5,5)] + [
    (5+4*f+j,6+f,2+j) for f in range(4) for j in range(4)]


def node_records(info):
    return [{k:int(getattr(n,k)) for k in NODE_FIELDS} for n in info]


def mp_mapping(nodes):
    """Same MP node selection as production ManusKeypointSource, strict topology."""
    ids=[n['node_id'] for n in nodes]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate Manus node IDs')
    roots=[i for i,n in enumerate(nodes) if n['chain_type']==13 and
           (n['parent_id']==n['node_id'] or n['parent_id'] not in ids)]
    if len(roots)!=1:raise ValueError('Expected exactly one hand root; cannot guess wrist')
    rows=[roots[0]]+[-1]*20
    for mp,chain,joint in FINGER_SPEC:
        matches=[i for i,n in enumerate(nodes) if n['chain_type']==chain and n['finger_joint_type']==joint]
        if len(matches)!=1:raise ValueError(f'Missing/ambiguous Manus node for MP point {mp}')
        rows[mp]=matches[0]
    if len(set(rows))!=21:raise ValueError('MP mapping reuses nodes')
    return np.asarray(rows,dtype=np.int64)


def canonicalize(points):
    """GeoRT palm convention; returns canonical meters and canonical-to-source T."""
    points=np.asarray(points,dtype=np.float64)
    if points.shape!=(21,3) or not np.isfinite(points).all():raise ValueError('Invalid 21-point skeleton')
    z=points[9]-points[0];aux=points[5]-points[13]
    if min(np.linalg.norm(z),np.linalg.norm(aux))<1e-6:raise ValueError('Degenerate palm anchors')
    z/=np.linalg.norm(z);aux/=np.linalg.norm(aux);x=np.cross(aux,z)
    if np.linalg.norm(x)<1e-4:raise ValueError('Collinear palm anchors')
    x/=np.linalg.norm(x);y=np.cross(z,x);y/=np.linalg.norm(y)
    rotation=np.column_stack((x,y,z));transform=np.eye(4)
    transform[:3,:3]=rotation;transform[:3,3]=points[0]
    result=(points-points[0])@rotation
    chains=np.stack([result[end-3:end+1] for end in (4,8,12,16,20)])
    lengths=np.linalg.norm(np.diff(chains,axis=1),axis=-1)
    if (lengths<.001).any() or (lengths>.15).any() or np.linalg.norm(result,axis=-1).max()>.5:
        raise ValueError('Implausible meter-scale skeleton; inspect units/mapping/calibration')
    return result.astype(np.float32),transform


def write_json(path,value):
    """Atomically replace an owned manifest, never a recording artifact."""
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.flush()
    tmp.replace(path)


def save_npz(path,**arrays):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    temporary=path.with_suffix('.partial')
    with temporary.open('xb') as f:np.savez_compressed(f,**arrays)
    temporary.rename(path)


class CaptureWriter:
    """Five-second chunks keep completed data if the process is interrupted."""
    def __init__(self,path,nodes,rows,metadata,chunk_size=300):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=False)
        self.nodes=nodes;self.rows=np.asarray(rows);self.chunk_size=chunk_size
        self.buffer=[];self.count=0;self.chunks=0;self.previous=None;self.last_time=None
        self.meta={**metadata,'schema_version':1,'node_info':nodes,'mp_rows':self.rows.tolist(),
                   'status':'recording','time_basis':'host_poll_monotonic; NOT SDK acquisition time',
                   'sdk_frame_id_available':False,'sdk_timestamp_available':False,
                   'raw_columns':'pos_xyz_m,quat_wxyz,scale_xyz','canonical_frame':'geort_canonical',
                   'confidence_available':False,'freshness_verified':False,'chunk_size':chunk_size}
        write_json(self.path/'metadata.json',self.meta)

    def add(self,raw,host_mono_s,host_wall_s,source_connected=True):
        if not np.isfinite([host_mono_s,host_wall_s]).all() or (self.last_time is not None and host_mono_s<=self.last_time):
            raise ValueError('Host poll times must be finite and increasing')
        self.last_time=host_mono_s
        raw=np.asarray(raw,dtype=np.float32)
        valid_shape=raw.shape==(len(self.nodes),10)
        reason='';canonical=np.full((21,3),np.nan,dtype=np.float32);transform=np.full((4,4),np.nan)
        # Preserve malformed values separately instead of silently reshaping/dropping.
        if not valid_shape:
            save_npz(self.path/f'rejected_{self.count:06d}.npz',raw=raw)
            raw=np.full((len(self.nodes),10),np.nan,dtype=np.float32);reason='node_shape_changed'
        elif not source_connected:reason='source_disconnected'
        elif not np.isfinite(raw).all():reason='nonfinite_raw'
        else:
            norms=np.linalg.norm(raw[:,3:7],axis=-1)
            if np.any(abs(norms-1)>.1):reason='invalid_quaternion_norm'
            else:
                try:canonical,transform=canonicalize(raw[self.rows,:3])
                except ValueError as e:reason=str(e)
        repeated=self.previous is not None and np.array_equal(raw,self.previous,equal_nan=True)
        self.previous=raw.copy()
        self.buffer.append((raw,canonical,transform,host_mono_s,host_wall_s,not bool(reason),repeated,reason,self.count))
        self.count+=1
        if len(self.buffer)>=self.chunk_size:self.flush()
        return not bool(reason),repeated,reason

    def flush(self):
        if not self.buffer:return
        rows=list(zip(*self.buffer))
        fields=('raw_skeleton','keypoints','canonical_to_source','host_mono_s','host_wall_s','valid_frame','same_raw_as_previous','invalid_reason','poll_index')
        save_npz(self.path/f'chunk_{self.chunks:05d}.npz',**{key:np.asarray(value) for key,value in zip(fields,rows)})
        self.chunks+=1;self.buffer=[]
        self.meta.update(poll_count=self.count,completed_chunks=self.chunks)
        write_json(self.path/'metadata.json',self.meta)

    def finish(self,status='complete'):
        self.flush();self.meta.update(status=status,poll_count=self.count,completed_chunks=self.chunks)
        write_json(self.path/'metadata.json',self.meta)


def load_capture(path):
    path=Path(path);meta=json.loads((path/'metadata.json').read_text());chunks=[]
    for file in sorted(path.glob('chunk_*.npz')):
        with np.load(file,allow_pickle=False) as f:chunks.append({k:f[k].copy() for k in f.files})
    if not chunks:raise ValueError('No completed capture chunks')
    data={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    if not np.array_equal(data['poll_index'],np.arange(len(data['poll_index']))):raise ValueError('Missing/out-of-order chunks')
    if np.any(np.diff(data['host_mono_s'])<=0):raise ValueError('Nonmonotonic host time')
    return meta,data


def export_static(path):
    """Training geometry only: no fabricated acquisition timestamps or continuity."""
    path=Path(path);meta,data=load_capture(path)
    keep=data['valid_frame'];points=data['keypoints'][keep]
    if len(points)<2:return None
    output=path/'keypoints.npy'
    with output.open('xb') as f:np.save(f,points,allow_pickle=False)
    with (path/'source_poll_index.npy').open('xb') as f:np.save(f,data['poll_index'][keep],allow_pickle=False)
    write_json(path/'export.json',{'kind':'static_geometry_only','keypoints_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'valid_frames':len(points),'omitted_invalid_polls':int((~keep).sum()),'session_id':meta['session_id'],
        'task':meta['task'],'split':meta['split'],'source_type':meta['source_type'],
        'temporal_training_allowed':False,'note':'Host polling timestamps stay in raw chunks. All clips from one session must share one split.'})
    return output
