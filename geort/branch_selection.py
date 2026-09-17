"""Training-only geometric-neighbor branch selection, without time assumptions."""
import numpy as np
from scipy.spatial import cKDTree


def geometric_neighbors(tips, axes, valid, neighbors=8):
    """Symmetric edges for nearby positions AND matching observed bone axes."""
    n = len(tips)
    if tips.shape != (n,3) or axes.shape != (n,3,3) or valid.shape != (n,3):
        raise ValueError('Invalid geometric neighbor feature shape')
    if not np.isfinite(tips).all() or not np.isfinite(axes).all():
        raise ValueError('Nonfinite neighbor features')
    features = np.concatenate((tips/.01, axes.reshape(n,9)/.25, valid.astype(float)), axis=1)
    distances, ids = cKDTree(features).query(features, k=min(n, neighbors+1))
    if distances.ndim == 1:
        return [[] for _ in range(n)]
    edges = {}
    for i in range(n):
        for distance,j in zip(distances[i], ids[i]):
            j=int(j)
            if i==j or distance > 3 or not np.array_equal(valid[i], valid[j]):
                continue
            # Only complete observed skeletons define direction-consistent neighbors.
            if not valid[i].all() or np.linalg.norm(tips[i]-tips[j]) > .01:
                continue
            if np.min(np.sum(axes[i]*axes[j], axis=-1)) < np.cos(np.deg2rad(20)):
                continue
            edges[min(i,j),max(i,j)] = float(np.exp(-.5*distance**2)/neighbors)
    graph = [[] for _ in range(n)]
    for (i,j),weight in edges.items():
        graph[i].append((j,weight));graph[j].append((i,weight))
    return graph


def select_branches(candidates, unary, graph, strength=.05, sweeps=5, seed=42):
    """Coordinate descent of explicit undirected graph energy; never averages q."""
    candidates=np.asarray(candidates,dtype=np.float64);unary=np.asarray(unary,dtype=np.float64)
    if candidates.ndim!=3 or unary.shape!=candidates.shape[:2] or len(graph)!=len(candidates):
        raise ValueError('Expected candidates [frames, choices, joints] and unary [frames, choices]')
    if not np.isfinite(candidates).all() or np.isnan(unary).any() or not np.isfinite(unary).any(axis=1).all():
        raise ValueError('Each frame needs at least one finite valid candidate')
    if not np.isfinite(strength) or strength<0 or sweeps<0:
        raise ValueError('Invalid graph optimization budget')
    count=len(candidates);labels=np.argmin(unary,axis=1);initial=labels.copy()
    rows=np.arange(count);scale=np.deg2rad(10)
    def energy():
        value=float(unary[rows,labels].sum())
        q=candidates[rows,labels]
        for i,edges in enumerate(graph):
            for j,w in edges:
                if i<j:value+=strength*w*float(np.mean(((q[i]-q[j])/scale)**2))
        return value
    energies=[energy()];rng=np.random.default_rng(seed)
    for _ in range(sweeps):
        changed=0
        for i in rng.permutation(count):
            cost=unary[i].copy()
            for j,w in graph[i]:
                cost+=strength*w*np.mean(((candidates[i]-candidates[j,labels[j]])/scale)**2,axis=-1)
            selected=int(np.argmin(cost))
            changed+=selected!=labels[i];labels[i]=selected
        energies.append(energy())
        if energies[-1]>energies[-2]+1e-6:raise RuntimeError('Graph energy increased')
        if not changed:break
    return labels, {'initial_energy':energies[0],'final_energy':energies[-1],
                    'energy_trace':energies,'changed_fraction':float(np.mean(labels!=initial)),
                    'edge_count':sum(len(e) for e in graph)//2}
