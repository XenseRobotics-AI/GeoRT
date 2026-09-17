"""Measure all frozen targets and actual PD states used in the closing videos."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from geort.coordination import skeleton_features
from geort.coordination_loss import ExactHand
from geort.evaluate_differences import angle, distribution, sha
from geort.manus_sessions import load_session_split

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('Original', 'R2_lp03', 'SDK', 'Wuji')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    video = ROOT / 'videos/final_comparison_test01'
    protocol = json.loads((video / 'protocol.json').read_text())
    manifest = ROOT / 'data/manus_stage4_v1/manifest.json'
    caches = [ROOT / 'reports/final_main_retrained_reference/outputs.npz',
              ROOT / 'reports/stage6/test_full/outputs.npz']
    assert sha(manifest) == protocol['manifest_sha256']
    assert sha(caches[0]) == protocol['main_reference']['outputs_sha256']
    assert sha(caches[1]) == protocol['input_cache_sha256']
    data = load_session_split(manifest, 'test')
    cfg_path = ROOT / 'checkpoint/stage5_R2_seed42/config.json'
    cfg = json.loads(cfg_path.read_text())
    exact = ExactHand(cfg)
    original, others = [np.load(p, allow_pickle=False) for p in caches]
    pools = {mode: {m: {'axis': [], 'opening': [], 'tracking': []} for m in METHODS}
             for mode in ('target', 'pd')}
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in [manifest, cfg_path, *caches, video / 'protocol.json']}
    by_task = {}
    for index, clip in enumerate(data['entry']['clips'], 1):
        task = clip['task']
        a, b = clip['range']
        pd_path = video / f'{index:02}_{task}_pd.npz'
        hashes[str(pd_path.relative_to(ROOT))] = sha(pd_path)
        pd = np.load(pd_path, allow_pickle=False)
        assert np.array_equal(pd['source_ids'], np.arange(a, b))
        with torch.inference_mode():
            human = skeleton_features(torch.tensor(data['keypoints'][a:b]), [4, 8, 12, 16, 20])
        assert human['bone_valid'].all()
        tips = human['tips'].numpy()
        axes = human['axes'][:, :, -1].numpy()
        gap = np.linalg.norm(tips[:, 1:] - tips[:, :1], axis=-1)
        desired = gap * np.asarray(cfg['objectives']['relation_scale'])
        by_task[task] = {}
        for mode in pools:
            by_task[task][mode] = {}
            for method in METHODS:
                target = (original if method == 'Original' else others)[task + '__' + method]
                q = target if mode == 'target' else pd[method]
                assert q.shape == (b-a, 20) and np.isfinite(q).all()
                with torch.inference_mode():
                    robot = exact(2 * (torch.tensor(q) - exact.lower) / (exact.upper - exact.lower) - 1)
                rt = robot['tips'].numpy()
                rg = np.linalg.norm(rt[:, 1:] - rt[:, :1], axis=-1)
                values = {'axis': angle(axes, robot['axes'][:, :, -1].numpy()),
                          'opening': (abs(rg-desired)*1000)[gap >= .015],
                          'tracking': np.rad2deg(abs(q-target))}
                by_task[task][mode][method] = {k: distribution(v) for k, v in values.items()}
                for k, v in values.items():
                    pools[mode][method][k].append(v.reshape(-1))
    aggregate = {mode: {m: {k: distribution(np.concatenate(v)) for k, v in row.items()}
                        for m, row in models.items()} for mode, models in pools.items()}
    result = {'frames': len(data['keypoints']), 'split': 'test01', 'methods': protocol['methods'],
              'units': {'axis': 'deg', 'opening': 'mm', 'tracking': 'deg'},
              'scope': 'All nine clips, pooled frames; same frozen targets and PD states as final video; no inference or retraining. Tracking is deviation from own target, not retargeting accuracy. Opening uses fixed R2 train-derived scale, not contact truth. No new static collision comparison.',
              'aggregate': aggregate, 'by_task': by_task, 'sha256': hashes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
