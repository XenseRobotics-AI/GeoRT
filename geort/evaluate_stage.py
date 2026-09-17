"""Paired exact-FK stage metrics; validation selects, test only confirms."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from geort.coordination_data import load_recording, batch, split_indices
from geort.coordination_loss import ExactHand
from geort.export import resolve_checkpoint
from geort.model import build_ik_model, ik_input
from geort.stage_metrics import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--weights', default='best', help='best, last, or step_N')
    parser.add_argument('--split', choices=['validation', 'test', 'train'], default='validation')
    parser.add_argument('--wuji-cache', type=Path, help='Confirmed production Wuji baseline cache; required for future acceptance')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = resolve_checkpoint(args.checkpoint)
    if args.weights not in ('best', 'last') and not (args.weights.startswith('step_') and args.weights[5:].isdigit()):
        raise ValueError('Invalid weights name')
    metadata = json.loads((path / 'experiment.json').read_text())
    config = json.loads((path / 'config.json').read_text())
    parent_config = json.loads((path / 'baseline_config.json').read_text())
    data_path = Path(metadata['data']).resolve()
    expected = [value for name, value in metadata['hashes'].items() if Path(name).resolve() == data_path]
    if len(expected) != 1 or hashlib.sha256(data_path.read_bytes()).hexdigest() != expected[0]:
        raise ValueError('Recording differs from the training provenance')
    recording = load_recording(data_path)
    split = split_indices(recording, metadata['experiment']['split'])
    sample = batch(recording, split[args.split])
    model = build_ik_model(config).eval()
    weights = path / f'{args.weights}.pth'
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    parent = build_ik_model(parent_config).eval()
    parent.load_state_dict(torch.load(path / 'baseline.pth', map_location='cpu', weights_only=True))
    exact = ExactHand(config)
    with torch.inference_mode():
        output = exact(model(sample))
        reference = exact(parent(ik_input(parent_config, sample['keypoints'])))
    report = summarize(output['q'].numpy(), output['tips'].numpy(), reference['q'].numpy(),
                       reference['tips'].numpy(), sample['keypoints'].numpy())
    report['external_baseline_status'] = 'missing; future acceptance requires Wuji comparison'
    if args.wuji_cache:
        from geort.wuji_baseline import load_wuji_cache, comparison_report
        cache = load_wuji_cache(args.wuji_cache, data_path, config['joint_order'])
        report['wuji_comparison'] = comparison_report(cache, split[args.split], exact, output['q'].numpy(),
            output['tips'].numpy(), reference['q'].numpy(), reference['tips'].numpy(), sample['keypoints'].numpy())
        report['external_baseline_status'] = 'included'
    report.update(checkpoint=str(path), weights=args.weights, split=args.split, frames=len(split[args.split]),
                  weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
