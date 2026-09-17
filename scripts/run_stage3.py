"""Frozen-base, equal-architecture local information ablation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arms',nargs='+',choices=['H0','H1'],default=['H0','H1'])
    p.add_argument('--prefix',default='stage3')
    args=p.parse_args()
    if not args.prefix.replace('_','').isalnum():raise ValueError('Use an alphanumeric output prefix')
    root=Path(__file__).resolve().parents[1]
    report=root/'reports/stage3';report.mkdir(parents=True,exist_ok=True)
    for arm in args.arms:
        output=f'checkpoint/{args.prefix}_{arm}_seed42'
        if (root/output).exists():raise FileExistsError(output)
        command=[sys.executable,'-m','geort.train_coordination','--config',f'geort/experiments/S3_{arm}.json',
            '--parent','wuji_hand2_beta1_right_2026-09-15_18-01-42_std_collision_seed0_w0',
            '--init-base','stage2_T2_seed42','--freeze-base','--head-lr','0.001',
            '--local-features','tip_control' if arm=='H0' else 'skeleton',
            '--output',output,'--evaluation-split','validation','--save-every','100',
            '--warmup-targets','reports/stage2/train_targets_seed42.npz','--warmup-steps','1000',
            '--posture-policy','feasible','--wuji-cache','reports/baselines/wuji_manus_right_human_alex.npz']
        (report/f'{args.prefix}_{arm}_command.json').write_text(json.dumps(command,indent=2))
        print(f'Starting {arm}',flush=True)
        with (report/f'{args.prefix}_{arm}.log').open('x') as log:
            subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'Completed {arm}',flush=True)


if __name__=='__main__':main()
