"""Run the recorded stage-two matrix explicitly; existing results are preserved."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms',nargs='+',choices=['T0','T1','T2','T3'],default=['T0','T1','T2','T3'])
    parser.add_argument('--prefix',default='stage2')
    args=parser.parse_args()
    if not args.prefix.replace('_','').isalnum():raise ValueError('Use an alphanumeric output prefix')
    root=Path(__file__).resolve().parents[1]
    report=root/'reports/stage2';report.mkdir(parents=True,exist_ok=True)
    for arm in args.arms:
        output=f'checkpoint/{args.prefix}_{arm}_seed42'
        if (root/output).exists():raise FileExistsError(output)
        command=[sys.executable,'-m','geort.train_coordination','--config',f'geort/experiments/S2_{arm}.json',
            '--parent','wuji_hand2_beta1_right_2026-09-15_18-01-42_std_collision_seed0_w0',
            '--output',output,'--evaluation-split','validation','--save-every','100',
            '--warmup-targets','reports/stage2/train_targets_seed42.npz','--warmup-steps','1000',
            '--target-variant','independent' if arm=='T0' else 'consistent',
            '--posture-policy','feasible' if arm in ['T2','T3'] else 'uniform',
            '--wuji-cache','reports/baselines/wuji_manus_right_human_alex.npz']
        (report/f'{args.prefix}_{arm}_command.json').write_text(json.dumps(command,indent=2))
        print(f'Starting {arm}: {output}',flush=True)
        with (report/f'{args.prefix}_{arm}.log').open('x') as log:
            subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'Completed {arm}',flush=True)


if __name__=='__main__':main()
