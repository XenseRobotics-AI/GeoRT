"""Summarize the frozen two-round gates without changing their thresholds."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(r):
    rows={}
    for name in r['collision']:
        rows[name]={'axis_deg':float(np.mean([t['models'][name]['axis_deg']['mean'] for t in r['by_task'].values()])),
            'opening_mm':float(np.mean([t['models'][name]['opening_mm']['mean'] for t in r['by_task'].values()])),
            **{k:v for k,v in r['collision'][name]['aggregate'].items() if k.startswith('over_')},
            'stationary_proxy_mm':r['stationary']['models'][name]}
    checks={}
    for name in [n for n in ('R1','R1_lp03','R2','R2_lp03') if n in rows]:
        baseline='Wuji' if name.endswith('_lp03') else 'Wuji_raw'
        pinch=[];micro=[]
        for c in r['pinch']:
            if c['n']>=30 and c['human_gap_mm'][0] in (15,30):
                v=c['models'][name]['opening_error_mm']['mean'];b=c['models']['Wuji']['opening_error_mm']['mean']
                pinch.append(dict(finger=c['finger'],human_gap_mm=c['human_gap_mm'],n=c['n'],candidate_mm=v,wuji_mm=b,passed=v<=b+1))
        for c in r['micro']:
            v=c['models'][name];b=c['models'][baseline]
            micro.append(dict(lag=c['lag_polls'],input_mm=c['input_mm'],n=c['n'],
                direction_deg=v['direction_deg_responsive_only']['mean'],baseline_direction_deg=b['direction_deg_responsive_only']['mean'],
                direction_pass=v['direction_deg_responsive_only']['mean']<=b['direction_deg_responsive_only']['mean']+2,
                gain_deviation=v['gain_deviation_from_one']['mean'],baseline_gain_deviation=b['gain_deviation_from_one']['mean'],
                low_response=v['low_response_fraction'],baseline_low_response=b['low_response_fraction']))
        collision_pass=(rows[name]['over_2mm_fraction']<=rows['Wuji']['over_2mm_fraction']+.02 and
                        rows[name]['over_5mm_fraction']<=rows['Wuji']['over_5mm_fraction']+.01)
        checks[name]=dict(baseline=baseline,collision_aggregate_pass=collision_pass,
            collision_by_task={task:{'candidate_over5':v['over_5mm_fraction'],
                                    'wuji_over5':r['collision']['Wuji']['by_task'][task]['over_5mm_fraction']}
                               for task,v in r['collision'][name]['by_task'].items()},
            pinch=pinch,pinch_pass=all(c['passed'] for c in pinch),micro=micro,
            micro_direction_pass_count=sum(c['direction_pass'] for c in micro),
            gain_worse_cell_count=sum(c['gain_deviation']>c['baseline_gain_deviation'] for c in micro),
            low_response_worse_cell_count=sum(c['low_response']>c['baseline_low_response'] for c in micro),
            axis_pass=rows[name]['axis_deg']<=rows[baseline]['axis_deg'])
    return dict(scope='validation screening, no test inference or physical contact truth; per-task collision and gain require explicit review',aggregate=rows,checks=checks)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=summarize(json.loads(a.report.read_text()));a.output.write_text(json.dumps(result,indent=2))
    for name,row in result['aggregate'].items():
        print(name,'axis',round(row['axis_deg'],2),'opening',round(row['opening_mm'],2),'>2%',round(row['over_2mm_fraction']*100,2),'>5%',round(row['over_5mm_fraction']*100,2))
    for name,c in result['checks'].items():print(name,{k:v for k,v in c.items() if k not in ('pinch','micro','collision_by_task')})


if __name__=='__main__':main()
