"""Render complete frozen test metrics; no inference or selected examples."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


NAMES={'R2_lp03':'GeoRT R2（LP=0.3）','Wuji':'仓库 Wuji（LP=0.3）','SDK':'SDK 2026.8.31'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,default=Path('reports/stage6/test_full/metrics.json'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=json.loads(a.report.read_text())
    if r['split']!='test':raise ValueError('Expected held-out test results')
    a.output.mkdir(parents=True,exist_ok=False)
    names=list(NAMES);agg=r['aggregate']
    lines=['# 三种方法的完整测试集表现','',
        f"test01：**{r['frames']:,}条有效读取，全部{len(r['all_clips'])}段动作**；每段完整推理，三种主要方法逐条碰撞检测，无片段选择、无碰撞抽检。冻结R2后首次测试，不用结果继续调参。",'',
        '## 全量读数合并统计','',
        '| 方法 | 末节方向误差（均值/P95）° | 开口代理误差（均值/P95）mm | >2mm穿透 | >5mm穿透 |',
        '|---|---:|---:|---:|---:|']
    for n in names:
        v=agg[n];d=v['axis_deg'];o=v['opening_mm'];c=v['collision']
        lines.append(f"| {NAMES[n]} | {d['mean']:.2f} / {d['p95']:.2f} | {o['mean']:.2f} / {o['p95']:.2f} | {c['over_2mm_fraction']*100:.2f}% | {c['over_5mm_fraction']*100:.2f}% |")
    lines+=['','方向覆盖全部读数×五指；开口采用训练时固定人机尺度，在人手间距≥15mm的全部关系上计算。穿透是共同仿真资产的最大自穿透，不是实机接触成功率。均值以所有读数/有效关系合并计算，较长片段权重更大。','',
        '## 九段等权平均','', '| 方法 | 方向 ° | 开口 mm |','|---|---:|---:|']
    for n in names:lines.append(f"| {NAMES[n]} | {agg[n]['equal_task_axis_deg']:.2f} | {agg[n]['equal_task_opening_mm']:.2f} |")
    lines+=['','## 全部片段中的小幅动作响应','',
        '每段内相邻读数，合并全部九段；输入位移0.5–10mm。方向仅对输出位移≥0.1mm的响应计算，同时保留低响应比例。增益为机器人/人手位移幅度比，不是速度比。','',
        '| 方法 | 方向误差 ° | 平均增益 | 平均 abs(增益−1) | 低响应比例 |', '|---|---:|---:|---:|---:|']
    for n in names:
        v=agg[n]['response']['fine_point5_to10mm']
        lines.append(f"| {NAMES[n]} | {v['direction_deg_responsive_only']['mean']:.2f} | {v['gain']['mean']:.3f} | {v['gain_deviation_from_one']['mean']:.3f} | {v['low_response_fraction']*100:.3f}% |")
    lines+=['',f"这组幅度条件覆盖{agg[names[0]]['response']['fine_point5_to10mm']['n']:,}次指尖位移；全部相邻指尖位移计{agg[names[0]]['response']['all_transition_finger_count']:,}次。≥0.1mm的全运动统计、微动输出代理、逐指方向和近捏合site距离均保存在完整JSON中。没有只使用micro片段。",'',
        '## 全部九段明细','', '| 动作 | 读数 | R2方向° | Wuji方向° | SDK方向° | R2 >2mm | Wuji >2mm | SDK >2mm |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    csv_rows=[]
    for task,entry in r['by_task'].items():
        v=entry['models'];ds=[v[n]['axis_deg']['mean'] for n in names];cs=[v[n]['collision']['over_2mm_fraction']*100 for n in names]
        lines.append(f"| {task} | {entry['frames']} | "+' | '.join(f'{x:.2f}' for x in ds)+' | '+' | '.join(f'{x:.2f}%' for x in cs)+' |')
        for n in names:
            value=v[n];c=value['collision'];csv_rows.append(dict(task=task,frames=entry['frames'],model=n,
                axis_mean_deg=value['axis_deg']['mean'],axis_p95_deg=value['axis_deg']['p95'],
                opening_mean_mm=value['opening_mm']['mean'],opening_p95_mm=value['opening_mm']['p95'],
                collision_over2_fraction=c['over_2mm_fraction'],collision_over5_fraction=c['over_5mm_fraction'],
                collision_p95_mm=c['depth_mm']['p95'],collision_max_mm=c['depth_mm']['maximum']))
    with (a.output/'all_tasks.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    failures=sum(v['wuji_solver_failures'] for v in r['provenance']['sources'].values())
    lines+=['','全部27行（9段×3方法）含开口、P95、深穿透等明细见[CSV](all_tasks.csv)。','',
        '## 计算边界','',f"- 仓库Wuji记录到{failures}条求解器异常/回退；未排除其对应帧。SDK不暴露等价的内部求解状态。",
        '- R2、Wuji同为LP=0.3；SDK内部滤波和目标配置不公开，所以这是固定实际使用配置的端到端几何输出比较。',
        '- 同一操作者、同一标定下的独立采集会话；没有统计成独立的两万次实验，也没有跨人泛化或实机验证。',
        '- 近闭合site距离不是接触误差。缺少物体和接触真值，不能从这些数字推断抓取成功率。',
        '- 原始GeoRT及R2/Wuji无滤波结果在完整JSON附录保留，主要结论始终围绕用户要求的三种方法。',
        '', '![全量测试概览](summary.png)','']
    (a.output/'RESULTS.md').write_text('\n'.join(lines))
    plt.rcParams['font.family']='Noto Sans CJK JP';plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(2,2,figsize=(13,9));colors=['#307854','#b97b3e','#7768a1']
    specs=[('axis_deg','全量末节方向误差','度'),('opening_mm','全量开口代理误差','mm'),('collision','全量自穿透比例','%'),('response','全片段0.5–10mm响应方向误差','度')]
    for ax,(metric,title,unit) in zip(axs.flat,specs):
        if metric=='collision':values=[agg[n][metric]['over_2mm_fraction']*100 for n in names]
        elif metric=='response':values=[agg[n][metric]['fine_point5_to10mm']['direction_deg_responsive_only']['mean'] for n in names]
        else:values=[agg[n][metric]['mean'] for n in names]
        ax.barh(list(NAMES.values()),values,color=colors);ax.invert_yaxis();ax.set_xlim(0,max(values)*1.23);ax.set_title(title+('（>2mm）' if metric=='collision' else ''));ax.set_xlabel(unit)
        for i,v in enumerate(values):ax.text(v+max(values)*.02,i,f'{v:.2f}',va='center')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f"完整 test01 · {r['frames']:,} 条读数 · 全部9段 · 碰撞100%覆盖",fontsize=17)
    fig.text(.5,.025,'模型与配置在测试前冻结；主表合并全部读数，另附九段等权均值。\n开口/碰撞为共同资产几何代理；SDK内部滤波未知；非实机接触或跨人泛化结论。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(a.output/'summary.png',dpi=150)
    print(a.output/'RESULTS.md')


if __name__=='__main__':main()
