"""Plot completed offline difference audits (all numbers read from metrics.json)."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    r = json.loads(args.report.read_text())
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'
    plt.rcParams['axes.unicode_minus'] = False
    names = ['Original','M0','M1','M1_lp03','Wuji','Wuji_raw']
    labels = ['原始 GeoRT','M0 指尖','M1 骨骼','M1 + LP 0.3','现用 Wuji','Wuji 未滤波']
    colors = ['#888888','#6a5acd','#2575ba','#569f78','#d17925','#bb537d']
    has_sdk = 'SDK' in r['collision']
    if has_sdk:
        names.append('SDK');labels.append('Wuji SDK 2026.8.31');colors.append('#39857d')
    fig, axs = plt.subplots(2, 2, figsize=(15, 10.8), gridspec_kw={'height_ratios':[1,1.6]})
    for ax, metric, title, unit in [(axs[0,0], 'axis_deg', '静态末节方向误差', '度'),
                                   (axs[0,1], 'opening_mm', '张开姿态的开口代理误差', 'mm')]:
        values = [np.mean([t['models'][n][metric]['mean'] for t in r['by_task'].values()]) for n in names]
        ax.barh(labels, values, color=colors, height=.6)
        ax.invert_yaxis();ax.set_xlim(0, max(values)*1.2);ax.set_xlabel(unit+' · 九段动作等权平均')
        ax.set_title(title)
        for i,v in enumerate(values):ax.text(v+max(values)*.015,i,f'{v:.2f}',va='center')
        ax.spines[['top','right']].set_visible(False)
    ax = axs[1,0]
    selected = ['M1','Wuji_raw','M1_lp03','Wuji']
    micro_labels = ['M1 无滤波','Wuji 无滤波','M1 LP 0.3','Wuji LP 0.3']
    if has_sdk:
        selected.append('SDK');micro_labels.append('SDK 内置滤波')
    values = np.array([[cell['models'][n]['direction_deg_responsive_only']['mean'] for n in selected] for cell in r['micro']])
    matrix = ax.imshow(values, cmap='YlOrRd', vmin=0, vmax=max(25,float(values.max())), aspect='auto')
    ax.set_xticks(range(len(selected)), micro_labels,fontsize=9)
    ax.set_yticks(range(12), [f"{c['lag_polls']} 读数 / {c['input_mm'][0]:g}–{c['input_mm'][1]:g} mm" for c in r['micro']])
    for y in range(12):
        for x in range(len(selected)):ax.text(x,y,f'{values[y,x]:.1f}°',ha='center',va='center',fontsize=11,color='black')
    ax.set_title('微调：位移响应方向误差（按幅度 × 读数间隔）')
    ax.tick_params(axis='both',length=0)
    ax = axs[1,1];y=np.arange(len(names))
    v2=[r['collision'][n]['aggregate']['over_2mm_fraction']*100 for n in names]
    v5=[r['collision'][n]['aggregate']['over_5mm_fraction']*100 for n in names]
    ax.barh(y-.16,v2,height=.3,color='#6e94be',label='穿透 >2 mm')
    ax.barh(y+.16,v5,height=.3,color='#d58c4c',label='穿透 >5 mm')
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('抽检比例 / %');ax.set_title('共同资产：1800 帧自穿透抽检')
    for i,(v,w) in enumerate(zip(v2,v5)):
        ax.text(v+1,i-.16,f'{v:.1f}%',va='center',fontsize=10)
        ax.text(w+1,i+.16,f'{w:.1f}%',va='center',fontsize=10)
    ax.legend(loc='lower right');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('原始 GeoRT / M1 / 现用 Wuji / SDK：共同验证集对照' if has_sdk else 'M1 与现用 Wuji：姿态误差更低，微调优势不稳定，穿透明显更多',fontsize=17,y=.995)
    fig.text(.5,.014,'val01 · 9 段 / 20988 条主机读数 · seed 42 · 固定模型输出 / 共同 FK\n开口代理误差排除人手开口 <15 mm；微调低响应单独计数；无实机接触、速度或延迟结论。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.055,1,.975));args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160)
    print(args.output)


if __name__ == '__main__': main()
