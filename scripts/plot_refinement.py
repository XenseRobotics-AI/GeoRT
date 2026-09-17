"""Export a static, JSON-derived comparison of the completed two rounds."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gates',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=json.loads(a.gates.read_text());plt.rcParams['font.family']='Noto Sans CJK JP';plt.rcParams['axes.unicode_minus']=False
    names=['M1_lp03','R1_lp03','R2_lp03','Wuji','SDK'];labels=['M1 + LP .3','R1 + LP .3','R2 + LP .3','生产 Wuji LP .3','SDK 内置状态']
    fig,axs=plt.subplots(2,2,figsize=(14,10))
    for ax,key,title,unit in [(axs[0,0],'axis_deg','末节方向误差','度'),(axs[0,1],'opening_mm','开口代理误差（人手 ≥15mm）','mm')]:
        values=[r['aggregate'][n][key] for n in names];ax.barh(labels,values,color=['#9298a6','#5597b4','#307854','#ba763a','#7867a0'])
        ax.invert_yaxis();ax.set_title(title);ax.set_xlabel(unit+' · 九任务等权平均');ax.set_xlim(0,max(values)*1.2)
        for i,v in enumerate(values):ax.text(v+max(values)*.02,i,f'{v:.2f}',va='center')
    ax=axs[1,0];y=np.arange(len(names));v2=[r['aggregate'][n]['over_2mm_fraction']*100 for n in names];v5=[r['aggregate'][n]['over_5mm_fraction']*100 for n in names]
    ax.barh(y-.15,v2,height=.3,label='>2mm',color='#5597b4');ax.barh(y+.15,v5,height=.3,label='>5mm',color='#ba763a');ax.set_yticks(y,labels);ax.invert_yaxis();ax.legend();ax.set_title('共同几何穿透：固定1800条读数');ax.set_xlabel('比例 / %')
    for i,(v,w) in enumerate(zip(v2,v5)):
        ax.text(v+.4,i-.15,f'{v:.2f}',va='center',fontsize=9);ax.text(w+.4,i+.15,f'{w:.2f}',va='center',fontsize=9)
    ax=axs[1,1];cells=r['checks']['R2_lp03']['micro'];r1=r['checks']['R1_lp03']['micro']
    values=np.array([[x['direction_deg'],z['direction_deg'],z['baseline_direction_deg']] for x,z in zip(r1,cells)])
    ax.imshow(values,cmap='YlOrRd',aspect='auto',vmin=0,vmax=25);ax.set_xticks([0,1,2],['R1 LP .3','R2 LP .3','Wuji LP .3']);ax.set_yticks(range(12),[f"{c['lag']}读数 / {c['input_mm'][0]:g}–{c['input_mm'][1]:g}mm" for c in cells],fontsize=9)
    for i in range(12):
        for j in range(3):ax.text(j,i,f'{values[i,j]:.1f}°',ha='center',va='center',fontsize=10)
    ax.set_title('微调位移方向误差：双方同滤波')
    fig.suptitle('两轮 GeoRT 改进 · val01 验证结果',fontsize=18)
    fig.text(.5,.018,'训练仅用 train01；每轮2000步，seed42。验证包含选型，不是独立泛化结果。\n穿透由共同仿真资产计算；开口是几何代理；SDK内部滤波未知；无物理接触或真人遥操验收。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.95));a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=150)


if __name__=='__main__':main()
