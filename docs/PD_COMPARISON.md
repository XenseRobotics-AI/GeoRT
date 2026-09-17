# 三方法同屏 PD 回放

完整test01按原顺序播放：9段、20,989条有效读取。读取已经冻结的全量输出缓存，不重新推理、不训练、不连接手套或机器人。

```bash
conda activate geort
cd /home/xense-wufan/GeoRT
python -m geort.mocap.replay_pd_comparison
```

从左至右为人手骨架、GeoRT R2（LP=0.3）、生产Wuji（LP=0.3）、SDK2026.8.31（内置滤波）。默认显示**PD执行后的实际关节位置**。三个独立物理场景使用同一资产、碰撞过滤、固定手腕、重力与控制参数，彼此不会发生碰撞。显示场景只呈现各物理场景产生的状态，不对目标关节做逐帧瞬移来冒充PD。

- 空格：暂停/继续。暂停期间物理时间也停止。
- D：切换显示PD实际姿态或模型目标；顶部明确标注模式。物理计算仍继续，切回时保留同一轨迹。
- R：恢复统一正交视角。
- Esc：关闭。全部九段播完后停在末帧，不跨末尾重复推进滤波或PD状态。

面板显示当前片段、总读数进度和三路关节跟踪平均绝对误差。所有读数均推进相同物理时间；机器来不及计算时播放会变慢，不丢帧追赶。因此这不是实际系统吞吐率或手套端到端延迟测量。

默认参数沿用项目PD增益：`kp=400`、`kd=10`、关节力矩上限`10 Nm`。假定数据60Hz，控制600Hz，物理1200Hz，每输入间隔10次线性插值目标更新、20次物理子步。旧official回放的每帧0.1秒驻留不适用于这里的60Hz假设，所以没有沿用旧时间放大方式。

每段起点三路均重置到相同的限位内零位和零速度；最初跟踪过渡包含在回放中。仅片段起点直接设置初态，后续使用drive targets。原有0.001rad驱动限位边距保留，不添加姿态修正或额外滤波。上述参数是统一仿真对照设置，未标定为Wuji实机执行器模型，也不代表实机安全参数。

显式参数命令：

```bash
python -m geort.mocap.replay_pd_comparison \
  --fps 60 --control-hz 600 --physics-hz 1200 \
  --kp 400 --kd 10 --force-limit 10 --window-size 1800 1000
```

有限帧验证及离线全量记录（输出目录必须不存在）：

```bash
python -m geort.mocap.replay_pd_comparison --headless --frames 120 \
  --output /tmp/geort-pd-check
python -m geort.mocap.replay_pd_comparison --headless \
  --output reports/stage7/pd_full_repeat
```

`actual.npz`保存连续source_ids、三路实际关节位置及每个输入区间最后物理子步的接触穿透深度；`protocol.json`记录参数、缓存哈希和是否完整。未完成的记录明确标记`complete=false`，不能用于全测试集结论。接触深度来自真实推进的最后子步，不能和先前静态目标的微小步几何探测冒充同一种测量。

本次已完成120帧三路物理计算、20帧实际GUI回放及截图检查；固定根姿态和重置可重复性检查通过，7项现有控制时序/相机测试通过。完整PD统计单独计算，不能直接把Stage 6的模型目标指标解释为PD结果。
