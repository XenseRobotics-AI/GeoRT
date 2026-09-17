# 阶段四结果：基本映射恢复，骨骼信息出现收益，碰撞仍未过关

2026-09-16。按 [开发目标](DEVELOPMENT_GOALS.md)、[本轮协议](STAGE4_PROTOCOL.md) 完成 M0/M1 两组 seed42、各 2000 步实际训练，以及完整 val01 的 9 个动作片段、20988 条读数的对照。**已获得可预览的阶段成果，但不替换默认模型，不作为实机验收版本。**

## 修复了哪一层问题

之前截图中旧 H1 的张手变成畸形握拳，来自模型真实关节输出，并非显示错位。旧模型的 BatchNorm 统计量和映射不适应新输入；train-only 重估统计量虽能降低饱和，方向误差仍大，因此没有把它当最终补丁。

本轮新增可选 `manus_v1`：按训练集统计量归一化输入，去掉旧 BatchNorm 和冻结基础网络，不沿用只能微调的残差限制。M0/M1 结构、参数量、初始化逐 tensor 完全相同，所有权重可学习；M0 只读指尖，M1 增加真实骨轴。没有全手上下文或机器人关节标签。两组 best 均为第 2000 步，由验证目标选择。

相对旧模型同时改了数据、网络与目标，不能把全部恢复归因于骨骼。**信息收益仅通过 M0/M1 同条件比较。** 历史模型和原默认入口保留不变。

## 完整验证结果

下面按 9 个动作片段等权平均。方向为共同精确 FK 的末节骨轴差；开口是人机训练骨长比例缩放后的指尖间距代理指标，排除人手开口 <15 mm 的区域，不是接触真值。严重 PIP/DIP 为非拇指任一关节 <−15°。

| 模型 | 全手末节方向误差 | 开口误差 | 接近关节限位比例 | 小指严重 PIP/DIP |
|---|---:|---:|---:|---:|
| 原始 GeoRT | 80.13° | 44.14 mm | 46.20% | 4.82% |
| 旧 H1 | 86.46° | 46.50 mm | 55.33% | 0.82% |
| M0，指尖输入 | 5.21° | 3.98 mm | 0.045% | 0.315% |
| M1，骨骼输入 | **3.70°** | 4.21 mm | 0.041% | **0%** |
| 现用 Wuji，lp_alpha=0.3 | 12.80° | 7.56 mm | 0.348% | 0% |

M1 在本次验证的全部非拇指上严重 PIP/DIP 与 MCP <−15°比例均为 0。其方向误差相对 M0 下降约 29%，但平均开口误差略增。neutral / open_close 方向误差分别为 M1 3.02° / 5.39°，M0 4.45° / 7.04°，Wuji 10.15° / 13.48°。

在 micro 片段，M1/M0 的局部位移响应方向误差为 10.9°/8.9°，平均增益约 1.17/1.11，说明骨骼收益没有覆盖所有指标。它们是相邻主机读数的位移关系，不是传感器时间下的速度或时延。Wuji 同片段响应方向约 19.0°，包括其原生滤波的影响。

## 真正与多解相关的证据

val01/ring_branch 中的小指有 384 对严格样本（134 条不同读数）：指尖 ≤2 mm、末节方向差 ≥20°、主机时间间隔 ≥0.5 秒。人手方向变化均值为 21.55°：

| 模型 | 对应机器人方向变化 | 方向变化向量误差 | 机器人指尖配对距离 |
|---|---:|---:|---:|
| M0 | 10.51° | 0.1965 | 4.59 mm |
| M1 | **20.64°** | **0.0469** | 5.25 mm |
| Wuji（滤波） | 3.99° | 0.3084 | 2.64 mm |

M1 能表达这批近同指尖姿态的额外方向变化，这是本轮骨骼信息收益的具体证据。同时 M1 的机器人指尖漂移更大，不能描述为“固定指尖换姿态已经解决”。配对强相关，不等于 384 次独立手势，不能以此作统计显著性或广泛泛化声明。

## 尚未解决：相邻手指碰撞和近捏合

共同 GeoRT SAPIEN 资产上，每个验证片段等距抽 100 帧，共 900 帧，检测自穿透 >2 mm：

| 模型 | 抽检比例 | 最大穿透 |
|---|---:|---:|
| 原始 GeoRT | 85.78% | 19.00 mm |
| 旧 H1 | 73.11% | 17.75 mm |
| M0 | 47.00% | 14.86 mm |
| M1 | **45.22%** | **12.74 mm** |
| Wuji（滤波） | **10.33%** | **6.08 mm** |

M1 的主要问题是中指和无名指中末节相互穿透；其中 distal/distal 在 900 个抽检帧中出现 350 帧，多个碰撞对可能来自同一帧。小指专项片段的碰撞比例为 82%，因此良好的小指方向和零反曲不能代表整手可执行性。Wuji 在同一资产上也有穿透，这不是实机碰撞成功率或安全认证。

在人手两指尖距离 <15 mm 的 pinch_axis 子集，M1 的拇食/拇中/拇小机器人 site 间距分别为 12.80/8.69/12.78 mm；Wuji 为 11.86/9.02/10.01 mm。没有指腹接触真值，不能把这些间距当接触误差或认为越接近 0 越好。

下一阶段应在训练中处理邻指间距和协调可执行性，复查碰撞几何并测量方向、近捏合和精细响应代价。不能只把方向 loss 再加大，也不能直接把某个历史碰撞权重恢复为默认。M0 保留为信息对照，M1 保留为本轮可预览候选；尚不运行测试集模型效果。

## 验证和证据

- 91 项 unittest 通过，覆盖旧模型兼容、新模型导出、输入信息区分、梯度、独立 split、边界和不读取 test 文件。
- 两组训练完成；实际初始 state_dict 逐 tensor 相同；导出与批量 FK 单帧最大误差均小于 1e-7 rad。
- Wuji 在完整验证集所有片段上独立起始，失败记录均为 0，配置/源版本/资产哈希与用户确认的基线一致。
- 已实际生成 SAPIEN 网格离屏图并目视检查张手、握拳和用户截图同一片段首帧。共同正交相机、相同模型资产和姿态，无 PD。预览 CLI 的模型、数据与缓存核验通过；没有驱动实机。
- 默认模型未替换，原数据/检查点未覆盖，未自动提交或推送。测试集未执行模型推理，严格多解测试覆盖薄弱的限制仍保留。

本机证据：`reports/stage4/validation/metrics.json`、`outputs.npz`、`saturation_audit.json`、`extra_diagnostics.json`、`collision_pairs.json`、`preview.png`。训练权重和源码快照分别在 `checkpoint/stage4_M0_seed42`、`checkpoint/stage4_M1_seed42`。

## 预览与复跑

```bash
conda activate geort
cd /home/xense-wufan/GeoRT

# 左人手、中现用 Wuji、右新 M1；查看之前失效的小指片段。
python -m geort.mocap.replay_evaluation \
  -hand wuji_hand2_beta1_right -ckpt_tag stage4_M1_seed42 --weights best \
  --compare-wuji-cache reports/stage4/wuji_val01/little_branch.npz \
  -data /home/xense-wufan/GeoRT/data/manus/val01/little_branch/keypoints.npy --fps 60

# 基本张合：将上面两处 little_branch 都改成 open_close。
# 信息对照：左人手、中 M0、右 M1，查看严格区分样本所在片段。
python -m geort.mocap.replay_evaluation \
  -hand wuji_hand2_beta1_right -ckpt_tag stage4_M1_seed42 --weights best \
  --compare-checkpoint stage4_M0_seed42 --reference-weights best \
  -data /home/xense-wufan/GeoRT/data/manus/val01/ring_branch/keypoints.npy --fps 60

# 复跑请使用新的输出目录；每组固定 seed42/2000步。
python -m geort.train_manus --manifest data/manus_stage4_v1/manifest.json \
  --mode tip_control --output checkpoint/stage4_repeat_M0_seed42
python -m geort.train_manus --manifest data/manus_stage4_v1/manifest.json \
  --mode skeleton --output checkpoint/stage4_repeat_M1_seed42
python -m geort.evaluate_manus --manifest data/manus_stage4_v1/manifest.json \
  --checkpoints stage4_repeat_M0_seed42 stage4_repeat_M1_seed42 \
  --wuji-cache-dir reports/stage4/wuji_val01 --output reports/stage4_repeat_validation
```
