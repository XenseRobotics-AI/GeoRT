# 阶段三：统一loss量化与骨骼分支受控实验

2026-09-16。按[开发目标](DEVELOPMENT_GOALS.md)和[本轮协议](STAGE3_PROTOCOL.md)，完成Wuji完整几何loss重算，以及两组冻结基础网络的实际训练。**新增分支已经学起来，但未证明骨骼信息带来总体收益；不替换默认模型。**

## 用我们的loss评价Wuji

使用同一348帧验证段、同一个T2目标配置、训练段得到的骨长比例、seed42扰动、同一256点机器人覆盖集合。Wuji保留用户确认的原生预处理、优化器、URDF和滤波，输出按关节名映射后，用共同GeoRT几何评价。不是重新训练Wuji，也不把Wuji输出当机器人真值。

以下是**精确FK口径、乘权重后的loss贡献**，数字越小仅表示该项训练目标更小。响应方向为负cos之和，最优为−5，故合计可以为负。完整JSON另含raw、normalized、weighted、有效样本数及Neural FK训练口径。

| 模型 | coverage | 响应方向 | pinch | 末节方向 | 骨段形状 | 姿态 | anchor | 去anchor合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始GeoRT | 1.5249 | −4.8821 | 0.0623 | 0.3235 | 0.0312 | 5.4209 | 0 | 2.4807 |
| T1 | 1.5240 | −4.4665 | 0.0680 | 0.1770 | 0.0296 | 0.0287 | 0.4027 | −2.6391 |
| T2 | 1.5256 | −4.7675 | 0.0701 | 0.1314 | 0.0279 | 1.2792 | 0.5318 | −1.7333 |
| H0，只读指尖的局部头 | 1.5224 | −4.7653 | 0.0724 | 0.1247 | 0.0297 | 1.3974 | 0.5039 | −1.6187 |
| H1，骨骼局部头 | 1.5234 | −4.7665 | 0.0729 | 0.1300 | 0.0302 | 1.4224 | 0.4587 | −1.5875 |
| Wuji，现用滤波 | 1.7081 | −1.6181 | 0.1621 | **0.0932** | 0.0775 | **0.0049** | 95.0896 | 0.4278 |
| Wuji，未滤波 | 1.6564 | −1.6138 | 0.1343 | **0.0817** | 0.0775 | **0.0061** | 93.4035 | 0.3422 |

flatness贡献介于0.000007–0.000080，已计入合计，表格未单列。relation权重仍为0，但真实计算其原始值；归一化relation分别为原始6.0329、T1 6.0586、T2 5.8967、H0 5.9452、H1 6.0029、Wuji滤波5.1176、未滤波4.5959。

**不能用总loss认定Wuji更差。** Wuji滤波完整精确FK总loss为95.5174，其中95.0896来自anchor。anchor要求复刻原始GeoRT指尖位置，天然偏向原始模型，本来只应控制继续训练时的行为漂移。Wuji在末节方向、姿态、相对向量方面更好，骨段形状、原pinch和原响应方向项更差，反映目标之间的取舍。

即使去掉anchor，总loss仍不是最终质量评分：姿态采用T2的参考MCP下界；原pinch用两site重合，不能代替真实指腹接触；coverage是抽样几何Chamfer，不是可操作空间覆盖率。响应方向用GeoRT原训练的整条手指链平移扰动，Wuji仍经过原生掌坐标归一化，包含预处理响应差异；不等价于相邻帧运动方向或实机跟随体验。不能用此表单独决定替换哪个模型。

Neural FK口径也全部计算了：Wuji滤波/未滤波完整总loss为95.5354/93.7716，精确FK为95.5174/93.7456。各模型Neural FK指尖误差均值约2.21–2.46 mm；这里的大anchor差距不是这一级别的FK近似误差造成的。

### 完整响应重算与历史状态

顺序输出缓存无法计算局部响应/flatness。新增`cache_wuji_probes.py`为每帧计算原始、+offset、−offset、独立delta四个输入，位移为2 mm标准差的高斯整链平移。每个分支都恢复**该帧之前**的原始关节热启动值及滤波状态，不让扰动推进真实录像历史。

348帧原始输出重算与已验证缓存的最大关节差为0 rad；前8帧反向遍历扰动分支，结果差为0 rad；全部重算无求解失败。所有模型使用相同扰动。`baseline_from_outputs`复用原几何公式；新增测试保证在线调用和保存输出调用一致、线性映射平坦性为零、同向响应方向为−5。另保留来源哈希、关节顺序、限位与原始输出复核，不静默裁剪。

## 下一步实验已经执行

H0/H1都从T2 best提取相同逐指基础网络，并冻结其全部参数和BN缓冲；同样的15→64→4局部头，初始化和采样seed42，2500步，其中1000步拟合原有训练标签。局部头学习率1e−3，logit残差上界仍0.1，原机械限位不变；全手上下文关闭。

H0只读取指尖，额外12维显式置零；H1读取真实骨轴及有效性。两组结构/参数数量相同，输入信息不同，H0零输入对应的参数不会获得相同信息或梯度，不能声称有效输入维度相同。标签、loss、位置参考和姿态mask不变。训练版元数据v2显式记录局部输入策略；旧v1模型保持严格兼容。

| 验证指标 | 冻结T2 | H0 | H1 |
|---|---:|---:|---:|
| best步数 | 原2500 | 2000 | 2200 |
| 小指严重PIP/DIP反曲 | 34.77% | 38.22% | 35.92% |
| 最大逐指位置偏移P95 | 6.56 mm | 6.37 mm | 6.47 mm |
| 末节方向平均误差 | 22.44° | 21.90° | 22.19° |
| 开口平均误差 | 12.88 mm | 12.89 mm | 12.91 mm |
| 相邻帧位移响应方向均值 | 12.27° | 12.28° | 12.51° |
| 局部头实际关节修正均值 | 无新增头 | 0.711° | 0.939° |

两组中指MCP低于−15°的比例均15.52%，父模型8.91%；两组都未通过反曲改善、MCP、5 mm位置门槛。50个保存快照全部未过关，没有继续用测试段挑模型。每个快照均验证基础网络与T2逐tensor完全一致；完整模型保存后，通过实际导出推理入口重载，误差约3.1e−6 rad。

本轮H1比上一轮T3的0.0417°平均修正明显增大，但两轮基础网络与训练策略不同，这仅说明该头有实际作用，不能单独归因于学习率或宣称质量提高。H1标签预训练后的归一化MSE为0.00614，H0为0.00686，冻结基础网络为0.00910；更好拟合标签没有转化为验证段的明确方向/开口优势。

## 定位到的两个限制

1. **残差修正范围限制了换分支。** 对冻结T2，允许的完整logit±0.1区间下，90.5%的训练帧小指标签至少一个关节仍在可表达区间外（容差0.5°）；食/中/无名分别41.9%/50.4%/47.45%。这约束的是当前标签与当前冻结基础网络，不证明其他映射无解。H1 best在验证段并未大量饱和，因此也不能把本轮失败全部归因于残差上界。
2. **用于检验新增信息的关键样本不足。** 分别在训练/验证段搜索相隔至少30帧、指尖≤2 mm、末节方向差≥20°的真实手势对：五指均为0对。不是没找到近邻：训练段小指有25560对≤2 mm近邻，方向差最大仅8.91°。放宽至5 mm/20°，训练段仅拇指11对、食指5对，其他指仍为0；验证段全为0。这只说明在记录与阈值下覆盖不足，不证明骨骼没有额外信息。

因此不继续盲目堆loss或直接放宽残差。**现在Manus专项采集有了明确用途：补充“相近指尖、不同骨段方向/弯曲分配”的可区分样本。** 建议先做可行性短录制，再按完整会话划分训练和验收。优先动作：小指指尖尽量保持位置的直指/勾曲变化；同一捏合开口下改变末节方向；单指微调而其他指保持；握拳与伸展往返。记录完整21点、真实采集时间/消息时间及会话边界；保留实际可用的原始姿态和标定记录，不能从回放FPS补时间。该采集建议尚未执行，现有Manus链路时间同步仍需单独核验。

## 交付与验证

新增/调整：`geort/evaluate_losses.py`统一评分；`scripts/cache_wuji_probes.py`原生扰动重算；`coordination_loss.py`共用几何公式；`coordination.py`版本化局部信息消融；`train_coordination.py`基础网络初始化/冻结和独立头学习率；S3_H0/H1配置、`scripts/run_stage3.py`、`scripts/audit_stage3.py`及测试。没有改原始训练器默认行为、机器人限位或实机接口。

74项本地测试通过；两组实际训练、完整三方loss与操作指标、50快照冻结检查均完成。模型/报告保留本机，未自动提交或推送。原始训练代码副本保存于`reports/stage3/training_sources/`并与实验元数据哈希核对；随后只将相同几何公式提取为共用函数，等价测试通过。

本机证据：`reports/stage3/common_losses_verified.json`、`H0_validation.json`、`H1_validation.json`、`head_audit_verified.json`；扰动缓存`reports/baselines/wuji_manus_right_validation_probes_seed42.npz`。训练日志与权重在`checkpoint/stage3_H0_seed42`、`stage3_H1_seed42`。数据与权重不随Git分发。

没有独立会话泛化、物体接触、自碰撞或实机验收；无可信采集时间戳，不报告真实速度、加速度和延迟。原始父模型已经见过部分开发数据，本轮是诊断实验。

## 可直接运行的命令

```bash
conda activate geort
cd /home/xense-wufan/GeoRT

# 同帧预览：左人手、中现用Wuji、右H1。诊断模型，尚未过关。
python -m geort.mocap.replay_evaluation \
  -hand wuji_hand2_beta1_right -ckpt_tag stage3_H1_seed42 --weights best \
  --compare-wuji-cache reports/baselines/wuji_manus_right_human_alex.npz \
  -data human_alex --fps 100

# 隔离信息作用：中H0、右H1；按R恢复同一正交视角。
python -m geort.mocap.replay_evaluation \
  -hand wuji_hand2_beta1_right -ckpt_tag stage3_H1_seed42 --weights best \
  --compare-checkpoint stage3_H0_seed42 --reference-weights best \
  -data human_alex --fps 100

# 统一loss重算；输出必须为新路径。console显示分项，JSON保留完整记录。
python -m geort.evaluate_losses \
  --checkpoints stage2_T1_seed42 stage2_T2_seed42 stage3_H0_seed42 stage3_H1_seed42 \
  --wuji-probes reports/baselines/wuji_manus_right_validation_probes_seed42.npz \
  --output /tmp/geort_common_losses.json

# 复跑同预算实验，避免覆盖已有checkpoint。
python scripts/run_stage3.py --prefix stage3_repeat

# 检查当前stage3固定实验的冻结、标签表达范围与样本覆盖。
python -m scripts.audit_stage3 --output /tmp/geort_head_audit.json

# 重新生成Wuji扰动缓存：只做离线求解；输出路径必须尚不存在。
.venv-wuji-baseline/bin/python scripts/cache_wuji_probes.py \
  --output /tmp/geort_wuji_validation_probes.npz
```
