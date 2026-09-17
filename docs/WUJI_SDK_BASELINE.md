# Wuji SDK 内置重定向对照（2026-09-17）

当前 M1 的整体遥操质量尚未达到现用 Wuji。它的静态末节方向误差更低，但穿透显著更多，同等外部 LP 条件下的微调优势不成立。SDK 内置版现已作为第三类方法接入离线评估和实时仿真；它并非旧 Wuji 的全面升级替代品。

## 对照定义

- **现用 Wuji**：保留用户确认的 `adaptive_analytical_manus_wuji_hand_2_right.yaml`、源 commit 和 LP=0.3，定义不变，见 [原有基线](WUJI_BASELINE.md)。
- **SDK**：隔离环境 `.venv-wuji-sdk`，`wuji-sdk==2026.8.31`、本次 NumPy 2.5.3。调用 `RetargetSession.for_hand(HandModel.WujiHand2, Handedness.Right)`，只调用 `step/reset`，不创建设备管理器。
- 输入为相同的右手 Manus、MediaPipe 顺序、米制 `(21,3)` 骨架。SDK 输出为 firmware 顺序，显式按关节名称转为公共 FK 顺序；拇指、食指、中指、无名指、小指各4关节。映射与 LeRobot `WUJI_DEVICE_JOINT_BASENAMES` 一致。
- SDK 保留内置热启动与滤波，每段动作独立 reset。公开接口不提供滤波系数、未滤波输出或内部求解状态，不能声称其 LP 与旧版相同，也不能报告“零内部 fallback”。无异常且输出有效只说明接口运行成功。
- SDK 使用内置调参，不加载原 Manus YAML；其内部资产没有通过该接口导出。本轮比较的是这些配置在同一输入、同一 GeoRT Hand2 beta1 资产上的输出表现，不能解释成不同算法在相同内部模型/目标/滤波下的消融，也不能据此断言 SDK 原生资产上的真实接触质量。

版本依据：[官方 API](https://docs.wuji.tech/docs/en/wuji-sdk/latest/retargeting/)；[2026.8.31 发布说明](https://docs.wuji.tech/docs/en/wuji-sdk/latest/release-notes/)说明捏合的拇指—指尖接触有调整。本机 LeRobot 依赖声明仍固定 SDK 2026.8.3，本次没有修改或升级该环境。

## 本次实测

固定 Stage4 模型与 seed42 抽样；`val01` 9段、20,988条主机读数，测试集未运行模型。全部 SDK 输出有限、帧数和公共关节范围检查通过，无截断。原有6种条件的碰撞结果与抽样索引完全复现。

| 方法 | 末节方向误差 ° | 开口代理误差 mm | 穿透 >2 mm | 穿透 >5 mm |
|---|---:|---:|---:|---:|
| 原始 GeoRT | 80.13 | 44.14 | 85.44% | 74.56% |
| M0 | 5.21 | 3.98 | 46.89% | 14.67% |
| M1 | 3.70 | 4.21 | 45.56% | 16.00% |
| M1 + LP=0.3 | 4.24 | 4.38 | 45.61% | 16.61% |
| 现用 Wuji，LP=0.3 | 12.80 | 7.56 | 10.11% | 0.72% |
| Wuji 未滤波 | 12.60 | 7.43 | 10.67% | 0.78% |
| SDK 2026.8.31，内置滤波 | 13.27 | 5.60 | 25.94% | 3.33% |

方向和开口为9段等权；开口只计人手间距≥15 mm，使用训练集确定的尺度，是代理目标。穿透为共同 SAPIEN 资产的相同1,800条分层抽检读数，不能视作真实碰撞率或安全认证。

微调12个幅度/间隔分组中，SDK 的响应方向误差全部高于现用 Wuji；M1未滤波胜10/12，M1加LP=0.3后胜1/12，旧Wuji也去滤波时M1胜3/12。分组有关联，不是独立试验；相同外部 LP 也不等于相同总延迟。

接近捏合的人手间距<15 mm样本中，拇小指公共 site 间距：M1 12.78 mm、现用 Wuji 10.01 mm、SDK 6.35 mm（54条）；拇食指分别12.80/11.86/10.83 mm（43条），拇中指8.69/9.02/10.84 mm（73条）。无名指该区间无样本。site靠近不是接触成功，SDK这项变化也伴随更多穿透。

SDK `step()` 本机逐帧时间中位0.183 ms、P95 0.279 ms，不含进程通信、手套或渲染，不是端到端延迟，也不能直接与批量 GeoRT 推理比较。SDK已经很快，因此“神经网络更快”暂不构成实际优势证据。

完整缓存：`reports/stage4/wuji_sdk_20260831_validation/`；包含SDK版本、native二进制SHA256、worker SHA256、NumPy版本、输入/配置/输出哈希及逐任务指标。

五组对照：`reports/stage4/difference_audit_sdk/metrics.json`、`collision_depths.npz`、`summary.png`。记录包括方向与开口、相近指尖/不同方向配对、微调、捏合分组、自穿透。

## 预览与复现

本机隔离环境已经安装；其他机器可在仓库根目录运行：

```bash
conda activate geort
python -m venv .venv-wuji-sdk
.venv-wuji-sdk/bin/python -m pip install 'wuji-sdk==2026.8.31' 'numpy==2.5.3'
```

实时右手 Manus 仿真：

```bash
conda activate geort
cd /home/xense-wufan/GeoRT
python -m geort.mocap.live_comparison --sdk-reference
```

左起 **人手 → SDK 2026.8.31 → 现用 Wuji → M1**。`L`只切换M1滤波；`F/T/G/R`行为不变。默认不加选项时仍显示原始GeoRT列。

不用手套，先看微调录像：

```bash
python -m geort.mocap.live_comparison --sdk-reference \
  --replay data/manus/val01/micro/keypoints.npy
```

缓存与完整评估复现（输出目录必须尚不存在）：

```bash
python -m geort.evaluate_wuji_sdk \
  --output reports/stage4/wuji_sdk_recheck
python -m geort.evaluate_differences \
  --sdk-evaluation reports/stage4/wuji_sdk_recheck \
  --output reports/stage4/difference_sdk_recheck
python scripts/plot_difference_audit.py \
  --report reports/stage4/difference_sdk_recheck/metrics.json \
  --output reports/stage4/difference_sdk_recheck/summary.png
```

验证：14项定向测试通过（含SDK逐帧/批量一致、reset复现、关节名称重排、非法输入失败）；`--sdk-reference --check`通过；实际桌面窗口90帧录像回放并正常退出，正交视角确认。没有连接手套或驱动实机；本轮不宣称SDK版真人遥操验收通过。

## 后续判断与工作量

**相近水平的目标**是常用张合、捏合和微调不出现明显退步，穿透降至现用Wuji附近，并得到操作者同场对照认可。更小的方向误差不单独构成通过。

按现有数据与环境可继续使用估计，先投入 **2–3轮受控实验，约1–2周集中开发工作量**，争取达到上述有限范围的相近体验。这是带风险的工程估计，不是已验证的收敛时间。跨操作者、校准变化与真实抓取的全面水平尚无法给出可靠工期。

下一轮保持输入、结构和训练预算一致、seed42，先做 M1 / 加局部响应约束 / 加核验过的自碰撞约束 / 两者同时 四组。碰撞几何和代理loss须先检查；过去仅增加碰撞项曾牺牲捏合，本轮必须同时看小开口、低响应、方向和穿透，不能只追求穿透比例。全手上下文作为后续独立消融，不一次改变全部因素。

有机会超越的地方是对细微输入更可预测的响应、方向变化与指尖位置的合理协调、适配本系统的多指关系；目前尚无证据说明能同时得到这些收益。若2–3轮仍只改善方向指标、无法缩小穿透和微调差距，应调整路线，考虑以Wuji为基础的受控学习修正，而不继续凭训练轮数押注纯GeoRT。最终选定候选后才启用保留测试会话，再安排真人验收。
