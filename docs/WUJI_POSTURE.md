# Wuji 姿态问题与实验状态

本页记录截至 2026-09-16 的开发结论；代码位于 `codex/wuji-posture-disambiguation`。
姿态候选尚未成为默认模型，历史在线修正也默认关闭。

## 问题与原因

目标是同时保持操作所需的指尖/接触精度、合理整指姿态、连续运动和可执行性，
不是强制逐关节复制人手。原网络每指由三维指尖输入预测四个关节，任务损失主要作用于指尖，
不足以排除异常姿态分支。原 direction loss 是指尖位移方向，并非指骨方向。
URDF 的非拇指 PIP/DIP 均允许约 −60°，不是小指独有的范围问题。

同配置不同随机种子的 950 帧模型目标中，严重反弯率如下（PIP/DIP 任一个 < −15°）：

| seed | 食指 | 中指 | 无名指 | 小指 |
|---|---:|---:|---:|---:|
| 0 | 1.79% | 26.42% | 31.26% | 96.84% |
| 1 | 31.37% | 20.84% | 7.58% | 3.58% |
| 2 | 48.42% | 1.26% | 18.00% | 0.95% |
| 3 | 53.26% | 0.42% | 13.05% | 38.84% |
| 4 | 45.68% | 0.00% | 30.53% | 4.74% |

原始证据为本地 `reports/collision_standard_20260915/seed*_w0/evaluation.npz` 的 target。
这支持选解分支与训练结果有关，而非只对小指设置补丁。指长、安装角和数据分布的独立贡献尚未消融验证。

## 最终版本取舍

- 保留碰撞语义修复、直接输出/PD 对照、共享精确 FK、旧模型加载兼容与可复现评估。
- 在线逐帧优化和固定帧差限幅保留作诊断实验，不能作为解决训练歧义的默认路径。
- 训练姿态约束值得继续研究，但须覆盖四根非拇指，并评估指尖、接触、整指姿态与连续性。
- 尚未支持把候选模型、碰撞 loss、硬人手角映射或固定 DIP/PIP 耦合设为默认。
- 当前结果来自同一录像的开发评估，缺少独立跨会话、小指对捏及真机验证。

## 文件与复现前提

源码、测试和本页纳入 Git；`checkpoint/`、`reports/`、`runs/` 的本地实验输出保持忽略。
下列具名 checkpoint 未随仓库分发；精确复现实验需要保留的父模型权重。
新训练得到的父模型不能用来宣称复现了同一组数值。不要覆盖已有输出目录或模型别名。

已有上游示例 `data/human_alex.npy` 时，可按历史切分生成实验输入：

```bash
python - <<'PYDATA'
from pathlib import Path
import numpy as np
out = Path('reports/collision_standard_20260915')
out.mkdir(parents=True, exist_ok=True)
points = np.load('data/human_alex.npy')
assert points.shape == (3498, 21, 3)
for name, values in [('train.npy', points[:2448]), ('test.npy', points[2548:])]:
    with (out / name).open('xb') as f:
        np.save(f, values)
PYDATA
```

### Wuji 小指：训练阶段姿态候选

预览本地已训练的候选（完整 21 点输入，**不加运行时姿态修正**）：

```bash
python -m geort.mocap.replay_evaluation \
  -hand wuji_hand2_beta1_right -ckpt_tag wuji_posture_train_v1_lr1e5 \
  -data human_alex --weights best --direct-qpos --fps 100
# 将 --direct-qpos 换成 --compare-pd，可查看人手 / 模型目标 / PD 实际姿态。
```

该 checkpoint 是本机实验产物，不随 Git 分发，也没有覆盖原来的 `*_last`。
它是在旧模型上进行第二阶段微调：只训练小指，添加三段人手指骨方向，
用精确可微 URDF FK 约束指尖位置、PIP/DIP 背伸、MCP 过伸与相邻帧变化。
其余四指冻结；没有把人手机械角一一复制，也没有锁死 DIP/PIP 比例。
`model_type=wuji_posture_v1` 会让加载器使用完整骨架，旧 checkpoint 仍走原来五指尖输入。

复现（输出目录必须不存在；数据必须是等间隔采样的单段连续录像）：

```bash
python -m geort.train_posture \
  --parent wuji_hand2_beta1_right_2026-09-15_18-01-42_std_collision_seed0_w0 \
  --weights last --data reports/collision_standard_20260915/train.npy \
  --output checkpoint/wuji_posture_reproduce --epochs 1500 --lr 0.00001

python -m geort.mocap.evaluate_posture \
  --checkpoint wuji_posture_train_v1_lr1e5 --weights best --raw-only \
  --reference-checkpoint wuji_hand2_beta1_right_2026-09-15_18-01-42_std_collision_seed0_w0 \
  --reference-weights last --data reports/collision_standard_20260915/test.npy \
  --output reports/wuji_posture_reproduce_eval
```

微调使用前 2000 帧，跳过 100 帧，用余下 348 帧选 best；父模型见过这段验证数据。
另外的 950 帧测试与父模型训练之间隔离 100 帧，但仍来自同一录像。
测试中严重反弯帧（PIP/DIP < −15°）由 96.84% 降至 0%，其他四指逐值不变；
相对父模型的小指尖偏移均值 5.64 mm、P95 15.32 mm，仍有 27.47% 帧超过约 −5° 的轻微背伸容差，
2.63% 帧出现 MCP < −15.1°。因此这是候选模型，尚不作为默认；不是全姿态或真机保证。
完整录像仍有 31.19% 帧 MCP 过伸，最大关节帧间差 81.17°，尚未完整解决整指姿态与连续性。
完整录像含训练数据，不能作为独立测试。结果与限制见下方实验记录。

### 实验性 Wuji 小指姿态修正（历史对照）

此路径保留作诊断对照。逐帧求解会切换分支，连续性限幅会引入指尖滞后；新训练候选默认不叠加它。

在不重训、不修改物理关节限位的情况下，比较 **人手 / 原始输出 / 修正输出**：

```bash
python -m geort.mocap.replay_evaluation \
  -hand wuji_hand2_beta1_right -ckpt_tag wuji_hand2_beta1_right_last \
  -data human_alex --weights best --compare-posture --fps 100
```

`--compare-posture` 自动启用修正，三栏均不做物理步进。它与 `--compare-pd`、
`--direct-qpos` 互斥。普通双栏直接显示可用 `--direct-qpos --posture-correction`；
检查修正目标的 PD 表现可用 `--compare-pd --posture-correction`，此时中间仍为原始输出，
右侧为修正目标经过 PD 后的结果。不开启这些修正选项时保持原行为。

修正器只调整小指的 4 个关节，其他手指逐值保持不变。使用真实 URDF FK 和解析 Jacobian，
以原模型指尖位置为目标，减小 PIP/DIP 过度反弯，并用人手末节方向辅助选解。
默认增加每数据帧 8° 的小指关节变化上限；位置预算与连续性冲突时，优先连续性并明确报告预算放宽。
它不要求逐关节复制人手，也不固定 PIP/DIP 比例。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--posture-budget-mm` | 2 | 修正前后小指尖的位置预算；连续性冲突时可放宽并报告，不约束 PD 实际轨迹 |
| `--posture-extension-deg` | 5 | PIP/DIP 允许的轻微背伸容差，超过后施加单侧软惩罚；并非硬保证 |
| `--posture-mcp-extension-deg` | 15 | 避免新增超容差的 MCP 背伸；与连续性边界冲突时报告放宽 |
| `--posture-direction-weight` | 0.05 | 归一化 DIP→指尖方向的软权重；设为 0 可单独检查反弯惩罚 |
| `--posture-max-step-deg` | 8 | 每次数据帧更新的关节变化上限；0 恢复第一轮独立帧模式。不是 rad/s 限速或硬件速度保证 |
| `--posture-starts` | 3 | 多起点求解；设为 1 减少计算，但可能更容易保留局部解 |

独立帧模式重新检查物理限位、指尖预算及反弯惩罚；失败时回退原输出。
连续模式进一步约束所有小指关节的帧间变化，包括正常原始姿态和求解失败的帧。
约束冲突或未求得可行解时，在关节变化范围内尽量接近指尖目标；不会直接跳回原命令。
Python 结果的 `tip_budget_satisfied`、`mcp_guard_satisfied` 表示是否满足对应预算/偏好，
`tip_budget_relaxed` 状态明确提示指尖偏移超预算。恢复阶段不保证反弯惩罚逐帧降低。
首帧或 `reset()` 后没有前帧约束，因此不包含从真实设备初始姿态到首帧的启动过渡。
方向输入须与 GeoRT 数据一样位于手部坐标系；零长度末节向量会关闭该帧的方向项。
跨动作序列使用 Python API 时调用 `corrector.reset()` 清除上一帧参考。
这是 CPU 数值优化，`--fps 100` 不保证实测 100 FPS。修正器没有碰撞约束，
姿态改善仍可能伴随碰撞、位置滞后和分支过渡，需要分别检查，不能当作硬件安全控制器。

可复现的 CPU 评估（默认在完整 `human_alex` 上对比基线、仅惩罚，以及方向项的 1/2/5 mm 预算）：

```bash
python -m geort.mocap.evaluate_posture \
  --checkpoint wuji_hand2_beta1_right_last --weights best \
  --data data/human_alex.npy --collision \
  --output reports/wuji_posture_new_run
```

添加 `--continuity-comparison` 可对比第一轮独立帧模式与 4/8/12° 每帧连续性上限。
输出目录必须尚不存在；保存模型/数据/代码哈希、每帧关节与指尖数据和 `metrics.json`。
指标包含反弯、末节方向、关节跳变、捏合、位置变化、采样轨迹体素覆盖和可选真实自碰撞深度。
体素覆盖只描述本次输入轨迹，不等于完整可达工作空间；当前模型曾用此数据训练，
全量回放结果不应称为独立泛化验证。`--stride` 采样时，关节差分是采样帧间差，不是每原始帧速度。
该命令不做 PD 回放、不训练模型、不更新 checkpoint 别名。


---

# Wuji 训练阶段姿态实验

第二阶段微调完成，候选模型 `wuji_posture_train_v1_lr1e5/best.pth`；尚不替换默认。

## 已回顾并调整的旧方案

- 碰撞过滤属于模型语义修复，保留。碰撞 loss 未能解决姿态多解且曾损害对捏，默认仍为 0。
- 逐帧 SLSQP 虽能守住指尖预算，但有解分支跳变；仅保留诊断入口。
- 8° 事后限幅消除了大跳变但引入跟踪滞后；新模型预览不用该限幅。
- 精确 FK 提取到共享模块，训练和历史校正器共用，避免两份运动学实现漂移。

## 训练与模型

父模型是标准 seed0 / w_collision=0 / last；小指输入由指尖位置扩展为位置 + 三段单位指骨方向（12 维）。其他四指权重及 BN 统计冻结。小指也使用固定 BN 统计，避免连续帧训练与单帧推理差异。

Loss = 指尖相对父模型偏移 / 5mm 的平方 + 10 ×（PIP/DIP 单侧背伸惩罚 + 0.25 × MCP 惩罚）+ 0.1 × 末节方向误差 + 0.1 × 相邻帧关节差分 / 0.1rad 的平方。反弯容差为 5°，MCP 为 15°，均是软偏好；物理关节范围仍由原模型 Tanh 参数化。没有硬复制人手角度或 DIP/PIP 耦合。

微调 train=[0,2000)，validation=[2100,2448)，Adam lr=1e-5，1500 epochs，验证最优为 epoch1000。父模型训练见过 validation；测试为原录像 [2548,3498)，与父模型训练间隔 100 帧。同一录像的单次训练结果，不是跨人/跨动作泛化证据。

首组 lr=1e-3 试跑输出饱和，已拒绝采用；保留实验目录便于追溯。降低学习率后训练稳定；测试数据未用于这次学习率调整或 checkpoint 选择。

## 950 帧测试（无运行时修正的新模型）

| 方法 | 严重反弯帧 | 最大关节帧间差 |
|---|---:|---:|
| 父模型 | 96.84% | 34.40° |
| 第一轮逐帧修正 | 1.68% | 108.34° |
| 第二轮 8° 限幅 | 9.37% | 8.00° |
| 训练候选 | 0.00% | 29.25° |

小指尖相对父模型偏移 mean/P95/max = 5.64/15.32/17.96 mm。其他四指逐值一致，故拇指—食指对捏目标也保持一致。

轻微背伸（PIP/DIP < −5.1°）仍有 27.47% 帧，MCP < −15.1° 有 2.63%。末节方向平均误差 32.84° → 28.67°。5mm 指尖占用体素 164 → 168，但不等价于灵活性完全保持；关节范围重新分配，PIP 范围缩小，MCP/DIP 范围扩大。该测试无小指对捏帧，不能声称小指对捏能力保留。

自碰撞穿透 >2mm 为 47.26%；此次没有解决其他手指及整手碰撞。完整 3498 帧严重反弯率为 1.32%，MCP < −15.1° 为 31.19%，最大关节帧间差 81.17°。这说明存在弯曲向 MCP 转移及连续性残留问题。完整录像包含训练数据，不能作为独立测试。

## 本地验证

31 项 unittest 通过，包括精确 FK/Jacobian/自动微分一致性、旧权重等价初始化、冻结其他手指、序列退化输入、历史校正器约束。
实际导出加载器逐帧跑完 950 帧，与批量评估最大差 5.8e-05 rad；CPU 单帧平均 0.27 ms（本机单线程，不含渲染与控制）。
SAPIEN 无界面 PD：official 每目标 10×0.01s，400/10/10，首帧预热100步。小指实际严重反弯为0，指尖跟踪自身目标 P95=3.04mm，实际最大帧间角差29.56°。这不是100Hz真机验证，也不包含新旧目标偏移。

## 下一步合理方向

增加跨人/跨动作及小指对捏数据；对大偏移姿态加入接触目标和位置优先的约束训练，再在验证集做权重消融。继续直接评估模型输出，避免用事后限幅掩盖训练问题。

## 预览

```bash
cd GeoRT
conda activate geort
python -m geort.mocap.replay_evaluation \
  -hand wuji_hand2_beta1_right -ckpt_tag wuji_posture_train_v1_lr1e5 \
  -data human_alex --weights best --direct-qpos --fps 100
```

改用 `--compare-pd` 检查物理跟踪。不要叠加 `--posture-correction`，否则不能独立观察训练效果。
