# 可选骨骼输入与全手协调：第一版

组织仓库：XenseRobotics-AI/GeoRT。审计见 [COORDINATION_AUDIT.md](COORDINATION_AUDIT.md)。
本版是可运行的受控实验入口；默认入口和原 checkpoint 保持可用。
历史小指修正实验见 [WUJI_POSTURE.md](WUJI_POSTURE.md)，不会默认叠加到本版输出。

## 增加的信息和构型选择

| 路径 | 真实可用输入 | 网络 | 目标 |
|---|---|---|---|
| 原始 GeoRT | 配置指定的指尖3D位置 | 每指独立IK | 原覆盖、响应方向、平坦度、pinch、可选碰撞 |
| tip_only + context | 同一帧各指指尖 | 原逐指网络 + 全手残差 | 可保持原几何目标 |
| skeleton | 指尖 + 每指3段轴向 + 有效性mask | 每指局部残差，可独立开关context | 可加入末节轴向和相对骨段手型项 |
| pose/指腹法向 | 当前离线文件和ZMQ接口不提供 | **未支持，显式报错** | 需要保留旋转、定义局部法向并完成人机标定 |

同样指尖、不同中间骨段，可以产生不同的增强输入；同一手指输入不变、其他手指变化，全手模块可表达不同输出。
测试仅证明信息和表达通路存在，未经充分训练不代表输出姿态正确。
骨段方向没有绕骨轴的旋转信息，不能当作完整指腹姿态。

每指残差和共享残差先相加，再以 `0.1*tanh(delta)` 加到原网络最后Tanh前的logit上。
最终Tanh仍限制到[-1,1]，按原URDF上下限换算为rad；不放宽机械范围。
新head的最后一层为0、前层为随机非零初始化，初始输出与父模型一致；首步最后一层有梯度，随后前层也能学习。
0.1是保守实验起点，限制新增信息对相同指尖输入的修正幅度，并不保证足以表达所有大幅手型差异。
续训时原逐指网络参数也更新，BN运行统计固定；原网络独立的baseline快照保持不变。

## 文件职责

| 文件 | 本轮职责 |
|---|---|
| `geort/coordination.py` | 特征/mask、版本元数据、有界局部与全手残差 |
| `geort/coordination_data.py` | 不改原schema的npy/npz适配、完整序列拆分验证 |
| `geort/coordination_loss.py` | 原几何目标复用、精确FK和可配置任务损失 |
| `geort/coordination_diagnostics.py` | 局部响应、时间、轨迹占用、逐指与操作关系指标 |
| `geort/train_coordination.py` | CPU有限步训练、校准、baseline快照、保存重载与离线评估 |
| `geort/experiments/B0.json`–`B5.json` | 固定划分/预算的逐项消融配置 |
| `geort/model.py`、`geort/export.py` | 新版本分派和可选mask；旧接口默认保持 |
| `geort/kinematics.py` | 在原FK接口上可选返回实际骨段轴与长度 |
| `tests/test_coordination.py` | 19项兼容性、几何、梯度、信息/协调能力和诊断测试 |
| `docs/COORDINATION_AUDIT.md`、本页、README | 审计、使用、限制与预览入口 |

## 输入、标定和缺失数据

训练和推理接受GeoRT手部canonical坐标、米、21点顺序。角度输出为rad。
`features`保存版本、模式、context开关、目标手侧、关键点顺序、骨段顺序、尺度、置信度阈值和机器人摘要。
`handedness`检查目标机器人侧，不会从坐标猜测源数据左右手；旧文件没有源侧身份信息，调用者必须确认采集约定。
没有自动左右镜像；skeleton首版明确限定Wuji右手。Allegro左右的旧入口/指尖路径有兼容性测试。
`adapt_points`是供显式预处理使用的单位/刚体变换工具，测试覆盖m/mm和正确旋转；不会自动推测未标定输入的变换。

旧 `.npy` (T,21,3) 可直接使用；可选 `.npz` 适配器字段：

| 字段 | 格式 | 行为 |
|---|---|---|
| keypoints | T×21×3 | 必需 |
| valid | T×21 bool | 可选，显式标记无效关键点 |
| confidence | T×21 float | 可选，范围0..1，阈值0.5；非有限/超范围无效 |
| timestamps | T float，秒 | 仅与sequence_id共同出现时用于时间诊断 |
| sequence_id | T标识 | 每条序列必须连续存放，时间严格递增；完整序列不能跨split |

非有限、低置信度及零长骨段通过mask排除，补位0不作为有效方向监督。
必需指尖无效会报错；全无效骨骼却显式开启skeleton也报错。
未知字段、未标定pose模式、机器人摘要不匹配、state_dict不匹配均不会静默忽略。
置信度缺失时，只能使用有限性/长度检查，日志明确标注未提供置信度。
推理可使用 `model.forward(keypoints, valid=..., confidence=...)`，有效骨段比例在 `model.last_input_validity`。

## 损失与可配置项

配置路径 `geort/experiments/B0.json` 至 `B5.json`。
每项输出raw、normalized、weighted、valid_count、total_count、valid_fraction，不只输出total。

- 原几何公式复用 `geort.loss`；保留80/1/1/1000覆盖/方向/平坦度/pinch权重及原物理尺度，不擅自重新归一化baseline。
- `task.axis`：有向末节轴误差 `1-dot`，除以 `1-cos(axis_tolerance_deg)`；不取绝对点积。轴由精确FK最后关节原点→site定义，非指腹法向。
- `task.shape`：相邻骨段点积之差的平方，除以 `shape_tolerance_cos²`；排除同源关系不明确的拇指。不是人机逐关节角标签。
- `task.posture`：非拇指MCP/PIP/DIP分别超过−15°/−5°/−5°的单侧平方惩罚，除以 `posture_tolerance_deg` 的弧度平方。它是自然性偏好，不是安全保证。
- `task.relation`：机器人拇指→其他指相对向量与缩放后人手向量差的平方，除以 `relation_tolerance_m²`。比例仅由训练段的人手双指骨链长度中位数与机器人骨链长度确定，并保存到checkpoint。
- 关系项只在开口≥`relation_min_opening_m`（初值15mm）的姿态上启用；小开口缺少接触标定，排除并报告比例。**不实现接触保持或强制吸附。**
- 开启relation时必须明确将baseline.pinch置0，否则报错；B5替换原pinch以避免重复计罚。近接触操作的最终目标设计仍待数据与标定。
- 所有新权重和容差是待调参起点，不是最优设置。方向/关系是任务目标，posture是选解偏好；第一版加权和不构成严格层级优先级保证。

精确Torch FK可回传网络参数，Neural FK冻结参数但保留输入梯度。
实验入口使用已保存的原Neural FK，不自动训练大型依赖；碰撞训练默认仍为0，原trainer的可选碰撞路径保留。

## 消融与公平性边界

| 配置 | 输入 | context | 新增任务项 |
|---|---|---|---|
| B0 | tip_only | 关 | 无 |
| B1 | tip_only | 开 | 无 |
| B2 | skeleton | 开 | 无 |
| B3 | skeleton | 开 | axis |
| B4 | skeleton | 开 | axis、shape、posture |
| B5 | skeleton | 开 | 前项 + relation，替代pinch |

mode与context可在JSON中独立配置，包括skeleton且context关闭。
各项weight=0关闭。修改配置另存新文件，不覆盖已有实验目录。

原始baseline继续使用 `geort.trainer`。本实验B0是从相同父模型开始的**受控短续训参考**：
B0-B5采用相同seed、预算、同帧均匀采样、固定BN统计和256个固定精确FK目标样本。
它保留原几何目标公式，但没有复刻原trainer的1mm体素重采样、每epoch采样量与BN统计更新流程。
不能将其称作原始200epoch训练的数值复现。checkpoint内另外保存未修改的 `baseline.pth` / `baseline_config.json`。

默认划分：train=[0,2000)，validation=[2100,2448)，test=[2548,3498)，留100帧间隔。
父模型曾见过微调validation；旧录像没有session/operator标签，test也被前几轮开发使用过，不能作为最终独立验收。
有sequence_id时必须按完整序列配置split，不能用默认帧边界随意切割。

## 可执行命令

所有命令在仓库根目录执行，`conda activate geort`。
示例父模型和Neural FK为本机已有文件，不随Git分发；缺失时会明确报错，需由原trainer生成或提供已有权重。

```bash
# 旧baseline推理，不加载任何新结构
python -m geort.mocap.replay_evaluation \
  -hand wuji_hand2_beta1_right \
  -ckpt_tag wuji_hand2_beta1_right_2026-09-15_18-01-42_std_collision_seed0_w0 \
  --weights last -data human_alex --direct-qpos --fps 100

# 原训练入口仍可使用；该命令由用户选择执行，本轮没有启动长训练
python -m geort.trainer -hand wuji_hand2_beta1_right \
  -human_data human_alex -ckpt_tag original_baseline --no-update-last

# 最小增强链路：4步，16帧batch，CPU；输出目录必须不存在
python -m geort.train_coordination \
  --config geort/experiments/B5.json \
  --parent wuji_hand2_beta1_right_2026-09-15_18-01-42_std_collision_seed0_w0 \
  --weights last --data data/human_alex.npy \
  --output checkpoint/coordination_B5_new --steps 4 --batch-size 16 --collision

# 逐步消融：每个配置相同预算；默认配置为20步，不是质量验收训练
for experiment in B0 B1 B2 B3 B4 B5; do
  python -m geort.train_coordination \
    --config "geort/experiments/${experiment}.json" \
    --parent wuji_hand2_beta1_right_2026-09-15_18-01-42_std_collision_seed0_w0 \
    --output "checkpoint/coordination_${experiment}_new" --steps 20 --batch-size 32
done

# 本机已生成的smoke模型预览；仅用于检查接口与输出，不是改进后的成品
python -m geort.mocap.replay_evaluation \
  -hand wuji_hand2_beta1_right -ckpt_tag coordination_B5_smoke_final \
  --weights best -data human_alex --direct-qpos --fps 100

# 本地测试
python -m unittest discover -s tests -q
```

`--seed`、`--steps`、`--batch-size`覆盖实验配置；`--fk-checkpoint`可指定已训练Neural FK。
`--collision`只额外评估目标姿态自穿透，不启用在线修正或运动硬件。
输出包括 `config.json`、`experiment.json`（输入/权重/代码哈希）、`training.json`、best/last、
`selection.json`、`evaluation.json` 和每帧输出 `evaluation.npz`。

## 诊断解释

- 末节轴角度、多指开口/相对向量、手型项及逐指反弯和历史pinch site距离；有效样本比例必须同时查看。
- 真实相邻帧位移比作为局部增益代理；同时统计死区和方向有效率。无响应样本保留为dead zone，不删除后宣称方向正确。
- 分别记录共同移动、相对开口、轴向变化；按开口区域和边界附近分组。它们是自然运动数据上的关联诊断，不是已控制其他变量的因果实验。
- 单指串扰只在“该指变化≥0.2mm、其他各指骨点变化<0.1mm”的真实帧对上统计；样本不足为null，不制造不合理单骨点姿态。
- 输出逐指/多指组合5mm轨迹占用、运动幅度、极限附近帧比例、静止输入时输出变化。没有独立可行参考集时覆盖率为null；不把占用大当作操作更好。
- 缺时间戳时只报告每采样步差分。有真实timestamps/sequence_id时按秒计算速度、非均匀间隔加速度和静止段输出运动；跨序列不求导。
- 缺少可靠机器人目标轨迹时延迟仍为null；没有凭模型耗时推算跟踪滞后。推理耗时仅CPU批量模型耗时，非端到端延迟。
- exact vs Neural FK误差、物理关节越界，以及可选SAPIEN穿透；软先验不构成安全控制器。

## 本轮验证与尚未证明的能力

50项unittest通过：旧Wuji与Allegro左右入口、全关闭/零初始化等价、mask/NaN/退化骨段、单位/刚体变换、
方向正反、理想几何、网络参数梯度、模块可学习、严格元数据和state_dict、边界、序列拆分、死区与实际导出入口。

B0-B5各跑4个优化步骤、batch16；各自保存/重载并离线运行950帧。最大逐帧/批量重载差约3.22e-6 rad，目标越界计数0。
B5最终链路包含SAPIEN自碰撞评估，>2mm穿透约49.37%，因此绝不能宣称输出已经安全可执行。
方向数据几何有效率100%，但没有采集置信度。单指独立变化样本很少，无法据此证明协调质量。
本轮未做图形窗口人工验收、真实Manus在线验证、真机运行、长训练或远端CI。

已实现但未用真实数据验证：带时间戳/sequence_id的时间诊断；低置信度适配；多会话完整拆分。
缺数据/标定而未实现：完整pose/指腹法向监督、真实接触保持、物体稳定性、可靠系统响应延迟。
smoke通过不代表姿态、精度或可操作性改善，也不能用4步输出决定替换默认模型。

后续独立验收需要：固定一套独立采集的动作序列（包含同指尖不同末节方向、单指微调和多指开口），
按会话划分，以相同预算对比B0/B1/B2，先分离“结构”与“信息”收益，再调整新损失。

2026-09-16：已先用现有录像开展实际训练对照，暂不要求补采。阶段门槛见 [STAGE1_PROTOCOL.md](STAGE1_PROTOCOL.md)，结果见 [STAGE1_RESULTS.md](STAGE1_RESULTS.md)。
