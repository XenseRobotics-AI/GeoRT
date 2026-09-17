# 固定对照：现用 wuji-retargeting

2026-09-16，用户确认使用LeRobot默认的 `adaptive_analytical_manus_wuji_hand_2_right.yaml`。后续实验固定比较：原始GeoRT、候选GeoRT、这套现用方案。原始开发目标见 [DEVELOPMENT_GOALS.md](DEVELOPMENT_GOALS.md)，不是只比较反弯率。

## 版本与接口

- 本机源代码：`/home/xense-wufan/lerobot-xensehand/third_party/wuji-retargeting`。
- 固定commit：`6eafdb22085f0e29c1d58c62f88f77ae1e971d8c`。
- 配置：`/home/xense-wufan/lerobot-xensehand/configs/manus_wuji/adaptive_analytical_manus_wuji_hand_2_right.yaml`。
- 配置SHA256：`d9f0f6738bd47ba5730daffe143eeb0b27e10ce37c8150d755fb3cf622012aa0`。
- 规范存储：`geort/baselines/wuji_manus_right.json`，包含原生URDF哈希。
- 主对照保留 `AdaptiveOptimizerAnalytical`、原有热启动/速度正则、骨段缩放、方向和关节耦合权重、坐标旋转及 `lp_alpha=0.3` 滤波。另存未滤波输出作诊断，不把关闭滤波的结果冒充现用方案。
- LeRobot实际入口：`src/lerobot/teleoperators/manus_wuji/manus_wuji.py`，`Retargeter.from_yaml(..., hand_side='right')` → `retarget(keypoints)` → 按关节名称重排。这里比较的是重定向输出，不是机器人驱动反馈或驱动层启动平滑后的动作。

## 数据和公平性边界

现有数据是3498帧、21个MediaPipe顺序关键点、米制右手骨骼。通过库的原生 `retarget()` 坐标变换输入，不猜测额外旋转。抽样刚体变换不变性检查已通过；这只验证坐标变换性质，不证明真实在线标定一致。

整个录像保持顺序，帧0重置状态，后续保留求解器热启动与滤波历史。验证段从完整录像缓存切片，因此有前序输入状态；这不是独立会话冷启动评估。当前入口仅支持单条连续 `.npy`，不能把多会话拼接后不重置。

两份机器人资产并不相同：现用Hand 2 URDF和GeoRT beta1在基座定义、部分拇指几何和指尖位置等处有差异。**不改现用方案内部URDF或参数**，保留来源哈希；输出按相同关节名称映射，在共同GeoRT精确FK上评估/显示。因此是现用方案的整体输出对照，不能把所有差异都归因于优化算法。超出公共关节限位会报错，不静默裁剪。

基线不是机器人关节真值，不把“更像wuji输出”直接当训练目标。旧父模型位置偏移门槛只做行为差异参考，不能要求新算法必须复刻旧模型错误。

现已支持[同一套GeoRT loss完整量化](STAGE3_RESULTS.md)：包括原生扰动重算的响应方向/flatness、Neural FK训练口径与精确FK复核。每个扰动恢复同一帧之前的求解器/滤波状态；anchor单列，不将其巨大的父模型偏好当作公平排名。

## 接入与复现

Pinocchio/NLopt与现有GeoRT依赖隔离，使用仓库内 `.venv-wuji-baseline`。没有升级GeoRT原环境。锁定本次实际离线依赖（不代表已核实实机全部依赖版本）。

```bash
cd /home/xense-wufan/GeoRT
# 新机器先创建独立环境；源仓库和配置路径需要存在。
python -m venv .venv-wuji-baseline
.venv-wuji-baseline/bin/python -m pip install \
  --index-url https://mirrors.ustc.edu.cn/pypi/simple \
  -r geort/baselines/requirements.lock.txt

# 本机已经生成，勿覆盖；复跑请换一个新的输出路径。
.venv-wuji-baseline/bin/python scripts/cache_wuji_baseline.py \
  --source /home/xense-wufan/lerobot-xensehand/third_party/wuji-retargeting \
  --config /home/xense-wufan/lerobot-xensehand/configs/manus_wuji/adaptive_analytical_manus_wuji_hand_2_right.yaml \
  --data data/human_alex.npy \
  --output reports/baselines/wuji_manus_right_human_alex.npz

conda activate geort
# 已有模型的三方量化评估；输出文件需不存在。
python -m geort.evaluate_stage --checkpoint stage1_G2_seed0 \
  --wuji-cache reports/baselines/wuji_manus_right_human_alex.npz \
  --split validation --output /tmp/geort_G2_wuji_validation.json

# 后续训练在原命令中加入此参数，训练结束自动输出同一划分的三方报告：
# --wuji-cache reports/baselines/wuji_manus_right_human_alex.npz
# 使用seed42。历史命令仍能运行，但缺少外部对照会显式标记missing，不能作为新一轮完整验收。

# 左人手、中现用wuji（含其原有滤波）、右G2；统一GeoRT资产，无PD。
python -m geort.mocap.replay_evaluation \
  -hand wuji_hand2_beta1_right -ckpt_tag stage1_G2_seed0 --weights best \
  --compare-wuji-cache reports/baselines/wuji_manus_right_human_alex.npz \
  -data human_alex --fps 100
# 可增加 --wuji-output unfiltered 检查滤波前结果。
```

缓存含输入/配置/源代码/原生资产哈希、依赖版本、原始帧号、关节名、两种输出、NLopt状态和失败回退记录。加载时核验固定版本、配置、数据哈希、完整帧顺序、关节集合和有限性。

并排对比默认采用正交投影，避免左右位置造成透视角度差；按 **R** 恢复统一视角。SAPIEN实际窗口首次渲染会重建为透视相机，因此每帧渲染后检查投影，必要时恢复正交并立即重绘，保留当前相机姿态。已用实际窗口连续帧确认恢复后保持正交；仅做离屏相机检查无法覆盖这个问题。模型输出与基座旋转不随投影方式改变。

## 当前验证与结果边界

- 实际跑完3498帧，记录到的求解失败/回退为0。前32帧缓存与直接调用现用 `retarget()` 的输出误差为0。
- 原始数据及历史模型未改；两步训练接入验证通过，仅验证报告/导出链路，不算新训练成果。
- 同一348帧验证段，在共同精确FK上：原始GeoRT末节轴向平均误差约35.48°，G2约28.34°，现用wuji含滤波约18.89°、滤波前约17.74°；方向有效比例100%。wuji四根非拇指的PIP/DIP低于−15°比例均为0。这些结果不足以证明整体操作质量，**当前G2也没有优于现用方案的充分证据**。
- 报告同时包括手型、开口、局部响应/增益/死区、整体/相对运动、边界响应、静止输入输出位移、幅度和关节限制。原始数据无时间戳，真实速度、加速度与跟随延迟不可验证；缺少物体/指腹标定，不声称接触稳定性或完整指腹朝向。无独立可行集合，不报告真实覆盖率。
- 求解耗时是本机预处理+优化+滤波的wall time，不是实机端到端延迟；不能直接与GeoRT批量均摊耗时计算加速比。
- 59项本地测试通过；CLI、缓存/指标和训练接入已验证。GUI目视和实机对照尚未完成。

本机详细量化报告：`reports/stage1/G2_with_wuji_final_validation.json`。今后每轮应围绕“增加了什么可表达信息、改变了什么选解、是否保住精细控制”汇报，不能只汇报反弯率或total loss。
