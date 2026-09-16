# Geometric Retargeting

[![CC BY-NC 4.0 License](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

Welcome! This repository contains the code for the paper "Geometric Retargeting: A Principled, Ultrafast Neural Hand Retargeting Algorithm".

## 常用命令速查

所有命令都在仓库根目录、`geort` 环境中运行：

```bash
conda activate geort
# 如当前不在仓库根目录，先 cd 到你的 GeoRT 目录。
```

[训练](#训练-allegro--wuji) · [回放](#回放查看模型输出或-pd-效果) ·
[训练曲线](#查看本地训练曲线) · [资产预览](#不加载模型的资产预览) ·
[参数与模型文件](#常用参数与模型文件) · [安装](#installation)

### 训练 Allegro / Wuji

```bash
# Allegro 右手
python -m geort.trainer \
  -hand allegro_right -human_data human_alex -ckpt_tag allegro_paper

# Wuji Hand2 Beta1 右手（独立训练，不能加载 Allegro 权重）
python -m geort.trainer \
  -hand wuji_hand2_beta1_right -human_data human_alex -ckpt_tag wuji_paper
```

默认开启 **paper loss、统一 BatchNorm 映射、同帧采样**，训练 200 epochs，
seed 为 0；Chamfer / Flatness / Pinch 权重为 **80 / 1 / 1000**。
CLI 使用 tqdm 显示进度。首次训练新手型会额外生成机器人 FK 数据并训练 Neural FK；
已有同名缓存和 FK 权重时直接复用，修改 URDF 后需核验缓存是否仍适用。
这些默认值在 Allegro 上改善了效果，不代表 Wuji 的全部姿态已验证正确。

### 回放：查看模型输出或 PD 效果

检查模型本身时，使用 **`--direct-qpos`**：人手骨骼与机器人显示同一帧，
直接设置机器人关节位置，不进行物理步进或碰撞响应。

```bash
# Allegro：最新一次训练中的 best
python -m geort.mocap.replay_evaluation \
  -hand allegro_right -ckpt_tag allegro_right_last \
  -data human_alex --weights best --direct-qpos --fps 100

# Wuji：最新一次训练中的 best
python -m geort.mocap.replay_evaluation \
  -hand wuji_hand2_beta1_right -ckpt_tag wuji_hand2_beta1_right_last \
  -data human_alex --weights best --direct-qpos --fps 100
```

同一窗口同时比较 **人手骨骼 / 模型直接输出 / PD 结果**（从左到右）：

```bash
python -m geort.mocap.replay_evaluation \
  -hand wuji_hand2_beta1_right -ckpt_tag wuji_hand2_beta1_right_last \
  -data human_alex --weights best --fps 100 --compare-pd
```

每帧只推理一次，两只手共享同一模型输出。中间的显示副本不加载碰撞形状，
在 PD 步进后设置原始关节角，不会与右侧手或地面发生碰撞。
`--compare-pd` 与 `--direct-qpos` 互斥；Allegro 替换手型与 checkpoint 即可。

对比 **PD 回放**时，去掉 `--direct-qpos` 即可，其余参数保持相同。PD 是默认模式，
默认 `--pd-timing official`：每个目标保持 10 个 0.01 秒物理步（共 0.1 秒），
沿用官方的步长和目标保持时间，不插值、不限速。为对齐人手骨骼，先设置当前目标，
步进后只渲染一次；这并非逐行复现官方先渲染上一目标的循环。
`--fps 100` 仅设定目标播放速率，不表示官方模式按真实时间仿真。
实验选项 `--pd-timing realtime` 每帧只推进 `1/fps` 秒，默认控制/物理频率为 500/1000 Hz，
启用插值。100 FPS 下目标保持时间缩短到 0.01 秒，可能明显增加 PD 滞后。
算力不足时播放会变慢，不会跳帧或偷偷增大仿真步长。按 **R** 重置视角，
关闭窗口或 **Ctrl+C** 退出。

### 查看本地训练曲线

```bash
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

浏览器打开 <http://127.0.0.1:6006>。无需账号或上传云端。
`runs/<run_name>/ik.csv` 保存原始与加权 loss；`training.json` 保存实际训练参数。
只有本次训练了 FK 时才会生成对应的 FK 曲线。

### 不加载模型的资产预览

```bash
python -m geort.mocap.replay_evaluation \
  --preview -hand taccap_slave --animate --fps 60

python -m geort.mocap.replay_evaluation \
  --preview -hand wuji_hand2_beta1_right --animate --fps 60
```

静态查看时，用 `--position 0.5` 替换 `--animate`，范围为 0–1。
TacCap 当前只支持资产与开合预览，不支持这里的手指重定向训练。

### 常用参数与模型文件

| 入口 | 参数 | 默认值 / 用途 |
|---|---|---|
| 训练 | `--epochs 200 --seed 0` | 训练轮数、随机种子 |
| 训练 | `--w_chamfer 80 --w_curvature 1 --w_pinch 1000` | 覆盖、平滑、捏合权重；显式参数覆盖默认值 |
| 训练 | `--w_collision 0.01` | 启用论文二值自碰撞惩罚；默认 0，关闭 |
| 训练 | `--no-paper-loss` | 关闭 paper 归约及统一 BN 处理；未显式指定的 Flatness / Pinch 权重变为 0.1 / 1 |
| 训练 | `--no-paired-sampling` | 关闭同帧采样，各 loss 使用独立手指点云；与 paper loss 开关独立 |
| 训练 | `--no-update-last` | 不移动 `<hand>_last` 别名，便于保留当前回放模型 |
| 训练 | `--save-every 25` | 额外保留 `epoch_24.pth`、`epoch_49.pth` 等；默认 0，仅 best / last |
| 训练 | `--log-dir runs` | 训练日志目录 |
| 回放 | `--weights best` / `--weights last` | 默认 last；best 按该次训练的最低加权总 loss 选择 |
| 回放 | `--direct-qpos` | 直接显示关节输出；省略则使用 PD |
| 回放 | `--fps 100` | 目标播放帧率；回放默认 100，资产预览默认 60 |
| PD 回放 | `--pd-timing official` / `--pd-timing realtime` | 默认 official，沿用官方每目标 0.1 秒；realtime 按 1/fps 推进 |
| PD 回放 | `--control-hz 500 --physics-hz 1000` | 仅 realtime； 控制频率必须是 fps 的整数倍，物理频率必须是控制频率的整数倍 |
| PD 回放 | `--max-joint-velocity 0` | 仅 realtime：指令变化速度限制，rad/s；默认 0 关闭。不是物理关节速度的硬限制 |
| PD 回放 | `--no-interpolation` | realtime 模式禁用线性插值，每帧直接更新目标；限速仍独立生效 |
| PD 回放 | `--kp 400 --kd 10 --force-limit 10` | 仿真增益和旋转关节力矩上限（N·m），不是 SDK 的电流参数 |

每次训练输出 `checkpoint/<hand>_<timestamp>_<tag>/`，默认保存 `best.pth`、
`last.pth`、`config.json` 和 `checkpoint.json`。**best 是训练 loss 最优，不是验证集最优。**
`<hand>_last` 是指向最近更新它的训练目录的符号链接；下一次训练会移动该别名，
它不是“所有实验中最好的模型”。即使新训练尚未完成，也可能已经更新别名。

需要固定一次实验时，先列出目录，再把完整目录名传给 `-ckpt_tag`：

```bash
ls -lt checkpoint/
# 将下面的完整目录名替换成实际存在的实验目录，不含 checkpoint/ 前缀。
python -m geort.mocap.replay_evaluation \
  -hand wuji_hand2_beta1_right -ckpt_tag YOUR_FULL_RUN_NAME \
  -data human_alex --weights best --direct-qpos --fps 100
```

标签必须精确匹配目录名或唯一匹配子串；重复使用训练 tag 后，推荐完整目录名。
数据名 `human_alex` 对应 `data/human_alex.npy`，回放需要 `(T, 21, 3)` 人手骨骼数据。

```bash
# 查看全部可用参数
python -m geort.trainer --help
python -m geort.mocap.replay_evaluation --help
```

### PD 控制对照

```bash
python -m geort.mocap.replay_evaluation \
  -hand wuji_hand2_beta1_right -ckpt_tag wuji_hand2_beta1_right_last \
  -data human_alex --weights best --fps 100 \
  --pd-timing realtime --control-hz 500 --physics-hz 1000 --max-joint-velocity 0 \
  --kp 400 --kd 10 --force-limit 10
```

用 `--max-joint-velocity 0 --no-interpolation` 对比同频率下的阶跃目标。
改变数据帧率时注意整除关系，例如 60 FPS 可配 `--control-hz 300 --physics-hz 1200`。
可用 `--max-joint-velocity 3` 试验限速，但会引入额外滞后，因此默认关闭。
400/10/10 保留原 drive 参数，不是 Wuji 硬件标定值。
默认 official 每帧推进 0.1 秒；实验性 realtime 在 100 FPS 每帧推进 0.01 秒，不能将两者的跟踪误差直接比较。
3498 帧 Wuji 对照中，插值没有明显改善碰撞导致的误差；限速 3 rad/s 降低速度但加重目标滞后。
直接模式 `--direct-qpos` 不受插值、限速或 PD 参数影响。

### 实验性碰撞 loss

```bash
python -m geort.trainer \
  -hand wuji_hand2_beta1_right -human_data human_alex \
  -ckpt_tag wuji_collision_paper --w_collision 0.01 --no-update-last
```

正的 `--w_collision` 会启用碰撞项；0 保留原训练行为。建议保留默认同帧采样，
让碰撞项约束真实手势对应的整手关节姿势。该命令不会移动现有回放别名，
回放新模型时请使用训练输出的完整目录名。

按照 [GeoRT 论文 §II-F、式 (8)](https://arxiv.org/html/2503.07541v1#S2.SS6)：
首次启用时，用 SAPIEN 对 32768 个随机关节姿势生成二值标签（任一手内接触点
穿透深度大于 0 为碰撞；外部物体不计），以 BCEWithLogitsLoss 训练碰撞分类器。
按类别划分 90% 训练、10% 验证，训练 200 epochs。冻结分类器后，IK 的碰撞项为
`mean(softplus(C_logits(q))) * w_collision`，数值稳定地实现 `-log(1-C(q))`。
论文给出的权重范围为 `1e-4..1e-2`，示例取 0.01；默认仍为 0，便于保持基线。
这替换了此前本地的最大穿透深度平方惩罚，两种 loss 的数值与权重不可直接比较。

预测网络缓存在 `checkpoint/collision_model_<hand>.pth`；算法版本、配置、URDF、mesh 或
SAPIEN 版本变化会使缓存失效。TensorBoard 的 `collision` 分组记录 BCE、按类别平均 BCE、
碰撞召回率和无碰撞召回率；按类别平均 BCE 选择最佳分类器，避免多数类准确率误导。
`ik/raw/collision` 和 `ik/weighted/collision` 记录 IK 的原始与加权碰撞项。
类别极度不平衡会给出警告；任一类别少于 10 个样本时中止，先排查碰撞几何。

论文未指定这里使用的网络宽度、采样数量、验证划分和具体穿透容差，这些是本地实现选择。
Wuji 配置已补齐官方 Beta1 MJCF 的 10 对腕部/手指根部碰撞排除关系，
同时作用于标签生成与回放；其他碰撞保持开启。旧 checkpoint 可直接回放，
无需重训即可验证过滤修复；旧碰撞分类器需重建。
分类器不是无碰撞硬约束，也不会修复错误碰撞几何。必须检查真实穿透及 PD 回放。
官方公开代码的碰撞项是占位零值；这里补上的是论文方法，并非声称完整复现官方实验。
同帧采样仍是本地扩展。
以下保留原项目的安装、手型接入与数据采集说明。

## Installation
If you have already got a conda environment with torch, you just need these packages
```
pip install numpy scipy trimesh open3d sapien zmq tqdm tensorboard
pip install -e .
```

Otherwise, we recommend using a virtual environment to install the required packages. To install the required packages, run the following command:
```
conda create --name geort python=3.8
conda activate geort
pip install -r requirements.txt
pip install -e .
```
The original setup above uses Python 3.8. The local workflow has also been tested
with Python 3.12 and SAPIEN 3; use a PyTorch build compatible with your GPU.

## Quick Overview
Upon completion, you will be able to train GeoRT and deploy the checkpoint in a clean and straightforward way. 
### Training:
```
python ./geort/trainer.py -hand allegro_right -human_data human_alex -ckpt_tag geort_1
```
### Deploy in code
```
import geort
model = geort.load_model('allegro_right_last', weights='best')
mocap = ...
qpos = model.forward(mocap.get())
```
But before this, we need to complete some one-time system setup steps outlined below.

**Useful Links**: [Notes and Troubleshooting](#notes-and-troubleshooting)
## Getting Started
We use the native Allegro Hand as an example. 

### Step 1: Import your robot hand (one-time setup).
Note: For the Allegro Hand, you can actually skip this step. However, please follow it if you want to import a customized robot hand.

We just need to complete a quick setup process outlined below:

1. Place your robot hand URDF file in the ``assets`` folder. (We have included the Allegro example there.)
2. Create a config file named ``your_robot_name.json`` in the ``geort/config`` directory. Below is an example for the Allegro hand. For brevity, the details are omitted here, but you can refer to the [this](./geort/config/allegro_right.json) for full information. For setup instructions, please read [this](./geort/config/template.py).

```
{
    "name": "allegro_right",  
    "urdf_path": "./assets/allegro_right/allegro_hand_right.urdf",
    "base_link": "base_link",
    "joint_order": [
        "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",
        "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",
        "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0",
        "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"
    ],
    "fingertip_link": [
        {
            "name": "index",
            "link": "link_3.0_tip",
            "joint": ["joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0"],
            "center_offset": [0.0, 0.0, -0.005],
            "human_hand_id": 8,
        },
        ...
    ]
}

```
Now, you can run this command to visualize your hand.
```
python geort/env/hand.py --hand [YOUR_HAND_CONFIG_NAME]
```
such as 
```
python geort/env/hand.py --hand allegro_right
```
<span style="color:red"> If there is any segmentation error, please simplify the collision meshes or just remove all the `<collision>` fields in your URDF. </span> See the [Notes and Troubleshooting](#notes-and-troubleshooting) section.

### Step 2: Collect human hand mocap data.
Now we need to collect some human hand data for training the retargeting model. We put an example human recording dataset in data folder. You can add your own data to that folder and here is a template python script to do this.

```
import geort
import time

# Dataset Name
data_output_name = "human" # TODO(): Specify a name for this (e.g. your name)

# Your data collection loop.
mocap = YourAwesomeMocap() # TODO(): your mocap system.
                           # Define a mocap.get() method.
                           # Apologies, you still have to do this...
 
data = []

for step in range(5000):       # collect 5000 data points.
    hand_keypoint = mocap.get() # mocap.get() return [N, 3] numpy array.
    data.append(hand_keypoint)
    
    time.sleep(0.01)            # take a short break.

# finish data collection.
geort.save_human_data(data, data_output_name)
```
Use ``geort.save_human_data`` API -- this can simplify your effort in specifying the path. This dataset can be reloaded later using **data_output_name**. 

During the data collection process, try to 1. fully stretch each finger and explore its fingertip moving range and 2. perform pinch grasps. Ensure that your fingers feel natural and comfortable—since during teleoperation deployment, you will use these recorded gestures to control the robot! Please avoid any unnatural or strained movements.

We understand that most users likely have their own mocap systems. However, for demonstration purposes, we provide a simple mocap solution based on MediaPipe. Please note, this is intended only for demo use and not for deployment; we will explain this in more detail later.

```
python ./geort/mocap/mediapipe_mocap.py --name human
```
to generate a dataset named ``human``. Refered to the file for instructions. When you see the pop-up window, press ``s`` to start recording and ``q`` to finish. 

**Note:** Please ensure that the hand frame orientation is consistent between your motion capture system and the hand URDF (but fortunately the origin does not require any alignment and you can just set it to palm center). In our provided mocap example, we support the **right** hand using the following convention:+Y axis: from the palm center to the thumb. +Z axis: from the palm center to the middle fingertip. +X axis: palm normal (pointing out of the palm). 

### Step 3: Train the Model
Assuming you have placed ``your_robot_name.json`` in the ``geort/config`` folder as described in Step 1, and set ``data_output_name`` to ``human`` in Step 2, run the following command. TAG is the checkpoint id to use in later deployment.

```
python ./geort/trainer.py -hand your_robot_name -human_data human -ckpt_tag TAG
```

The default run completes 200 epochs. Use `--epochs N` to change this; wall time depends on the hardware. Ctrl+C stops early, leaving checkpoints from completed epochs.

The first run for a new hand also generates robot samples and trains Neural FK. Later runs reuse the cached data and FK weights.

For demo purpose, we have put ``human_alex.npy`` data in the ``data`` folder. For adapting it to a right Allegro hand, just run

```
python ./geort/trainer.py -hand allegro_right -human_data human_alex -ckpt_tag geort_1
```
This creates a timestamped checkpoint directory ending in `geort_1`. Use its full directory name or the `allegro_right_last` alias when loading it; the tag alone must uniquely identify a run.

### Local training curves (TensorBoard)

Training now writes TensorBoard events and CSV files to `runs/<run_name>/`.
No account or cloud upload is required. Install `tensorboard` in the training
environment if updating an existing installation:

```bash
python -m pip install tensorboard
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

Open http://127.0.0.1:6006. Existing training commands work unchanged; use
`--log-dir PATH` to choose another log root. Each run records sample-weighted
epoch means: FK MSE (when FK is trained), IK raw loss components, weighted
components, total loss, and learning rate. `training.json` records the hand
config, input path and training options. Logs flush after every completed epoch.
Cached FK weights do not produce a new FK training curve. CLI progress uses tqdm with the current epoch, batch progress, speed and ETA.
The postfix shows running sample-weighted means for the current epoch; TensorBoard
and CSV receive the final epoch means.

Each run keeps `last.pth` and `best.pth` by default. `best` minimizes the
sample-weighted epoch mean of the weighted training loss, not a validation
metric. `checkpoint.json` records the selected epochs and losses. Writes replace
weights atomically; `<hand>_last` is a relative symlink to the run, not another
copy of every checkpoint. Existing legacy alias directories must be renamed
before updating them, or use `--no-update-last`.

Use `--save-every 25` to additionally retain `epoch_24.pth`, `epoch_49.pth`, etc.
The default `--save-every 0` disables numbered snapshots. Model API calls accept
`geort.load_model(TAG, weights='best')`; the replay entry point accepts
`--weights best` or `--weights last` (default).

Git ignores local checkpoints, recordings/FK caches, logs, experiments, editor
settings and build metadata. The bundled Allegro FK and `human_alex.npy` remain
versioned; robot assets, source and tests remain visible. Investigation reports stay local.

### Step 4: Deploy!
Ok, now we are all set. Use the following code to import and deploy the trained model. 

```
import geort

checkpoint_tag = 'allegro_right_last'  # Or a full experiment directory name.
model = geort.load_model(checkpoint_tag, weights='best')  # weights='last' for the last completed epoch.

mocap = YourAwesomeMocap()      # TODO: your mocap.
robot = YourRobustRobotHand()   # TODO: your robot.

while True:
    qpos = model.forward(mocap.get()) # This is the retargeted qpos. 
                                      # (Note: unnormalized joint angle)
    robot.command(qpos)               # execute!

```
We provide some examples in ``geort/mocap/mediapipe_evaluation.py`` and ``geort/mocap/replay_evaluation``. If you have manus glove, you can also refer to ``geort/mocap/manus_evaluation.py``. We recommend (insist) you use a glove-based mocap system instead of MediaPipe, as for vision-based mocap there is significant input distribution shift during deployment!

The simplest way for testing is to use the replay evaluation as below. This will show the retargeted trajectory in the viewer. 
```
python ./geort/mocap/replay_evaluation.py -hand allegro_right -ckpt_tag YOUR_CKPT -data YOUR_TRAINING_DATA
```
For instance, if we have ``human.npy`` in the ``data`` folder
```
python ./geort/mocap/replay_evaluation.py -hand allegro_right -ckpt_tag YOUR_CKPT -data human
```
The replay viewer shows the recorded human skeleton on the left and the retargeted
robot on the right, with an upright camera covering both. Human data must have
shape `(T, 21, 3)`, ordered as wrist, then four points each for thumb, index,
middle, ring, and little finger. These are the stored hand-frame coordinates,
not camera images; no MediaPipe installation or live camera is needed.

Both sides use the same input frame and scale. Default PD timing holds each target
for ten 0.01-second physics steps, following the official dwell time. The experimental
`--pd-timing realtime` mode interpolates targets and advances 1/fps simulated seconds
per frame, which can increase tracking lag. Press **R**
to restore the camera; close the window or press **Ctrl+C** to exit.

Add `--direct-qpos` to display the model's joint positions directly, without
physics steps or collision response. PD replay remains the default. Both modes
support `--fps 100` to target 100 data frames per second, subject to processing
and rendering speed. Direct mode ignores the PD rate, gain and slew settings. For example:

```bash
python -m geort.mocap.replay_evaluation -hand wuji_hand2_beta1_right \
  -ckpt_tag wuji_hand2_beta1_right_last -data human_alex \
  --weights best --direct-qpos --fps 100
```

### Imported models: TacCap slave and Wuji Hand2 Beta1

Preview either imported asset without training:

```bash
python -m geort.mocap.replay_evaluation --preview -hand taccap_slave --animate
python -m geort.mocap.replay_evaluation --preview -hand wuji_hand2_beta1_right --animate
```

Omit `--animate` and set `--position 0.5` for a static pose. For TacCap,
0 means closed and 1 means fully open in the source simulation range. Both
previews preserve an upright camera; press **R** to reset it.

Wuji is the **right Hand2 Beta1**, with 20 joints and five fingertips. Train
a separate model, then reuse the side-by-side human/robot replay:

```bash
python -m geort.trainer -hand wuji_hand2_beta1_right -human_data human_alex -ckpt_tag wuji_paper
python -m geort.mocap.replay_evaluation -hand wuji_hand2_beta1_right -ckpt_tag wuji_hand2_beta1_right_last -data human_alex --weights best --direct-qpos --fps 100
```

Allegro checkpoints cannot be reused for Wuji. TacCap is currently an asset
and coupled-opening preview only: the existing independent per-finger networks
do not implement its single shared opening command. See the asset READMEs for
[Wuji](assets/wuji_hand2_beta1_right/README.md) and
[TacCap](assets/taccap_slave/README.md) for coordinate transforms and provenance.

Wuji optimizer comparison tools are not part of the current command interface.

## Contributing
Feel free to contribute your robot model and mocap system to the GeoRT repository!

## [Notes and Troubleshooting](#notes-and-troubleshooting)
1. **Note:Joint Range Clipping.** One core assumption of GeoRT is that the motion range of robot fingertips resembles that of human hands. To maintain realistic fingertip poses, please clip your robot's joint movement ranges appropriately and avoid unnatural configurations.

2. **Simulation Errors with New Hands?** Simulation errors (segmentation fault) may occur when importing new robotic hands (e.g. [this issue](https://github.com/facebookresearch/GeoRT/issues/7)), and this is usually caused by collision meshes. To avoid this, ensure that the collision meshes defined in your URDF are simple—such as boxes or basic convex shapes. Alternatively, you can remove all <collision> elements from the URDF to eliminate these issues entirely. 

3. **Hand Coordinate System (Frame) Convention** Please ensure that the hand frame orientation is consistent between your motion capture system and the hand URDF (but fortunately the origin does not require any alignment and you can just set it to palm center). In our provided mocap example, we support the **right** hand using the following convention:+Y axis: from the palm center to the thumb. +Z axis: from the palm center to the middle fingertip. +X axis: palm normal (pointing out of the palm). 


## Contact Us
For any inquiries, please open an issue or contact the authors via email at ``zhaohengyin@cs.berkeley.edu``
<!-- ## Bibliography -->

## License
CC-by-NC license
