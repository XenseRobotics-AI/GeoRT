# 最终 test01 四方法回放

直接打开完整合辑：

```bash
xdg-open /home/xense-wufan/GeoRT/videos/final_comparison_test01/00_complete_test.mp4
```

打开视频目录：

```bash
xdg-open /home/xense-wufan/GeoRT/videos/final_comparison_test01
```

## 内容与对照条件

每个视频均为 1920×1080、60 fps。四列从左到右为：主分支 GeoRT、最终 GeoRT R2、Wuji SDK 2026.8.31、生产 wuji-retargeting。上排显示冻结的模型目标关节姿态，下排显示四个独立物理场景推进后的实际 PD 姿态。两排四列使用相同视觉模型、大小、固定手掌方向与正交投影。

主分支 GeoRT 使用主分支 main@8e93d0d 的原始 IKModel 与 HandFormatter，加载当前 train01 从零重训的 `wuji_hand2_beta1_right_2026-09-17_14-13-59_main_manus_train01_seed42/best.pth`。训练200轮、seed42，best按训练loss选择；test没有参与训练或选择。没有骨架模板适配、BN重估、附加低通或姿态修正。源码、权重哈希及完整test输出保存在 `main_reference/`。

R2 使用 `stage5_R2_seed42/best.pth`，沿用最终比较配置 LP=0.3；wuji-retargeting 使用用户确认的 `adaptive_analytical_manus_wuji_hand_2_right.yaml` 配置及 LP=0.3；SDK 保留其内置状态和滤波。这是一组已部署/验收配置的对照，未人为把四者改成相同滤波。

主分支原始列读取 `reports/final_main_retrained_reference/outputs.npz`；R2、SDK 与 wuji-retargeting 三列直接读取 `reports/stage6/test_full/outputs.npz` 的既有冻结输出。没有重新训练、调参或按效果筛选片段。涵盖 test01 的全部 9 段、20,989 条有效读取，保持片段内原始顺序，不跳帧。各段开头均从相同的限位内零位和零速度启动 PD，保留起步过渡；片段之间不继承 PD 状态。

统一 PD 参数：kp=400、kd=10、力矩上限 10 Nm；输入假定 60 Hz，控制 600 Hz，物理 1200 Hz，每输入间隔 10 次线性插值目标更新、20 次物理子步。除片段起点设置初态外，PD 场景使用驱动目标推进，没有逐帧设置实际关节位置。视频右下/各列底部的关节跟踪 MAE 衡量实际关节偏离对应模型目标的程度，不是人手重定向质量分数。

60 Hz 是离线回放时间假设；数据没有 SDK 采集时间/帧号，不代表实测采集频率或系统延迟。上述 PD 设置为统一仿真条件，未标定为实机执行器参数。视频生成过程没有连接手套或控制实机。

## 文件

| 文件 | 内容 | 帧数 | 时长（秒） |
|---|---|---:|---:|
| `00_complete_test.mp4` | 全部九段顺序合辑 | 20,989 | 349.817 |
| `01_neutral.mp4` | 基础手型 | 1,200 | 20.000 |
| `02_little_branch.mp4` | 小指构型 | 3,598 | 59.967 |
| `03_index_branch.mp4` | 食指构型 | 2,399 | 39.983 |
| `04_ring_branch.mp4` | 无名指构型 | 2,399 | 39.983 |
| `05_pinch_axis.mp4` | 捏合与方向 | 3,597 | 59.950 |
| `06_micro.mp4` | 小幅动作 | 2,399 | 39.983 |
| `07_context.mp4` | 多指上下文 | 2,398 | 39.967 |
| `08_open_close.mp4` | 张手与握拳 | 1,799 | 29.983 |
| `09_wrist.mp4` | 手腕动作 | 1,200 | 20.000 |

协议、输入哈希、各视频帧数与校验值分别保存在 `protocol.json`、`frozen_inference_protocol.json`、`frozen_inference_metadata.json`、`videos.json`。每段的 `*_pd.npz` 保存实际 PD 关节状态及源帧编号，`*_frame*.jpg` 为首/中/末帧布局检查图，`*_encode.log` 保存编码错误日志。最终完整性检查结果保存在 `verification.json`。

## 重录命令

输出目录必须尚不存在，避免覆盖已归档视频：

```bash
cd /home/xense-wufan/GeoRT
/home/xense-wufan/miniforge3/envs/geort/bin/python -m geort.mocap.record_final_comparison \
  --main-reference reports/final_main_retrained_reference \
  --workers 2 --output videos/final_comparison_test01_repeat
```

## 最终完成状态

9个分段与1个合辑均已生成；合辑20,989帧、349.817秒，1920×1080、60fps；完整解码无错误。PD状态数组与源帧编号完整连续。已检查九段中间帧总览及合辑解码抽帧，记录见 `verification.json`。全部文件约111 MiB。
