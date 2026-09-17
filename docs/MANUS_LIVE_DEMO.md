# Manus 实时动态对照场景

直接读取现用右手 Manus SDK，在同一窗口显示 **人手骨架 → 原始 GeoRT → 现用 Wuji → M1**。每次显示的四列来自同一条输入读数；统一资产、基座旋转和正交视角。没有真实灵巧手驱动接口。

## 启动

佩戴右手手套，在现用 Manus 环境确认已连接且校准正常，运行：

```bash
conda activate geort
cd /home/xense-wufan/GeoRT
python -m geort.mocap.live_comparison
```

默认 `integrated`、`world_coordinates=True`，沿用现有校准，不写入新校准。默认候选是 `stage4_M1_seed42/best`，原始参考为原 seed0 checkpoint/last；Wuji 固定为已经确认的 Manus / Hand 2 right 配置，保留热启动与 LP=0.3。

窗口默认1800×1000，四列手模型已统一放大。可通过 `--window-size 1920 1080` 指定窗口大小；`R`恢复放大后的公共正交视角。已有窗口需退出并重新运行才能加载新默认值。

无需启动旧的 ROS/ZMQ Manus 广播，也不需要连接 Wuji 实机。Wuji 优化器运行在本仓库 `.venv-wuji-baseline` 子进程中，避免与 GeoRT 环境依赖冲突。

## 怎么体验

新增 SDK 版可替换原始参考列，保留现用 Wuji 和 M1 同屏：

```bash
python -m geort.mocap.live_comparison --sdk-reference
# 无手套也可先预览：
python -m geort.mocap.live_comparison --sdk-reference \
  --replay data/manus/val01/micro/keypoints.npy
```

左起为 **人手骨架 → Wuji SDK 2026.8.31 → 现用 Wuji LP=0.3 → M1**。
SDK 使用内置右手 Hand2 配置和内部滤波，系数未通过接口暴露；`L` 只改变 M1。
SDK 子进程也按输入切换/长暂停重置状态。[安装、离线对照与边界](WUJI_SDK_BASELINE.md)。

1. **张合与手型**：缓慢张手、半握、握拳，比较三只模型；`R` 恢复统一视角，鼠标可调整视角。
2. **停住与微调**：做一个半捏姿态，按 `F` 留下五指指尖的金色参考点，再仅微动一根手指；按 `T` 显示最近90次输出的轨迹。参考点是固定位置标记，不是关节锁定。
3. **开口跟随**：按 `G` 显示缓慢变化的 10–40 mm 双点开口标尺（12秒一周期），观察顶部的三组拇食 site 间距，练习逐渐靠近目标、停住再打开。绿色标尺位于每只手下方，不是可抓取物体，也不计抓取成功率。
4. **控制滤波差异**：按 `L` 切换 M1 原始输出 / LP=0.3，Wuji始终保留生产滤波。切换不重置手套，不改变权重；会清除旧轨迹和冻结参考，避免混看。相同 LP 系数不代表相同总延迟。

`Space` 暂停显示与新读数提交，再按恢复；暂停超过0.5秒后重置求解器/滤波历史。`C` 清除参考点和轨迹；`Esc` 或关闭窗口退出，释放SDK与子进程。当前仅支持右手。

想让候选从一开始就使用同样的输出 LP：

```bash
python -m geort.mocap.live_comparison --m1-filter lp03
```

想把原始模型那一列换成 M0，单独看骨骼输入带来的差异：

```bash
python -m geort.mocap.live_comparison \
  --reference stage4_M0_seed42 --reference-weights best
```

## 可见状态与边界

- 顶部显示 live/replay、实际完成频率、主机读数距今时间、被替换的待处理读数数、求解状态和每次三模型计算耗时。
- 最多缓存一条待处理输入，避免越积越慢；Wuji 滤波按实际完成的输入序列运行，跳帧数量明确显示。不同实时负载下的结果不能直接冒充固定60 Hz的离线缓存。
- 无效骨骼或断连时显示 HOLD，保留最后有效姿态，不发送零姿态；设备身份或节点拓扑变化要求重新启动。
- SDK没有采集时间戳/帧号，主机读数时间不能证明传感器数据新鲜度，计算耗时也不是端到端延迟。
- 直接设置仿真关节位置，无PD、物理接触或抓取动力学；可见穿插是模型输出的表现，不由物理引擎替模型纠正。
- 输出有限性、公共关节范围、机器人配置、Wuji固定源版本/配置/URDF都会核验；不偷偷截断或更换基线。
- 本入口不保存新训练样本、不修改默认模型、不驱动真实灵巧手。

## 无手套检查和回放

`--check` 只验证SDK可导入、模型可加载、原生Wuji可计算，不连接手套：

```bash
python -m geort.mocap.live_comparison --check
```

同一场景先用已有录制演示，界面明确标为 RECORDED REPLAY：

```bash
python -m geort.mocap.live_comparison \
  --replay data/manus/val01/micro/keypoints.npy
```

离屏验证不访问手套，不打开窗口；快照路径必须不存在：

```bash
python -m geort.mocap.live_comparison --headless \
  --replay data/manus/val01/micro/keypoints.npy --frames 60 \
  --snapshot /tmp/geort_live_demo_preview.png
```

如果提示 `Renderer does not support display`，需要在有桌面显示和图形设备访问权限的终端运行；离屏成功不等于图形窗口可用。连接超时则检查 Manus Core 和右手设备；SDK导入成功不代表手套已连接。

## 本机验证记录（2026-09-16）

- 9项定向测试通过：输入映射、断连/身份/拓扑变化、非法关节输出、最新输入队列、重置标记保留、求解异常、正交视角恢复。
- 原生Wuji独立进程与已有micro缓存前64帧最大关节差为3.65e-7 rad，重置后首帧复现通过。
- 实际桌面窗口运行120个同步帧后正常退出，退出前确认orthographic；离屏运行90帧以及视野调整后的60帧通过。
- 当前会话中的实时连接尝试：SDK连接成功，landscape为gloves=0、dongles=1、right_id=0，无骨骼数据，等待超时后已释放连接。因此尚未完成真实手套动作验收；这不是校准质量结论。
- 证据在 `reports/stage4/live_demo/`，图形窗口在具备桌面显示权限的进程中验证；未连接真实灵巧手、未提交或推送。
