# 原始主分支 GeoRT：当前 Manus 数据重训

已完成主分支 `8e93d0d4e32c00b992bf0346bc74a5e22caeaa5c` 原始网络与训练目标的 200 轮训练，seed=42，使用 train01 全部 20,990 帧，从随机初始化训练。没有旧权重热启动、骨架模板适配、骨段输入、全手上下文或 R2 损失。

权重目录：`/home/xense-wufan/GeoRT/checkpoint/wuji_hand2_beta1_right_2026-09-17_14-13-59_main_manus_train01_seed42`。训练完成 epoch=199（从0计数），best 为 epoch=196，按主分支最低训练 epoch 平均 loss 选择。验证数据仅用于固定模型训练后核查；测试数据没有用于本次训练、超参数选择或 best 选择。旧默认 alias、旧 checkpoint、FK 缓存与数据保持不变。

## 预览

```bash
cd /home/xense-wufan/GeoRT
/home/xense-wufan/miniforge3/envs/geort/bin/python -m geort.mocap.replay_evaluation \
  -hand wuji_hand2_beta1_right -ckpt_tag wuji_hand2_beta1_right_2026-09-17_14-13-59_main_manus_train01_seed42 --weights best \
  -data /home/xense-wufan/GeoRT/data/manus/val01/open_close/keypoints.npy \
  --compare-wuji-cache /home/xense-wufan/GeoRT/reports/stage4/wuji_val01/open_close.npz --fps 60
```

左侧人手、中间现用 Wuji、右侧重训主分支模型，均为直接输出。将数据和缓存路径中的 `open_close` 同时改为 `neutral`、`little_branch` 等可查看对应完整片段。没有实机连接。

同帧静态图：

```bash
xdg-open /home/xense-wufan/GeoRT/reports/main_geort_manus_seed42/preview.jpg
```

## 完整验证结果

完整 val01 共 20,988 帧、9 段，按动作等权平均；共同精确 FK 和原已冻结的 R2 人机关系比例，仅用于评价，没有作为主分支训练目标。

| 方法 | 末节方向误差 | 开口误差 | 接近限位的关节输出占比 | 小指严重 PIP/DIP 反曲 |
|---|---:|---:|---:|---:|
| 旧主分支权重 | 84.56° | 29.83 mm | 45.800% | 14.669% |
| 当前数据重训主分支 | 25.06° | 7.90 mm | 0.090% | 0.154% |
| R2 LP=0.3 | 5.38° | 3.18 mm | 0.009% | 0.000% |
| 现用 Wuji LP=0.3 | 12.80° | 7.56 mm | 0.348% | 0.000% |

接近限位定义为归一化输出绝对值 >0.95；反曲定义为小指 PIP/DIP 任一个 <-15°。开口指标只评价预定义开放手势区间，不是物体接触真值。这些结果支持重训解决了大部分旧数据失配；不能用训练 loss 或这些几何指标宣称实机操作质量优于 Wuji。未做本轮完整碰撞和实机验收；完整测试集四方法视频已录制并校验，见 `videos/final_comparison_test01/README.md`。

## 训练与复核

实际使用隔离的主分支 `source/geort`，未改动其代码。原训练配置为五个独立指尖 MLP、BatchNorm、AdamW 1e-4、batch 2048、5000 updates；paper_loss 与 paired_sampling 开启，direction/Chamfer/curvature/collision/pinch 权重为 1/80/1/0/1000。保留每50轮快照、best与last，不更新旧默认入口。

`protocol.json` 保存代码和输入哈希；`train.py` 为启动器，`train.log` 为实际日志；`curves/` 保存 CSV 与 TensorBoard；`validation.json` 保存逐动作和汇总结果，`validation_outputs.npz` 保存所有模型的完整验证输出；`validate.py` 为评价源码。源码与训练输入、机器人 FK 数据、Neural FK 权重训练后校验通过。

复跑（每次生成独立时间戳 checkpoint，保留旧模型）：

```bash
cd /home/xense-wufan/GeoRT
PYTHONPATH=/home/xense-wufan/GeoRT/reports/main_geort_manus_seed42/source \
  /home/xense-wufan/miniforge3/envs/geort/bin/python reports/main_geort_manus_seed42/train.py
```
