# Manus专项采集：先试采，再分会话正式录制

目的：补充“相近指尖位置、不同末节方向/弯曲分配”，以及单指微调、全手上下文变化。当前旧数据小指在≤2 mm近邻下最大方向差只有约8.91°，不能充分验证骨骼输入的价值。采集人手原始骨骼，机器人动作不是标签。

本轮入口只导入`manus_glove`，没有机器人或重定向控制接口。沿用现用LeRobot的右手SDK原始骨骼链路、`integrated`连接模式、`world_coordinates=True`和相同MediaPipe节点选择；不使用旧ROS→ZMQ重复广播链路，也不把LeRobot的action数据当原始手套数据。右手以外暂不支持。

## 现在先做什么

1. 佩戴右手手套，在现用Manus环境中确认跟踪与校准正常。前臂可轻放桌面，掌部和手指保持活动空间。
2. 执行下方`--check`检查环境；它不会实例化或连接手套。
3. 先录`pilot01`：基准静止20秒、小指构型60秒、捏合方向60秒，净录制**2分20秒**。每项前显示动作说明，按回车才开始，倒数3秒。
4. 检查原始手型及信息样本数后再录完整会话。不要一开始就花半小时采大量重复握拳数据。

```bash
conda activate geort
cd /home/xense-wufan/GeoRT

# 已在本机验证SDK可导入；不连接任何设备。
python -m geort.mocap.record_manus --check

# 查看全部动作内容，不需要SDK或手套。
python -m geort.mocap.record_manus --list --plan full

# 连接右手手套并试采。op01为匿名操作者标识，可自行替换。
python -m geort.mocap.record_manus \
  --session pilot01 --operator op01 --plan pilot --split pilot \
  --calibration-note "右手已在现用Manus环境完成校准，沿用当前校准"
```

默认SDK目录：`~/lerobot-xensehand/third_party/manussdk/build/python`，与本机现用实机配置一致。其他机器通过`--module-dir /实际路径/build/python`指定；Python版本要与pybind模块匹配。本机模块为CPython 3.12，`geort`环境已通过导入检查。默认`--mode integrated`；如果现场实际使用其他模式，显式设为`local`或`remote`，不要猜测切换。

**默认沿用当前校准，不自动把仓库中的Right.mcal写进手套。** 校准备注用于记录实际步骤，不能证明设备已应用某份校准。若有本次操作者确认可用的校准文件，可显式增加`--calibration-file /绝对路径/Right.mcal`；这会在录制前应用该文件，要求SDK返回成功，并把文件及哈希保存在本会话中。不要将其他人的校准直接当作本人的校准。

每项完成后保存文件，再提示下一项，期间可以休息。提示处输入`q`结束；录制过程中Ctrl+C会先保存当前数据，再断开SDK。已有会话目录不会覆盖；重录用`pilot02`等新ID。只补一个动作可用`--tasks little_branch`，仍创建新会话。

## 采集内容清单

| 顺序 / ID | 内容 | 时长 | 做法与用途 |
|---|---|---:|---|
| 1 `neutral` | 基准静止 | 20秒 | 自然张手与放松半握各约10秒。看手型、比例、静止读数是否合理。 |
| 2 `little_branch` | **小指相近指尖、不同弯曲分配** | **60秒** | 选近/中/远三个舒服位置，每处约20秒。在指尖大致停在附近时交替改变根部弯曲和中末节勾曲，端点停2秒。 |
| 3 `index_branch` | 食指构型对照 | 40秒 | 两个位置，各往返约3次，改变根部与中末节弯曲分配，端点停2秒。 |
| 4 `ring_branch` | 无名指构型对照 | 40秒 | 同食指；其他手指保持自然，不要求完全独立。 |
| 5 `pinch_axis` | **相近开口、不同末节方向** | **60秒** | 拇食、拇中、拇小各20秒；留舒适间隙，改变末节朝向，再轻触与分开。 |
| 6 `micro` | 单指小幅微调 | 40秒 | 小指/食指各20秒；小幅屈伸和侧向移动，移动约2秒、停约2秒。用于局部响应与死区诊断。 |
| 7 `context` | 全手上下文变化 | 40秒 | 小指大致保持，其他手指张开、半握、拇食捏合；第二轮允许小指自然参与。 |
| 8 `open_close` | 连续握拳与展开 | 30秒 | 慢速张手→半握→握拳→展开约5轮，端点短暂停顿，覆盖自然过渡。 |
| 9 `wrist` | 掌坐标稳定性 | 20秒 | 半握姿态不变，缓慢转动手腕/手掌，检查是否引入手型跳变。 |

完整版净时长**350秒，5分50秒**，加倒数和休息约7–10分钟。动作保持舒适，允许指尖漂移；**2 mm/20°是录后诊断阈值，不是要求你肉眼达到的动作精度**。可用桌面上方的一个视觉参考点帮助保持位置，不按压固定指尖，不强扳关节，不追求背伸极限。轻触不自动标成精确接触，首轮不需要夹物体。

## 录后检查和预览

```bash
# 整个pilot会话的质量与信息覆盖报告；输出文件不能已存在。
python -m geort.mocap.inspect_manus_capture data/manus/pilot01 \
  --output data/manus/pilot01/quality.json

# 只显示录到的人手骨骼，拖动鼠标可观察视角，不加载机器人。
python -m geort.mocap.inspect_manus_capture \
  data/manus/pilot01/little_branch --preview

# 可选：录到的人手 + H1模型直接输出；这仍只是仿真诊断。
python -m geort.mocap.replay_evaluation \
  -hand wuji_hand2_beta1_right -ckpt_tag stage3_H1_seed42 --weights best \
  -data /home/xense-wufan/GeoRT/data/manus/pilot01/little_branch/keypoints.npy \
  --direct-qpos --fps 60
```

重点看以下结果：

- 有效率：低于99%先排查记录中的失败原因；它只表示节点、数值、骨长和坐标计算有效，不是传感器置信度。
- 原始手型：手指索引、拇指方向、手掌朝向合理，无明显错位；先确认原始手型，再评价机器人模型。
- `little`的`2mm/20°`、`5mm/20°`样本对：看小指动作是否确实带来方向差，而不只是移动指尖。严格阈值为0不代表采集失败，可先看5 mm下覆盖、动作可行性和预览。
- 完全相同读数：不能直接等同于丢帧；静止时可能正常，运动时大量不变则检查跟踪。SDK缓存停更与真正静止目前无法完全区分。
- 主机读取间隔：最大间隔超过250 ms会提示检查，但不将其直接称为传感器丢帧。

信息覆盖搜索使用最多1200个均匀抽取的有效读数，同对读数的主机时间间隔≥0.5秒，报告配对数与不同读数数量。大量相关样本对不等于大量独立动作。试采缺少关键姿态时先调整动作、再录短片，不靠重复采同一动作刷帧数。

`--preview`按主机读取间隔显示；机器人回放的`--fps 60`只控制播放速度。二者都不能证明真实采集频率或端到端延迟。

## 正式会话安排

试采确认可用后，建议同一操作者做三次完整会话，每次独立启动录制，中间休息并重新检查手型。若摘下重戴/重新校准，写入新的校准备注。不要把同一次录制的相邻帧随机分配到训练和测试。

```bash
python -m geort.mocap.record_manus \
  --session train01 --operator op01 --plan full --split train \
  --calibration-note "填写本次实际校准与佩戴情况"

python -m geort.mocap.record_manus \
  --session val01 --operator op01 --plan full --split validation \
  --calibration-note "填写本次实际校准与佩戴情况"

python -m geort.mocap.record_manus \
  --session test01 --operator op01 --plan full --split test \
  --calibration-note "填写本次实际校准与佩戴情况"
```

三个会话净录制17分30秒，含准备/休息约25分钟。`pilot01`留作采集流程调试，不加入最终验收。所有属于同一session的动作片段必须保留在同一个split。测试会话可以检查录制质量与姿态覆盖，但在方案冻结之前不使用其机器人模型效果来选模型。

## 正式数据打包

会话完成后生成带来源清单的静态数据包（输出目录必须不存在）：

```bash
python -m geort.mocap.prepare_manus_dataset \
  --train data/manus/train01 --validation data/manus/val01 --test data/manus/test01 \
  --output data/manus_stage4_repeat
```

程序校验会话用途、片段完整性、节点映射、原始分块与导出一致性及哈希，拒绝跨划分复用同一会话或完全相同的片段。每个 split 单独保存关键点数组、片段编号和原始读取序号；`manifest.json` 保留全部来源和片段范围。不添加采集时间、不去重、不随机拆分会话。**数组是静态样本集合，不是连续录像**；训练/评估接入时必须读取清单，不能直接套旧录像的帧区间配置。Wuji 热启动/滤波和相邻帧诊断必须在每个片段边界重置或截断。现有单个片段的预览命令不变。正式会话检查见 [阶段四数据就绪报告](STAGE4_DATA_READINESS.md)。

## 保存了什么

每个会话目录有`session.json`，保存匿名操作者、用途split、右手glove ID、SDK模块哈希、采集代码和动作表哈希、节点映射、校准说明与状态。每个动作有独立目录：

- `chunk_00000.npz`等：约每5秒落盘。包含完整SDK `(N,10)`节点位置/四元数wxyz/缩放、canonical 21点、canonical到源坐标的4×4变换、主机单调时钟与墙钟、读取序号、几何有效标记、重复读数标记、无效原因。
- `metadata.json`：节点拓扑、21点映射、数据单位/坐标、任务说明、完成/中断状态。节点数改变不会被静默截断；异常原始数组另存`rejected_*.npz`。
- `keypoints.npy`：有效canonical骨骼，可被现有训练数据加载器和回放读取。
- `source_poll_index.npy`、`export.json`：导出帧与原始读取序号对应，包含被排除无效读数数量、哈希、session/split和“仅静态几何训练”说明。

SDK使用的是估计骨骼，不是机器人关节真值；保存完整四元数也不等于已有可靠指腹法向标定。未提供的传感器置信度、SDK帧号和采集时间保持缺失。**主机时间不会填入训练数据的`timestamps`字段**，本轮数据不用于速度/加速度或时序训练。原始数据保留重复读数；后续筛样须显式记录，不能先去重再假装均匀时间序列。

中断后，已完成的分块可读取；硬退出最多可能丢失尚未保存的当前约5秒缓冲。若未生成`keypoints.npy`，可以显式导出已保存分块：

```bash
python -m geort.mocap.inspect_manus_capture \
  data/manus/pilot01/little_branch --recover-export
```

不会覆盖已有导出。录制完成只表示数据保存，不自动表示质量合格；不自动合并训练集、训练模型、推送数据或连接机器人。

## 本机验证边界

SDK模块导入已通过，未实例化或连接真实手套。80项本地测试通过；新增覆盖节点映射、退化/毫米单位误输入、刚体变换、原始数据保留、异常/重复读数、分块恢复、禁止覆盖、信息样本检测、Ctrl+C保存与显式路径回放。使用明确标注的SDK测试夹具完成分块→导出→检查器联调；它不是新采集的Manus数据。实际跟踪、校准、设备连接与图形窗口仍待接手套后核验。
