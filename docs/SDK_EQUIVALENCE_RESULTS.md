# SDK / wuji-retargeting 条件对齐核验

2026-09-17，第一阶段实际完成72个验证姿态的持姿与控制变量测试；**严格内部等价尚未建立**。用户随后明确授权开始两轮GeoRT训练，按可核验条件对照继续，并保留SDK内部配置未知的限制。

## SDK 相比仓库版有哪些可确认的改进

[官方SDK更新记录](https://docs.wuji.tech/docs/en/wuji-sdk/latest/release-notes/)明确写到：2026.7.21减少单步重定向延迟、去掉额外的`[retarget]`安装依赖、加入C重定向API并改进拇指活动；2026.8.31改善拇指到其他指尖的捏合接触。这些是SDK版本更新说明，并不是针对本仓库固定生产配置的同参数A/B证据。Wuji Glove用户标定更新也不能直接解释为Manus输入收益。

我们的固定Manus验证集上，SDK开口代理误差5.60mm，生产仓库版7.56mm；但末节方向误差13.27°对12.80°，共同资产>2mm穿透25.94%对10.11%，12个微调方向分组均更差。故目前只支持“工程接口更集中、部分开口几何更好”，不支持“SDK在当前Manus遥操上全面优于仓库版”。内部尺度和滤波仍未知；不把输出差异全部归因于求解算法。完整输出和统计见`reports/stage4/difference_audit_sdk/metrics.json`。

## 已核验条件

- 同一Manus右手、米制MediaPipe 21点；样本取val01全部9段，每段8个均匀位置。不读取测试集。
- SDK2026.8.31；旧库commit与生产YAML及原生URDF哈希严格校验。
- SDK按官方firmware顺序输出，旧库按名称重排，两者使用同一公共FK评估。三列显示另已核验共同正交投影、相同掌部旋转。
- 每个输入独立reset，重复80次，比较末10次均值。SDK末10次最大相邻关节变化0.054°；旧生产版0.099°。这是固定输入的准稳态诊断，不能声称数学上完全收敛或已经剥离所有求解历史。
- SDK刚体旋转/平移对照在72个姿态上的最大均值关节差0.253°，远小于SDK与旧生产版差异，但不宣称数值严格旋转不变。
- SDK包内唯一匹配解剖关节名的Hand2右手URDF，与旧库生产原生URDF的26个关节定义全部一致：type、parent、child、origin、axis、limit逐项相同。公开API无法导出运行时资产，因此这里证明的是包内资产一致性；不把它扩大成对全部内置配置的证明。

注意：两份原生URDF均不同于当前GeoRT beta1显示/公共评估资产。上一轮“SDK包内资产与GeoRT有差异”的发现，不能据此推断SDK与旧Wuji之间存在同样的资产差异。本轮交叉控制排除了这种推断。

## 控制变量结果

每个数字比较相同72个姿态；差异值不是任务误差，也不是优劣排名。

| 旧版求解器条件 | 与SDK的平均绝对关节差 | 与SDK的公共FK指尖平均距离 |
|---|---:|---:|
| 现用Manus生产配置 | 14.785° | 24.96 mm |
| 仅替换SDK包内Hand2资产 | 14.785° | 24.96 mm |
| 仅把骨段尺度设为1 | 13.017° | 23.34 mm |
| 同时替换资产和尺度 | 13.017° | 23.34 mm |

资产替换前后的整段关节输出相同，与关节定义核验一致。取消Manus专用骨段尺度只能解释部分差异；不能把全1尺度称为SDK真实尺度，因为其参数未公开。

输入、显示与简单滤波滞后不是当前差异的充分解释。SDK内部目标、手型正则、几何预处理、求解器选择/终止条件等仍是未核验因素。当前不能声称SDK实现较旧库退化，也不能宣称二者内部等价。

## 接口边界与下一步

已安装SDK的公开 `RetargetSession` 只有 `for_hand / step / reset`；官方说明它使用内置配置并维护热启动、低通状态。没有配置/权重/尺度/滤波系数的getter或setter、未滤波输出、运行时URDF导出或内部求解状态。

用户已明确回复可以开始GeoRT优化训练，故按**统一可观测条件并明确剩余差异**继续已限定的两轮。内部配置/目标严格等价仍需要对应版本内部配置、源码或官方接口，不在此次结果中冒称已完成。

两轮计划及停止规则见 [协议](STAGE5_TWO_ROUND_PROTOCOL.md)。历史碰撞分类器的46.95%穿透检出率问题也已记录，不能直接继续沿用旧临时方案。

## 复现与预览

```bash
conda activate geort
cd /home/xense-wufan/GeoRT
# 每次使用尚不存在的新输出目录：
python scripts/audit_wuji_sdk_equivalence.py \
  --output reports/stage5/sdk_equivalence_recheck
# 当前真实输出对照，未为“看起来一致”添加修正：
python -m geort.mocap.live_comparison --sdk-reference \
  --replay data/manus/val01/micro/keypoints.npy
```

证据：`reports/stage5/sdk_equivalence_controls/report.json`、`asset_comparison.json`、完整hold输出NPZ、配置快照、run.log、as-run脚本快照。

验证：4项定向测试通过（资产提取唯一性/歧义拒绝、SDK顺序与reset、非法输入错误传播），`git diff --check`通过。没有修改生产配置、旧checkpoint或默认模型，没有设备连接、硬件运行或新训练。

来源：[SDK官方接口](https://docs.wuji.tech/docs/en/wuji-sdk/latest/retargeting/)、[官方无硬件示例及firmware手指顺序](https://github.com/wuji-technology/wuji-sdk/blob/main/examples/python/retargeting/0.retarget_session.py)、本机SDK2026.8.31的类型声明与二进制。
