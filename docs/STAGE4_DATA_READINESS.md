# 阶段四：真实 Manus 会话就绪与实验起点

2026-09-16。遵循 DEVELOPMENT_GOALS.md，区分输入信息、协调结构和损失的作用，同时保留方向、多指关系与精细控制。当前完成数据准备和已有模型的验证集诊断，尚未完成新训练或质量验收。

## 采集核验

| 会话 | 用途 | 动作片段 | 有效读数 |
|---|---|---:|---:|
| train01 | 训练 | 9 | 20990 / 20990 |
| val01 | 验证 | 9 | 20988 / 20988 |
| test01 | 测试 | 9 | 20989 / 20989 |

会话用途、分块序号、导出来源索引、SHA-256、节点映射和掌坐标逆变换均核验通过。沿用用户已目视核验的校准，没有发现要求重新校准或整轮重录的证据。几何有效不等于传感器置信度或标定精度。

`geort.mocap.prepare_manus_dataset` 已生成本机 `data/manus_stage4_v1/manifest.json`、三个独立 split 的关键点数组及来源索引，保留重复读数和全部 27 个片段边界。不添加时间戳、不混入 pilot、不修改原始记录。10 项定向 unittest 通过，覆盖分块、损坏、错误 split、改名复制片段泄漏、来源与边界及禁止覆盖。

新增录像路径的 Wuji 并排预览也已修复：缓存核验复用 ReplayMocap 实际解析的路径，不再错误地按旧数据集名称查找。连同相机回归共 13 项测试通过；实际预览命令通过参数、模型兼容性和缓存哈希检查，在创建渲染器前停止，没有宣称 GUI 或实机验收。

现有训练入口仍按旧录像中的区间划分数据，不能直接套用新数据包。下一步须接入清单并保护边界，不能通过拼接或填充伪帧绕过旧 guard-gap 要求。

## 小指信息覆盖

全量搜索每个片段内、相隔至少 0.5 秒主机读取时间的样本对。严格阈值为掌坐标指尖 ≤2 mm、末节骨轴差 ≥20°，放宽阈值仍为预先使用的 ≤5 mm、≥20°。

- train01：little_branch 有 498 对严格样本，涉及 203 条不同读数。
- val01：little_branch 严格为 0 对、放宽为 25643 对；ring_branch 中小指有 384 对严格样本，涉及 134 条读数。不能只按动作名称筛选手指。
- test01：little_branch 严格为 0 对、放宽为 17967 对；ring_branch 中只有 1 对严格小指样本。测试集可评价总体方向和操作代理指标，但严格近同指尖的多解结论证据薄弱。不能依据测试覆盖事后降低阈值，放宽阈值的收益也不能全部归因于骨骼信息。

这些配对强相关，不等于独立手势次数。现有数据可继续实验，暂不要求重复整轮采集；专项结论须保留覆盖限制。测试集仅检查录制质量与输入覆盖，没有计算机器人模型效果。

## 已有模型在新验证片段的表现

使用 val01/little_branch 全部 3597 条读数、共同 GeoRT 精确 FK。原始 GeoRT 使用原冻结 last，旧 H1 使用 best；Wuji 使用已确认配置和 lp_alpha=0.3，在片段起点重置、内部按原始顺序推进，求解失败记录为 0。这不是新训练或完整验收。

| 模型 | 小指 PIP/DIP 任一低于 −15° | 小指末节方向平均误差 | 全手末节方向平均误差 |
|---|---:|---:|---:|
| 原始 GeoRT | 9.42% | 31.26° | 79.27° |
| 上一轮 H1 | 3.45% | 49.84° | 84.33° |
| 现用 Wuji（滤波） | 0% | 27.46° | 16.01° |

H1 小指 MCP 低于 −15°的比例为 57.19%，原始为 10.54%，Wuji 为 0%。不能因 PIP/DIP 反曲变少而宣称改善。新会话还暴露了旧模型其他手指的问题，不能只盯小指。

新验证片段逐指指尖到整段旧 human_alex 最近邻的距离中位数约 15.5–26.9 mm，提示旧数据覆盖不足，但不能单独区分手长、动作分布和预处理因素，也不等于校准失败。下一轮先核对这些因素，再建立新数据上的指尖基线；不能直接套用旧 H1 冻结基础网络及 ±0.1 logit 残差的结论。

本机证据：`reports/manus_formal_review/` 中的 `val01_quality.json`、`test01_quality.json`、`little_coverage_full.json`、`validation_existing_models.json`、`wuji_val01_little_branch.npz`。报告保存模型与输入哈希；Wuji 缓存保存已确认源版本、配置和原生资产哈希。开口比例仅由 train01 骨长计算。

## 下一轮顺序

1. 接入会话清单：训练只采 train，验证按片段分别计算，测试不参与 best 选择；采样策略在运行前固定。相邻位移及 Wuji 滤波不能跨片段。
2. 审计新旧坐标、手长和覆盖差异，在新数据上建立指尖基线；原始冻结 GeoRT 保留为历史对照。
3. 固定初始化、预算、loss、seed42，比较只读指尖与读取真实骨骼信息的模型。先不冻结已表现失配的旧基础网络，离线选解标签及残差放宽不与输入消融同时引入。
4. 再单独启用全手上下文。持续包含现用 Wuji 对照；报告方向、开口关系、各指构型和局部响应，anchor 不作公平排名。
5. 验证方案冻结后才运行测试模型效果。严格多解专项覆盖限制单独报告；无 SDK 采集时间，不报告真实速度/延迟，无物体真值不报告接触成功率。

## 预览

```bash
conda activate geort
cd /home/xense-wufan/GeoRT

# 左人手、中现用 Wuji、右旧 H1，仅用于诊断。
python -m geort.mocap.replay_evaluation \
  -hand wuji_hand2_beta1_right -ckpt_tag stage3_H1_seed42 --weights best \
  --compare-wuji-cache reports/manus_formal_review/wuji_val01_little_branch.npz \
  -data /home/xense-wufan/GeoRT/data/manus/val01/little_branch/keypoints.npy \
  --fps 60

# 验证集中包含严格小指区分样本的原始片段。
python -m geort.mocap.inspect_manus_capture data/manus/val01/ring_branch --preview
```
