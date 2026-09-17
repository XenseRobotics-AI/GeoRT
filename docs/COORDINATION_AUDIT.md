# 全手协调第一版审计（2026-09-16）

仓库：XenseRobotics-AI/GeoRT。未找到本地 AGENTS.md，遵循会话提供的全局约定。
原始 baseline 为 `258e371` 中未修改的 `geort.trainer` / `IKModel`；本轮不覆盖历史 checkpoint，不推送。

## 输入与数据链

- `geort/mocap/manus_client/src/right_hand_ros.cpp`：发布 xyzw 四元数。
- `geort/mocap/manus_mocap_core.py::ManusForwardKinematicsSolver`：用 SciPy xyzw 计算21点，位置使用代码内固定骨长（米），不直接传感测量机器人角。
- `hand_to_canonical`：腕为原点；z 沿腕→中指根，x 为掌面叉积，y 由正交叉积得到。不是世界坐标。
- `Manus.run` / `ManusMocap`：ZMQ 只传 canonical float32 (21,3)，丢弃原四元数；各 ROS 回调无时间同步保证。
- `ReplayMocap`：直接读 .npy，无额外归一化；`human_alex.npy` 实测 (3498,21,3)、float64、有限，腕位置为0，最大坐标约0.166m。
- 腕=0，拇/食/中/无名/小指各4点=1:5/5:9/9:13/13:17/17:21。非拇指为MCP/PIP/DIP/tip。
- 数据有真实可用的不同中间点，能区分同指尖不同骨段轴；没有完整指腹朝向或绕骨轴旋转。
- .npy 没有置信度、时间戳、会话、操作者或可核实采样率。不能把回放 `--fps` 当作采集时间。
- Manus在线与离线形状/预期坐标一致，但该旧文件没有传感器、标定或操作者元数据，不能证明实际来源一致。
- MediaPipe有单独掌坐标变换及可选旋转EMA；README已说明动态尺度不适合真实遥操作。
- Allegro左右配置独立；现有Manus示例主要为右手，无统一的左右镜像/尺度标定元数据。增强方向项首版限定有明确约定的Wuji右手，其他旧入口保留。

## 模型、FK和目标函数

- `geort/model.py::IKModel`：每指独立3→128→128→DOF，分支不共享参数；BN、LeakyReLU、Tanh，输出[-1,1]。
- `formatter.HandFormatter`：线性转换为URDF物理关节范围（rad）。旧 `PostureIKModel` 仅扩充小指12维，属于历史候选。
- `GeoRTTrainer._train`：只取配置 human_hand_id 指尖；覆盖样本逐指独立、1mm体素降采样，其余几何项默认同帧采样。
- `FKModel`：归一化关节→每指3D指尖的学习近似；训练IK时冻结FK参数但保留输入梯度。
- `geort/kinematics.py`：已有精确URDF FK、解析Jacobian和Torch可微FK，可作为新损失与独立核验，不需要新增依赖。
- baseline paper loss：Σ指 Chamfer双向最小**平方米**距离×80；指尖扰动方向负cos（均值×指数量）×1；对称扰动二阶差平方和×1；人手两指<15mm时机器人两site距离平方、无序对×2×1000；可选碰撞softplus(logit)×0（默认）。各项不是天然同量纲。
- 原方向项是指尖响应方向，不是末节轴向；原曲率项也不是关节时间平滑。
- `geort/loss.py::pinch_distance_loss` 的机器人目标是两个site重合，无指腹/物体模型；本轮保留原模式作B0，替代关系项不再重复该吸附目标。
- checkpoint的config保存机器人、joint上下限；旧权重是严格state_dict。已有新版factory依据model_type分派，无strict=False。

## 机器人与运行链

- 正式JSON配置：Allegro右/左、Wuji右。手型与指尖顺序以 `fingertip_link` 为准，Allegro顺序不是Wuji顺序。
- `HandKinematicModel`按joint_order↔URDF模拟关节顺序映射；精确FK支持独立revolute/fixed，遇mimic明确拒绝。
- Wuji关节是独立20DOF，末端site在配置center_offset处；骨段轴可由最后移动关节原点→site确定。没有经标定的指腹法向。
- 原资产有碰撞几何和显式排除对；`self_collision_depth`可检查穿透，不是接触稳定性验证。
- `export.forward`不滤波；`set_qpos_target`再裁剪到物理范围内1e-3。回放可选PD插值/限速/在线姿态修正，默认输出路径不额外启用这些实验补丁。
- 本仓库提供仿真入口，未发现可在本轮验证的Wuji硬件发送/安全闭环。训练惩罚不是运行时安全保证。

## 本轮实现选择

1. 新增可选协调模型，原指尖分支保持，骨段局部残差和全手上下文残差独立开关；残差在Tanh前的归一化logit空间施加，有界且初始化零输出，最后一层保留可学习梯度。
2. 统一21点/可选有效性和置信度适配器；严格元数据记录单位、坐标、顺序、机器人摘要。无完整pose标定时明确拒绝pose模式。
3. 使用真实骨段方向和显式mask，新增末节轴向、非拇指背伸偏好、按训练集尺寸比例校准的拇指相对向量项。关系项开启时替代原pinch，避免双重计罚。
4. 原trainer保持不动；新增有限步数实验入口，复用原几何公式/Neural FK，使用统一同帧采样控制消融变量。其B0是受控续训参考，不声称与原200epoch/体素采样流程逐步相同。
5. 首轮时序只诊断；真实相邻姿态差分没有可信时间时只报每采样步变化，不标速度/加速度。可选npz timestamps+sequence_id适配器提供时间质量计算。
6. 局部增益通过真实近邻帧对诊断，单指串扰仅在其他手指变化小的可观察子集报告，无足够样本时标null。不合成任意骨点扰动，不报告没有可行参考集的覆盖率。

缺失能力：pose/指腹法向、物体接触稳定性、跨会话泛化与真机安全。首版不伪造这些信息。
