# Glossary

| Term | Working definition in this course |
|---|---|
| ego | 自车状态、坐标原点和运动；不要和 actor 混用。 |
| actor / agent | 其他交通参与者；至少包含位置、速度、尺寸、ID 和不确定性。 |
| scene | 某时刻的 sensor bundle、ego pose、agent state、map/route 和 scenario metadata。 |
| ODD | 系统允许运行的地理、道路、天气、光照、速度、地图和健康度边界。 |
| frame | 坐标系；本课程 toy convention 是 ego `x forward/y left/z up`。 |
| extrinsic / intrinsic | 传感器相对 frame 的位姿 / 相机成像内参。 |
| BEV | bird's-eye-view；把观测或特征放到 ego/map 的俯视空间。 |
| occupancy | 空间 cell 是否被占据；它不是自动带有类别、ID 或意图。 |
| tracking state | 连续时间的 agent state，例如 ID、position、velocity、age、covariance。 |
| prediction | 对其他 agent 未来轨迹/行为的可能分布。 |
| planning | 在规则、route、障碍物、舒适性和车辆约束下选择 ego trajectory/maneuver。 |
| control | 将 trajectory/ego state 转换为 steering/throttle/brake 等车辆动作。 |
| open-loop | 固定记录上评估模型，不把模型动作反馈给下一帧。 |
| closed-loop | 模型动作改变后续 state/observation，再继续 rollout。 |
| degraded mode | 输入/定位/运行时退化后，降低速度或切换保守策略。 |
| minimal risk maneuver | 不能继续 nominal operation 时的最小风险动作。 |
| corner case | 低频、高风险或 distribution-shift 的场景/切片；不是“任何模型错例”的同义词。 |
