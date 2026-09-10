# 术语速查

第一单元先掌握状态、参考、控制、动作和反馈。坐标约定随环境而定：当前单元使用 MetaDrive 的世界坐标与车道局部坐标；旧 `urban_cut_in` 材料的自车坐标另行定义，不能直接混用。

| Term | Working definition in this course |
|---|---|
| ego | 当前被研究或控制的自车。自车位置可以用世界坐标或自车坐标表达。 |
| actor / agent | 场景中的行动实体；依 API 可包含自车。MetaDrive 的 `env.agent` 就是受控自车，不能一概译作他车。 |
| state / observation | 状态描述系统的物理情况；观测是策略可获得的信息。当前控制器读取模拟器真值状态，后续感知实验再引入估计误差。 |
| reference | 希望车辆跟踪的目标，如车道中心位置、方向和目标速度。它与车辆实际状态分开记录。 |
| command / applied action | 控制器本步发出的命令 / 本步实际送入仿真器的动作；存在延迟时两者可能不同。 |
| scene | 某时刻的 sensor bundle、ego pose、agent state、map/route 和 scenario metadata。 |
| ODD | 系统允许运行的地理、道路、天气、光照、速度、地图和健康度边界。 |
| frame | 坐标系。旧点集练习使用 ego `x forward/y left/z up`；当前单元的车道横向坐标与转向正方向见单元公式。 |
| extrinsic / intrinsic | 传感器相对 frame 的位姿 / 相机成像内参。 |
| BEV | bird's-eye-view；把观测或特征放到 ego/map 的俯视空间。 |
| occupancy | 空间 cell 是否被占据；它不是自动带有类别、ID 或意图。 |
| tracking state | 连续时间的 agent state，例如 ID、position、velocity、age、covariance。 |
| prediction | 对其他 agent 未来轨迹/行为的可能分布。 |
| planning | 在规则、route、障碍物、舒适性和车辆约束下选择 ego trajectory/maneuver。 |
| control | 将 trajectory/ego state 转换为 steering/throttle/brake 等车辆动作。 |
| open-loop | 固定记录上评估模型，不把模型动作反馈给下一帧。 |
| closed-loop | 动作改变后续状态/观测，并反馈到下一次决策；控制器可以是规则或学习模型。 |
| terminated / truncated | 任务终止 / 达到时间等外部限制后截断；应结合到达、碰撞、出界原因判断，跑满时长不等于到达目的地。 |
| degraded mode | 输入/定位/运行时退化后，降低速度或切换保守策略。 |
| minimal risk maneuver | 不能继续 nominal operation 时的最小风险动作。 |
| corner case | 低频、高风险或 distribution-shift 的场景/切片；不是“任何模型错例”的同义词。 |
