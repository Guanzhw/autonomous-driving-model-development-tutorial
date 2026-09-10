# 教材与研究脉络

每一阶段只选一套主教材，带着实验中的问题阅读。先理解一种方法解决了什么，再判断它引入了什么新假设。

## 第一单元的选读

用 60–90 分钟选读 [TUM 第 08 节 Control 实践](https://github.com/TUMFTM/Lecture_ADSE/tree/master/08_control/practice)，关注速度控制器和微分方程求解。先读 notebook 的解释与图，不必安装整套外部环境。写出其中的状态、目标速度、动作和反馈，并与本项目的速度控制公式逐项对应。第一单元不要求先学 RL。

完成[第一单元](../course/first_loop/README.md)后，应能用自己的结果解释“相同控制器在执行延迟下为什么可能出现不同表现”，并指出简化直道实验尚未检验哪些驾驶能力。

## 后续主教材

| 需要解决的问题 | 材料 | 建议用法 |
|---|---|---|
| 智驾系统有哪些模块，它们怎样协作？ | [TUM ADSE](https://github.com/TUMFTM/Lecture_ADSE) | 按定位、预测、规划、控制选读与实验 |
| 如何从示范和奖励学习？ | [Robot Learning: A Tutorial](https://arxiv.org/abs/2510.12403) | 理论伴读，串起 BC、RL 与通用策略 |
| RL 的更新怎样实现与验证？ | [Berkeley CS 185/285](https://rail.eecs.berkeley.edu/deeprlcourse/) | BC、策略梯度、价值学习、model-based/offline 选读 |
| 机械臂和汽车有什么不同？ | [MIT Robotic Manipulation](https://manipulation.csail.mit.edu/) | 空间变换、FK/IK、接触、抓取与规划 |
| 生成式动作与 VLA 怎样发展？ | [ETH Robot Learning 2026](https://cvg.ethz.ch/lectures/Robot-Learning/) | 配合公开作业和原论文 |
| 怎样收集数据并训练操作策略？ | [LeRobot 文档](https://huggingface.co/docs/lerobot/) | 固定版本，先跑 ACT 数据—训练—评估链 |
| 平衡、动力学和腿式运动怎样理解？ | [MIT Underactuated Robotics](https://underactuated.mit.edu/) | 后续控制与运动专题 |

[Hugging Face Robotics Course](https://huggingface.co/learn/robotics-course/en/unit0/1) 的部分单元仍标为 Coming Soon（2026-09-10 查阅）。其完整 Tutorial 正文与分单元网页是不同资源。[Datawhale every-embodied](https://github.com/datawhalechina/every-embodied) 可作中文主题补充；[dive-into-embodied-ai](https://github.com/datawhalechina/dive-into-embodied-ai) 明示 Alpha 和占位状态，选择已经有正文和实验的部分。

## 从问题看发展

| 问题的变化 | 代表工作 | 实验时追问 |
|---|---|---|
| 真实传感器与车辆约束下如何稳定运行？ | [Stanley，2006](https://robots.stanford.edu/papers/thrun.stanley05.pdf) | 感知、估计、控制与测试怎样共同支撑结果？ |
| 能否从驾驶示范学习像素到转向？ | [DAVE-2，2016](https://arxiv.org/abs/1604.07316) | 偏离专家轨迹后能否恢复？ |
| 多相机如何进入共同空间？ | [LSS，2020](https://arxiv.org/abs/2008.05711)、[BEVFormer，2022](https://arxiv.org/abs/2203.17270) | 标定、深度和时间误差如何传播？ |
| 模块目标如何共同服务驾驶？ | [UniAD，2023](https://arxiv.org/abs/2212.10156) | 中间指标改善是否改变实际驾驶？ |
| 离线好成绩为何不一定开得好？ | [NAVSIM，2024](https://arxiv.org/abs/2406.15349)、[Bench2Drive，2024](https://arxiv.org/abs/2406.03877) | 评测是否有环境反馈、交通反应和恢复能力？ |
| 示范数据为何不能覆盖自己行动后的状态？ | [DAgger，2011](https://arxiv.org/abs/1011.0686) | 数据来自专家还是学习者访问的状态？ |
| 一个观测有多种合理动作，怎样生成动作序列？ | [ACT，2023](https://arxiv.org/abs/2304.13705)、[Diffusion Policy，2023](https://arxiv.org/abs/2303.04137) | chunk 长度、反馈频率与延迟如何影响成功率？ |
| 如何迁移跨任务的视觉语言知识？ | [OpenVLA，2024](https://arxiv.org/abs/2406.09246)、[π0，2024](https://arxiv.org/abs/2410.24164) | 同数据下比简单策略好在哪里，泛化划分是什么？ |

这些路线相互交织。BEV 是空间表示，RL 是通过奖励学习的方法，VLA 是一种输入条件和动作模型范式；它们可以组合。经典控制在学习型系统中仍承担执行与反馈职责。

## 常用查询

[术语](glossary.md) · [指标](metrics.md) · [数据与仿真](datasets.md) · [原有材料](legacy/README.md)
