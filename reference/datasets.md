# 数据与仿真

| 资源 | 当前用途 | 运行状态 |
|---|---|---|
| [MetaDrive](https://github.com/metadriverse/metadrive) | 第一单元的车辆动力学与闭环环境 | 单元依赖与命令见[第一单元](../course/first_loop/README.md)；验证结果见维护记录 |
| [nuScenes](https://www.nuscenes.org/) | 后续坐标、传感器与真实数据练习 | 历史材料有 optional scaffolding；尚无固定真实数据结果 |
| [NAVSIM](https://github.com/autonomousvision/navsim) | 后续规划与评估协议选读 | 尚未集成 runner；先核对 v1/v2 与 split |
| [CARLA](https://carla.org/) / [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) | 后续城市道路闭环实验 | 尚未集成；Python/API/引擎按 benchmark 固定 |
| [LeRobot](https://huggingface.co/docs/lerobot/) | 后续操作数据、策略训练与评估 | 尚未集成；拟从 MuJoCo 与 ACT 小实验起步 |

第一单元用仿真真值构造控制观测，当前不训练视觉感知。真实道路数据适合检验传感器、时间和分布；交互式仿真适合检验动作后果，两者在后续课程中分别安排。

NAVSIM v1 用初始真实观测后固定计划进行短时推进；v2 pseudo-simulation 引入预生成偏移观测。使用时应写清协议，不能统称完整顺序交互式闭环。参见 [v1 论文](https://arxiv.org/abs/2406.15349)、[v2 论文](https://arxiv.org/abs/2506.04218)。

每次使用公开数据固定版本、场景 ID、划分和许可；下载预算包括压缩包、解压、缓存、checkpoint 和录像。第一单元无需下载这些道路数据集。
