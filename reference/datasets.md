# Public datasets and runners

| Resource | Use in the course | Current evidence level |
|---|---|---|
| [nuScenes](https://www.nuscenes.org/) / [devkit](https://github.com/nutonomy/nuscenes-devkit) | Chapter 01/02/08 的 optional real-data checkpoint；多传感器、3D boxes、地图和时间链。 | 本仓库提供 adapter/checkpoint 形状；除非提交结果文件，不声称 benchmark 已复现。 |
| [nuPlan](https://www.nuscenes.org/nuplan) | planning/log replay 和 closed-loop 公开入口。 | next integration target。 |
| [NAVSIM](https://github.com/autonomousvision/navsim) | planning/simulation benchmark 入口。 | next integration target。 |
| [CARLA](https://carla.org/) | 可控仿真和 failure replay。 | next integration target。 |
| [Waymo Open Dataset](https://waymo.com/open/) | 大规模 perception/motion 数据；遵守许可。 | next integration target。 |
| [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) | 3D detection/BEV 代码生态参考。 | code reference，不等于本仓库依赖。 |

所谓 real-data checkpoint 至少要固定：dataset/version、sample/scenario ID、frame chain、timestamp policy、split、license、命令和输出 artifact。一个资源链接、一个 import 或一张手工截图都不构成公开 benchmark 结果。
