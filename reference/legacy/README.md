# 历史材料与复用建议

这里保存重构前的 11 课与 3 个 Lab。几何、坐标、时间、扰动与失败分析思路仍有参考价值；部分系统连接与指标定义存在已知缺口，已在对应 notebook 开头标出。

学习主线从[驾驶闭环单元](../../course/first_loop/README.md)开始。这里按问题查阅，不作为一套已经完成的智驾课程顺序修读。

| 材料 | 可参考的内容 | 需要注意 |
|---|---|---|
| [00 System & ODD](course/00_system_and_odd.ipynb) | 任务范围、场景与系统术语 | 原岗位目标属于历史叙事 |
| [01 Sensors & Geometry](course/01_sensors_geometry.ipynb) | 坐标变换、投影与时间 | 真实数据实验需另行准备 |
| [02 BEV & Fusion](course/02_bev_and_fusion.ipynb) | 栅格化与数据形状 | camera_points 是点集近似；occupancy 输入与标签重复 |
| [03 Temporal State](course/03_temporal_state.ipynb) | 时序估计、观测扰动 | 简化场景中的机制练习 |
| [04 Localization](course/04_localization_mapping.ipynb) | 漂移和观测中断 | 简化状态估计 |
| [05 Learnable BEV](course/05_learnable_bev_model.ipynb) | PyTorch 空间 query 与训练接口 | 先比较复制 LiDAR 的 identity baseline；不能据此证明学会真实 BEV |
| [06 Prediction](course/06_prediction.ipynb) | 他车多模态未来与误差 | 跨 mode 的阈值比例并非公开 benchmark 的跨样本 miss rate |
| [07 Planning](course/07_planning_closed_loop.ipynb) | 识别预测/规划语义错误 | 他车未来被选作轨迹；rollout 未消费选中轨迹 |
| [08 Evaluation](course/08_data_and_evaluation.ipynb) | 切片、回放和指标组织 | 不能验证第 05 章模型对驾驶的改善 |
| [09 Safety & Runtime](course/09_safety_runtime.ipynb) | 状态机和计时实验 | 阈值与 CPU 计时限于该机制 |
| [10 Capstone](course/10_capstone.ipynb) | artifact 加载与结果整理 | 模型推理没有重新驱动规划、执行和评测 |
| [Flow Matching / Action Chunk](labs/flow_matching_action_chunk.ipynb) | 生成式动作的局部机制 | 需另接真实策略训练与执行 |
| [VLM Conditions](labs/vlm_structured_driving_conditions.ipynb) | 结构化条件的接口 | 手写示例，未加载预训练 VLM |
| [VLA / World-Action](labs/vla_world_action_interface.ipynb) | 动作提议的接口 | 手写动作、加噪下一状态和固定 margin，未训练 VLA/world model |

## 复现实验

从仓库根目录运行 Jupyter，按旧 00→10 顺序执行时，数据仍写入 `artifacts/urban_cut_in/`。依赖见根目录的 `requirements-ml.txt`；它是历史 BEV 训练的独立环境，不必安装到第一单元环境。第 05/10 章需要 PyTorch，真实 nuScenes 样本仍需单独数据和依赖。

生成源为 `scripts/build_legacy_materials.py`；生成器统一附加归档说明，重新生成不会让这些材料回到当前课程目录。运行不报错只证明代码执行，通过指标获得领域结论仍须考虑上表的限制。
