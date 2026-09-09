# Advanced Labs

这些实验在核心 11 课之后进行。它们不是进入智驾模型开发岗位的先修路径；它们把已有 `state → prediction → planning → safety` 接口延伸到生成式动作和多模态 foundation model 方向。

| Lab | Notebook | 前置 | 关注点 |
|---|---|---|---|
| 1 | [Flow Matching / Action Chunk](flow_matching_action_chunk.ipynb) | 06–07 | action space、horizon、replan、closed-loop drift |
| 2 | [VLM Structured Driving Conditions](vlm_structured_driving_conditions.ipynb) | 00、08–09 | schema、evidence、abstain、safety gate |
| 3 | [VLA / WA / π0 Interface](vla_world_action_interface.ipynb) | 03、07、09 | action proposal、world consequence、uncertainty |

默认 labs 使用合成机制实验，不加载大 checkpoint。真实 Hugging Face processor/backbone 扩展才安装 `requirements-frontier.txt`。无论使用 VLM 还是 VLA，输出都必须回到既有的 frame/time、planner contract 和 independent safety layer，不能直接控车。
