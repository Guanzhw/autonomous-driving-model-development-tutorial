# 视觉素材与来源

本目录保存一张课程封面和四张 AI 原理图，供首页、Markdown 单元说明与 notebook 导读复用。四张原理图包含可核对的步骤、公式和手算示例；网页与 notebook 配有正文解释和展开答案，实验结果仍由真实运行产生。

| 文件 | 用途 | 读图提示 |
|---|---|---|
| [driving-lab.png](driving-lab.png) | 首页、项目与课程入口 | 弯道与研究工作台是课程主题意象 |
| [feedback-loop.png](feedback-loop.png) | 第一单元、01–02 课 | 四步延迟的发出/执行时间表，0.4 s 时执行 u₀ |
| [state-estimation.png](state-estimation.png) | 第二单元、03–04 课 | 世界点 (2,3) 变换为 ego (2,−1)；K=0.5 的标量更新得到 3 m |
| [imitation-learning.png](imitation-learning.png) | 第三单元、05–06 课 | 完整 episode 划分、动作反馈与 8/5/7 步轨迹的共同前 5 步 |
| [reinforcement-learning.png](reinforcement-learning.png) | 第四单元、07–08 课 | 纸上 MDP 的 4.6 回报；另一个驾驶任务只学习 2/6 m/s 目标速度 |

2026-09-13 使用 Codex 内置 `image_gen` 生成与编辑。用户请求 GPT image-2.5；该工具未提供模型选择参数，也未返回可核验的模型版本，因此实际底层版本记为未知。未调用图片 API 或 CLI，未新增课程运行依赖。

完整生成及修订提示词见 [prompts.json](prompts.json)；最终 PNG 的尺寸、字节数和 SHA-256 见 [manifest.json](manifest.json)。保存工具返回的原始 PNG，网页复用本地文件，并通过原始宽高、响应式尺寸和延迟加载呈现。后续维护先修订对应提示词，再检查生成结果和消费者中的实际显示。
