# 视觉呈现增强验证

## 范围

新增一张封面和四张 AI 原理图，接入首页、项目与课程入口、四个单元说明及八份 notebook 的生成源。原理图配合正文解释和可展开的手算答案，分别讲解延迟执行、SE(2)/Kalman、BC闭环/共同前缀和MDP/速度策略。课程模型、指标、依赖与可执行实验逻辑保持原有实现。素材、提示词与来源见 [视觉素材说明](../assets/visuals/README.md)。

## 本机执行

2026-09-13，Windows 项目 `.venv/Scripts/python.exe`：

| 命令 | 实际结果 |
|---|---|
| `.venv/Scripts/python.exe -m pytest tests -q` | 32 passed，9.98 秒 |
| `.venv/Scripts/python.exe scripts/run_first_unit.py` | PASS |
| `.venv/Scripts/python.exe scripts/run_state_estimation.py` | PASS |
| `.venv/Scripts/python.exe scripts/run_imitation.py` | PASS |
| `.venv/Scripts/python.exe scripts/run_rl_foundations.py` | PASS，默认三 seed 配置 |

CLI 原始日志位于本机 `artifacts/visual-refresh/run_*.py.log`；真实实验输出仍由各 CLI 写入原有 artifact 目录。

## 生成、结构与手算核对

- `.venv/Scripts/python.exe scripts/validate_project.py`：8份当前及14份归档 notebook 的语法、结构、本地链接和索引通过；额外用同一 `check_links` 检查 `assets/visuals/README.md`。
- `.venv/Scripts/python.exe scripts/execute_notebooks.py`：01–08 全部通过，各课耗时 4.1、4.8、1.2、3.4、5.9、11.4、1.0、4.7 秒。日志保留于本机 `artifacts/visual-refresh/notebooks.log`。
- `.venv/Scripts/python.exe artifacts/visual-refresh/verify_presentation.py`：八课重复生成 SHA-256 相同；删除新增图解 cell 后，原有 cell 类型和 source 与基线完全一致；八课均导出 HTML 检查。
- 使用实际 `DelayedActuator(4)` 验证前四个决策区间为零动作、随后为 u₀/u₁；实际 `world_to_ego` 验证 `(2,3),(1,1),90° → (2,−1)`；`ScalarKalman` 在教学方差设定下验证 `2 → 3 m`；实际回报函数验证 `1+0.9×4=4.6`。`8/5/7` 为共同前缀规则的教学长度，非实验结果。
- 五张最终 PNG 均为 1536×1024、不透明 RGB，总计 8,132,887 字节。文件 SHA-256、大小和尺寸登记在素材 manifest；原始和修订提示词完整保留。底层图像模型版本无法由工具核验，来源明确记录这一点。
- `git diff --check` 通过。

## 浏览器验证

使用独立 agent-browser 会话及本机静态 HTTP 服务检查实际渲染。1440×1080 桌面和 390×844 手机视口均无横向溢出；五张图片均完整载入，四个手算答案都通过实际点击展开。手机显示四个“查看原图”链接；notebook 03 的导出 HTML 在 390 px 下图片可载入且无横向溢出。原理图采用完整宽度呈现，原始图片没有裁切。页面错误日志为空；axe 工具报告 0 条 WCAG A/AA 违规。截图保存在本机 `artifacts/visual-refresh/`。

## 独立审查与处理

Reviewer 为独立 `gpt-5.6-terra`，Devil's Advocate 为独立 `gpt-5.6-luna`，均未参与实现。最终两者结论分别为 PASS 与 GO，无未解决的阻断问题。

| 审查问题 | 最终处理与复核 |
|---|---|
| 原理解释需要步骤与可核对数字 | 四张单元图改为机制图，附时间表、坐标与滤波数值、共同前缀和MDP回报 |
| 生成器拼接可能影响 Markdown 缩进 | 独立新增 Markdown cell；全量对比确认原有 cell 内容保持一致 |
| 第一课会误以为已接入四步延迟 | 图注明第02课预览，正文解释零索引 t=4 是第5个决策区间 |
| 坐标变换与测量滤波可能被误画为一个串行处理链 | 正文分别解释坐标变换和真值→合成测量→滤波→控制器 |
| 纸上 MDP 与驾驶速度策略可能混同 | 原理图明确分栏；区分 short/long 与 2/6 m/s、γ=.9 与 γ=.99、采样训练与 argmax 评测 |
| BC共同前缀格数与颜色 | 最终图核对为8/5/7格，均只涂前5格，完整结果单列 |
| 透明背景与缺少固有尺寸影响阅读 | 最终图为不透明白底；HTML、文档与 notebook 都有固有尺寸和响应式显示 |

两位审查者均查看最终四张原理图，逐项确认公式、箭头、分组、格数与相邻文案一致。学习者能否据此独立解释与迁移，仍需后续试学反馈；本轮未将图解的发布视为学习效果实证。
