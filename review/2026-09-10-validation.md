# 重构验证记录 · 2026-09-10

## 环境

本机 Windows，第一单元新建 Python 3.11.15 环境 `.venv`。安装元数据确认 MetaDrive 0.4.3 来自上游 commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`，Panda3D 1.10.16、NumPy 2.4.6、Matplotlib 3.11.1、pytest 8.4.2、nbformat 5.11.1、nbclient 0.11.0、ipykernel 6.31.0。驾驶实验使用 CPU 物理仿真和真值状态；未进行 GPU 模型训练。

## 已完成的迁移检查

- 归档生成器生成 11 课与 3 个 Lab，每份开头包含归档用途和具体缺口。第 10 章的图改为实际文件读取/推理与缺失连接的审计图。
- 归档环境按顺序执行全部 14 份 notebook 成功，包含第 05/10 章 PyTorch 路径。执行副本保存在 `artifacts/executed/legacy/`。运行使用原 RL 项目已安装的 Python 环境，没有变更该环境；首次尝试曾遇到 pandas 导入的内存不足，检查后单独导入成功，第二次完整运行通过。
- 结构检查确认 2 份当前 notebook 与 14 份归档 notebook；Python 语法、notebook 格式、索引与本地文档链接通过。
- 实际浏览器检查中文首页桌面与 390px 手机布局；手机文档宽度与视口均为 390px，未出现横向溢出。主入口链接到现有单元 README。截图在 `artifacts/home-desktop.png` 与 `artifacts/home-mobile.png`。

归档执行成功只确认运行路径；已标注的概念、数据与系统连接缺陷仍作为历史材料保留。

## 第一单元最终验证

最终主线是固定长直段、单车道、无交通；起点在参考车道145m处，横向偏移+0.5m。保留默认连续线终止规则。MetaDrive 的有效地图配置为 block_sequence `S`，参考车道长290m。初版短入口段与默认+0.8m的配置已由实际运行发现问题并替换，旧 smoke/probe 文件不作为最终交付证据。

在第一单元 `.venv` 中执行：

```powershell
python scripts/build_first_unit.py
python scripts/validate_project.py
python -m pytest tests -q
python scripts/run_first_unit.py --failure-gif
python scripts/execute_notebooks.py
```

- **9项测试通过**：手算动作、精确延迟队列、非法配置、同条件策略干预、固定参考车道和前后状态衔接、从导出轨迹重算指标、同种子重现、参考点改变实际车辆轨迹、物理决策周期改变实际推进。
- **2份新 notebook 执行通过**：最后一次执行分别约2.4秒、4.0秒（不含首次安装）。执行副本位于 `artifacts/executed/first_loop/`；数值断言与轨迹重算 cell 均通过。
- **默认 CLI 实际三组对照**，结果位于 `artifacts/first_loop/summary.json` 及对应 `*_seed7.json/csv/png`：

| 条件 | 时长 | 行驶距离 | 平均绝对横向误差 | 终止结果 |
|---|---:|---:|---:|---|
| 无延迟、6m/s | 18.0s | 98.6386m | 0.0414m | 时长截断，未到终点 |
| 延迟0.4s、6m/s | 3.7s | 12.1678m | 0.3225m | 连续白线接触，出界终止 |
| 延迟0.4s、2m/s | 18.0s | 33.1224m | 0.0827m | 时长截断，未到终点 |

第二与第三组仅目标速度不同。完整时长不等，不能只按平均误差排名；第二课另算共同时间窗口。route_completion 是整条模拟器路线的累计位置比例，包含中途起点之前的部分，不等于本次完成比例。

另执行 `python scripts/run_first_unit.py --unit recovery --delay-steps 12 --failure-gif --output artifacts/delay12`：目标2m/s但延迟1.2秒时，在3.9秒、4.4565m处触发连续黄线终止，说明减速方案仍有失效范围。0.8秒延迟在最终+0.5m起点条件下跑满18秒；初步探查的不同条件下曾失败，最终结论采用这里的同条件结果。

检查实际轨迹 PNG、失败 GIF 最后一帧及其对应 trace：默认延迟失败回放共19帧，最后一帧为3.7秒终止状态，停留1000ms。车身矩形按实际长度/宽度与heading绘制，图像坐标等比例；它是几何示意回放。首次试验+0.8m时三组都在首步压白线，未据此宣称延迟效果；最终起点+0.5m避免了这项混杂。

修改的运行、生成与测试代码经过 Ruff 格式化和检查（使用本机已有RL环境的Ruff 0.15.12，不变更依赖）。未执行 Linux/WSL 的 MetaDrive 验证；GitHub Actions 状态见下方。

最后检查16份 notebook 的 SHA-256：重新运行两个生成器前后逐字节一致；结构、本地链接、Ruff与 `git diff --check` 通过。独立技术与教学审查的本次 P0/P1 问题均已关闭。

## 审查与远端状态

独立技术审查与教学反方审查分别记录在 [技术审查](2026-09-10-technical-review.md) 和 [教学审查](2026-09-10-devils-advocate.md)。初步发现及其最终处理状态由各审查者记录。

GitHub Actions 工作流已更新为生成一致性、结构检查、实际驾驶测试与 notebook 执行；本次尚未推送，因此没有本次远端 CI 运行结果。
