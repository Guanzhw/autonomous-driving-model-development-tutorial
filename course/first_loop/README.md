# 第一单元：让一辆车跟住路线，并理解它为什么失败

从一辆车的反馈行为入门智驾。先理解状态、参考、动作和动力学，再用执行延迟制造可解释的失败，尝试减速恢复方案。面向会 Python、做过深度学习训练但没有 RL 背景的学习者。几何、单位、公式、实验和推理答案都在 notebook 中展开。

## 先跑起来

本机 Windows 已安装本单元环境。进入仓库根目录运行：

```powershell
.venv\Scripts\python.exe scripts\run_first_unit.py --failure-gif
.venv\Scripts\python.exe -m jupyter lab course\first_loop\01_drive_and_observe.ipynb
```

在另一台机器首次安装时，需要 Git 和 uv；使用 Python 3.11：

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-driving.txt
```

Linux/WSL 的解释器路径为 `.venv/bin/python`。本次实际验证平台是 Windows；远端 Linux CI 尚未运行。首次运行 MetaDrive 会下载对应 assets，需要网络。默认使用 CPU 物理仿真、无渲染窗口，输出 PNG 和 GIF 回放，不需要 GPU 训练或真实道路数据。

MetaDrive 0.4.3 固定到 commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`。JSON 中记录实际安装版本、commit、地图、初始状态、车道、车身尺寸和时间间隔。运行报错应保留日志并检查依赖，不要把出错改成假结果。

## 两课如何连起来

1. [01 · 车为什么能跟住一条线？](01_drive_and_observe.ipynb)：认识坐标与反馈，手算第一步动作，核对前后状态，改变初始偏移和参考距离。
2. [02 · 动作晚到之后，怎样恢复？](02_delay_and_recovery.ipynb)：解释延迟队列，比较三组条件，查看真实失败标志与车身回放，检查平均指标和减速代价。

先写下预测，再执行相应 cell。答案可折叠展开，但要保留自己预测错在哪里。notebook 中的代码模板可以直接修改、重跑和保存；关键控制实现位于 [driving.py](../../src/ad_tutorial/driving.py)。

## 明确的实验条件

本单元使用单车道长直段，固定参考车道长 290m，从纵向位置145m、横向偏移+0.5m起步，无其他车辆。默认18秒实验保持在该直段内；延长时长或提高速度可能走到参考段末端，需重新设计路线跟踪后再解释结果。保留 MetaDrive 默认的连续线终止规则。+0.8m 接近车身与白线的接触边界，不作为默认起点。

控制器直接读取模拟器真值位置、朝向、速度和车道投影；MetaDrive 返回的数值观测向量只记录维度，没有作为控制输入。传感器估计与视觉模型在后续单元接入。

| 对照 | 初始状态、地图、种子、时长 | 延迟 | 目标速度 |
|---|---|---|---|
| baseline | 相同 | 0s | 6m/s |
| delay | 相同 | 0.4s | 6m/s |
| recovery | 相同 | 0.4s | 2m/s |

recovery 只降低目标速度，保留造成失败的延迟。它是重新运行同一条件时采用的策略干预，不是碰撞后的救车过程。实际结果应一起报告失败、进度与用时；时长截断不等于到达终点。固定地图上改变 seed 也不等于测试了新道路。

## 输出与再实验

默认保存至 `artifacts/first_loop/`：每组 JSON、CSV、轨迹/误差/动作 PNG；失败时另有 GIF。JSON 的 before_* 是控制输入，无前缀状态是执行后的结果。GIF 根据真实轨迹和矩形车身绘制，保留最后失败帧，是俯视示意回放。

每次独立实验使用不同目录，保留配置和完整失败结果：

```powershell
# 只增加延迟，检查减速方案的失效范围
.venv\Scripts\python.exe scripts\run_first_unit.py --unit recovery --delay-steps 12 --failure-gif --output artifacts\delay12
# 改变起点，检验未调参的迁移表现
.venv\Scripts\python.exe scripts\run_first_unit.py --initial-offset -0.4 --failure-gif --output artifacts\offset_minus04
```

## 两周怎样安排

每周26小时是可投入预算，按理解程度推进；不必为凑满52小时反复运行默认实验。两本 notebook 是指导部分，主要时间用于修改、排错、选读与独立分析。

| 时间块 | 第一周：解释基线 | 第二周：解释失败 |
|---|---|---|
| 工作日5×2小时 | 逐段学习01；手算、核对输入/输出；每次只改一项 | 学习02；队列、三组对照、共同窗口与失败标志 |
| 周六8小时 | 初始偏移与前视距离实验；查代码；画图复核 | 延迟1/2/8/12步；保存全部结果；试减速干预 |
| 周日8小时 | TUM控制选读60–90分钟；写解释；环境与理解缓冲 | DAgger Introduction选读45分钟；未调参迁移；一页研究笔记 |

选读入口和问题在 notebook 中，暂不安装整套外部课程环境。

## 完成时应留下什么

- 手算第一步转向，并从实际 trace 重算一次控制输出与平均误差。
- 给出发出动作、延迟后动作和下一状态的对应关系，单位与时间准确。
- 保存同条件 baseline/delay/recovery 配置、轨迹和至少一个真实失败片段。
- 解释减速改善了什么、付出了什么；区分到达、失败与时长截断。
- 完成一个未调参起点的实验，用一页笔记说明预测、对照、结果、局限和下一项实验。

实际软件验证与独立审查见 [验证记录](../../review/2026-09-10-validation.md)。学习效果仍要通过你的试学和独立解释来判断。后续方向见 [课程路线](../README.md)。
