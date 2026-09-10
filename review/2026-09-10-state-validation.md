# 第二阶段验证：坐标、测量与状态估计

## 实际环境与命令

Windows、本仓库 Python 3.11.15 环境；MetaDrive 0.4.3 固定提交与第一单元相同；Panda3D 1.10.16。CPU Torch 2.10.0 已为后续学习单元安装，但本单元依赖仍为 requirements-driving.txt。

```powershell
.venv/Scripts/python.exe -m pytest tests/test_driving.py tests/test_estimation.py -q
.venv/Scripts/python.exe scripts/run_state_estimation.py --output artifacts/state_verified
.venv/Scripts/python.exe scripts/execute_notebooks.py --unit state_estimation
.venv/Scripts/python.exe scripts/execute_notebooks.py --unit first_loop
.venv/Scripts/python.exe scripts/build_active_units.py
.venv/Scripts/python.exe scripts/validate_project.py
```

完整默认实验为 seed=7、noise seed=19、120步、每步0.1秒。首次完整运行测得：oracle/raw 均到12秒时长截断；filter 在8秒触黄实线结束。raw位置RMSE为0.233207m，filter为2.448323m；其自身原始测量位置RMSE为0.222313m。两组轨迹长度不同，后两数用于同一filter轨迹上评价原始测量与估计。

图和trace显示滤波位置滞后，状态转移的随机游走假设不适合此处运动速度；这是一项模型失配反例。没有把平滑或短时存活解释为定位/驾驶成功。输出包含JSON、CSV、轨迹PNG、比较图与真实失败GIF。

## 已发现问题与修正

- Windows实际子进程探针复现 Panda3D先于已安装Torch载入时 c10.dll WinError1114；Torch先载入则正常。共享环境构建入口据此调整初始化顺序。异常不会被吞掉；不要求无Torch环境安装Torch。
- 观测器使用同一 noisy world position 投影固定lane的s/e_y；truth、原始measurement、实际controller input分别记录。raw/filter同seed的噪声按sample index配对。
- observer每次run只reset一次，policy输出真正经过动作队列进入env.step。共享闭环的第一单元两课已重新执行通过。
- 独立审查：[技术审查](2026-09-10-state-technical.md)、[教学反方](2026-09-10-state-devils-advocate.md)。最终修正与复核结论见两份报告追加记录。

## 交付范围

主线登记四课；生成、执行和结构检查共用course_catalog。原14份材料保持归档，Python语法和本地链接继续验证。GitHub CI会在此次阶段推送后重新执行四课，实际运行结果以对应提交的Actions记录为准。

最终复核：16项测试通过（4.36秒）；第一单元两课与第二单元两课全部实际执行。Notebook04与CLI使用相同的120步和默认带偏差测量配置；CLI再次重现filter80步失败。角度跨±π回归通过；Q字段明确为每秒过程方差。Ruff、结构与链接检查通过。两位独立审查者已追加修正闭环结论。
