from pathlib import Path
from textwrap import dedent
import json


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def cell(cell_type, source):
    item = {
        "cell_type": cell_type,
        "metadata": {},
        "source": dedent(source).strip() + "\n",
    }
    if cell_type == "code":
        item.update({"execution_count": None, "outputs": []})
    return item


def notebook(cells, title):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "title": title,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(name, cells, title):
    path = NOTEBOOK_DIR / name
    path.write_text(json.dumps(notebook(cells, title), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


write(
    "00_odd_and_system_contract.ipynb",
    [
        cell("markdown", r'''
        # 00 · ODD 与 L4 系统契约

        L4 不是单个神经网络的准确率等级，而是一个自动驾驶系统在明确的 **Operational Design Domain（ODD）** 内完成任务、检测退化并安全退出的系统属性。本 notebook 先把“模型输入输出”放进可验证的系统契约中。

        学习目标：

        - 用结构化字段定义 ODD，而不是用“城市道路”这种模糊描述；
        - 为 sensor bundle、localization、planner 和 safety monitor 写输入/输出/时间约束；
        - 区分 `NOMINAL`、`DEGRADED`、`MINIMAL_RISK` 与 `ODD_EXIT`；
        - 计算 ODD coverage、契约违规率和延迟预算余量。

        下面的实现是可运行的教学版。它不构成真实车辆的 safety case，也不能替代 ISO 26262、SOTIF 或公司的系统安全流程。
        '''),
        cell("code", r'''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from dataclasses import dataclass
        from ipywidgets import interact, FloatSlider

        plt.rcParams["figure.figsize"] = (10, 4.5)
        plt.rcParams["axes.grid"] = True
        '''),
        cell("markdown", r'''
        ## Part A — 把 ODD 写成可计算的约束

        这里定义一个简化的 robotaxi ODD：限定地理区域、最高速度、允许的雨量、是否允许夜间运行，以及定位和传感器的最低健康度。真实系统还会加入道路类型、施工、地图版本、交通规则、通信和车辆平台等约束。
        '''),
        cell("code", r'''
        @dataclass(frozen=True)
        class ODD:
            geography: str = "urban_mapped"
            max_speed_mps: float = 13.9
            max_rain_mm_h: float = 8.0
            night_allowed: bool = False
            min_sensor_health: float = 0.70
            max_localization_sigma_m: float = 0.80


        odd = ODD()


        def make_scenarios(n=320, seed=7):
            rng = np.random.default_rng(seed)
            return pd.DataFrame(
                {
                    "geography": rng.choice(["urban_mapped", "rural_unmapped", "urban_mapped"], n),
                    "speed_mps": rng.uniform(4.0, 18.0, n),
                    "rain_mm_h": rng.uniform(0.0, 14.0, n),
                    "night": rng.random(n) < 0.25,
                    "sensor_health": rng.uniform(0.45, 1.0, n),
                    "localization_sigma_m": rng.uniform(0.15, 1.35, n),
                }
            )


        def classify_odd(frame, definition=odd):
            checks = pd.DataFrame(
                {
                    "geography_ok": frame["geography"].eq(definition.geography),
                    "speed_ok": frame["speed_mps"].le(definition.max_speed_mps),
                    "rain_ok": frame["rain_mm_h"].le(definition.max_rain_mm_h),
                    "light_ok": frame["night"].le(definition.night_allowed),
                    "sensor_ok": frame["sensor_health"].ge(definition.min_sensor_health),
                    "localization_ok": frame["localization_sigma_m"].le(definition.max_localization_sigma_m),
                }
            )
            return checks, checks.all(axis=1)


        scenarios = make_scenarios()
        checks, scenarios["in_odd"] = classify_odd(scenarios)
        coverage = scenarios["in_odd"].mean()
        print(f"synthetic ODD coverage: {coverage:.1%}")
        display(scenarios.head())
        '''),
        cell("code", r'''
        check_rate = checks.mean().sort_values()
        ax = check_rate.plot(kind="barh", color="#4f8cc9")
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction satisfying the constraint")
        ax.set_title("Which ODD constraint removes the most scenarios?")
        plt.show()
        '''),
        cell("markdown", r'''
        ## Part B — 为模块定义输入、输出和时间契约

        一个模型接口至少应说明：

        - 输入字段和单位；
        - 时间戳、最大允许 age 和坐标系；
        - 输出的语义及其置信度；
        - 最大 latency 和缺失数据处理方式。

        `validate_sensor_bundle` 只做静态检查；它不会证明传感器数据正确，也不会证明模型在 OOD 场景中安全。
        '''),
        cell("code", r'''
        SENSOR_CONTRACT = {
            "required_keys": {"camera", "lidar", "imu", "timestamp_s", "frame_id"},
            "frame_id": "base_link",
            "max_age_s": {"camera": 0.15, "lidar": 0.10, "imu": 0.05},
            "max_latency_ms": 100.0,
        }


        def validate_sensor_bundle(bundle, contract=SENSOR_CONTRACT):
            issues = []
            missing = contract["required_keys"] - set(bundle)
            if missing:
                issues.append(f"missing keys: {sorted(missing)}")
            if bundle.get("frame_id") != contract["frame_id"]:
                issues.append("frame_id does not match the module contract")
            for sensor, max_age in contract["max_age_s"].items():
                age = bundle.get(f"{sensor}_age_s")
                if age is None:
                    issues.append(f"missing age for {sensor}")
                elif age > max_age:
                    issues.append(f"{sensor} age {age:.3f}s exceeds {max_age:.3f}s")
            latency = bundle.get("latency_ms")
            if latency is not None and latency > contract["max_latency_ms"]:
                issues.append(f"latency {latency:.1f}ms exceeds budget")
            return {"valid": not issues, "issues": issues}


        valid_bundle = {
            "camera": np.zeros((8, 8, 3)), "lidar": np.zeros((20, 4)), "imu": np.zeros(6),
            "timestamp_s": 12.0, "frame_id": "base_link", "camera_age_s": 0.04,
            "lidar_age_s": 0.03, "imu_age_s": 0.01, "latency_ms": 61.0,
        }
        broken_bundle = {**valid_bundle, "frame_id": "camera_front", "lidar_age_s": 0.21, "latency_ms": 124.0}
        print("valid:", validate_sensor_bundle(valid_bundle))
        print("broken:", validate_sensor_bundle(broken_bundle))
        '''),
        cell("markdown", r'''
        ## Part C — 从健康度到系统状态

        L4 系统需要把模型输出和运行时健康度连接起来。下面的状态机是最小示例：

        - `NOMINAL`：正常运行；
        - `DEGRADED`：降低速度或切换到保守策略；
        - `MINIMAL_RISK`：执行最小风险动作；
        - `ODD_EXIT`：当前条件超出 ODD，停止接受新的自动驾驶任务。
        '''),
        cell("code", r'''
        def system_state(sensor_health, localization_sigma, latency_ms, odd_ok,
                         min_health=0.70, max_sigma=0.80, max_latency=100.0):
            if not odd_ok:
                return "ODD_EXIT"
            if sensor_health < 0.45 or localization_sigma > 1.50 or latency_ms > 1.5 * max_latency:
                return "MINIMAL_RISK"
            if sensor_health < min_health or localization_sigma > max_sigma or latency_ms > max_latency:
                return "DEGRADED"
            return "NOMINAL"


        health = np.linspace(0.95, 0.35, 80)
        sigma = np.linspace(0.20, 1.70, 80)
        latency = np.linspace(55.0, 140.0, 80)
        states = [
            system_state(h, s, l, odd_ok=(i < 68))
            for i, (h, s, l) in enumerate(zip(health, sigma, latency))
        ]
        print(pd.Series(states).value_counts().to_dict())

        state_code = {"NOMINAL": 0, "DEGRADED": 1, "MINIMAL_RISK": 2, "ODD_EXIT": 3}
        plt.step(np.arange(len(states)), [state_code[s] for s in states], where="post")
        plt.yticks(list(state_code.values()), list(state_code))
        plt.xlabel("cycle")
        plt.ylabel("system state")
        plt.title("A system state is not the same thing as a model confidence score")
        plt.show()
        '''),
        cell("markdown", r'''
        ### 交互练习

        调整最低传感器健康度，观察 ODD coverage 和状态分布如何变化。然后完成以下 TODO：

        1. 增加 `map_version` 和 `route_available` 两个契约字段；
        2. 给 `DEGRADED` 增加恢复滞回（连续若干个健康周期后才回到 `NOMINAL`）；
        3. 计算每条约束单独造成的 coverage loss；
        4. 写出一个“模型置信度很高，但输入已经 stale”的反例。
        '''),
        cell("code", r'''
        def inspect_odd(min_sensor_health=0.70, max_localization_sigma=0.80):
            definition = ODD(
                min_sensor_health=min_sensor_health,
                max_localization_sigma_m=max_localization_sigma,
            )
            _, in_odd = classify_odd(scenarios, definition)
            print(f"coverage = {in_odd.mean():.1%}")
            print(pd.Series([
                system_state(row.sensor_health, row.localization_sigma_m, 70.0, bool(ok),
                             min_health=min_sensor_health,
                             max_sigma=max_localization_sigma)
                for row, ok in zip(scenarios.itertuples(), in_odd)
            ]).value_counts().to_dict())


        interact(
            inspect_odd,
            min_sensor_health=FloatSlider(value=0.70, min=0.45, max=0.95, step=0.05),
            max_localization_sigma=FloatSlider(value=0.80, min=0.30, max=1.50, step=0.05),
        )
        '''),
        cell("markdown", r'''
        ## 完成标准

        你应能提交一页 system contract：列出 ODD、每个模块的输入输出、时间预算、坐标系、降级动作和验证指标。只写“模型准确率 95%”不能替代这份契约。
        '''),
    ],
    "00 ODD and system contract",
)


write(
    "05_training_evaluation_baseline.ipynb",
    [
        cell("markdown", r'''
        # 05 · PyTorch Transformer / BEV Query 训练基线

        当前主流的 BEV、检测、tracking 和 planning 模型大量使用 attention，但“知道 attention 这个词”不等于能训练和评测一个时序/多模态模型。本 notebook 用一个小型 PyTorch Transformer 处理 camera/LiDAR token，学习一个 BEV query 的分类任务。

        核心公式为

        \[
        \operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
        \]

        学习目标：

        - 用 `nn.TransformerEncoder` 实现 token mixing；
        - 用一个 learned BEV query 汇聚多模态 token；
        - 训练、验证、数据扰动和指标报告全部可复现；
        - 比较 LiDAR dropout 对模型性能和推理 latency 的影响。

        这里不依赖 Hugging Face `transformers`：我们学习的是架构和训练接口，不是加载预训练语言模型。`transformers` 会在 VLM/VLA 前沿分支中作为可选依赖出现。
        '''),
        cell("code", r'''
        import time
        import numpy as np
        import matplotlib.pyplot as plt
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from ipywidgets import interact, FloatSlider

        torch.set_num_threads(1)
        torch.manual_seed(23)
        np.random.seed(23)
        plt.rcParams["figure.figsize"] = (9, 4.5)
        plt.rcParams["axes.grid"] = True
        device = torch.device("cpu")
        print("torch:", torch.__version__, "device:", device)
        '''),
        cell("markdown", r'''
        ## Part A — 构造带有模态交互的 token 数据

        每个样本有 6 个 camera token 和 6 个 LiDAR token。标签依赖于两种模态的统计量以及一个交互项，因此只看单一模态不能稳定完成任务。真实 BEV 模型会把 token 换成 image feature、voxel feature、map feature 或 temporal memory。
        '''),
        cell("code", r'''
        def make_token_dataset(n=900, seed=23, d_in=12):
            rng = np.random.default_rng(seed)
            camera = rng.normal(size=(n, 6, d_in)).astype(np.float32)
            lidar = rng.normal(size=(n, 6, d_in)).astype(np.float32)
            camera_signal = camera[:, :, :4].mean(axis=(1, 2))
            lidar_signal = lidar[:, :, :4].mean(axis=(1, 2))
            interaction = camera_signal * lidar_signal
            score = 1.00 * camera_signal + 1.30 * lidar_signal + 0.50 * interaction
            bins = np.quantile(score, [1 / 3, 2 / 3])
            label = np.digitize(score, bins).astype(np.int64)
            tokens = np.concatenate([camera, lidar], axis=1)
            modality = np.concatenate([np.zeros(6, dtype=np.int64), np.ones(6, dtype=np.int64)])
            return torch.tensor(tokens), torch.tensor(label), torch.tensor(modality)


        tokens, labels, modality = make_token_dataset()
        split = int(0.8 * len(tokens))
        train_x, val_x = tokens[:split], tokens[split:]
        train_y, val_y = labels[:split], labels[split:]
        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
        print("train:", tuple(train_x.shape), "val:", tuple(val_x.shape), "class counts:", torch.bincount(labels).tolist())
        '''),
        cell("code", r'''
        class TinyBEVQueryModel(nn.Module):
            def __init__(self, d_in=12, d_model=48, nhead=4, layers=2, n_classes=3):
                super().__init__()
                self.input_proj = nn.Linear(d_in, d_model)
                self.modality_embedding = nn.Embedding(2, d_model)
                self.query = nn.Parameter(torch.zeros(1, 1, d_model))
                self.position = nn.Parameter(torch.zeros(1, 13, d_model))
                nn.init.normal_(self.query, std=0.02)
                nn.init.normal_(self.position, std=0.02)
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
                    dropout=0.0, batch_first=True, norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
                self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

            def forward(self, x, modality_ids):
                h = self.input_proj(x) + self.modality_embedding(modality_ids)
                query = self.query.expand(x.size(0), -1, -1)
                h = torch.cat([query, h], dim=1) + self.position[:, : x.size(1) + 1]
                return self.head(self.encoder(h)[:, 0])


        def macro_f1(y_true, y_pred, n_classes=3):
            scores = []
            for cls in range(n_classes):
                tp = ((y_true == cls) & (y_pred == cls)).sum()
                fp = ((y_true != cls) & (y_pred == cls)).sum()
                fn = ((y_true == cls) & (y_pred != cls)).sum()
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                scores.append(2 * precision * recall / max(precision + recall, 1e-8))
            return float(np.mean(scores))


        def evaluate(model, x, y, dropout=0.0, noise_std=0.0, seed=91):
            local = torch.Generator().manual_seed(seed)
            perturbed = x.clone()
            if dropout > 0:
                mask = torch.rand(perturbed[:, 6:].shape, generator=local) < dropout
                perturbed[:, 6:] = perturbed[:, 6:].masked_fill(mask, 0.0)
            if noise_std > 0:
                noise = torch.randn(perturbed.shape, generator=local) * noise_std
                perturbed = perturbed + noise
            with torch.no_grad():
                logits = model(perturbed.to(device), modality.to(device))
            pred = logits.argmax(dim=1).cpu()
            accuracy = float((pred == y).float().mean())
            return {"accuracy": accuracy, "macro_f1": macro_f1(y.numpy(), pred.numpy()), "pred": pred}


        def train_model(epochs=42):
            model = TinyBEVQueryModel().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            history = {"train_loss": [], "val_accuracy": []}
            for _ in range(epochs):
                model.train()
                losses = []
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(batch_x, modality)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss))
                model.eval()
                history["train_loss"].append(float(np.mean(losses)))
                history["val_accuracy"].append(evaluate(model, val_x, val_y)["accuracy"])
            return model, history


        model, history = train_model()
        print(evaluate(model, val_x, val_y))
        '''),
        cell("code", r'''
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["train_loss"])
        axes[0].set_title("training loss")
        axes[0].set_xlabel("epoch")
        axes[1].plot(history["val_accuracy"], color="#d97706")
        axes[1].set_title("validation accuracy")
        axes[1].set_xlabel("epoch")
        plt.show()


        def show_robustness(lidar_dropout=0.0, noise_std=0.0):
            model.eval()
            result = evaluate(model, val_x, val_y, dropout=lidar_dropout, noise_std=noise_std)
            print({k: round(v, 4) for k, v in result.items() if k != "pred"})


        interact(
            show_robustness,
            lidar_dropout=FloatSlider(value=0.0, min=0.0, max=1.0, step=0.1),
            noise_std=FloatSlider(value=0.0, min=0.0, max=0.8, step=0.05),
        )
        '''),
        cell("code", r'''
        def benchmark(model, sample, repeats=80):
            model.eval()
            sample = sample[:1]
            times = []
            with torch.no_grad():
                for _ in range(10):
                    _ = model(sample, modality)
                for _ in range(repeats):
                    start = time.perf_counter()
                    _ = model(sample, modality)
                    times.append((time.perf_counter() - start) * 1000)
            return {"p50_ms": np.percentile(times, 50), "p95_ms": np.percentile(times, 95), "p99_ms": np.percentile(times, 99)}


        print("batch=1 latency:", {k: round(v, 3) for k, v in benchmark(model, val_x).items()})
        print("parameters:", sum(p.numel() for p in model.parameters()))
        '''),
        cell("markdown", r'''
        ### 练习与依赖边界

        1. 把 learned query 改成两个 query，分别输出 road occupancy 和 interaction risk；
        2. 增加 causal temporal mask，把 3 个历史时刻作为 token；
        3. 比较 `nhead=1/2/4/8` 的精度与 latency；
        4. 用 `torch.profiler` 记录 CPU kernel 和 memory；
        5. 安装 `requirements-frontier.txt` 后，只用 `transformers` 加载一个公开 encoder，写一个 adapter 把它的 hidden states 接到本 notebook 的 BEV query head。

        关键区分：`torch` 是训练自定义 AD 模型的基础；Hugging Face `transformers` 是预训练 Transformer/VLM 的生态入口，不是所有 BEV、tracking 或 planning notebook 的必需依赖。
        '''),
    ],
    "05 PyTorch Transformer BEV baseline",
)


write(
    "16_localization_and_mapping.ipynb",
    [
        cell("markdown", r'''
        # 16 · Localization 与 Mapping：从漂移到可恢复定位

        感知模型输出目标之后，系统仍需要知道 ego vehicle 在地图和路网中的位置。L4 研发岗位通常同时关注 state estimation、地图匹配、传感器退化和 relocalization。

        本 notebook 使用二维道路中心线作为简化地图，模拟：

        - wheel/IMU odometry 的累积漂移；
        - GNSS 噪声、outlier 和 outage；
        - 基于地图最近点的 map matching；
        - odometry + GNSS + map prior 的轻量融合。

        这不是完整的 factor graph 或 LiDAR SLAM，但接口和误差分析与真实系统一致：明确坐标系、观测时间和退化模式。
        '''),
        cell("code", r'''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from ipywidgets import interact, FloatSlider

        plt.rcParams["figure.figsize"] = (10, 5)
        plt.rcParams["axes.grid"] = True

        def make_route(steps=260, dt=0.1):
            t = np.arange(steps) * dt
            x = 0.65 * t
            y = 3.5 * np.sin(x / 14.0) + 0.8 * np.sin(x / 5.0)
            return t, np.c_[x, y]


        time_s, truth = make_route()
        map_x = np.linspace(truth[:, 0].min() - 3, truth[:, 0].max() + 3, 700)
        map_points = np.c_[map_x, 3.5 * np.sin(map_x / 14.0) + 0.8 * np.sin(map_x / 5.0)]
        '''),
        cell("markdown", r'''
        ## Part A — 观测模型与漂移

        设真值位置为 \(p_t\)。里程计提供增量

        \[
        \Delta \hat p_t = \Delta p_t + b\Delta t + \epsilon_t,
        \]

        因而积分后的误差会随时间增长。GNSS 是绝对观测，但会出现噪声和离群点；地图匹配则提供一个带有地图先验的几何约束。
        '''),
        cell("code", r'''
        def simulate_observations(gnss_noise=1.4, outage_start=120, outage_end=175, seed=12):
            rng = np.random.default_rng(seed)
            delta = np.diff(truth, axis=0, prepend=truth[:1])
            odom_delta = delta + np.array([0.012, -0.008]) + rng.normal(0, 0.035, delta.shape)
            odom = truth[0] + np.cumsum(odom_delta, axis=0)
            gnss = truth + rng.normal(0, gnss_noise, truth.shape)
            valid = np.ones(len(truth), dtype=bool)
            valid[outage_start:outage_end] = False
            outlier_ids = np.arange(55, len(truth), 83)
            gnss[outlier_ids] += np.array([7.0, -5.0])
            valid[outlier_ids] = False
            return odom, gnss, valid


        def nearest_map_point(position):
            distance = np.linalg.norm(map_points - position, axis=1)
            return map_points[np.argmin(distance)]


        def fuse_localization(odom, gnss, gnss_valid, gnss_gain=0.18, map_gain=0.08):
            estimate = np.zeros_like(odom)
            estimate[0] = odom[0]
            for i in range(1, len(odom)):
                prediction = estimate[i - 1] + (odom[i] - odom[i - 1])
                if gnss_valid[i]:
                    prediction = (1 - gnss_gain) * prediction + gnss_gain * gnss[i]
                matched = nearest_map_point(prediction)
                estimate[i] = (1 - map_gain) * prediction + map_gain * matched
            return estimate


        odom, gnss, valid = simulate_observations()
        fused = fuse_localization(odom, gnss, valid)

        def report(name, estimate):
            error = np.linalg.norm(estimate - truth, axis=1)
            return {"system": name, "RMSE_m": np.sqrt(np.mean(error ** 2)), "p95_m": np.percentile(error, 95), "max_m": error.max()}


        display(pd.DataFrame([report("odometry", odom), report("fused", fused)]))
        '''),
        cell("code", r'''
        fig, ax = plt.subplots()
        ax.plot(truth[:, 0], truth[:, 1], label="ground truth", linewidth=3)
        ax.plot(map_points[:, 0], map_points[:, 1], "--", label="map centerline", alpha=0.7)
        ax.plot(odom[:, 0], odom[:, 1], label="odometry drift")
        ax.plot(gnss[valid, 0], gnss[valid, 1], ".", label="valid GNSS", alpha=0.35)
        ax.plot(fused[:, 0], fused[:, 1], label="fused estimate")
        ax.set_aspect("equal")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.set_title("Localization with odometry drift and map prior")
        ax.legend()
        plt.show()
        '''),
        cell("code", r'''
        def inspect_localization(gnss_noise=1.4, map_gain=0.08, outage_length=55):
            odom, gnss, valid = simulate_observations(gnss_noise=gnss_noise, outage_end=120 + int(outage_length))
            estimate = fuse_localization(odom, gnss, valid, map_gain=map_gain)
            error = np.linalg.norm(estimate - truth, axis=1)
            print({k: round(v, 3) for k, v in report("fused", estimate).items() if k != "system"})
            plt.plot(time_s, error, label="position error")
            plt.axvspan(120 * 0.1, (120 + outage_length) * 0.1, color="red", alpha=0.12, label="GNSS outage")
            plt.axhline(0.8, color="black", linestyle="--", label="ODD sigma threshold")
            plt.xlabel("time / s")
            plt.ylabel("position error / m")
            plt.legend()
            plt.show()


        interact(
            inspect_localization,
            gnss_noise=FloatSlider(value=1.4, min=0.1, max=4.0, step=0.1),
            map_gain=FloatSlider(value=0.08, min=0.0, max=0.5, step=0.02),
            outage_length=FloatSlider(value=55, min=0, max=120, step=5),
        )
        '''),
        cell("markdown", r'''
        ### 练习

        1. 把 map matching 改成带 heading 的最近车道匹配，拒绝横向误差过大的匹配；
        2. 加入 IMU yaw bias，并报告 heading RMSE；
        3. 统计 GNSS outage 结束后恢复到 0.8 m 以内需要多少秒；
        4. 构造一个“地图版本错误但 GNSS 正常”的 failure case；
        5. 把融合输出写成 `localization_state = {pose, covariance, map_version, timestamp}`，供后续 planner 使用。
        '''),
        cell("markdown", r'''
        ## 完成标准

        除了画出轨迹，还要记录 outage、outlier、地图匹配错误对 RMSE/p95/max error 的影响，并说明哪些误差会直接改变 ODD 状态。
        '''),
    ],
    "16 Localization and mapping",
)


write(
    "17_scenario_runner_log_replay.ipynb",
    [
        cell("markdown", r'''
        # 17 · Scenario Runner 与 Log Replay：从一条轨迹到可回归的测试集

        只在静态日志上计算 ADE/FDE 或控制误差，无法观察 planner 偏离专家轨迹之后的误差累积。L4 开发需要把道路事件变成可重复执行的 scenario，并区分：

        - `log_replay`：ego 和其他交通参与者都按记录回放；
        - `closed_loop_nonreactive`：ego 由当前策略控制，其他参与者按记录运行；
        - `closed_loop_reactive`：ego 和其他参与者都根据当前状态作出反应。

        这里用一个轻量 runner 模拟前车急刹场景，接口可以进一步接到 nuPlan、NAVSIM 或 CARLA ScenarioRunner。
        '''),
        cell("code", r'''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from dataclasses import dataclass, replace

        plt.rcParams["figure.figsize"] = (10, 4.5)
        plt.rcParams["axes.grid"] = True

        @dataclass(frozen=True)
        class Scenario:
            name: str
            initial_gap_m: float = 28.0
            ego_speed_mps: float = 8.0
            lead_speed_mps: float = 6.0
            lead_brake_time_s: float = 4.0
            duration_s: float = 10.0
            dt: float = 0.1
            reactive_agents: bool = False


        scenario = Scenario("lead_vehicle_hard_brake")
        '''),
        cell("markdown", r'''
        ## Part A — 记录轨迹和 scenario contract

        一个 scenario 不应只保存视频文件名，还需要初始状态、交通参与者行为、地图/天气条件、随机种子和可计算的 pass/fail criteria。下面的最小 contract 只包含纵向运动。
        '''),
        cell("code", r'''
        def lead_profile(s, sc):
            speed = sc.lead_speed_mps if s < sc.lead_brake_time_s else 0.0
            return speed


        def run_scenario(sc, mode="closed_loop_reactive", policy="cautious"):
            time_s = np.arange(0.0, sc.duration_s, sc.dt)
            lead_x = sc.initial_gap_m
            ego_x = 0.0
            ego_speed = sc.ego_speed_mps
            rows = []
            for now in time_s:
                lead_speed = lead_profile(now, sc)
                gap = lead_x - ego_x
                if mode == "closed_loop_reactive" and sc.reactive_agents and gap < 16.0:
                    lead_speed = min(lead_speed, max(0.0, ego_speed - 1.0))

                if mode == "log_replay":
                    target_speed = sc.ego_speed_mps
                elif policy == "naive":
                    target_speed = sc.ego_speed_mps
                else:
                    target_speed = sc.ego_speed_mps
                    if gap < 18.0:
                        target_speed = min(target_speed, max(0.0, lead_speed - 1.0))
                    if gap < 8.0:
                        target_speed = 0.0

                acceleration = np.clip((target_speed - ego_speed) * 1.4, -4.0, 2.0)
                if mode == "log_replay":
                    acceleration = 0.0
                ego_speed = max(0.0, ego_speed + acceleration * sc.dt)
                ego_x += ego_speed * sc.dt
                lead_x += lead_speed * sc.dt
                rows.append({"time_s": now, "ego_x": ego_x, "lead_x": lead_x,
                             "ego_speed": ego_speed, "lead_speed": lead_speed,
                             "gap": lead_x - ego_x, "acceleration": acceleration})
            result = pd.DataFrame(rows)
            result["collision"] = result["gap"] < 2.0
            return result


        def metrics(result):
            acceleration = result["acceleration"].to_numpy()
            jerk = np.diff(acceleration, prepend=acceleration[0]) / 0.1
            return {
                "collision": bool(result["collision"].any()),
                "min_gap_m": float(result["gap"].min()),
                "progress_m": float(result["ego_x"].iloc[-1]),
                "max_decel_mps2": float(-min(acceleration.min(), 0.0)),
                "max_jerk_mps3": float(np.abs(jerk).max()),
            }


        results = {}
        for mode in ["log_replay", "closed_loop_nonreactive", "closed_loop_reactive"]:
            results[mode] = run_scenario(replace(scenario, reactive_agents=True), mode=mode)
        display(pd.DataFrame({mode: metrics(result) for mode, result in results.items()}))
        '''),
        cell("code", r'''
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for mode, result in results.items():
            axes[0].plot(result["time_s"], result["gap"], label=mode)
            axes[1].plot(result["ego_x"], result["lead_x"], label=mode)
        axes[0].axhline(2.0, color="red", linestyle="--", label="collision threshold")
        axes[0].set(xlabel="time / s", ylabel="lead gap / m", title="Open-loop and closed-loop gap")
        axes[1].set(xlabel="ego x / m", ylabel="lead x / m", title="Trajectory interaction")
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
        plt.show()
        '''),
        cell("markdown", r'''
        ## Part B — Counterfactual scenario sweep

        真实 data engine 会从 intervention、近碰撞、planner disagreement 等事件挖掘场景，并生成参数化变体。这里扫描初始 gap 和急刹时间，得到一个小型 regression matrix。
        '''),
        cell("code", r'''
        rows = []
        for gap in np.linspace(12, 36, 9):
            for brake_time in np.linspace(2.0, 6.0, 9):
                sc = replace(scenario, initial_gap_m=float(gap), lead_brake_time_s=float(brake_time), reactive_agents=True)
                result = run_scenario(sc, mode="closed_loop_reactive", policy="cautious")
                rows.append({"gap_m": gap, "brake_time_s": brake_time, **metrics(result)})
        sweep = pd.DataFrame(rows)
        pivot = sweep.pivot(index="gap_m", columns="brake_time_s", values="collision")
        plt.imshow(pivot.to_numpy(), origin="lower", aspect="auto", cmap="RdYlGn_r")
        plt.colorbar(label="collision")
        plt.xticks(range(len(pivot.columns)), [f"{x:.1f}" for x in pivot.columns], rotation=45)
        plt.yticks(range(len(pivot.index)), [f"{x:.0f}" for x in pivot.index])
        plt.xlabel("lead brake time / s")
        plt.ylabel("initial gap / m")
        plt.title("Scenario regression matrix")
        plt.show()
        print("collision rate:", sweep["collision"].mean())
        '''),
        cell("markdown", r'''
        ### 练习

        1. 增加横穿行人和红灯两个 scenario type；
        2. 给 reactive agent 加入基于 TTC 的让行策略；
        3. 将 `metrics` 扩展为 collision、off-road、progress、jerk、fallback latency 和 scenario coverage；
        4. 设计一个 scenario ID，使同一变体可以在 log replay、nuPlan 和 CARLA runner 中对应；
        5. 解释为什么一个 open-loop 低误差模型可能在 closed-loop 中发生碰撞。
        '''),
        cell("markdown", r'''
        ## 完成标准

        交付一张 scenario regression 表和一个失败场景回放图；不能只报告平均轨迹误差。
        '''),
    ],
    "17 Scenario runner and log replay",
)


write(
    "18_safety_state_machine_degraded_mode.ipynb",
    [
        cell("markdown", r'''
        # 18 · Safety State Machine 与 Degraded Mode

        corner-case rule 不应只是一个孤立的二分类器。它需要读取 sensor health、localization covariance、planner disagreement、TTC、时间戳 age 和 ODD 状态，并决定车辆是否继续运行、降速、执行最小风险动作或退出 ODD。

        本 notebook 实现一个带有恢复滞回的 safety state machine，并报告：

        - fault 到降级状态的检测 latency；
        - MINIMAL_RISK 的触发原因；
        - nominal 条件下的 false fallback；
        - 传感器恢复后重新进入 nominal 的稳定时间。
        '''),
        cell("code", r'''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from dataclasses import dataclass
        from ipywidgets import interact, FloatSlider

        plt.rcParams["figure.figsize"] = (11, 6)
        plt.rcParams["axes.grid"] = True

        STATES = ["NOMINAL", "DEGRADED", "MINIMAL_RISK", "ODD_EXIT"]
        ACTIONS = {
            "NOMINAL": "learned_policy",
            "DEGRADED": "slow_down_and_monitor",
            "MINIMAL_RISK": "pull_over_or_stop",
            "ODD_EXIT": "stop_accepting_mission",
        }


        @dataclass
        class SafetyStateMachine:
            health_threshold: float = 0.70
            localization_threshold_m: float = 0.80
            ttc_threshold_s: float = 2.50
            recovery_cycles: int = 8
            state: str = "NOMINAL"
            good_cycles: int = 0

            def update(self, sensor_health, localization_sigma, ttc, planner_disagreement, odd_ok, age_s):
                hard_fault = sensor_health < 0.40 or localization_sigma > 1.50 or age_s > 0.60
                immediate_risk = ttc < self.ttc_threshold_s or planner_disagreement > 0.65
                degraded = sensor_health < self.health_threshold or localization_sigma > self.localization_threshold_m or age_s > 0.15
                if not odd_ok:
                    self.state = "ODD_EXIT"
                elif hard_fault or immediate_risk:
                    self.state = "MINIMAL_RISK"
                elif degraded:
                    self.state = "DEGRADED"
                    self.good_cycles = 0
                else:
                    self.good_cycles += 1
                    if self.state in {"DEGRADED", "MINIMAL_RISK"} and self.good_cycles >= self.recovery_cycles:
                        self.state = "NOMINAL"
                return self.state


        def make_fault_trace(n=600, seed=31):
            rng = np.random.default_rng(seed)
            t = np.arange(n) * 0.1
            health = np.clip(0.96 + rng.normal(0, 0.015, n), 0, 1)
            sigma = np.clip(0.25 + rng.normal(0, 0.03, n), 0, None)
            ttc = np.full(n, 8.0)
            disagreement = np.clip(rng.normal(0.10, 0.03, n), 0, 1)
            age = np.clip(rng.normal(0.04, 0.01, n), 0, None)
            odd_ok = np.ones(n, dtype=bool)
            health[180:250] = 0.58
            sigma[180:250] = 1.00
            age[180:250] = 0.22
            ttc[225:245] = 1.8
            disagreement[225:245] = 0.80
            odd_ok[430:480] = False
            return pd.DataFrame({"time_s": t, "sensor_health": health, "localization_sigma": sigma,
                                 "ttc": ttc, "planner_disagreement": disagreement, "age_s": age, "odd_ok": odd_ok})


        trace = make_fault_trace()
        machine = SafetyStateMachine()
        trace["state"] = [machine.update(**row) for row in trace.drop(columns="time_s").to_dict("records")]
        trace["action"] = trace["state"].map(ACTIONS)
        display(trace["state"].value_counts())
        '''),
        cell("code", r'''
        state_code = {state: i for i, state in enumerate(STATES)}
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
        axes[0].plot(trace["time_s"], trace["sensor_health"], label="sensor health")
        axes[0].axhline(0.70, color="black", linestyle="--")
        axes[0].set_ylabel("health")
        axes[1].plot(trace["time_s"], trace["ttc"], label="TTC")
        axes[1].axhline(2.5, color="red", linestyle="--")
        axes[1].plot(trace["time_s"], trace["localization_sigma"], label="localization sigma")
        axes[1].legend()
        axes[1].set_ylabel("risk signal")
        axes[2].step(trace["time_s"], trace["state"].map(state_code), where="post")
        axes[2].set_yticks(list(state_code.values()), list(state_code))
        axes[2].set_xlabel("time / s")
        axes[2].set_ylabel("state")
        plt.show()
        '''),
        cell("code", r'''
        def transition_report(frame):
            transitions = frame["state"].ne(frame["state"].shift()).sum() - 1
            minimal_risk = frame["state"].eq("MINIMAL_RISK")
            nominal_window = frame.iloc[:170]
            return {
                "transition_count": int(transitions),
                "minimal_risk_seconds": float(minimal_risk.sum() * 0.1),
                "nominal_false_fallback_rate": float(nominal_window["state"].ne("NOMINAL").mean()),
                "first_degraded_or_risk_s": float(frame.loc[frame["state"].ne("NOMINAL"), "time_s"].iloc[0]),
            }


        print(transition_report(trace))
        '''),
        cell("markdown", r'''
        ### 交互练习

        1. 调大 `ttc_threshold_s`，观察 safety recall 与 false fallback 的变化；
        2. 给状态机增加 `map_version_valid` 和 `compute_budget_ok`；
        3. 让 `ODD_EXIT` 只能由系统健康恢复后重置，而不是每帧自动回退；
        4. 为每次状态转换记录 `reason`，形成可以审计的 event log；
        5. 设计独立于 learned policy 的 rule monitor，并说明它为什么不能复用同一模型的置信度。
        '''),
        cell("code", r'''
        def inspect_threshold(ttc_threshold_s=2.5, recovery_cycles=8):
            machine = SafetyStateMachine(ttc_threshold_s=ttc_threshold_s, recovery_cycles=int(recovery_cycles))
            local = trace.copy()
            local["state"] = [machine.update(**row) for row in local.drop(columns=["time_s", "state", "action"], errors="ignore").to_dict("records")]
            print(transition_report(local))


        interact(
            inspect_threshold,
            ttc_threshold_s=FloatSlider(value=2.5, min=1.0, max=5.0, step=0.25),
            recovery_cycles=FloatSlider(value=8, min=1, max=20, step=1),
        )
        '''),
        cell("markdown", r'''
        ## 完成标准

        最终报告必须包含状态转换表、触发原因、检测 latency、恢复 latency 和 nominal false fallback。单独报告一个 rule 的 precision/recall 不足以说明系统安全性。
        '''),
    ],
    "18 Safety state machine and degraded mode",
)


write(
    "19_l4_model_development_capstone.ipynb",
    [
        cell("markdown", r'''
        # 19 · L4 Model Development Capstone

        这个 capstone 把前面的接口串成一条最小的 L4 模型开发闭环：

        \[
        \text{scenario} \rightarrow \text{sensor/model signals} \rightarrow \text{prediction/planning} \rightarrow \text{safety gate} \rightarrow \text{metrics}.
        \]

        目标不是声称一个合成实验“实现了 L4”，而是练习岗位真正需要的工程证据：

        - 一组有 ODD 标签的场景；
        - 一个可替换的 learned-policy 接口；
        - 一个独立 safety gate；
        - open-loop 与 closed-loop 风险指标；
        - latency、fallback、舒适性和失败案例报告。

        你最后应把本 notebook 的报告替换成 nuPlan/NAVSIM/CARLA 或公司内部 scenario runner 的输出。
        '''),
        cell("code", r'''
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from dataclasses import dataclass

        plt.rcParams["figure.figsize"] = (10, 5)
        plt.rcParams["axes.grid"] = True

        @dataclass(frozen=True)
        class Episode:
            episode_id: str
            weather: str
            density: str
            sensor_fault: str
            initial_gap_m: float
            obstacle_speed_mps: float
            sensor_health: float
            localization_error_m: float
            odd_ok: bool
            base_latency_ms: float


        episodes = [
            Episode("urban_nominal_01", "clear", "medium", "none", 28, 4.0, 0.96, 0.25, True, 48),
            Episode("urban_lidar_dropout", "clear", "high", "lidar_dropout", 20, 3.0, 0.58, 0.65, True, 55),
            Episode("rain_relocalization", "rain", "medium", "gnss_outage", 24, 4.0, 0.78, 1.10, True, 63),
            Episode("construction_odd_exit", "clear", "high", "map_stale", 18, 2.0, 0.62, 1.55, False, 71),
            Episode("rare_close_cut_in", "night", "high", "camera_glare", 10, 1.0, 0.52, 0.90, True, 67),
        ]
        '''),
        cell("markdown", r'''
        ## Part A — 可替换的 policy 与 safety gate

        `naive` 代表只根据 nominal training distribution 生成速度，`guarded` 则使用健康度、定位误差、ODD 和 gap 触发独立保护。真实项目中，policy 可以是 Transformer、diffusion/flow matching 或 VLA，gate 仍应有独立验证路径。
        '''),
        cell("code", r'''
        def run_episode(ep, policy="guarded", dt=0.1, horizon_s=8.0, seed=4):
            rng = np.random.default_rng(seed)
            steps = int(horizon_s / dt)
            ego_x, ego_speed = 0.0, 7.0
            obstacle_x = ep.initial_gap_m
            rows = []
            for step in range(steps):
                t = step * dt
                obstacle_speed = ep.obstacle_speed_mps if t < 2.8 else max(0.0, ep.obstacle_speed_mps - 3.0)
                gap = obstacle_x - ego_x
                risk = (gap < 12.0 or ep.sensor_health < 0.70 or ep.localization_error_m > 0.80 or not ep.odd_ok)
                if policy == "naive":
                    target_speed = 7.0
                    action = "learned_policy"
                elif risk:
                    target_speed = max(0.0, min(3.0, obstacle_speed - 0.5))
                    action = "minimal_risk" if gap < 7.0 or not ep.odd_ok else "degraded"
                else:
                    target_speed = 7.0
                    action = "learned_policy"
                acceleration = np.clip((target_speed - ego_speed) * 1.6, -4.0, 2.0)
                ego_speed = max(0.0, ego_speed + acceleration * dt)
                ego_x += ego_speed * dt
                obstacle_x += obstacle_speed * dt
                rows.append({"time_s": t, "gap": obstacle_x - ego_x, "ego_speed": ego_speed,
                             "acceleration": acceleration, "action": action,
                             "latency_ms": ep.base_latency_ms + rng.lognormal(-2.2, 0.25)})
            frame = pd.DataFrame(rows)
            acceleration = frame["acceleration"].to_numpy()
            jerk = np.diff(acceleration, prepend=acceleration[0]) / dt
            fallback = frame["action"].ne("learned_policy")
            return frame, {
                "episode_id": ep.episode_id,
                "policy": policy,
                "collision": bool((frame["gap"] < 2.0).any()),
                "min_gap_m": float(frame["gap"].min()),
                "progress_m": float((frame["ego_speed"] * dt).sum()),
                "fallback_seconds": float(fallback.sum() * dt),
                "max_jerk_mps3": float(np.abs(jerk).max()),
                "p95_latency_ms": float(frame["latency_ms"].quantile(0.95)),
                "odd_violation": not ep.odd_ok,
            }


        reports = []
        traces = {}
        for policy in ["naive", "guarded"]:
            for ep in episodes:
                traces[(policy, ep.episode_id)], report = run_episode(ep, policy=policy)
                reports.append(report)
        report_df = pd.DataFrame(reports)
        display(report_df)
        '''),
        cell("code", r'''
        summary = report_df.groupby("policy").agg(
            collision_rate=("collision", "mean"),
            mean_min_gap_m=("min_gap_m", "mean"),
            mean_fallback_s=("fallback_seconds", "mean"),
            p95_latency_ms=("p95_latency_ms", "max"),
            mean_max_jerk=("max_jerk_mps3", "mean"),
        )
        display(summary)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for policy in ["naive", "guarded"]:
            for ep in episodes:
                frame = traces[(policy, ep.episode_id)]
                axes[0].plot(frame["time_s"], frame["gap"], alpha=0.45, label=policy if ep is episodes[0] else None)
        axes[0].axhline(2.0, color="red", linestyle="--", label="collision threshold")
        axes[0].set(xlabel="time / s", ylabel="gap / m", title="Per-episode gap traces")
        axes[0].legend()
        summary["collision_rate"].plot(kind="bar", ax=axes[1], color=["#b91c1c", "#15803d"])
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("collision rate")
        axes[1].set_title("Guarded policy must be evaluated on safety and cost")
        plt.show()
        '''),
        cell("markdown", r'''
        ## Part B — 作品集交付物

        这个 capstone 不应以“guarded policy 的分数更高”结束。请输出：

        1. ODD 说明和场景矩阵；
        2. policy/safety gate 的输入输出契约；
        3. naive 与 guarded 的 collision、fallback、progress、jerk、p95 latency 对比；
        4. 至少一个 failure replay，指出是感知、定位、规划、系统健康还是评测设计导致；
        5. 一个下一步真实数据接入计划，包括数据许可、坐标/时间同步、scenario ID 和 regression gate。

        ### 练习

        - 增加一个 `open_loop` 指标，并解释它为什么不能替代 closed-loop collision rate；
        - 把 scenario coverage 按 weather、density、sensor_fault 分组；
        - 对 latency budget 增加 p99 和 watchdog violation；
        - 把 safety gate 拆成可以独立单元测试的函数，并构造至少三个 adversarial case。
        '''),
        cell("code", r'''
        coverage = pd.DataFrame([
            {"weather": ep.weather, "density": ep.density, "sensor_fault": ep.sensor_fault}
            for ep in episodes
        ]).value_counts().rename("episodes").reset_index()
        display(coverage)
        print("portfolio checklist:")
        print("- scenario matrix:", len(episodes), "episodes")
        print("- policies compared: ", report_df["policy"].unique().tolist())
        print("- safety metric: collision rate + minimum gap + fallback duration")
        print("- runtime metric: p95 latency; add p99 and watchdog in your submission")
        '''),
        cell("markdown", r'''
        ## 边界声明

        该 notebook 是一个接口和验证骨架，不是自动驾驶系统认证证据。真正的 L4 项目还需要真实传感器、车辆动力学、地图、交通参与者、硬件 runtime、系统冗余和组织级 safety case。
        '''),
    ],
    "19 L4 model development capstone",
)
