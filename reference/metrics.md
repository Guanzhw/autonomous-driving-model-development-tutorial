# Metrics and boundaries

| Layer | Typical metrics | What it does not prove |
|---|---|---|
| Occupancy / detection | IoU、precision/recall、mAP、range/slice breakdown | 不证明 tracking、planning 或 closed-loop 安全。 |
| Tracking | IDF1、MOTA/HOTA、position/velocity RMSE、track age/dropout recovery | 不证明 future behavior prediction 正确。 |
| Prediction | ADE、FDE、minADE/minFDE、miss rate、NLL/calibration | 不证明 planner 会选对 mode 或闭环不碰撞。 |
| Planning / control | path/lateral error、constraint violation、jerk、progress、intervention | open-loop path error 不等于 closed-loop risk。 |
| Closed-loop | collision rate、minimum gap、route progress、off-road、comfort、fallback duration | 合成 runner/CPU 数字不等于道路安全证据。 |
| Safety | fallback recall、false alarm、TTC coverage、ODD violations、state transition latency | 阈值实验不等于 ISO 26262/SOTIF safety case。 |
| Runtime | batch=1 throughput、p50/p95/p99 latency、memory、watchdog violation、quantization error | 一台 CPU 上的 toy latency 不等于车端 runtime。 |

课程中的每个报告都应写清楚：数据版本、split、坐标和时间语义、seed、硬件、warm-up、batch、阈值、slice breakdown 和已知失效边界。
