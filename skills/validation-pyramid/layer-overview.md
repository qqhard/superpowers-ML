# Validation Pyramid Layer Overview

| Layer | Skill | Cost | Time | What it catches |
|-------|-------|------|------|-----------------|
| **L0** | vp-engineering-efficiency | Very low | Minutes | Wrong backend, bad I/O, low MFU/TCA, memory waste |
| **L1** | vp-process-metrics | Low | Seconds-minutes (first N steps) | NaN/Inf, gradient issues, activation collapse, loss spikes, parameter drift |
| **L2** | vp-overfitting-test | Medium | ~10 minutes | Model can't fit small data = implementation bug or architecture issue |
| **L3** | vp-e2e-pipeline | Medium-high | 10-30 minutes | Pipeline integration issues, data-to-evaluation flow broken |

## Pass/Fail Criteria Summary

### L0: Engineering Efficiency
- Backend: Expected backends enabled (FA, MoE kernel, etc.)
- All metrics measured during steady-state (skip warmup, confirm data loading stable)
- Sample speed: samples/sec or tokens/sec meets expectation
- TCA: >= user-defined threshold
- MFU: >= user-defined threshold (typically 0.3-0.6 depending on model)
- Memory: No obvious waste (reserved >> allocated = fragmentation)
- I/O: Data loading not bottlenecking steady-state training
- Bandwidth: NCCL/HBM/PCIE meeting expected throughput

### L1: Process Metrics
- No NaN/Inf in any tensor
- Gradient norms in reasonable range (no vanishing < 1e-7, no exploding > 1e3)
- Loss decreasing in first N steps (for supervised tasks)
- No loss spikes > 10x moving average
- Parameter drift from init in expected range
- Architecture-specific checks pass (attention entropy, MoE balance, etc.)

### L2: Overfitting Test
- Training loss steadily decreasing on 100-1000 samples (前半 avg > 后半 avg, 下降比例 >= 60%)
- Task-specific metric showing clear improvement trend
- Completed within expected time

### L3: End-to-End Pipeline
- Full flow completes without error on tiny data
- Inference produces non-degenerate output
- Evaluation metrics computable (not NaN)
