# Validation Pyramid Layer Overview

| Layer | Skill | Cost | Time | What it catches |
|-------|-------|------|------|-----------------|
| **L0** | vp-engineering-efficiency | Very low | Seconds | Wrong backend, bad I/O, low GPU utilization, memory waste |
| **L1** | vp-process-metrics | Low | Seconds-minutes (first N steps) | NaN/Inf, gradient issues, activation collapse, loss spikes, parameter drift |
| **L2** | vp-overfitting-test | Medium | ~10 minutes | Model can't fit small data = implementation bug or architecture issue |
| **L3** | vp-e2e-pipeline | Medium-high | 10-30 minutes | Pipeline integration issues, data-to-evaluation flow broken |

## Pass/Fail Criteria Summary

### L0: Engineering Efficiency
- Backend: Expected backends enabled (FA, MoE kernel, etc.)
- MFU: >= user-defined threshold (typically 0.3-0.6 depending on model)
- TCA: >= user-defined threshold
- Memory: No obvious waste (reserved >> allocated = fragmentation)
- I/O: Data loading not bottlenecking GPU (GPU utilization not dropping during data fetch)
- Bandwidth: NCCL/HBM/PCIE meeting expected throughput

### L1: Process Metrics
- No NaN/Inf in any tensor
- Gradient norms in reasonable range (no vanishing < 1e-7, no exploding > 1e3)
- Loss decreasing in first N steps (for supervised tasks)
- No loss spikes > 10x moving average
- Parameter drift from init in expected range
- Architecture-specific checks pass (attention entropy, MoE balance, etc.)

### L2: Overfitting Test
- Training loss monotonically decreasing to near 0 on 100-1000 samples
- Task-specific metric near perfect (NDCG near 1.0, perplexity near 1.0)
- Completed within expected time

### L3: End-to-End Pipeline
- Full flow completes without error on tiny data
- Inference produces non-degenerate output
- Evaluation metrics computable (not NaN)
