---
name: vp-engineering-efficiency
description: Use when checking L0 engineering efficiency in the Validation Pyramid - backend verification, MFU/TCA, memory analysis, I/O speed, and bandwidth checks
---

# L0: Engineering Efficiency

## Overview

The cheapest, fastest layer. Run this FIRST before any training. Catches configuration errors, wrong backends, low MFU/TCA, and infrastructure issues in minutes.

**If L0 fails, don't waste time on L1-L3.** Fix infrastructure first.

**TDD reminder:** Follow the RED → GREEN → REFACTOR rhythm defined in `validation-pyramid/SKILL.md`.

## Three-Phase Architecture (5 min total)

### Phase 1: Static Analysis (~10 seconds)

No training needed. Runs instantly.

1. **Precise FLOPs via FlopCounterMode** — `torch.utils.flop_counter.FlopCounterMode` counts at aten operator level. Automatically handles GQA, MoE, SwiGLU, any standard PyTorch op. One fwd+bwd pass.
2. **GPU hardware info** — model, peak BF16 TFLOPS, memory capacity, DCGM availability
3. **Backend pre-check** — parameter dtype, FlashAttention availability
4. **Memory snapshot** — parameter count, theoretical memory footprint

### Phase 2: Steady-State Measurement (~3 minutes)

All metrics from one continuous training run. No separate profiling passes.

1. **Warmup detection** — runs training steps until step time variance converges (CV of last 5 steps < 5%). Not a fixed N steps — adaptive.
2. **DCGM background sampling** — `dcgmi dmon -e 1004 -d 1` runs as subprocess. Field 1004 = `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`. **Zero overhead** on training. Matches monitoring dashboard TCA exactly.
3. **Steady-state timing** — CUDA Events per step: `step_time_ms`, `samples/sec`, `tokens/sec`, memory stats.
4. **Backend verification** — `torch.profiler` captures 1 step of real kernel names. Detects FlashAttention, identifies top CUDA kernels.

### Phase 3: Report (~seconds)

1. **MFU** = `FlopCounterMode_flops / (median_step_time × peak_tflops)`
2. **TCA** = `mean(DCGM_tensor_active%)` over steady-state window
3. **Sample speed** = `batch_size / median_step_time`
4. **TCA vs MFU gap analysis** — qualitative contributors (clock throttling, tensor overhead, memory-bound)
5. **Memory report** — peak / allocated / reserved / fragmentation
6. **Backend report** — kernel names, FA status, dtype

## Using Toolkit

Primary entry point:

```python
from toolkit.profiling.l0_runner import run_l0, format_summary_table

report = run_l0(
    model=model,
    train_step_fn=train_one_step,   # user's single-step training function
    mock_input=mock_input,          # for FlopCounterMode
    batch_size=2048,
    seq_len=1024,                   # optional, for tokens/sec
    steady_state_minutes=3,         # steady-state measurement duration
)

print(format_summary_table(report))
```

`train_step_fn` signature: `def train_one_step(model, step_idx: int) -> None`

Individual tools also available:
- `toolkit/profiling/mfu_calculator.py` — `count_flops()`, `calculate_mfu()`, `measure_step_times()`, `detect_warmup_end()`
- `toolkit/profiling/dcgm_profiler.py` — DCGM lifecycle + TCA parsing
- `toolkit/profiling/gap_analyzer.py` — TCA vs MFU gap analysis
- `toolkit/profiling/layer_profiler.py` — per-layer timing decomposition (deep-dive)
- `toolkit/profiling/memory_profiler.py` — memory analysis

## Key Metrics

| Metric | Source | What It Measures |
|--------|--------|------------------|
| **MFU** | FlopCounterMode + CUDA Events | Useful model FLOPs / peak hardware FLOPs |
| **TCA** | DCGM field 1004 | % time tensor cores are active (hardware counter) |
| **Sample Speed** | CUDA Events | samples/sec or tokens/sec throughput |

## Report Output (no pass/fail)

L0 **reports metrics only**. It does not make pass/fail judgments. The user or agent interprets results based on context (model type, GPU, expected ranges).

## DCGM Requirements

TCA measurement requires DCGM installed on GPU nodes. If DCGM is unavailable:
- TCA is reported as `None` with a warning
- MFU and sample speed are still measured normally
- **Do not fake TCA** — real data or nothing

## Failure Decomposition

If MFU or TCA seem low:
1. Check gap analysis contributors in the report
2. Use `layer_profiler` to identify which layers are slow
3. Check backend report — are expected kernels being used?
4. For multi-node: check distributed training bandwidth (see `distributed-training.md`)
