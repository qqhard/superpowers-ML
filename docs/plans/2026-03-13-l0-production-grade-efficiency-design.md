# L0 Production-Grade Engineering Efficiency

## Summary

Upgrade L0 from toy-level checks (6N FLOPs estimation, TCA=MFU alias) to production-grade engineering efficiency validation that produces real MFU, real TCA, and real sample speed from a single 3-minute steady-state training run.

## Motivation

Current L0 has three critical gaps compared to production monitoring (reference: onetrans-mfu-analysis):

1. **FLOPs are guessed**: `6N` formula ignores GQA, MoE routing, SwiGLU, pyramid truncation, etc.
2. **TCA is fake**: `tca = mfu` — no actual tensor core activity measurement
3. **No gap analysis**: Cannot explain why MFU differs from TCA (clock throttling, tensor overhead, padding waste)

The goal: run L0 once for 5 minutes, get production-grade metrics identical to what monitoring dashboards report.

## Design

### Three-Phase Architecture (5 min total budget)

#### Phase 1: Static Analysis (~10 seconds, no training)

1. **Precise FLOPs via FlopCounterMode**
   - Use `torch.utils.flop_counter.FlopCounterMode` (PyTorch built-in, `__torch_dispatch__` based)
   - Run 1 forward + backward pass under the context manager
   - Automatically handles GQA, MoE, SwiGLU, custom ops — counts at aten operator level
   - Output: per-module FLOPs breakdown + total FLOPs per step
   - If user model has custom ops, support `custom_mapping` to register FLOPs formulas

2. **GPU Hardware Info**
   - GPU model, peak BF16 TFLOPS, memory capacity
   - DCGM availability check (`dcgmi discovery -l`)

3. **Backend Pre-check**
   - Parameter dtype verification (bf16/fp16/fp32 as expected)
   - FlashAttention availability

4. **Memory Snapshot**
   - Parameter count, theoretical memory footprint (params + grads + optimizer states)

#### Phase 2: Steady-State Training Measurement (~3 minutes, single run)

All metrics collected simultaneously from one continuous training run. No separate profiling passes.

1. **Warmup (auto-detect)**
   - Run training steps until step time variance converges
   - Criteria: coefficient of variation of last 5 steps < 5%
   - Typical: 10-30 steps depending on model/data pipeline

2. **Start DCGM background sampling**
   - `dcgmi dmon -e 1004 -d 100` (field 1004 = DCGM_FI_PROF_PIPE_TENSOR_ACTIVE, 100ms interval)
   - Runs as subprocess, zero overhead on training
   - Produces ~1800 samples over 3 minutes

3. **Steady-state measurement (N steps over ~3 minutes)**
   - Per-step recording via CUDA Events:
     - `step_time_ms` (start.record → end.record → synchronize → elapsed_time)
     - `samples_per_sec` = batch_size / step_time
     - `tokens_per_sec` = batch_size * seq_len / step_time (if applicable)
   - Memory per step: `torch.cuda.memory_allocated()`, `memory_reserved()`, `max_memory_allocated()`

4. **Stop DCGM, parse output**
   - Parse DCGM CSV output, extract TCA% per sample
   - Align with steady-state window (discard warmup-period samples)
   - Compute: mean, median, std, min, max of TCA over steady-state

5. **Backend verification (1 step)**
   - `torch.profiler.profile` for 1 step to capture actual CUDA kernel names
   - Check: flash_attn kernels present (if Transformer), correct GEMM kernels, no unexpected fallbacks

#### Phase 3: Report Generation (seconds)

1. **Core Metrics**
   - `MFU = flops_per_step / (median_step_time_s * peak_tflops * 1e12)`
   - `TCA = mean(DCGM_tensor_active%) over steady-state window`
   - `sample_speed = batch_size / median_step_time_s`
   - `tokens_per_sec = batch_size * seq_len / median_step_time_s` (if applicable)

2. **TCA vs MFU Gap Analysis**
   - `gap = TCA - MFU` (in percentage points)
   - Possible contributors (reported qualitatively):
     - Clock throttling: if GPU not running at max clock
     - Memory-bound ops: tensor cores active but waiting for data
     - Padding waste: tile alignment overhead in attention/GEMM
   - Note: full quantitative decomposition (clock_factor, tensor_overhead, pad_ratio) requires NCU per-kernel data — available as optional deep-dive

3. **Memory Report**
   - Peak / allocated / reserved / fragmentation
   - Parameter memory / gradient memory / activation memory

4. **Backend Verification**
   - Kernel names found, FA status, precision confirmed

5. **Output Format**
   - Structured dict (machine-readable)
   - Summary table (human-readable)
   - All raw data preserved for downstream analysis

### Toolkit Changes

| File | Change | Description |
|------|--------|-------------|
| `toolkit/profiling/mfu_calculator.py` | **Rewrite** | Replace 6N with FlopCounterMode, add CUDA Events steady-state timing |
| `toolkit/profiling/dcgm_profiler.py` | **New** | DCGM process management, output parsing, TCA extraction |
| `toolkit/profiling/gap_analyzer.py` | **New** | TCA vs MFU gap analysis |
| `toolkit/profiling/l0_runner.py` | **New** | Three-phase orchestration, report generation |
| `toolkit/profiling/ncu_profiler.py` | **New (optional)** | NCU deep-dive for per-kernel TCA and kernel classification |
| `toolkit/profiling/memory_profiler.py` | Minor | Adapt to steady-state measurement mode |
| `toolkit/profiling/layer_profiler.py` | No change | Remains available as deep-dive tool |
| `skills/vp-engineering-efficiency/` | Update | Point skill docs to new toolkit |
| `tests/toolkit/profiling/` | New tests | For all new/changed modules |

### User Interface

```python
from toolkit.profiling.l0_runner import run_l0

report = run_l0(
    model=model,
    train_step_fn=train_one_step,   # user's single-step training function
    mock_input=mock_input,          # for FlopCounterMode static analysis
    batch_size=2048,
    seq_len=1024,                   # optional, for tokens/sec
    steady_state_minutes=3,         # steady-state measurement duration
)

print(report.summary_table())
```

`train_step_fn` signature:
```python
def train_one_step(model, step_idx: int) -> None:
    """Execute one training step. Must handle data loading, forward, backward, optimizer step."""
    ...
```

### DCGM Integration Details

**Start:**
```python
proc = subprocess.Popen(
    ["dcgmi", "dmon", "-e", "1004", "-d", "100"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
)
```

**Parse output:**
```
# Entity  SMACT  TENACT
     0     85.2   23.7
     0     84.1   24.1
     ...
```

Column `TENACT` = `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` = TCA%.

**Fallback:** If DCGM not available, report TCA as "unavailable" with instructions to install DCGM. Do not fake it.

### NCU Deep-Dive (Optional, Not Part of Standard L0)

When per-kernel analysis is needed (e.g., to understand why TCA is low):

```python
from toolkit.profiling.ncu_profiler import run_ncu_analysis

ncu_report = run_ncu_analysis(
    training_script="train.py",
    training_args=["--config", "..."],
    launch_skip=10,
    launch_count=5,
)

# Per-kernel TCA, instruction type classification (WGMMA/HMMA/CUDA Core)
# Quantitative gap decomposition (clock_factor, tensor_overhead, pad_ratio)
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| FLOPs calculation | `FlopCounterMode` | PyTorch built-in, operator-level precision, zero user effort |
| TCA measurement | DCGM (`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`) | Zero overhead, matches monitoring dashboards, simultaneous with training |
| MFU timing | CUDA Events during normal training | Not polluted by profiling overhead |
| Single vs two-phase | Single phase | DCGM enables simultaneous MFU + TCA collection |
| Pass/fail judgment | Report only | User/agent interprets results |
| Warmup detection | Step time CV convergence | Adaptive, not fixed N steps |
| DCGM unavailable | Report "unavailable", no fallback | Real data or nothing, never fake |
| NCU | Optional deep-dive tool | For per-kernel analysis, not standard L0 |

### Validation Scope

- **L0 only**: This design covers L0 Engineering Efficiency
- **No L1-L3 changes**: Process metrics, overfitting test, E2E pipeline unchanged
- **Skill docs updated**: vp-engineering-efficiency skill points to new toolkit

### Dependencies

- PyTorch >= 2.1 (for `FlopCounterMode`)
- DCGM installed on GPU nodes (for TCA)
- CUDA toolkit (for CUDA Events)
