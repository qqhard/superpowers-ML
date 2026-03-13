# GPU Utilization: MFU, TCA, Sample Speed

## Three Core Metrics

### MFU (Model FLOPs Utilization)

**What:** Ratio of actual model FLOPs to theoretical peak hardware FLOPs.

**How it's measured:**
1. `FlopCounterMode` counts exact FLOPs per fwd+bwd pass (aten operator level, not 6N estimation)
2. CUDA Events measure median step time over steady-state
3. `MFU = flops_per_step / (step_time_s × peak_tflops × 1e12)`

```python
from toolkit.profiling.mfu_calculator import count_flops, calculate_mfu

# Step 1: Count FLOPs (once, static)
flops = count_flops(model, mock_input, include_backward=True)
print(f"FLOPs/step: {flops['total_flops'] / 1e12:.2f} T")

# Step 2: Calculate MFU (with measured step time)
result = calculate_mfu(
    flops_per_step=flops['total_flops'],
    step_time_ms=measured_median_step_time,
    gpu_peak_tflops=989.0,  # H100, or auto-detected
)
print(f"MFU: {result['mfu']:.2%}")
```

### TCA (Tensor Core Activity)

**What:** Percentage of time tensor cores are actively computing. Hardware counter, not estimation.

**How it's measured:**
- DCGM field 1004 (`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`)
- Sampled every 1 second by `dcgmi dmon` subprocess
- Zero overhead on training — runs independently
- Matches monitoring dashboard exactly

```python
from toolkit.profiling.dcgm_profiler import (
    check_dcgm_available, start_dcgm_sampling,
    stop_dcgm_sampling, parse_dcgm_output, compute_tca_stats,
)

if check_dcgm_available():
    proc = start_dcgm_sampling(interval_ms=1000)
    # ... run training ...
    output = stop_dcgm_sampling(proc)
    tca_values = parse_dcgm_output(output)
    stats = compute_tca_stats(tca_values)
    print(f"TCA: {stats['mean']:.2f}% (median={stats['median']:.2f}%)")
```

### Sample Speed

**What:** Training throughput in samples/sec or tokens/sec.

**How it's measured:**
- `sample_speed = batch_size / median_step_time_s`
- `tokens_per_sec = batch_size * seq_len / median_step_time_s` (if applicable)

## TCA vs MFU Gap

TCA > MFU is normal. The gap reveals inefficiencies:

| Gap Cause | Explanation |
|-----------|-------------|
| **Clock throttling** | GPU not at max frequency (power/thermal) |
| **Tensor overhead** | Tile padding, sub-optimal GEMM shapes |
| **Memory-bound ops** | Tensor cores active but starved for data |

```python
from toolkit.profiling.gap_analyzer import analyze_gap

gap = analyze_gap(
    tca_percent=23.7,
    mfu_percent=14.6,
    actual_clock_mhz=1761,  # optional
    max_clock_mhz=2550,      # optional
)
print(f"Gap: {gap['gap_pp']:.1f} pp")
for c in gap['contributors']:
    print(f"  - {c}")
```

## Steady-State Measurement Protocol

**All measurements MUST be taken during steady-state.** Warmup measurements are not representative.

1. Run training normally
2. Warmup auto-detected by step time variance convergence (CV < 5% over 5 steps)
3. Measure over 3 minutes of steady-state (configurable)
4. DCGM samples TCA simultaneously, zero overhead
5. Report uses median step time (robust to outliers)

The `run_l0()` function handles all of this automatically.

## Per-Layer Profiling (on investigation)

When MFU is unexpectedly low, decompose per-layer:

```python
from toolkit.profiling.layer_profiler import profile_layers

result = profile_layers(model, mock_input)
for layer in result['layers']:
    print(f"{layer['name']}: {layer['time_ms']:.2f}ms ({layer['percentage']:.1f}%)")
```

## Memory Analysis

```python
from toolkit.profiling.memory_profiler import analyze_memory

result = analyze_memory(model, mock_input)
print(f"Peak: {result['peak_gb']:.2f} GB, Fragmentation: {result['fragmentation']:.1%}")
```
