---
name: vp-engineering-efficiency
description: Use when checking L0 engineering efficiency in the Validation Pyramid - backend verification, MFU/TCA, memory analysis, I/O speed, and bandwidth checks
---

# L0: Engineering Efficiency

## Overview

The cheapest, fastest layer. Run this FIRST before any training. Catches configuration errors, wrong backends, low MFU/TCA, and infrastructure issues in minutes.

**If L0 fails, don't waste time on L1-L3.** Fix infrastructure first.

**TDD reminder:** Follow the RED → GREEN → REFACTOR rhythm defined in `validation-pyramid/SKILL.md`.

## Steady-State Measurement

**All efficiency metrics MUST be measured during steady-state training, not during warmup.**

Steps:
1. Run training for enough steps that data loading pipeline is warm (prefetch buffers full, no first-load overhead)
2. Skip the first 5-10 steps (warmup: CUDA lazy init, JIT compilation, data prefetch filling)
3. Measure over the next N steps (N >= 10) to get stable averages
4. Confirm data loading is not the bottleneck during measurement (GPU not idle waiting for data)

The three key metrics to report during steady-state:
- **Sample speed**: samples/sec or tokens/sec
- **TCA** (Tensor Core Activity): percentage of time tensor cores are active
- **MFU** (Model FLOPs Utilization): ratio of actual model FLOPs to theoretical peak

## What to Check

### 1. Backend Verification
See `backend-checks.md` for detailed checks.

**Quick version:**
- Is FlashAttention enabled? (if Transformer)
- Is the correct MoE kernel loaded? (if MoE)
- Are CUDA kernels the expected ones? (not falling back to slow paths)
- Is the correct precision being used? (fp16/bf16/fp32 as intended)

### 2. Steady-State Efficiency (MFU / TCA / Sample Speed)
See `gpu-utilization.md` for detailed checks.

**Quick version (all measured during steady-state, see above):**
- Sample speed: samples/sec or tokens/sec meets expectation
- MFU (Model FLOPs Utilization): >= user threshold
- TCA (Tensor Core Activity): >= user threshold
- Memory: allocated vs reserved, fragmentation check
- If below target: use `toolkit/profiling/layer_profiler.py` to decompose per-layer

### 3. Data I/O
- Confirm data loading is not bottlenecking steady-state training (GPU not idle waiting for data)
- Checkpoint load time: acceptable for the model size

### 4. Infrastructure
- W&B logging connected and receiving data
- Checkpoint save/load works correctly
- mmap data loading working (if applicable)

### 5. Distributed Training (if multi-GPU/multi-node)
See `distributed-training.md` for detailed checks.

**Quick version:**
- NCCL bandwidth meets expected throughput
- HBM bandwidth not bottlenecked
- PCIE bandwidth adequate
- Communication/computation overlap working

## Using Toolkit

When available, use these toolkit tools:
- `toolkit/profiling/mfu_calculator.py` — Calculate MFU/TCA
- `toolkit/profiling/layer_profiler.py` — Per-layer timing decomposition
- `toolkit/profiling/memory_profiler.py` — Memory analysis

If toolkit is not yet available, write equivalent checks following the guidance in this skill.

## Pass Criteria

All enabled checks pass. Any failure -> trigger **spml:diagnostics** with the specific failure data.

## Failure Decomposition

If MFU/TCA below target:
1. Use layer_profiler to identify which layers are slow
2. Check if slow layers are using expected kernels
3. Check if data loading is the bottleneck (GPU idle time)
4. For multi-node: check communication overhead
