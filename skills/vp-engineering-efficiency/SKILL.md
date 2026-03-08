---
name: vp-engineering-efficiency
description: Use when checking L0 engineering efficiency in the Validation Pyramid - backend verification, GPU utilization, memory analysis, I/O speed, and bandwidth checks
---

# L0: Engineering Efficiency

## Overview

The cheapest, fastest layer. Run this FIRST before any training. Catches configuration errors, wrong backends, low GPU utilization, and infrastructure issues in seconds.

**If L0 fails, don't waste time on L1-L3.** Fix infrastructure first.

## TDD Flow

Follow RED → GREEN → REFACTOR for each check:

1. **RED:** Write the assertion (e.g., `assert mfu >= target`). Run it. Confirm it fails.
2. **GREEN:** Optimize configuration/code until the assertion passes.
3. **REFACTOR:** Clean up.

If an assertion passes immediately, verify the threshold is meaningful — not too lenient.

## What to Check

All checks run on a few forward/backward steps with mock data. No real training needed.

### 1. Backend Verification
See `backend-checks.md` for detailed checks.

**Quick version:**
- Is FlashAttention enabled? (if Transformer)
- Is the correct MoE kernel loaded? (if MoE)
- Are CUDA kernels the expected ones? (not falling back to slow paths)
- Is the correct precision being used? (fp16/bf16/fp32 as intended)

### 2. GPU Utilization
See `gpu-utilization.md` for detailed checks.

**Quick version:**
- MFU (Model FLOPs Utilization): >= user threshold
- TCA (Tensor Core Activity): >= user threshold
- Memory: allocated vs reserved, fragmentation check
- If below target: use `toolkit/profiling/layer_profiler.py` to decompose per-layer

### 3. Data I/O
- Run training for a few steps, check if GPU is idle waiting for data
- Sample consumption speed: tokens/sec or samples/sec meets expectation
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

All enabled checks pass. Any failure -> trigger **spml:ml-diagnostics** with the specific failure data.

## Failure Decomposition

If MFU/TCA below target:
1. Use layer_profiler to identify which layers are slow
2. Check if slow layers are using expected kernels
3. Check if data loading is the bottleneck (GPU idle time)
4. For multi-node: check communication overhead
