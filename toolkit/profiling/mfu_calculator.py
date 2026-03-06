"""MFU (Model FLOPs Utilization) Calculator.

Computes theoretical FLOPS from model structure, measures actual throughput,
and reports MFU and TCA (Training Compute Achievable) metrics.

Usage:
    from toolkit.profiling.mfu_calculator import calculate_mfu
    result = calculate_mfu(model, input_shape=(8, 512), step_time_ms=150.0)
"""

import torch
import torch.nn as nn
from typing import Optional


# Peak TFLOPS (bf16/fp16 tensor core) for known GPU architectures
GPU_PEAK_TFLOPS = {
    "A100": 312.0,
    "A100-SXM": 312.0,
    "A100-PCIE": 312.0,
    "A800": 312.0,
    "H100": 989.0,
    "H100-SXM": 989.0,
    "H100-PCIE": 756.0,
    "H800": 989.0,
    "B200": 2250.0,
    "B100": 1750.0,
    "V100": 125.0,
    "L40": 362.0,
    "L40S": 362.0,
    "4090": 330.0,
    "3090": 142.0,
}


def _detect_gpu_peak_tflops() -> float:
    """Auto-detect GPU and return peak bf16 TFLOPS."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    name = torch.cuda.get_device_name(0).upper()
    for key, tflops in GPU_PEAK_TFLOPS.items():
        if key.upper() in name:
            return tflops
    raise ValueError(
        f"Unknown GPU: {torch.cuda.get_device_name(0)}. "
        f"Pass gpu_peak_tflops manually. Known GPUs: {list(GPU_PEAK_TFLOPS.keys())}"
    )


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops_per_step(
    model: nn.Module,
    input_shape: tuple,
    model_type: str = "dense",
    num_active_experts: int = 2,
    num_total_experts: int = 8,
) -> float:
    """Estimate FLOPs per training step (forward + backward).

    For a dense model: ~6 * N * tokens_per_step (2x forward, 4x backward including activation recompute).
    For MoE: scale by active/total expert ratio for expert layers.

    Args:
        model: PyTorch model
        input_shape: (batch_size, seq_len) or (batch_size,) for non-sequence models
        model_type: "dense" or "moe"
        num_active_experts: for MoE, how many experts are active per token
        num_total_experts: for MoE, total number of experts

    Returns:
        Estimated FLOPs for one training step
    """
    num_params = count_parameters(model)

    # tokens per step = product of input shape dimensions
    tokens_per_step = 1
    for dim in input_shape:
        tokens_per_step *= dim

    if model_type == "dense":
        # 6N * T: 2x for forward matmuls, 4x for backward (grad_input + grad_weight)
        flops = 6.0 * num_params * tokens_per_step
    elif model_type == "moe":
        # MoE: non-expert params get full compute, expert params get scaled
        expert_ratio = num_active_experts / num_total_experts
        # Approximate: assume ~50% of params are in experts
        # Better: user can provide exact split, but this is a reasonable default
        flops = 6.0 * num_params * tokens_per_step * (0.5 + 0.5 * expert_ratio)
    else:
        # Fallback: treat as dense
        flops = 6.0 * num_params * tokens_per_step

    return flops


def calculate_mfu(
    model: nn.Module,
    input_shape: tuple,
    step_time_ms: float,
    model_type: str = "dense",
    num_active_experts: int = 2,
    num_total_experts: int = 8,
    gpu_peak_tflops: Optional[float] = None,
    target_mfu: float = 0.5,
    num_gpus: int = 1,
) -> dict:
    """Calculate MFU and TCA for a model.

    Args:
        model: PyTorch model
        input_shape: (batch_size, seq_len) or (batch_size,) for non-sequence models
        step_time_ms: measured wall-clock time per training step in milliseconds
        model_type: "dense" or "moe"
        num_active_experts: for MoE models
        num_total_experts: for MoE models
        gpu_peak_tflops: override GPU peak TFLOPS (auto-detected if None)
        target_mfu: target MFU threshold (default 0.5 = 50%)
        num_gpus: number of GPUs used (for multi-GPU training)

    Returns:
        dict with keys: mfu, tca, theoretical_tflops, actual_tflops,
        flops_per_step, step_time_ms, num_params, status, gpu_name, gpu_peak_tflops
    """
    if step_time_ms <= 0:
        raise ValueError("step_time_ms must be positive")

    if gpu_peak_tflops is None:
        gpu_peak_tflops = _detect_gpu_peak_tflops()

    total_peak_tflops = gpu_peak_tflops * num_gpus

    num_params = count_parameters(model)
    flops_per_step = estimate_flops_per_step(
        model, input_shape, model_type, num_active_experts, num_total_experts
    )

    step_time_s = step_time_ms / 1000.0
    actual_tflops = flops_per_step / step_time_s / 1e12

    mfu = actual_tflops / total_peak_tflops
    # TCA: what fraction of peak you'd get if you include all overhead
    # For simplicity, TCA = MFU here; in practice TCA accounts for memory-bound ops
    tca = mfu

    if mfu >= target_mfu:
        status = "on_target"
    elif mfu >= target_mfu * 0.7:
        status = "below_target"
    else:
        status = "needs_investigation"

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    return {
        "mfu": mfu,
        "tca": tca,
        "theoretical_tflops": total_peak_tflops,
        "actual_tflops": actual_tflops,
        "flops_per_step": flops_per_step,
        "step_time_ms": step_time_ms,
        "num_params": num_params,
        "status": status,
        "gpu_name": gpu_name,
        "gpu_peak_tflops": gpu_peak_tflops,
    }
