"""MFU (Model FLOPs Utilization) Calculator.

Production-grade FLOPs counting via torch.utils.flop_counter.FlopCounterMode
and steady-state step timing via CUDA Events.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
import statistics

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


def count_flops(
    model: nn.Module,
    mock_input: torch.Tensor,
    include_backward: bool = True,
    loss_fn: Optional[Callable] = None,
) -> dict:
    """Count FLOPs using torch.utils.flop_counter.FlopCounterMode.

    Args:
        model: PyTorch model
        mock_input: example input tensor
        include_backward: if True, count fwd+bwd FLOPs
        loss_fn: how to compute loss for backward (default: output.sum())

    Returns:
        dict with: total_flops (int)
    """
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()  # eval for consistent FLOPs counting

    flop_counter = FlopCounterMode(model, display=False)
    with flop_counter:
        output = model(mock_input)
        if include_backward:
            if loss_fn is not None:
                loss = loss_fn(output)
            else:
                if isinstance(output, tuple):
                    loss = output[0].sum()
                else:
                    loss = output.sum()
            loss.backward()

    total_flops = flop_counter.get_total_flops()

    if was_training:
        model.train()

    # Zero out gradients from FLOPs counting pass
    model.zero_grad()

    return {
        "total_flops": total_flops,
    }


def detect_warmup_end(
    step_times: list, window: int = 5, cv_threshold: float = 0.05
) -> int:
    """Find the index where step times stabilize.

    Slides a window and checks if coefficient of variation < threshold.

    Args:
        step_times: list of step times (ms)
        window: number of recent steps to check
        cv_threshold: coefficient of variation threshold (default 5%)

    Returns:
        Index of first step in steady-state, or -1 if not converged.
    """
    if len(step_times) < window:
        return -1

    for i in range(window, len(step_times) + 1):
        recent = step_times[i - window : i]
        mean = statistics.mean(recent)
        if mean <= 0:
            continue
        stdev = statistics.stdev(recent) if len(recent) > 1 else 0.0
        cv = stdev / mean
        if cv < cv_threshold:
            return i - window

    return -1


def measure_step_times(
    model: nn.Module,
    train_step_fn: Callable,
    num_steps: int,
) -> list:
    """Measure per-step wall time using CUDA Events.

    Args:
        model: PyTorch model (on CUDA)
        train_step_fn: function(model, step_idx) that runs one training step
        num_steps: number of steps to measure

    Returns:
        list of step_time_ms for each step
    """
    step_times = []
    for step_idx in range(num_steps):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        train_step_fn(model, step_idx)
        end_evt.record()
        torch.cuda.synchronize()

        step_times.append(start_evt.elapsed_time(end_evt))

    return step_times


def calculate_mfu(
    flops_per_step: int,
    step_time_ms: float,
    gpu_peak_tflops: Optional[float] = None,
    num_gpus: int = 1,
) -> dict:
    """Calculate MFU from pre-computed FLOPs and measured step time.

    Args:
        flops_per_step: total FLOPs per training step (from count_flops)
        step_time_ms: median step time in milliseconds
        gpu_peak_tflops: peak GPU TFLOPS (auto-detected if None)
        num_gpus: number of GPUs

    Returns:
        dict with: mfu, actual_tflops, theoretical_tflops, step_time_ms
    """
    if step_time_ms <= 0:
        raise ValueError("step_time_ms must be positive")

    if gpu_peak_tflops is None:
        gpu_peak_tflops = _detect_gpu_peak_tflops()

    total_peak_tflops = gpu_peak_tflops * num_gpus
    step_time_s = step_time_ms / 1000.0
    actual_tflops = flops_per_step / step_time_s / 1e12

    mfu = actual_tflops / total_peak_tflops

    return {
        "mfu": mfu,
        "actual_tflops": actual_tflops,
        "theoretical_tflops": total_peak_tflops,
        "step_time_ms": step_time_ms,
    }
