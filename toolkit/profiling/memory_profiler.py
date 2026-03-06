"""Memory profiler for PyTorch models.

Analyzes GPU memory usage: peak, allocated, reserved, fragmentation,
and per-phase breakdown (forward, backward).

Usage:
    from toolkit.profiling.memory_profiler import analyze_memory
    result = analyze_memory(model, mock_input)
    print(f"Peak: {result['peak_gb']:.2f} GB, Fragmentation: {result['fragmentation']:.1%}")
"""

import torch
import torch.nn as nn
from typing import Optional


def analyze_memory(
    model: nn.Module,
    mock_input: torch.Tensor,
    mock_target: Optional[torch.Tensor] = None,
    loss_fn: Optional[nn.Module] = None,
    include_backward: bool = True,
    fragmentation_threshold: float = 0.2,
) -> dict:
    """Analyze GPU memory usage for a model forward (and optionally backward) pass.

    Args:
        model: PyTorch model (must be on CUDA)
        mock_input: example input tensor (must be on CUDA)
        mock_target: target tensor for backward pass
        loss_fn: loss function (defaults to sum of output)
        include_backward: whether to run backward pass
        fragmentation_threshold: flag if fragmentation exceeds this (default 20%)

    Returns:
        dict with memory breakdown in GB and bytes
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for memory profiling")

    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(f"Model must be on CUDA, got {device}")

    model.train()

    # Reset stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Baseline
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()

    # Forward
    output = model(mock_input)
    torch.cuda.synchronize()
    after_forward_allocated = torch.cuda.memory_allocated()
    after_forward_peak = torch.cuda.max_memory_allocated()

    # Backward
    if include_backward:
        if loss_fn is not None and mock_target is not None:
            loss = loss_fn(output, mock_target)
        else:
            if isinstance(output, tuple):
                loss = output[0].sum()
            else:
                loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()

    # Final measurements
    final_allocated = torch.cuda.memory_allocated()
    final_reserved = torch.cuda.memory_reserved()
    peak_allocated = torch.cuda.max_memory_allocated()

    # Compute derived metrics
    activation_memory = after_forward_allocated - baseline_allocated
    fragmentation = (final_reserved - final_allocated) / final_reserved if final_reserved > 0 else 0.0

    # Model parameter memory estimate
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_memory = sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    )

    warnings = []
    if fragmentation > fragmentation_threshold:
        warnings.append(
            f"High memory fragmentation: {fragmentation:.1%} "
            f"(threshold: {fragmentation_threshold:.1%})"
        )

    return {
        # In GB
        "peak_gb": peak_allocated / 1e9,
        "allocated_gb": final_allocated / 1e9,
        "reserved_gb": final_reserved / 1e9,
        "baseline_gb": baseline_allocated / 1e9,
        "activation_gb": activation_memory / 1e9,
        "param_gb": param_memory / 1e9,
        "grad_gb": grad_memory / 1e9,
        # In bytes (for precise comparisons)
        "peak_bytes": peak_allocated,
        "allocated_bytes": final_allocated,
        "reserved_bytes": final_reserved,
        "baseline_bytes": baseline_allocated,
        "activation_bytes": activation_memory,
        "param_bytes": param_memory,
        "grad_bytes": grad_memory,
        # Metrics
        "fragmentation": fragmentation,
        "fragmentation_threshold": fragmentation_threshold,
        "fragmentation_ok": fragmentation <= fragmentation_threshold,
        # Context
        "device": torch.cuda.get_device_name(0),
        "warnings": warnings,
    }
