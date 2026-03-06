"""Per-layer profiling for PyTorch models.

Measures forward and backward time per layer/module, with proper CUDA
synchronization and warmup handling.

Usage:
    from toolkit.profiling.layer_profiler import profile_layers
    result = profile_layers(model, mock_input)
    for layer in result['layers']:
        print(f"{layer['name']}: {layer['time_ms']:.2f}ms ({layer['percentage']:.1f}%)")
"""

import torch
import torch.nn as nn
from typing import Optional
import warnings


def profile_layers(
    model: nn.Module,
    mock_input: torch.Tensor,
    mock_target: Optional[torch.Tensor] = None,
    loss_fn: Optional[nn.Module] = None,
    n_warmup: int = 3,
    n_measure: int = 5,
    include_backward: bool = True,
    max_depth: int = 1,
) -> dict:
    """Profile per-layer timing using PyTorch profiler.

    Args:
        model: PyTorch model (must be on CUDA)
        mock_input: example input tensor (must be on CUDA)
        mock_target: target tensor for backward pass (auto-generated if None)
        loss_fn: loss function (defaults to sum of output for backward)
        n_warmup: warmup iterations before measuring
        n_measure: measurement iterations to average
        include_backward: whether to profile backward pass
        max_depth: module tree depth to profile (1 = top-level children only)

    Returns:
        dict with keys: layers (list of dicts), total_time_ms, device
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for layer profiling")

    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(f"Model must be on CUDA, got {device}")

    model.train()

    # Collect modules at the specified depth
    target_modules = _get_modules_at_depth(model, max_depth)

    if not target_modules:
        warnings.warn("No modules found at specified depth, profiling top-level model only")
        target_modules = [("model", model)]

    # Warmup
    for _ in range(n_warmup):
        output = model(mock_input)
        if include_backward:
            loss = _compute_loss(output, mock_target, loss_fn)
            loss.backward()
        torch.cuda.synchronize()

    # Profile using CUDA events for accurate per-layer timing
    layer_times = {name: [] for name, _ in target_modules}
    hooks = []

    start_events = {}
    end_events = {}

    def make_forward_pre_hook(name):
        def hook(module, input):
            start_events[name] = torch.cuda.Event(enable_timing=True)
            start_events[name].record()
        return hook

    def make_forward_hook(name):
        def hook(module, input, output):
            end_events[name] = torch.cuda.Event(enable_timing=True)
            end_events[name].record()
        return hook

    for name, module in target_modules:
        hooks.append(module.register_forward_pre_hook(make_forward_pre_hook(name)))
        hooks.append(module.register_forward_hook(make_forward_hook(name)))

    # Measure
    for _ in range(n_measure):
        start_events.clear()
        end_events.clear()

        torch.cuda.synchronize()
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)

        total_start.record()
        output = model(mock_input)
        if include_backward:
            loss = _compute_loss(output, mock_target, loss_fn)
            loss.backward()
        total_end.record()
        torch.cuda.synchronize()

        for name, _ in target_modules:
            if name in start_events and name in end_events:
                elapsed = start_events[name].elapsed_time(end_events[name])
                layer_times[name].append(elapsed)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute averages
    layers = []
    total_layer_time = 0.0
    for name, _ in target_modules:
        times = layer_times[name]
        if times:
            avg_time = sum(times) / len(times)
            layers.append({"name": name, "time_ms": avg_time})
            total_layer_time += avg_time

    # Compute percentages
    for layer in layers:
        layer["percentage"] = (layer["time_ms"] / total_layer_time * 100.0) if total_layer_time > 0 else 0.0

    # Sort by time descending
    layers.sort(key=lambda x: x["time_ms"], reverse=True)

    return {
        "layers": layers,
        "total_time_ms": total_layer_time,
        "device": torch.cuda.get_device_name(0),
    }


def _get_modules_at_depth(model: nn.Module, max_depth: int) -> list:
    """Get named modules at a specific depth in the module tree."""
    result = []
    for name, module in model.named_children():
        if max_depth <= 1:
            result.append((name, module))
        else:
            children = list(module.named_children())
            if children:
                for child_name, child_module in children:
                    sub = _get_modules_at_depth(child_module, max_depth - 2)
                    if sub:
                        result.extend(
                            (f"{name}.{child_name}.{sn}", sm) for sn, sm in sub
                        )
                    else:
                        result.append((f"{name}.{child_name}", child_module))
            else:
                result.append((name, module))
    return result


def _compute_loss(output, mock_target, loss_fn):
    """Compute loss for backward pass."""
    if loss_fn is not None and mock_target is not None:
        return loss_fn(output, mock_target)
    if isinstance(output, tuple):
        output = output[0]
    return output.sum()
