"""L0 Engineering Efficiency Runner.

Three-phase production-grade L0 validation:
1. Static analysis (FLOPs, GPU info, backend, memory)
2. Steady-state measurement (step timing + DCGM TCA, simultaneous)
3. Report generation (MFU, TCA, sample speed, gap analysis)

Usage:
    from toolkit.profiling.l0_runner import run_l0
    report = run_l0(model, train_step_fn, mock_input, batch_size=2048)
    print(format_summary_table(report))
"""

import time
import statistics
import torch
import torch.nn as nn
from typing import Optional, Callable

from toolkit.profiling.mfu_calculator import (
    count_flops,
    detect_warmup_end,
    calculate_mfu,
    _detect_gpu_peak_tflops,
    GPU_PEAK_TFLOPS,
)
from toolkit.profiling.dcgm_profiler import (
    check_dcgm_available,
    start_dcgm_sampling,
    stop_dcgm_sampling,
    parse_dcgm_output,
    compute_tca_stats,
    trim_warmup_samples,
)
from toolkit.profiling.gap_analyzer import analyze_gap
from toolkit.profiling.memory_profiler import analyze_memory


def run_l0(
    model: nn.Module,
    train_step_fn: Callable[[nn.Module, int], None],
    mock_input: torch.Tensor,
    batch_size: int,
    seq_len: Optional[int] = None,
    steady_state_minutes: float = 3.0,
    gpu_peak_tflops: Optional[float] = None,
    num_gpus: int = 1,
    dcgm_interval_ms: int = 1000,
    loss_fn: Optional[Callable] = None,
) -> dict:
    """Run full L0 Engineering Efficiency validation.

    Args:
        model: PyTorch model (on CUDA)
        train_step_fn: function(model, step_idx) that runs one training step
        mock_input: example input tensor for FLOPs counting and memory analysis
        batch_size: global batch size (for sample speed calculation)
        seq_len: sequence length (for tokens/sec, optional)
        steady_state_minutes: how long to measure in steady state (default 3 min)
        gpu_peak_tflops: peak GPU TFLOPS (auto-detected if None)
        num_gpus: number of GPUs
        dcgm_interval_ms: DCGM sampling interval in milliseconds
        loss_fn: optional loss function for FLOPs counting

    Returns:
        dict with all metrics + raw data
    """
    report = {}

    # === Phase 1: Static Analysis ===

    # 1a. Count FLOPs
    flops_result = count_flops(model, mock_input, include_backward=True, loss_fn=loss_fn)
    report["flops_per_step"] = flops_result["total_flops"]

    # 1b. GPU info
    if gpu_peak_tflops is None:
        gpu_peak_tflops = _detect_gpu_peak_tflops()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    report["gpu_info"] = {
        "name": gpu_name,
        "peak_tflops": gpu_peak_tflops,
        "num_gpus": num_gpus,
    }

    # 1c. Backend check (1 step with profiler)
    backend_info = _check_backend(model, train_step_fn)
    report["backend"] = backend_info

    # 1d. Memory snapshot
    memory_result = analyze_memory(model, mock_input, include_backward=True)
    report["memory"] = memory_result

    # === Phase 2: Steady-State Measurement ===

    # 2a. Start DCGM if available
    dcgm_available = check_dcgm_available()
    dcgm_proc = None
    dcgm_start_time = time.time()
    if dcgm_available:
        dcgm_proc = start_dcgm_sampling(interval_ms=dcgm_interval_ms)

    # 2b. Run training with CUDA Events timing
    step_times = []
    total_steps = 0
    warmup_end = -1
    steady_start_time = None
    steady_state_seconds = steady_state_minutes * 60

    while True:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        train_step_fn(model, total_steps)
        end_evt.record()
        torch.cuda.synchronize()

        step_times.append(start_evt.elapsed_time(end_evt))
        total_steps += 1

        # Check warmup convergence
        if warmup_end < 0:
            warmup_end = detect_warmup_end(step_times)
            if warmup_end >= 0:
                steady_start_time = time.time()

        # Check if steady-state duration reached
        if warmup_end >= 0 and steady_start_time is not None:
            elapsed = time.time() - steady_start_time
            if elapsed >= steady_state_seconds:
                break

        # Safety: if we haven't converged after many steps, force start
        if warmup_end < 0 and total_steps >= 50:
            warmup_end = total_steps
            steady_start_time = time.time()

    # 2c. Stop DCGM
    tca_stats = None
    tca_warning = None
    if dcgm_proc:
        dcgm_output = stop_dcgm_sampling(dcgm_proc)
        tca_values = parse_dcgm_output(dcgm_output)
        warmup_duration = (steady_start_time - dcgm_start_time) if steady_start_time else 0
        tca_steady = trim_warmup_samples(tca_values, dcgm_interval_ms, warmup_duration)
        tca_stats = compute_tca_stats(tca_steady)
    else:
        tca_warning = "DCGM not available — TCA not measured. Install DCGM for real tensor core activity."

    # === Phase 3: Report Generation ===

    steady_step_times = step_times[warmup_end:] if warmup_end >= 0 else step_times
    median_step_ms = statistics.median(steady_step_times)

    # MFU
    mfu_result = calculate_mfu(
        flops_per_step=flops_result["total_flops"],
        step_time_ms=median_step_ms,
        gpu_peak_tflops=gpu_peak_tflops,
        num_gpus=num_gpus,
    )
    report["mfu"] = mfu_result["mfu"]
    report["actual_tflops"] = mfu_result["actual_tflops"]

    # TCA
    report["tca"] = tca_stats["mean"] if tca_stats and tca_stats.get("mean") is not None else None
    report["tca_stats"] = tca_stats
    report["tca_warning"] = tca_warning

    # Sample speed
    step_time_s = median_step_ms / 1000.0
    report["sample_speed"] = batch_size / step_time_s

    # Tokens per second
    tokens_per_sec = None
    if seq_len is not None:
        tokens_per_sec = batch_size * seq_len / step_time_s
    report["tokens_per_sec"] = tokens_per_sec

    # Gap analysis
    tca_for_gap = tca_stats["mean"] if tca_stats and tca_stats.get("mean") is not None else None
    mfu_for_gap = mfu_result["mfu"] * 100  # convert to percent
    gap_result = analyze_gap(tca_percent=tca_for_gap, mfu_percent=mfu_for_gap)
    report["gap_analysis"] = gap_result

    # Step time stats
    report["step_time_ms"] = median_step_ms
    report["step_times"] = step_times
    report["warmup_steps"] = warmup_end if warmup_end >= 0 else total_steps
    report["steady_state_steps"] = len(steady_step_times)

    return report


def format_summary_table(report: dict) -> str:
    """Format L0 report as human-readable summary table."""
    lines = []
    lines.append("+" + "-" * 58 + "+")
    lines.append("| L0 Engineering Efficiency Report" + " " * 26 + "|")
    lines.append("+" + "-" * 28 + "+" + "-" * 29 + "+")

    def row(label, value):
        return f"| {label:<26} | {value:<27} |"

    # MFU
    mfu = report.get("mfu")
    lines.append(row("MFU", f"{mfu:.2%}" if mfu is not None else "N/A"))

    # TCA
    tca = report.get("tca")
    lines.append(row("TCA", f"{tca:.2f}%" if tca is not None else "N/A (DCGM unavailable)"))

    # Sample speed
    ss = report.get("sample_speed")
    lines.append(row("Sample Speed", f"{ss:,.0f} samples/sec" if ss else "N/A"))

    # Tokens/sec
    tps = report.get("tokens_per_sec")
    if tps is not None:
        lines.append(row("Tokens/sec", f"{tps:,.0f}"))

    # FLOPs
    flops = report.get("flops_per_step")
    if flops:
        lines.append(row("FLOPs/step", f"{flops / 1e12:.2f} TFLOPS"))

    # Step time
    st = report.get("step_time_ms")
    steady = report.get("steady_state_steps", 0)
    lines.append(row("Step Time", f"{st:.2f} ms (median, N={steady})" if st else "N/A"))

    # Memory
    mem = report.get("memory", {})
    peak = mem.get("peak_gb") if isinstance(mem, dict) else None
    if peak:
        lines.append(row("Peak Memory", f"{peak:.1f} GB"))

    # Gap
    gap = report.get("gap_analysis", {})
    gap_pp = gap.get("gap_pp") if isinstance(gap, dict) else None
    if gap_pp is not None:
        lines.append(row("TCA-MFU Gap", f"{gap_pp:.1f} pp"))

    lines.append("+" + "-" * 28 + "+" + "-" * 29 + "+")

    # Backend
    backend = report.get("backend", {})
    if backend and isinstance(backend, dict):
        fa_str = "FA ok" if backend.get("flash_attention_detected") else "FA ?"
        dtype_str = backend.get("param_dtype", "?")
        info = f"| Backend: {fa_str}, {dtype_str}"
        lines.append(info + " " * max(0, 59 - len(info)) + "|")

    # TCA warning
    if report.get("tca_warning"):
        warning_text = report["tca_warning"][:48]
        info = f"| Warning: {warning_text}"
        lines.append(info + " " * max(0, 59 - len(info)) + "|")

    # Gap contributors
    if isinstance(gap, dict):
        contributors = gap.get("contributors", [])
        for c in contributors[:3]:
            text = c[:56]
            info = f"| {text}"
            lines.append(info + " " * max(0, 59 - len(info)) + "|")

    lines.append("+" + "-" * 58 + "+")

    return "\n".join(lines)


def _check_backend(model: nn.Module, train_step_fn: Callable) -> dict:
    """Check backend: param dtypes and kernel usage via profiler."""
    result = {}

    # Param dtype check
    dtypes = set()
    for p in model.parameters():
        dtypes.add(str(p.dtype))
    result["param_dtype"] = ", ".join(sorted(dtypes))

    # Try to detect FA via profiler (1 step)
    result["flash_attention_detected"] = False
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
        ) as prof:
            train_step_fn(model, -1)  # -1 signals profiling step

        events = prof.key_averages()
        for evt in events:
            if "flash" in evt.key.lower() or "fmha" in evt.key.lower():
                result["flash_attention_detected"] = True
                break

        # Collect top kernels
        top_kernels = []
        for evt in sorted(events, key=lambda e: e.device_time_total, reverse=True)[:10]:
            if evt.device_time_total > 0:
                top_kernels.append({"name": evt.key, "cuda_time_us": evt.device_time_total})
        result["top_kernels"] = top_kernels
    except Exception as e:
        result["profiler_error"] = str(e)

    return result
