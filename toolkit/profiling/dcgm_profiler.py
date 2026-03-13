"""DCGM-based Tensor Core Activity (TCA) profiler.

Uses DCGM_FI_PROF_PIPE_TENSOR_ACTIVE (field 1004) to measure real tensor core
utilization during normal training with zero overhead.

Usage:
    from toolkit.profiling.dcgm_profiler import (
        check_dcgm_available, start_dcgm_sampling, stop_dcgm_sampling,
        parse_dcgm_output, compute_tca_stats
    )
"""
import subprocess
import statistics
from typing import Optional


def check_dcgm_available() -> bool:
    """Check if dcgmi is available on this system."""
    try:
        r = subprocess.run(["dcgmi", "discovery", "-l"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def start_dcgm_sampling(interval_ms: int = 1000, gpu_id: int = 0) -> subprocess.Popen:
    """Start DCGM background sampling.

    Runs: dcgmi dmon -e 1004 -d {interval_s} -i {gpu_id}
    Field 1004 = DCGM_FI_PROF_PIPE_TENSOR_ACTIVE

    Returns subprocess handle.
    """
    interval_s = max(1, interval_ms // 1000)
    cmd = ["dcgmi", "dmon", "-e", "1004", "-d", str(interval_s), "-i", str(gpu_id)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc


def stop_dcgm_sampling(proc: subprocess.Popen) -> str:
    """Stop DCGM sampling and return captured stdout."""
    proc.terminate()
    try:
        stdout, _ = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate()
    return stdout


def parse_dcgm_output(output: str) -> list:
    """Parse dcgmi dmon output, extract TENACT column.

    Handles comment lines (#), blank lines, N/A values.
    Returns list of TCA% as floats.
    """
    if not output or not output.strip():
        return []

    lines = output.strip().split("\n")
    tenact_col = -1
    values = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("#"):
            # Header line -- find TENACT column index
            parts = stripped.lstrip("#").split()
            for i, part in enumerate(parts):
                if part.upper() in ("TENACT", "TENSOR_ACTIVE"):
                    tenact_col = i
                    break
            continue

        if tenact_col < 0:
            continue

        parts = stripped.split()
        if len(parts) <= tenact_col:
            continue

        val_str = parts[tenact_col]
        if val_str.upper() in ("N/A", "NA", "-"):
            continue

        try:
            values.append(float(val_str))
        except ValueError:
            continue

    return values


def compute_tca_stats(tca_values: list) -> dict:
    """Compute statistics over TCA samples.

    Returns dict with: mean, median, std, min, max, num_samples
    """
    if not tca_values:
        return {
            "mean": None, "median": None, "std": None,
            "min": None, "max": None, "num_samples": 0,
        }

    return {
        "mean": statistics.mean(tca_values),
        "median": statistics.median(tca_values),
        "std": statistics.stdev(tca_values) if len(tca_values) > 1 else 0.0,
        "min": min(tca_values),
        "max": max(tca_values),
        "num_samples": len(tca_values),
    }


def trim_warmup_samples(
    tca_values: list, interval_ms: int, warmup_duration_s: float
) -> list:
    """Remove samples that fall within the warmup period.

    Args:
        tca_values: list of TCA% values
        interval_ms: DCGM sampling interval in ms
        warmup_duration_s: warmup duration in seconds

    Returns:
        list with warmup samples removed
    """
    if warmup_duration_s <= 0:
        return list(tca_values)
    samples_to_skip = int(warmup_duration_s * 1000 / interval_ms)
    return tca_values[samples_to_skip:]
