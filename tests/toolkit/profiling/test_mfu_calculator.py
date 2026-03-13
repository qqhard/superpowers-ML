"""Tests for toolkit/profiling/mfu_calculator.py

Tests MFU calculator with FlopCounterMode-based FLOPs counting
and CUDA Events-based step timing.
"""

import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from toolkit.profiling.mfu_calculator import (
    GPU_PEAK_TFLOPS,
    count_flops,
    detect_warmup_end,
    measure_step_times,
    calculate_mfu,
)


class SimpleMLP(nn.Module):
    def __init__(self, d_in=128, d_hidden=256, d_out=10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# ===== Pure logic tests (no CUDA) =====


def test_gpu_peak_tflops_known_gpus():
    """Verify GPU_PEAK_TFLOPS dict contains expected entries."""
    assert GPU_PEAK_TFLOPS["H100"] == 989.0
    assert GPU_PEAK_TFLOPS["A100"] == 312.0
    assert GPU_PEAK_TFLOPS["B200"] == 2250.0


def test_warmup_convergence_detection():
    """Given step times that converge, verify warmup detector finds steady state."""
    step_times = [500, 300, 200, 180, 175, 172, 173, 171, 172, 173]
    idx = detect_warmup_end(step_times, window=5, cv_threshold=0.05)
    assert idx >= 0, "Should detect convergence"
    # The last 5 values [172, 173, 171, 172, 173] have very low CV
    # Steady state should start no later than index 5
    assert idx <= 5


def test_warmup_no_convergence():
    """Given wildly varying step times, verify returns -1."""
    step_times = [500, 100, 800, 50, 900, 200, 700, 150, 600, 300]
    idx = detect_warmup_end(step_times, window=5, cv_threshold=0.05)
    assert idx == -1


def test_mfu_computation():
    """Given known values, verify MFU calculation is correct."""
    result = calculate_mfu(
        flops_per_step=int(1e12),
        step_time_ms=100.0,
        gpu_peak_tflops=100.0,
        num_gpus=1,
    )
    # actual_tflops = 1e12 / 0.1 / 1e12 = 10.0
    # mfu = 10.0 / 100.0 = 0.1
    assert result["mfu"] == pytest.approx(0.1)
    assert result["actual_tflops"] == pytest.approx(10.0)
    assert result["theoretical_tflops"] == pytest.approx(100.0)


def test_mfu_multi_gpu():
    """Verify peak scales by num_gpus."""
    result_1 = calculate_mfu(
        flops_per_step=int(1e12),
        step_time_ms=100.0,
        gpu_peak_tflops=100.0,
        num_gpus=1,
    )
    result_4 = calculate_mfu(
        flops_per_step=int(1e12),
        step_time_ms=100.0,
        gpu_peak_tflops=100.0,
        num_gpus=4,
    )
    assert result_4["theoretical_tflops"] == pytest.approx(400.0)
    assert result_4["mfu"] == pytest.approx(result_1["mfu"] / 4)


# ===== CUDA tests =====


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_count_flops_linear():
    """FlopCounterMode should report FLOPs for a Linear layer forward pass."""
    model = nn.Linear(128, 256).cuda()
    mock_input = torch.randn(8, 128, device="cuda")
    result = count_flops(model, mock_input, include_backward=False)
    total = result["total_flops"]
    assert total > 0
    # Matmul FLOPs for Linear: 2 * batch * in * out = 2 * 8 * 128 * 256 = 524288
    expected_approx = 2 * 8 * 128 * 256
    # Allow some tolerance (bias adds, etc.)
    assert total >= expected_approx * 0.5
    assert total <= expected_approx * 2.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_count_flops_with_backward():
    """Forward + backward FLOPs should exceed forward-only FLOPs."""
    model = nn.Linear(128, 256).cuda()
    mock_input = torch.randn(8, 128, device="cuda")

    fwd_only = count_flops(model, mock_input, include_backward=False)
    fwd_bwd = count_flops(model, mock_input, include_backward=True)

    assert fwd_bwd["total_flops"] > fwd_only["total_flops"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_measure_step_times():
    """Run a trivial model for 10 steps, verify returns list of 10 positive floats."""
    model = SimpleMLP().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def train_step(model, step_idx):
        x = torch.randn(8, 128, device="cuda")
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    times = measure_step_times(model, train_step, num_steps=10)
    assert len(times) == 10
    for t in times:
        assert isinstance(t, float)
        assert t > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_calculate_mfu_integration():
    """Full pipeline: count_flops + measure_step_times + calculate_mfu."""
    model = SimpleMLP().cuda()
    mock_input = torch.randn(8, 128, device="cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Count FLOPs
    flop_result = count_flops(model, mock_input, include_backward=True)
    assert flop_result["total_flops"] > 0

    # Measure step times
    def train_step(model, step_idx):
        x = torch.randn(8, 128, device="cuda")
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    step_times = measure_step_times(model, train_step, num_steps=5)
    import statistics
    median_time = statistics.median(step_times)

    # Calculate MFU
    result = calculate_mfu(
        flops_per_step=flop_result["total_flops"],
        step_time_ms=median_time,
        gpu_peak_tflops=100.0,  # use a known value for test stability
    )
    assert "mfu" in result
    assert "actual_tflops" in result
    assert "step_time_ms" in result
    assert result["mfu"] > 0
    assert result["step_time_ms"] > 0
