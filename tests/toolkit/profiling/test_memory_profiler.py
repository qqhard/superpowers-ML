"""Tests for toolkit/profiling/memory_profiler.py"""

import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from toolkit.profiling.memory_profiler import analyze_memory


class SimpleMLP(nn.Module):
    def __init__(self, d_in=128, d_hidden=256, d_out=10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class LargeModel(nn.Module):
    """Bigger model to ensure measurable memory usage."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
        )

    def forward(self, x):
        return self.layers(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_analyze_memory_basic():
    model = SimpleMLP().cuda()
    mock_input = torch.randn(32, 128, device="cuda")

    result = analyze_memory(model, mock_input)

    assert "peak_gb" in result
    assert "allocated_gb" in result
    assert "reserved_gb" in result
    assert "fragmentation" in result
    assert "device" in result
    assert "warnings" in result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_memory_values_positive():
    model = SimpleMLP().cuda()
    mock_input = torch.randn(32, 128, device="cuda")

    result = analyze_memory(model, mock_input)

    assert result["peak_gb"] >= 0
    assert result["allocated_gb"] >= 0
    assert result["reserved_gb"] >= 0
    assert result["param_gb"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_peak_gte_allocated():
    model = SimpleMLP().cuda()
    mock_input = torch.randn(32, 128, device="cuda")

    result = analyze_memory(model, mock_input)

    assert result["peak_bytes"] >= result["allocated_bytes"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reserved_gte_allocated():
    model = SimpleMLP().cuda()
    mock_input = torch.randn(32, 128, device="cuda")

    result = analyze_memory(model, mock_input)

    assert result["reserved_bytes"] >= result["allocated_bytes"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_param_memory_accurate():
    model = SimpleMLP(128, 256, 10).cuda()
    mock_input = torch.randn(32, 128, device="cuda")

    result = analyze_memory(model, mock_input)

    # fc1: (128*256 + 256) * 4 bytes = 132096 + fc2: (256*10 + 10) * 4 = 10280
    expected_bytes = (128 * 256 + 256 + 256 * 10 + 10) * 4
    assert result["param_bytes"] == expected_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_forward_only():
    model = SimpleMLP().cuda()
    mock_input = torch.randn(32, 128, device="cuda")

    result = analyze_memory(model, mock_input, include_backward=False)

    assert result["peak_gb"] >= 0
    assert "fragmentation" in result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_with_loss_fn():
    model = SimpleMLP().cuda()
    mock_input = torch.randn(32, 128, device="cuda")
    mock_target = torch.randint(0, 10, (32,), device="cuda")
    loss_fn = nn.CrossEntropyLoss()

    result = analyze_memory(model, mock_input, mock_target=mock_target, loss_fn=loss_fn)

    assert result["peak_gb"] >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fragmentation_threshold():
    model = SimpleMLP().cuda()
    mock_input = torch.randn(32, 128, device="cuda")

    result = analyze_memory(model, mock_input, fragmentation_threshold=0.99)
    assert result["fragmentation_ok"] is True

    # Note: can't guarantee fragmentation > 0 so we just check the field exists
    assert "fragmentation" in result
    assert 0 <= result["fragmentation"] <= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_larger_model_more_memory():
    small = SimpleMLP().cuda()
    large = LargeModel().cuda()
    mock_small = torch.randn(32, 128, device="cuda")
    mock_large = torch.randn(32, 1024, device="cuda")

    r_small = analyze_memory(small, mock_small)
    r_large = analyze_memory(large, mock_large)

    assert r_large["param_bytes"] > r_small["param_bytes"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bytes_and_gb_consistent():
    model = SimpleMLP().cuda()
    mock_input = torch.randn(32, 128, device="cuda")

    result = analyze_memory(model, mock_input)

    assert result["peak_gb"] == pytest.approx(result["peak_bytes"] / 1e9)
    assert result["allocated_gb"] == pytest.approx(result["allocated_bytes"] / 1e9)
    assert result["param_gb"] == pytest.approx(result["param_bytes"] / 1e9)
