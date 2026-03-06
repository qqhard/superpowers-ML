"""Tests for toolkit/profiling/mfu_calculator.py"""

import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from toolkit.profiling.mfu_calculator import (
    calculate_mfu,
    count_parameters,
    estimate_flops_per_step,
    GPU_PEAK_TFLOPS,
    _detect_gpu_peak_tflops,
)


class SimpleMLP(nn.Module):
    def __init__(self, d_in=128, d_hidden=256, d_out=10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class SimpleTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, vocab_size=1000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return self.fc(x)


# --- count_parameters ---

def test_count_parameters():
    model = SimpleMLP(128, 256, 10)
    n = count_parameters(model)
    # fc1: 128*256 + 256 = 33024, fc2: 256*10 + 10 = 2570
    assert n == 33024 + 2570


def test_count_parameters_frozen():
    model = SimpleMLP(128, 256, 10)
    for p in model.fc2.parameters():
        p.requires_grad = False
    n = count_parameters(model)
    assert n == 33024  # only fc1


# --- estimate_flops_per_step ---

def test_flops_dense():
    model = SimpleMLP(128, 256, 10)
    n = count_parameters(model)
    flops = estimate_flops_per_step(model, input_shape=(32,), model_type="dense")
    assert flops == pytest.approx(6.0 * n * 32)


def test_flops_sequence():
    model = SimpleMLP(128, 256, 10)
    n = count_parameters(model)
    flops = estimate_flops_per_step(model, input_shape=(8, 512), model_type="dense")
    assert flops == pytest.approx(6.0 * n * 8 * 512)


def test_flops_moe():
    model = SimpleMLP(128, 256, 10)
    n = count_parameters(model)
    flops = estimate_flops_per_step(
        model, input_shape=(8, 512), model_type="moe",
        num_active_experts=2, num_total_experts=8,
    )
    expected_ratio = 0.5 + 0.5 * (2 / 8)
    assert flops == pytest.approx(6.0 * n * 8 * 512 * expected_ratio)


# --- calculate_mfu ---

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_calculate_mfu_basic():
    model = SimpleMLP().cuda()
    result = calculate_mfu(
        model, input_shape=(32, 128), step_time_ms=10.0, gpu_peak_tflops=100.0
    )
    assert "mfu" in result
    assert "tca" in result
    assert "theoretical_tflops" in result
    assert "actual_tflops" in result
    assert "status" in result
    assert result["mfu"] > 0
    assert result["theoretical_tflops"] == 100.0
    assert result["num_params"] == count_parameters(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_calculate_mfu_status_on_target():
    model = SimpleMLP().cuda()
    # Force a very fast step time to get high MFU
    result = calculate_mfu(
        model, input_shape=(32, 128), step_time_ms=0.001,
        gpu_peak_tflops=0.001, target_mfu=0.01,
    )
    assert result["status"] == "on_target"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_calculate_mfu_status_needs_investigation():
    model = SimpleMLP().cuda()
    # Force a very slow step time
    result = calculate_mfu(
        model, input_shape=(32, 128), step_time_ms=1e9,
        gpu_peak_tflops=1000.0, target_mfu=0.5,
    )
    assert result["status"] == "needs_investigation"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_calculate_mfu_multi_gpu():
    model = SimpleMLP().cuda()
    result_1gpu = calculate_mfu(
        model, input_shape=(32, 128), step_time_ms=10.0, gpu_peak_tflops=100.0, num_gpus=1
    )
    result_2gpu = calculate_mfu(
        model, input_shape=(32, 128), step_time_ms=10.0, gpu_peak_tflops=100.0, num_gpus=2
    )
    # Same actual TFLOPS, but MFU halved with 2 GPUs (same step time)
    assert result_2gpu["mfu"] == pytest.approx(result_1gpu["mfu"] / 2)


def test_calculate_mfu_invalid_step_time():
    model = SimpleMLP()
    with pytest.raises(ValueError, match="step_time_ms must be positive"):
        calculate_mfu(model, input_shape=(32,), step_time_ms=0.0, gpu_peak_tflops=100.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_detect_gpu():
    # Should not raise on a machine with known GPU
    tflops = _detect_gpu_peak_tflops()
    assert tflops > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_calculate_mfu_auto_detect_gpu():
    """Test that auto-detection works end-to-end."""
    model = SimpleMLP().cuda()
    result = calculate_mfu(model, input_shape=(32, 128), step_time_ms=10.0)
    assert result["gpu_peak_tflops"] > 0
    assert result["gpu_name"] != "N/A"
