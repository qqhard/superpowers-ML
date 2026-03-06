"""Tests for toolkit/profiling/layer_profiler.py"""

import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from toolkit.profiling.layer_profiler import profile_layers


class ThreeLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(128, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_basic():
    model = ThreeLayerMLP().cuda()
    mock_input = torch.randn(32, 128, device="cuda")

    result = profile_layers(model, mock_input, n_warmup=2, n_measure=3)

    assert "layers" in result
    assert "total_time_ms" in result
    assert "device" in result
    assert len(result["layers"]) > 0
    assert result["total_time_ms"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_layer_fields():
    model = ThreeLayerMLP().cuda()
    mock_input = torch.randn(32, 128, device="cuda")

    result = profile_layers(model, mock_input, n_warmup=2, n_measure=3)

    for layer in result["layers"]:
        assert "name" in layer
        assert "time_ms" in layer
        assert "percentage" in layer
        assert layer["time_ms"] >= 0
        assert 0 <= layer["percentage"] <= 100


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_percentages_sum_to_100():
    model = ThreeLayerMLP().cuda()
    mock_input = torch.randn(32, 128, device="cuda")

    result = profile_layers(model, mock_input, n_warmup=2, n_measure=3)

    total_pct = sum(l["percentage"] for l in result["layers"])
    assert total_pct == pytest.approx(100.0, abs=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_sorted_descending():
    model = ThreeLayerMLP().cuda()
    mock_input = torch.randn(32, 128, device="cuda")

    result = profile_layers(model, mock_input, n_warmup=2, n_measure=3)

    times = [l["time_ms"] for l in result["layers"]]
    assert times == sorted(times, reverse=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_forward_only():
    model = ThreeLayerMLP().cuda()
    mock_input = torch.randn(32, 128, device="cuda")

    result = profile_layers(
        model, mock_input, include_backward=False, n_warmup=2, n_measure=3
    )
    assert len(result["layers"]) > 0
    assert result["total_time_ms"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_with_loss_fn():
    model = ThreeLayerMLP().cuda()
    mock_input = torch.randn(32, 128, device="cuda")
    mock_target = torch.randint(0, 10, (32,), device="cuda")
    loss_fn = nn.CrossEntropyLoss()

    result = profile_layers(
        model, mock_input, mock_target=mock_target, loss_fn=loss_fn,
        n_warmup=2, n_measure=3,
    )
    assert len(result["layers"]) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_finds_named_layers():
    model = ThreeLayerMLP().cuda()
    mock_input = torch.randn(32, 128, device="cuda")

    result = profile_layers(model, mock_input, n_warmup=2, n_measure=3)

    layer_names = [l["name"] for l in result["layers"]]
    # Should find our named layers
    assert any("layer1" in name for name in layer_names)
    assert any("layer2" in name for name in layer_names)
    assert any("layer3" in name for name in layer_names)
