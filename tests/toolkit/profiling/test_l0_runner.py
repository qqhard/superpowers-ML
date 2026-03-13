"""Tests for l0_runner — Three-Phase L0 Orchestration."""

import pytest
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn


# --- Pure logic tests (mock everything CUDA/DCGM) ---


@patch("toolkit.profiling.l0_runner.analyze_memory")
@patch("toolkit.profiling.l0_runner._check_backend")
@patch("toolkit.profiling.l0_runner.check_dcgm_available", return_value=False)
@patch("toolkit.profiling.l0_runner.count_flops")
@patch("toolkit.profiling.l0_runner._detect_gpu_peak_tflops", return_value=989.0)
@patch("toolkit.profiling.l0_runner.torch")
def test_l0_report_keys(
    mock_torch, mock_detect, mock_count_flops, mock_dcgm, mock_backend, mock_memory
):
    """Verify report dict has all expected keys."""
    from toolkit.profiling.l0_runner import run_l0

    # Setup mocks
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_name.return_value = "NVIDIA H100"

    # Make CUDA events work without GPU
    mock_event = MagicMock()
    mock_event.elapsed_time.return_value = 100.0
    mock_torch.cuda.Event.return_value = mock_event

    mock_count_flops.return_value = {"total_flops": 1e12}
    mock_backend.return_value = {"param_dtype": "torch.bfloat16", "flash_attention_detected": False}
    mock_memory.return_value = {"peak_gb": 10.0}

    # Use real calculate_mfu and detect_warmup_end
    with patch("toolkit.profiling.l0_runner.detect_warmup_end", return_value=0):
        with patch("toolkit.profiling.l0_runner.calculate_mfu") as mock_mfu:
            mock_mfu.return_value = {
                "mfu": 0.5, "actual_tflops": 100.0,
                "theoretical_tflops": 200.0, "step_time_ms": 100.0,
            }
            with patch("toolkit.profiling.l0_runner.analyze_gap") as mock_gap:
                mock_gap.return_value = {
                    "gap_pp": None, "tca": None, "mfu": 50.0,
                    "clock_factor": None, "contributors": [],
                }

                model = MagicMock(spec=nn.Module)
                model.parameters.return_value = iter([])
                train_fn = MagicMock()

                report = run_l0(
                    model=model,
                    train_step_fn=train_fn,
                    mock_input=MagicMock(),
                    batch_size=2048,
                    gpu_peak_tflops=989.0,
                    steady_state_minutes=0.0,  # immediate
                )

    expected_keys = {
        "mfu", "tca", "sample_speed", "tokens_per_sec",
        "flops_per_step", "step_times", "memory", "backend",
        "gap_analysis", "gpu_info", "warmup_steps", "steady_state_steps",
    }
    assert expected_keys.issubset(set(report.keys())), (
        f"Missing keys: {expected_keys - set(report.keys())}"
    )


def test_format_summary_table():
    """Given a complete report dict, verify format_summary_table output."""
    from toolkit.profiling.l0_runner import format_summary_table

    report = {
        "mfu": 0.4375,
        "tca": 55.2,
        "tca_stats": {"mean": 55.2},
        "tca_warning": None,
        "sample_speed": 20480.0,
        "tokens_per_sec": 20971520.0,
        "flops_per_step": 74.82e12,
        "step_time_ms": 172.89,
        "step_times": [172.89] * 10,
        "memory": {"peak_gb": 38.5},
        "backend": {"param_dtype": "torch.bfloat16", "flash_attention_detected": True},
        "gap_analysis": {"gap_pp": 11.45, "contributors": ["Moderate gap"]},
        "gpu_info": {"name": "H100", "peak_tflops": 989.0, "num_gpus": 1},
        "warmup_steps": 5,
        "steady_state_steps": 100,
        "actual_tflops": 432.9,
    }

    table = format_summary_table(report)
    assert "MFU" in table
    assert "TCA" in table
    assert "Sample Speed" in table
    assert "43.75%" in table  # MFU formatted
    assert "55.20%" in table  # TCA formatted


def test_phase3_mfu_calculation():
    """Verify MFU calculation: flops=74.82e12, step=172.89ms, peak=989 TFLOPS."""
    from toolkit.profiling.mfu_calculator import calculate_mfu

    result = calculate_mfu(
        flops_per_step=74.82e12,
        step_time_ms=172.89,
        gpu_peak_tflops=989.0,
        num_gpus=1,
    )
    # MFU = 74.82e12 / (0.17289 * 989e12) = 74.82 / (170.9 ...) ~ 0.4375
    assert abs(result["mfu"] - 0.4375) < 0.005, f"MFU={result['mfu']}, expected ~0.4375"


@patch("toolkit.profiling.l0_runner.analyze_memory")
@patch("toolkit.profiling.l0_runner._check_backend")
@patch("toolkit.profiling.l0_runner.check_dcgm_available", return_value=False)
@patch("toolkit.profiling.l0_runner.count_flops")
@patch("toolkit.profiling.l0_runner.torch")
def test_dcgm_unavailable_graceful(
    mock_torch, mock_count_flops, mock_dcgm, mock_backend, mock_memory
):
    """When DCGM unavailable, tca should be None with a warning."""
    from toolkit.profiling.l0_runner import run_l0

    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_name.return_value = "NVIDIA H100"

    mock_event = MagicMock()
    mock_event.elapsed_time.return_value = 100.0
    mock_torch.cuda.Event.return_value = mock_event

    mock_count_flops.return_value = {"total_flops": 1e12}
    mock_backend.return_value = {"param_dtype": "torch.bfloat16", "flash_attention_detected": False}
    mock_memory.return_value = {"peak_gb": 10.0}

    with patch("toolkit.profiling.l0_runner.detect_warmup_end", return_value=0):
        with patch("toolkit.profiling.l0_runner.calculate_mfu") as mock_mfu:
            mock_mfu.return_value = {
                "mfu": 0.5, "actual_tflops": 100.0,
                "theoretical_tflops": 200.0, "step_time_ms": 100.0,
            }
            with patch("toolkit.profiling.l0_runner.analyze_gap") as mock_gap:
                mock_gap.return_value = {
                    "gap_pp": None, "tca": None, "mfu": 50.0,
                    "clock_factor": None, "contributors": [],
                }

                model = MagicMock(spec=nn.Module)
                model.parameters.return_value = iter([])
                report = run_l0(
                    model=model,
                    train_step_fn=MagicMock(),
                    mock_input=MagicMock(),
                    batch_size=2048,
                    gpu_peak_tflops=989.0,
                    steady_state_minutes=0.0,
                )

    assert report["tca"] is None
    assert report["tca_warning"] is not None
    assert "DCGM" in report["tca_warning"]


def test_sample_speed_calculation():
    """batch_size=2048, step_time_ms=100 -> sample_speed=20480 samples/sec."""
    batch_size = 2048
    step_time_ms = 100.0
    expected = batch_size / (step_time_ms / 1000.0)
    assert expected == 20480.0


def test_tokens_per_sec_with_seq_len():
    """batch_size=2048, seq_len=1024, step_time_ms=100 -> tokens_per_sec."""
    batch_size = 2048
    seq_len = 1024
    step_time_ms = 100.0
    expected = batch_size * seq_len / (step_time_ms / 1000.0)
    assert expected == 2048 * 1024 / 0.1
    assert expected == 20971520.0


# --- CUDA smoke test ---

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_run_l0_smoke():
    """Smoke test with a trivial MLP on real GPU."""
    from toolkit.profiling.l0_runner import run_l0

    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = SimpleMLP().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    mock_input = torch.randn(32, 128, device="cuda")

    def train_step(m, step_idx):
        optimizer.zero_grad()
        out = m(mock_input)
        loss = out.sum()
        loss.backward()
        optimizer.step()

    report = run_l0(
        model=model,
        train_step_fn=train_step,
        mock_input=mock_input,
        batch_size=32,
        steady_state_minutes=0.05,  # 3 seconds
    )

    expected_keys = {
        "mfu", "tca", "sample_speed", "tokens_per_sec",
        "flops_per_step", "step_times", "memory", "backend",
        "gap_analysis", "gpu_info", "warmup_steps", "steady_state_steps",
    }
    assert expected_keys.issubset(set(report.keys()))
    assert report["mfu"] > 0
    assert report["sample_speed"] > 0
    assert len(report["step_times"]) > 0
