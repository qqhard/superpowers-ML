"""Tests for toolkit/profiling/dcgm_profiler.py"""
import sys, os, pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from toolkit.profiling.dcgm_profiler import (
    parse_dcgm_output,
    compute_tca_stats,
    trim_warmup_samples,
    check_dcgm_available,
)


class TestParseDcgmOutput:
    def test_parse_dcgm_output_normal(self):
        output = (
            "#Entity   SMACT  TENACT\n"
            "     0    85.2    23.7\n"
            "     0    84.1    24.1\n"
            "     0    83.5    23.9\n"
        )
        result = parse_dcgm_output(output)
        assert result == [23.7, 24.1, 23.9]

    def test_parse_dcgm_output_with_comments_and_blanks(self):
        output = (
            "# Some header comment\n"
            "#Entity   SMACT  TENACT\n"
            "\n"
            "     0    85.2    23.7\n"
            "\n"
            "     0    84.1    24.1\n"
            "# Another comment\n"
            "     0    83.5    23.9\n"
        )
        result = parse_dcgm_output(output)
        assert result == [23.7, 24.1, 23.9]

    def test_parse_dcgm_output_empty(self):
        assert parse_dcgm_output("") == []
        assert parse_dcgm_output("   ") == []

    def test_parse_dcgm_output_na_values(self):
        output = (
            "#Entity   SMACT  TENACT\n"
            "     0    85.2    N/A\n"
            "     0    84.1    24.1\n"
        )
        result = parse_dcgm_output(output)
        assert result == [24.1]


class TestComputeTcaStats:
    def test_compute_tca_stats(self):
        values = [20.0, 22.0, 24.0, 23.0, 21.0]
        stats = compute_tca_stats(values)
        assert stats["mean"] == 22.0
        assert stats["median"] == 22.0
        assert stats["min"] == 20.0
        assert stats["max"] == 24.0
        assert stats["num_samples"] == 5
        assert abs(stats["std"] - 1.5811388300841898) < 0.01

    def test_compute_tca_stats_single(self):
        stats = compute_tca_stats([25.0])
        assert stats["mean"] == 25.0
        assert stats["std"] == 0.0
        assert stats["num_samples"] == 1

    def test_compute_tca_stats_empty(self):
        stats = compute_tca_stats([])
        assert stats["mean"] is None
        assert stats["median"] is None
        assert stats["std"] is None
        assert stats["min"] is None
        assert stats["max"] is None
        assert stats["num_samples"] == 0


class TestTrimWarmupSamples:
    def test_trim_warmup_samples(self):
        samples = list(range(20))  # 20 samples
        result = trim_warmup_samples(samples, interval_ms=1000, warmup_duration_s=5.0)
        assert len(result) == 15
        assert result[0] == 5

    def test_trim_warmup_samples_zero(self):
        samples = list(range(10))
        result = trim_warmup_samples(samples, interval_ms=1000, warmup_duration_s=0)
        assert len(result) == 10
        assert result == list(range(10))


class TestCheckDcgmAvailable:
    @patch("toolkit.profiling.dcgm_profiler.subprocess.run")
    def test_check_dcgm_available_true(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assert check_dcgm_available() is True

    @patch("toolkit.profiling.dcgm_profiler.subprocess.run")
    def test_check_dcgm_available_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        assert check_dcgm_available() is False
