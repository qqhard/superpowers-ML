"""Tests for toolkit/profiling/gap_analyzer.py"""
import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from toolkit.profiling.gap_analyzer import analyze_gap


def test_gap_basic():
    result = analyze_gap(tca_percent=23.7, mfu_percent=14.6)
    assert abs(result["gap_pp"] - 9.1) < 0.01
    assert isinstance(result["contributors"], list)
    assert len(result["contributors"]) > 0


def test_gap_tca_equals_mfu():
    result = analyze_gap(tca_percent=30.0, mfu_percent=30.0)
    assert result["gap_pp"] == 0.0
    assert len(result["contributors"]) == 0


def test_gap_low_mfu_high_tca():
    result = analyze_gap(tca_percent=50.0, mfu_percent=10.0)
    assert abs(result["gap_pp"] - 40.0) < 0.01
    assert len(result["contributors"]) > 0


def test_gap_report_keys():
    result = analyze_gap(tca_percent=23.7, mfu_percent=14.6)
    for key in ("gap_pp", "tca", "mfu", "contributors"):
        assert key in result


def test_gap_with_clock_info():
    result = analyze_gap(
        tca_percent=23.7, mfu_percent=14.6,
        actual_clock_mhz=1761, max_clock_mhz=2550,
    )
    assert abs(result["clock_factor"] - (1761 / 2550)) < 0.01
    assert any("clock" in c.lower() or "throttl" in c.lower() for c in result["contributors"])


def test_gap_no_clock_info():
    result = analyze_gap(tca_percent=23.7, mfu_percent=14.6)
    assert result["clock_factor"] is None


def test_gap_tca_none():
    result = analyze_gap(tca_percent=None, mfu_percent=14.6)
    assert result["gap_pp"] is None
    assert any("TCA unavailable" in c for c in result["contributors"])
