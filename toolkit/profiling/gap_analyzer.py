"""TCA vs MFU Gap Analyzer.

Analyzes the gap between Tensor Core Activity (TCA) and Model FLOPs Utilization (MFU),
identifying likely contributors qualitatively.

Usage:
    from toolkit.profiling.gap_analyzer import analyze_gap
    result = analyze_gap(tca_percent=23.7, mfu_percent=14.6)
"""
from typing import Optional


def analyze_gap(
    tca_percent: Optional[float],
    mfu_percent: float,
    actual_clock_mhz: Optional[float] = None,
    max_clock_mhz: Optional[float] = None,
) -> dict:
    """Analyze the gap between TCA and MFU.

    TCA measures tensor core activity (% of time tensor cores are computing).
    MFU measures useful model FLOPs as fraction of peak.
    Gap (TCA > MFU) can be caused by:
    - Clock throttling: GPU not running at max frequency
    - Tensor overhead: tile padding, sub-optimal GEMM shapes
    - Memory-bound ops: tensor cores active but starved for data

    Args:
        tca_percent: TCA in percent (e.g., 23.7 means 23.7%), or None if unavailable
        mfu_percent: MFU in percent (e.g., 14.6 means 14.6%)
        actual_clock_mhz: actual GPU clock during measurement (optional)
        max_clock_mhz: max GPU clock (optional)

    Returns:
        dict with: gap_pp, tca, mfu, clock_factor, contributors (list of strings)
    """
    contributors = []
    clock_factor = None

    if tca_percent is None:
        return {
            "gap_pp": None,
            "tca": None,
            "mfu": mfu_percent,
            "clock_factor": None,
            "contributors": ["TCA unavailable — install DCGM for real tensor core activity measurement"],
        }

    gap_pp = tca_percent - mfu_percent

    # Clock throttling
    if actual_clock_mhz is not None and max_clock_mhz is not None and max_clock_mhz > 0:
        clock_factor = actual_clock_mhz / max_clock_mhz
        if clock_factor < 0.95:
            contributors.append(
                f"Clock throttling: GPU at {actual_clock_mhz:.0f}/{max_clock_mhz:.0f} MHz "
                f"({clock_factor:.0%} of max) — power or thermal throttling likely"
            )

    # Large gap heuristics
    if gap_pp > 15:
        contributors.append(
            "Large TCA-MFU gap suggests significant tensor overhead "
            "(tile padding, sub-optimal GEMM shapes) or memory-bound operations"
        )
    elif gap_pp > 5:
        contributors.append(
            "Moderate TCA-MFU gap — possible tensor overhead or memory-bound operations"
        )

    if gap_pp > 5 and clock_factor is None:
        contributors.append(
            "Consider running NCU deep-dive for per-kernel analysis to identify specific bottlenecks"
        )

    return {
        "gap_pp": gap_pp,
        "tca": tca_percent,
        "mfu": mfu_percent,
        "clock_factor": clock_factor,
        "contributors": contributors,
    }
