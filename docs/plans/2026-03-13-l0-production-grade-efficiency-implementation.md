# L0 Production-Grade Engineering Efficiency Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use spml:subagent-dev to implement this plan task-by-task. (If subagent-dev is not yet available, use spml:executing-plans.)

**Goal:** Replace toy-level L0 checks with production-grade MFU (FlopCounterMode), TCA (DCGM), and sample speed measurement from a single 3-minute steady-state training run.

**Validation scope:** L0 only. No L1-L3 changes. Tests run without GPU via mocking; CUDA-dependent tests marked `@pytest.mark.skipif`.

**Architecture:** Three-phase L0 runner (static analysis → steady-state measurement → report). FlopCounterMode for precise FLOPs. DCGM subprocess for real TCA. CUDA Events for step timing. All metrics collected simultaneously in one training run.

**Design doc:** `docs/plans/2026-03-13-l0-production-grade-efficiency-design.md`

---

## Shared Scaffold

### Existing infra (don't touch, advise if problems found)
- `toolkit/profiling/layer_profiler.py` — per-layer timing, keep as deep-dive tool
- `toolkit/profiling/memory_profiler.py` — memory analysis, minor adapt only
- `toolkit/profiling/__init__.py` — empty, will add exports
- `skills/vp-engineering-efficiency/` — skill docs, update after toolkit done
- `tests/toolkit/profiling/test_layer_profiler.py`
- `tests/toolkit/profiling/test_memory_profiler.py`

### Needs setup
- Nothing beyond the subtask implementations below

---

## Subtask 1: Rewrite mfu_calculator.py — FlopCounterMode + CUDA Events

**Goal:** Replace 6N estimation with `torch.utils.flop_counter.FlopCounterMode` for precise per-operator FLOPs. Add CUDA Events based steady-state step timing with automatic warmup detection.

**Implementation:** Rewrite `toolkit/profiling/mfu_calculator.py`
**Unit Tests:** `tests/toolkit/profiling/test_mfu_calculator.py` (rewrite)

### Step 1: Write unit tests

File: `tests/toolkit/profiling/test_mfu_calculator.py`

Test the following (all tests that need CUDA marked with `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")`):

**Pure logic tests (no CUDA):**
- `test_gpu_peak_tflops_known_gpus()` — verify GPU_PEAK_TFLOPS dict contains expected entries (H100=989, A100=312, B200=2250, etc.)
- `test_warmup_convergence_detection()` — given a list of step times `[500, 300, 200, 180, 175, 172, 173, 171, 172, 173]`, verify warmup detector identifies convergence at the right index (CV of last 5 < 0.05)
- `test_warmup_no_convergence()` — given wildly varying step times, verify detector returns None / raises
- `test_mfu_computation()` — given known flops=1e12, step_time_s=0.1, peak_tflops=100, verify mfu = 1e12 / (0.1 * 100e12) = 0.1
- `test_mfu_multi_gpu()` — verify peak scales by num_gpus

**CUDA tests:**
- `test_count_flops_linear()` — Linear(128, 256), input (8, 128). FlopCounterMode should report ~2*8*128*256 = 524288 FLOPs for forward
- `test_count_flops_transformer()` — small TransformerEncoderLayer, verify FLOPs > 0 and includes attention + FFN
- `test_count_flops_with_backward()` — verify fwd+bwd FLOPs > forward-only FLOPs
- `test_measure_step_time()` — run a trivial model for 10 steps, verify returns list of positive floats
- `test_calculate_mfu_integration()` — full pipeline: model + step function → returns dict with mfu, flops_per_step, step_time_ms, sample_speed

### Step 2: Run tests to verify they fail

```bash
cd /Users/bytedance/Project/superpowers-ML && python -m pytest tests/toolkit/profiling/test_mfu_calculator.py -v
```
Expected: FAIL (old API doesn't match new tests)

### Step 3: Implement mfu_calculator.py

File: `toolkit/profiling/mfu_calculator.py`

Key functions:

```python
GPU_PEAK_TFLOPS = { ... }  # keep existing dict

def _detect_gpu_peak_tflops() -> float:
    """Auto-detect GPU and return peak bf16 TFLOPS."""
    # keep existing implementation

def count_flops(
    model: nn.Module,
    mock_input: torch.Tensor,
    include_backward: bool = True,
    loss_fn: Optional[Callable] = None,
) -> dict:
    """Count FLOPs using torch.utils.flop_counter.FlopCounterMode.

    Args:
        model: PyTorch model (must be on CUDA)
        mock_input: example input tensor
        include_backward: if True, count fwd+bwd FLOPs
        loss_fn: how to compute loss for backward (default: output.sum())

    Returns:
        dict with: total_flops (int), flops_by_module (dict), flops_by_operator (dict)
    """

def detect_warmup_end(step_times: list[float], window: int = 5, cv_threshold: float = 0.05) -> int:
    """Find the index where step times stabilize.

    Returns index of first step in steady-state, or -1 if not converged.
    Criteria: coefficient of variation of last `window` steps < cv_threshold.
    """

def measure_step_times(
    model: nn.Module,
    train_step_fn: Callable,
    num_steps: int,
) -> list[float]:
    """Measure per-step wall time using CUDA Events.

    Returns list of step_time_ms for each step.
    """

def calculate_mfu(
    flops_per_step: int,
    step_time_ms: float,
    gpu_peak_tflops: Optional[float] = None,
    num_gpus: int = 1,
) -> dict:
    """Calculate MFU from pre-computed FLOPs and measured step time.

    Returns dict with: mfu, actual_tflops, theoretical_tflops, step_time_ms
    """
```

Important implementation notes:
- `count_flops` uses `FlopCounterMode(display=False)` as context manager
- `measure_step_times` uses `torch.cuda.Event(enable_timing=True)` per step
- `detect_warmup_end` slides a window and checks `std/mean < cv_threshold`
- `calculate_mfu` is pure math, no CUDA dependency (testable without GPU)
- Keep `GPU_PEAK_TFLOPS` dict and `_detect_gpu_peak_tflops` from existing code

### Step 4: Run tests to verify they pass

```bash
cd /Users/bytedance/Project/superpowers-ML && python -m pytest tests/toolkit/profiling/test_mfu_calculator.py -v
```
Expected: PASS

### Step 5: Commit

```bash
git add toolkit/profiling/mfu_calculator.py tests/toolkit/profiling/test_mfu_calculator.py
git commit -m "refactor: rewrite mfu_calculator with FlopCounterMode and CUDA Events timing"
```

---

## Subtask 2: New dcgm_profiler.py — Real TCA via DCGM

**Goal:** Manage DCGM subprocess lifecycle (start/stop/parse), extract real TCA% from `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` (field 1004) during normal training with zero overhead.

**Implementation:** New file `toolkit/profiling/dcgm_profiler.py`
**Unit Tests:** `tests/toolkit/profiling/test_dcgm_profiler.py`

### Step 1: Write unit tests

File: `tests/toolkit/profiling/test_dcgm_profiler.py`

**Pure logic tests (no DCGM needed — test parsing/stats logic):**

- `test_parse_dcgm_output_normal()` — given typical dcgmi dmon output string:
  ```
  #Entity   SMACT  TENACT
       0    85.2    23.7
       0    84.1    24.1
       0    83.5    23.9
  ```
  verify returns list of floats `[23.7, 24.1, 23.9]`

- `test_parse_dcgm_output_with_header_lines()` — handle `#` comment lines and blank lines
- `test_parse_dcgm_output_empty()` — returns empty list
- `test_parse_dcgm_output_not_available()` — handle "N/A" values gracefully (skip them)
- `test_compute_tca_stats()` — given `[20.0, 22.0, 24.0, 23.0, 21.0]`, verify returns dict with mean=22.0, median=22.0, std≈1.58, min=20.0, max=24.0
- `test_compute_tca_stats_empty()` — returns all None/NaN
- `test_trim_warmup_samples()` — given 100 samples and warmup_end_idx corresponding to a timestamp, verify only steady-state samples remain
- `test_check_dcgm_available()` — mock subprocess to test both available/unavailable paths

### Step 2: Run tests to verify they fail

```bash
cd /Users/bytedance/Project/superpowers-ML && python -m pytest tests/toolkit/profiling/test_dcgm_profiler.py -v
```
Expected: FAIL (module doesn't exist)

### Step 3: Implement dcgm_profiler.py

File: `toolkit/profiling/dcgm_profiler.py`

Key functions:

```python
def check_dcgm_available() -> bool:
    """Check if dcgmi is available on this system."""

def start_dcgm_sampling(
    interval_ms: int = 100,
    gpu_id: int = 0,
) -> subprocess.Popen:
    """Start DCGM background sampling process.

    Runs: dcgmi dmon -e 1004 -d {interval_ms // 1000 or 1} -i {gpu_id}
    Field 1004 = DCGM_FI_PROF_PIPE_TENSOR_ACTIVE (tensor core activity %)

    Returns the subprocess handle. Caller must call stop_dcgm_sampling() when done.
    """

def stop_dcgm_sampling(proc: subprocess.Popen) -> str:
    """Stop DCGM sampling and return captured stdout."""

def parse_dcgm_output(output: str) -> list[float]:
    """Parse dcgmi dmon output, extract TENACT (tensor activity) column.

    Handles:
    - Comment lines starting with #
    - Blank lines
    - N/A values (skipped)
    - Multiple entity columns (picks gpu_id=0)

    Returns list of TCA% values as floats.
    """

def compute_tca_stats(tca_values: list[float]) -> dict:
    """Compute statistics over TCA samples.

    Returns dict with: mean, median, std, min, max, num_samples
    """

def trim_warmup_samples(
    tca_values: list[float],
    interval_ms: int,
    warmup_duration_s: float,
) -> list[float]:
    """Remove samples that fall within the warmup period."""
```

Implementation notes:
- `dcgmi dmon -e 1004` outputs at 1-second minimum interval. Use `-d 1` (1 second).
- Parse by splitting lines, finding the TENACT column index from header
- All parsing is pure Python string processing, fully testable without DCGM

### Step 4: Run tests to verify they pass

```bash
cd /Users/bytedance/Project/superpowers-ML && python -m pytest tests/toolkit/profiling/test_dcgm_profiler.py -v
```
Expected: PASS

### Step 5: Commit

```bash
git add toolkit/profiling/dcgm_profiler.py tests/toolkit/profiling/test_dcgm_profiler.py
git commit -m "feat: add dcgm_profiler for real TCA measurement via DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"
```

---

## Subtask 3: New gap_analyzer.py — TCA vs MFU Gap Analysis

**Goal:** Analyze the gap between TCA and MFU, report possible contributors qualitatively.

**Implementation:** New file `toolkit/profiling/gap_analyzer.py`
**Unit Tests:** `tests/toolkit/profiling/test_gap_analyzer.py`

### Step 1: Write unit tests

File: `tests/toolkit/profiling/test_gap_analyzer.py`

**All pure logic tests (no CUDA/DCGM):**

- `test_gap_basic()` — TCA=23.7%, MFU=14.6% → gap=9.1pp, contributors non-empty
- `test_gap_tca_equals_mfu()` — TCA=30%, MFU=30% → gap=0, no contributors
- `test_gap_low_mfu_high_tca()` — TCA=50%, MFU=10% → large gap, flags possible memory-bound + padding
- `test_gap_report_format()` — verify returns dict with expected keys: gap_pp, tca, mfu, contributors (list of strings)
- `test_gap_with_clock_info()` — if actual_clock_mhz and max_clock_mhz provided, include clock_factor in report

### Step 2: Run tests to verify they fail

```bash
cd /Users/bytedance/Project/superpowers-ML && python -m pytest tests/toolkit/profiling/test_gap_analyzer.py -v
```
Expected: FAIL

### Step 3: Implement gap_analyzer.py

File: `toolkit/profiling/gap_analyzer.py`

```python
def analyze_gap(
    tca_percent: float,
    mfu_percent: float,
    actual_clock_mhz: Optional[float] = None,
    max_clock_mhz: Optional[float] = None,
) -> dict:
    """Analyze the gap between TCA and MFU.

    TCA measures tensor core activity (% of time tensor cores are computing).
    MFU measures useful model FLOPs as fraction of peak.
    The gap (TCA > MFU) can be caused by:
    - Clock throttling: GPU not running at max frequency
    - Tensor overhead: tile padding, sub-optimal GEMM shapes
    - Memory-bound ops: tensor cores active but starved for data

    Returns dict with:
        gap_pp: gap in percentage points
        tca: input TCA%
        mfu: input MFU%
        clock_factor: actual/max clock ratio (if provided)
        contributors: list of human-readable strings describing likely gap causes
    """
```

Implementation notes:
- Pure math + heuristics, zero dependencies beyond Python stdlib
- Contributors are qualitative: "Clock throttling: GPU running at {actual}/{max} MHz ({ratio:.0%} of max)"
- If gap > 5pp and no clock info: suggest "Run NCU deep-dive for per-kernel analysis"

### Step 4: Run tests to verify they pass

```bash
cd /Users/bytedance/Project/superpowers-ML && python -m pytest tests/toolkit/profiling/test_gap_analyzer.py -v
```
Expected: PASS

### Step 5: Commit

```bash
git add toolkit/profiling/gap_analyzer.py tests/toolkit/profiling/test_gap_analyzer.py
git commit -m "feat: add gap_analyzer for TCA vs MFU gap decomposition"
```

---

## Subtask 4: New l0_runner.py — Three-Phase Orchestration + Report

**Goal:** Orchestrate the full L0 flow: static analysis → steady-state measurement → report. Single entry point `run_l0()`.

**Implementation:** New file `toolkit/profiling/l0_runner.py`
**Unit Tests:** `tests/toolkit/profiling/test_l0_runner.py`

### Step 1: Write unit tests

File: `tests/toolkit/profiling/test_l0_runner.py`

**Pure logic tests:**

- `test_l0_report_keys()` — verify report dict contains all expected keys: mfu, tca, sample_speed, flops_per_step, step_time_ms, memory, backend, gap_analysis
- `test_l0_report_summary_table()` — verify `format_summary_table(report)` returns a formatted string containing MFU, TCA, sample speed lines
- `test_phase1_static_analysis_output()` — mock FlopCounterMode, verify returns flops_per_step + module breakdown
- `test_phase2_collects_all_metrics()` — mock train_step_fn + DCGM, verify output has step_times, tca_samples, memory_stats
- `test_phase3_computes_correct_mfu()` — given known phase1 + phase2 outputs, verify MFU calculation correct
- `test_dcgm_unavailable_graceful()` — when DCGM not available, TCA reported as None with warning message

**CUDA integration test:**
- `test_run_l0_smoke()` — trivial model + step fn, steady_state_minutes=0.05 (3 seconds), verify returns valid report

### Step 2: Run tests to verify they fail

```bash
cd /Users/bytedance/Project/superpowers-ML && python -m pytest tests/toolkit/profiling/test_l0_runner.py -v
```
Expected: FAIL

### Step 3: Implement l0_runner.py

File: `toolkit/profiling/l0_runner.py`

```python
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

    Three phases:
    1. Static analysis: FlopCounterMode FLOPs, GPU info, backend pre-check, memory snapshot
    2. Steady-state measurement: warmup detection, CUDA Events timing, DCGM TCA sampling
    3. Report generation: MFU, TCA, sample speed, gap analysis, memory report

    Args:
        model: PyTorch model (on CUDA)
        train_step_fn: function(model, step_idx) that runs one training step
        mock_input: example input for FlopCounterMode (static FLOPs counting)
        batch_size: for sample speed calculation
        seq_len: for tokens/sec calculation (optional)
        steady_state_minutes: how long to measure during steady-state
        gpu_peak_tflops: override GPU peak (auto-detected if None)
        num_gpus: number of GPUs
        dcgm_interval_ms: DCGM sampling interval (minimum 1000ms)
        loss_fn: for FlopCounterMode backward pass

    Returns:
        dict with all L0 metrics + raw data
    """

def format_summary_table(report: dict) -> str:
    """Format L0 report as human-readable table."""
```

Implementation flow inside `run_l0`:

```python
# Phase 1: Static Analysis
flops_result = count_flops(model, mock_input, include_backward=True, loss_fn=loss_fn)
gpu_info = _collect_gpu_info(gpu_peak_tflops, num_gpus)
backend_info = _check_backend(model, mock_input)
memory_snapshot = analyze_memory(model, mock_input)

# Phase 2: Steady-State Measurement
# 2a. Start DCGM (if available)
dcgm_proc = None
dcgm_available = check_dcgm_available()
if dcgm_available:
    dcgm_proc = start_dcgm_sampling(interval_ms=dcgm_interval_ms)
dcgm_start_time = time.time()

# 2b. Run training steps with CUDA Events timing
step_times = []
total_steps = 0
warmup_end = -1

while True:
    # Measure one step
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
    if warmup_end >= 0:
        elapsed = time.time() - steady_start_time
        if elapsed >= steady_state_minutes * 60:
            break

# 2c. Stop DCGM, parse
tca_stats = None
if dcgm_proc:
    output = stop_dcgm_sampling(dcgm_proc)
    tca_values = parse_dcgm_output(output)
    warmup_duration = steady_start_time - dcgm_start_time
    tca_steady = trim_warmup_samples(tca_values, dcgm_interval_ms, warmup_duration)
    tca_stats = compute_tca_stats(tca_steady)

# 2d. Backend verification (1 step with profiler)
kernel_info = _profile_kernels(model, train_step_fn, total_steps)

# Phase 3: Report
steady_step_times = step_times[warmup_end:]
median_step_ms = sorted(steady_step_times)[len(steady_step_times) // 2]
mfu_result = calculate_mfu(flops_result['total_flops'], median_step_ms, ...)
sample_speed = batch_size / (median_step_ms / 1000)
gap = analyze_gap(tca_stats['mean'] if tca_stats else None, mfu_result['mfu'] * 100, ...)
```

Helper functions (private, in same file):
- `_collect_gpu_info()` — GPU name, peak TFLOPS, memory, DCGM availability
- `_check_backend()` — param dtypes, FA check via 1-step profiler
- `_profile_kernels()` — torch.profiler for 1 step, extract kernel names, classify

### Step 4: Run tests to verify they pass

```bash
cd /Users/bytedance/Project/superpowers-ML && python -m pytest tests/toolkit/profiling/test_l0_runner.py -v
```
Expected: PASS

### Step 5: Commit

```bash
git add toolkit/profiling/l0_runner.py tests/toolkit/profiling/test_l0_runner.py
git commit -m "feat: add l0_runner for three-phase production-grade L0 orchestration"
```

---

## Subtask 5: Update skill docs + __init__.py exports

**Goal:** Update vp-engineering-efficiency skill to reference new toolkit. Update `__init__.py` exports.

**Implementation:** Edit skill markdown files + `toolkit/profiling/__init__.py`

### Step 1: Update `toolkit/profiling/__init__.py`

```python
from toolkit.profiling.mfu_calculator import count_flops, calculate_mfu, measure_step_times, detect_warmup_end, GPU_PEAK_TFLOPS
from toolkit.profiling.dcgm_profiler import check_dcgm_available, start_dcgm_sampling, stop_dcgm_sampling, parse_dcgm_output, compute_tca_stats
from toolkit.profiling.gap_analyzer import analyze_gap
from toolkit.profiling.l0_runner import run_l0, format_summary_table
from toolkit.profiling.layer_profiler import profile_layers
from toolkit.profiling.memory_profiler import analyze_memory
```

### Step 2: Update skill docs

**`skills/vp-engineering-efficiency/SKILL.md`** — rewrite to reflect three-phase L0:
- Phase 1: static analysis (FlopCounterMode, GPU info, backend, memory)
- Phase 2: steady-state measurement (3 min training, CUDA Events, DCGM)
- Phase 3: report (MFU, TCA, sample speed, gap analysis)
- Remove old "toy" measurement guidance
- Reference `run_l0()` as the primary entry point

**`skills/vp-engineering-efficiency/gpu-utilization.md`** — rewrite:
- Replace `6N` estimation with FlopCounterMode usage
- Replace `tca = mfu` with DCGM-based TCA
- Add gap analysis section
- Update code examples to use new API

### Step 3: Run all tests

```bash
cd /Users/bytedance/Project/superpowers-ML && python -m pytest tests/toolkit/profiling/ -v
```
Expected: ALL PASS

### Step 4: Commit

```bash
git add toolkit/profiling/__init__.py skills/vp-engineering-efficiency/
git commit -m "docs: update L0 skill docs and exports for production-grade efficiency toolkit"
```

---

## Subtask 6 (Optional): ncu_profiler.py — NCU Deep-Dive

**Goal:** Optional deep-dive tool for per-kernel TCA analysis and quantitative gap decomposition. Not part of standard L0 flow.

**Implementation:** New file `toolkit/profiling/ncu_profiler.py`
**Unit Tests:** `tests/toolkit/profiling/test_ncu_profiler.py`

### Step 1: Write unit tests

- `test_parse_ncu_csv()` — parse NCU CSV output, extract per-kernel metrics
- `test_compute_per_kernel_tca()` — given tensor_active and cycles_elapsed, verify TCA = tensor_active / (cycles_elapsed * 4) * 100
- `test_classify_kernel()` — flash_fwd → FA2 HMMA, nvjet_tst → WGMMA, triton_ → CUDA Core
- `test_build_ncu_command()` — verify correct ncu CLI args

### Step 2-5: Implement, test, commit

Same TDD rhythm as above. Key functions:

```python
def check_ncu_available() -> bool
def build_ncu_command(script, args, launch_skip=10, launch_count=5) -> list[str]
def run_ncu_profiling(script, args, ...) -> str  # returns CSV output
def parse_ncu_csv(csv_text) -> list[dict]  # per-kernel metrics
def compute_per_kernel_tca(tensor_active, cycles_elapsed, sub_cores=4) -> float
def classify_kernel(kernel_name) -> str  # FA2/WGMMA/CUDA Core/Other
def analyze_ncu_results(kernels) -> dict  # aggregate per-category stats
```

```bash
git commit -m "feat: add optional ncu_profiler for per-kernel TCA deep-dive"
```
