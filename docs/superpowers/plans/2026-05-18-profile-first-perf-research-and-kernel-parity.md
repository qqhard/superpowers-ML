# Profile-First Perf Research + Kernel Parity Guardrail — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two orthogonal guardrails to `autoresearch` and `ml-iteration` — (A) Profile-first discipline for perf-mode research and (B) Kernel I/O parity check for custom-kernel R&D — wired through `ml-brainstorming` (collection) and the two `*-handoff` skills (pre-validation).

**Architecture:** A single new carrier script (`kernel_parity.py`) implements the parity check. Two new primitive docs (`profile-first.md`, `kernel-parity.md`) describe the patterns. Six existing skill files (`ml-brainstorming`, `autoresearch`, `ml-iteration`, `autoresearch-handoff`, `training-handoff`) get targeted insertions for question collection, prompt augmentation, compliance sub-step, and handoff pre-checks. All changes are additive — accuracy research and non-kernel work see no behavior change.

**Tech Stack:** Python 3 + torch + PyYAML for `kernel_parity.py`; bash for the smoke test harness; markdown for skill files; existing `tests/ml-iteration/`-style end-to-end test pattern.

**Spec reference:** `docs/superpowers/specs/2026-05-17-profile-first-perf-research-and-kernel-parity-design.md` (commit `f3355b2`).

**Invocation-path note (spec correction):** The spec wrote `python -m _ml_loop_primitives.kernel_parity` but the directory is `_ml-loop-primitives` (hyphen), not a valid Python module. The implementation invokes the script by **absolute file path** instead: `python <SKILL_ROOT>/_ml-loop-primitives/kernel_parity.py …`. Supervisor derives `<SKILL_ROOT>` from the SKILL.md path it was loaded from. The primitive doc carries the exact discovery snippet.

---

## File Structure

**New files:**

```
skills/_ml-loop-primitives/profile-first.md           # primitive doc (Researcher prompt template)
skills/_ml-loop-primitives/kernel-parity.md           # primitive doc (Supervisor compliance protocol)
skills/_ml-loop-primitives/kernel_parity.py           # carrier script (CLI; torch + yaml)
tests/kernel-parity/run-test.sh                       # smoke test harness
tests/kernel-parity/fixtures/                         # fixture project for the harness
  ├── baseline_softmax.py                             # baseline kernel ref
  ├── new_softmax_ok.py                               # new kernel that matches
  ├── new_softmax_sig_bad.py                          # new kernel with wrong signature
  ├── new_softmax_shape_bad.py                        # new kernel returning wrong shape
  ├── new_softmax_num_bad.py                          # new kernel with numerical drift
  └── fixture.py                                      # fixture() returning (args, kwargs)
```

**Modified files:**

```
skills/ml-brainstorming/SKILL.md                      # +Q3.5 +Q3.6 +Q4 +Q4.x; design-doc schema
skills/autoresearch/SKILL.md                          # +perf prompt section; Step 2 -> 2a/2b; Shared Patterns; S2 task string
skills/ml-iteration/SKILL.md                          # same edits mirrored
skills/autoresearch-handoff/SKILL.md                  # +Pre-check A +Pre-check B; protocol template fields
skills/training-handoff/SKILL.md                      # iteration branch: +Pre-check A +Pre-check B; protocol template fields
.claude-plugin/plugin.json                            # version bump
.cursor-plugin/plugin.json                            # version bump
.claude-plugin/marketplace.json                       # version bump
gemini-extension.json                                 # version bump
RELEASE-NOTES.md                                      # entry for this release
```

**Dependency order:** Task 1 (script) → Tasks 2–3 (primitive docs referencing script) → Task 4–5 (brainstorming) → Tasks 6–7 (autoresearch) → Tasks 8–9 (ml-iteration) → Tasks 10–11 (autoresearch-handoff) → Tasks 12–13 (training-handoff) → Task 14 (version bump + release notes) → Task 15 (plugin cache sync + smoke test).

---

## Task 1: Build `kernel_parity.py` carrier script with smoke tests

**Files:**
- Create: `skills/_ml-loop-primitives/kernel_parity.py`
- Create: `tests/kernel-parity/run-test.sh`
- Create: `tests/kernel-parity/fixtures/baseline_softmax.py`
- Create: `tests/kernel-parity/fixtures/new_softmax_ok.py`
- Create: `tests/kernel-parity/fixtures/new_softmax_sig_bad.py`
- Create: `tests/kernel-parity/fixtures/new_softmax_shape_bad.py`
- Create: `tests/kernel-parity/fixtures/new_softmax_num_bad.py`
- Create: `tests/kernel-parity/fixtures/fixture.py`
- Create: `tests/kernel-parity/fixtures/protocol_ok.yaml`
- Create: `tests/kernel-parity/fixtures/protocol_sig_bad.yaml`
- Create: `tests/kernel-parity/fixtures/protocol_shape_bad.yaml`
- Create: `tests/kernel-parity/fixtures/protocol_num_bad.yaml`

- [ ] **Step 1.1: Write the failing smoke-test harness**

Create `tests/kernel-parity/run-test.sh`:

```bash
#!/usr/bin/env bash
# Smoke test for kernel_parity.py — exercises pass + 3 failure modes.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARITY="$SCRIPT_DIR/../../skills/_ml-loop-primitives/kernel_parity.py"
FIXTURES="$SCRIPT_DIR/fixtures"

assert_exit() {
  local label="$1" expected="$2" actual="$3"
  if [[ "$expected" == "$actual" ]]; then
    echo "  PASS: $label (exit=$actual)"
  else
    echo "  FAIL: $label (expected exit=$expected, got $actual)"
    exit 1
  fi
}

run_case() {
  local label="$1" protocol="$2" expected="$3"
  echo "Case: $label"
  cd "$FIXTURES"
  python3 "$PARITY" --protocol "$FIXTURES/$protocol" >/tmp/parity-stdout 2>/tmp/parity-stderr
  local actual=$?
  assert_exit "$label" "$expected" "$actual"
  if [[ "$expected" != "0" ]]; then
    echo "    stderr: $(head -1 /tmp/parity-stderr)"
  fi
}

run_case "all-pass"        "protocol_ok.yaml"         "0"
run_case "signature-bad"   "protocol_sig_bad.yaml"    "1"
run_case "shape-bad"       "protocol_shape_bad.yaml"  "1"
run_case "numerical-bad"   "protocol_num_bad.yaml"    "1"

echo "All cases passed."
```

Make executable: `chmod +x tests/kernel-parity/run-test.sh`

- [ ] **Step 1.2: Write fixture Python files**

Create `tests/kernel-parity/fixtures/baseline_softmax.py`:

```python
import torch

def softmax_ref(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x, dim=dim)
```

Create `tests/kernel-parity/fixtures/new_softmax_ok.py`:

```python
import torch

def fused_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Numerically equivalent to baseline (within fp32 noise)
    return torch.softmax(x, dim=dim)
```

Create `tests/kernel-parity/fixtures/new_softmax_sig_bad.py`:

```python
import torch

def fused_softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    # Wrong parameter name: 'axis' instead of 'dim'
    return torch.softmax(x, dim=axis)
```

Create `tests/kernel-parity/fixtures/new_softmax_shape_bad.py`:

```python
import torch

def fused_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Returns sum over last dim — wrong shape
    return torch.softmax(x, dim=dim).sum(dim=dim, keepdim=False)
```

Create `tests/kernel-parity/fixtures/new_softmax_num_bad.py`:

```python
import torch

def fused_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Off-by-constant — fails numerical tolerance
    return torch.softmax(x, dim=dim) + 0.05
```

Create `tests/kernel-parity/fixtures/fixture.py`:

```python
import torch

def softmax_inputs():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    return (x,), {"dim": -1}
```

- [ ] **Step 1.3: Write protocol fixture YAMLs**

Create `tests/kernel-parity/fixtures/protocol_ok.yaml`:

```yaml
kernel_targets:
  - name: fused_softmax
    new_module: new_softmax_ok:fused_softmax
    baseline_module: baseline_softmax:softmax_ref
    fixture: fixture:softmax_inputs
    tolerance: { atol: 1e-3, rtol: 1e-3 }
```

Create `tests/kernel-parity/fixtures/protocol_sig_bad.yaml`:

```yaml
kernel_targets:
  - name: fused_softmax
    new_module: new_softmax_sig_bad:fused_softmax
    baseline_module: baseline_softmax:softmax_ref
    fixture: fixture:softmax_inputs
    tolerance: { atol: 1e-3, rtol: 1e-3 }
```

Create `tests/kernel-parity/fixtures/protocol_shape_bad.yaml`:

```yaml
kernel_targets:
  - name: fused_softmax
    new_module: new_softmax_shape_bad:fused_softmax
    baseline_module: baseline_softmax:softmax_ref
    fixture: fixture:softmax_inputs
    tolerance: { atol: 1e-3, rtol: 1e-3 }
```

Create `tests/kernel-parity/fixtures/protocol_num_bad.yaml`:

```yaml
kernel_targets:
  - name: fused_softmax
    new_module: new_softmax_num_bad:fused_softmax
    baseline_module: baseline_softmax:softmax_ref
    fixture: fixture:softmax_inputs
    tolerance: { atol: 1e-3, rtol: 1e-3 }
```

(Module references use `module:callable` form — this avoids ambiguity with dotted package paths in user protocols.)

- [ ] **Step 1.4: Run the smoke test to verify it fails (no script yet)**

Run: `bash tests/kernel-parity/run-test.sh`

Expected: FAIL with `python3: can't open file '.../kernel_parity.py'` (script does not exist).

- [ ] **Step 1.5: Implement `kernel_parity.py`**

Create `skills/_ml-loop-primitives/kernel_parity.py`:

```python
#!/usr/bin/env python3
"""Kernel parity guardrail — invoked by autoresearch / ml-iteration Supervisor.

Reads kernel_targets from a YAML/markdown protocol file and verifies, for each
target, that the new kernel matches the baseline kernel on:
  1. inspect.signature
  2. output pytree structure (shape + dtype, recursively)
  3. torch.testing.assert_close with the declared tolerance

Exits 0 if all targets pass; exits 1 on first failure. Failure details go to
stderr in the form:
  PARITY_FAIL target=<name> kind=<signature|shape|dtype|numerical> detail=<...>
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


def _load_callable(spec: str, search_path: Path) -> Any:
    """Resolve 'module:attr' or 'pkg.mod:attr' against search_path.

    Search path is prepended to sys.path for the duration of the import so that
    user-provided modules in the experiment dir are reachable without packaging.
    """
    if ":" not in spec:
        raise ValueError(f"expected 'module:attr', got {spec!r}")
    mod_spec, attr = spec.split(":", 1)
    sys.path.insert(0, str(search_path))
    try:
        mod = importlib.import_module(mod_spec)
    finally:
        sys.path.pop(0)
    return getattr(mod, attr)


def _parse_protocol(path: Path) -> list[dict]:
    """Load kernel_targets block from YAML or YAML-front markdown.

    Accepts either a pure .yaml file or a markdown file that contains a
    fenced ```yaml block — Supervisor protocols are markdown today but the
    iteration-protocol form is YAML frontmatter. We do the simplest robust
    parse: try YAML first, then scan for the first ```yaml fence.
    """
    text = path.read_text()
    try:
        doc = yaml.safe_load(text)
        if isinstance(doc, dict) and "kernel_targets" in doc:
            return doc["kernel_targets"] or []
    except yaml.YAMLError:
        pass

    in_fence = False
    fence_lines: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("```yaml"):
            in_fence = True
            continue
        if in_fence and line.strip().startswith("```"):
            break
        if in_fence:
            fence_lines.append(line)
    if fence_lines:
        doc = yaml.safe_load("\n".join(fence_lines)) or {}
        return doc.get("kernel_targets") or []
    return []


def _structures_match(a: Any, b: Any) -> tuple[bool, str]:
    if type(a) is not type(b):
        return False, f"type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape:
            return False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
        if a.dtype != b.dtype:
            return False, f"dtype {a.dtype} vs {b.dtype}"
        return True, ""
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False, f"length {len(a)} vs {len(b)}"
        for ai, bi in zip(a, b):
            ok, why = _structures_match(ai, bi)
            if not ok:
                return False, why
        return True, ""
    if isinstance(a, dict):
        if set(a) != set(b):
            return False, f"keys {sorted(a)} vs {sorted(b)}"
        for k in a:
            ok, why = _structures_match(a[k], b[k])
            if not ok:
                return False, f"{k}: {why}"
        return True, ""
    return True, ""


def _check_target(target: dict, search_path: Path) -> tuple[bool, str]:
    name = target.get("name", "<unnamed>")
    new_fn = _load_callable(target["new_module"], search_path)
    base_fn = _load_callable(target["baseline_module"], search_path)

    new_sig = inspect.signature(new_fn)
    base_sig = inspect.signature(base_fn)
    if str(new_sig) != str(base_sig):
        return False, (
            f"PARITY_FAIL target={name} kind=signature "
            f"detail=new{new_sig} vs baseline{base_sig}"
        )

    fixture = _load_callable(target["fixture"], search_path)
    args, kwargs = fixture()

    out_new = new_fn(*args, **kwargs)
    out_base = base_fn(*args, **kwargs)

    ok, why = _structures_match(out_new, out_base)
    if not ok:
        return False, f"PARITY_FAIL target={name} kind=shape detail={why}"

    tol = target.get("tolerance") or {}
    atol = float(tol.get("atol", 1e-3))
    rtol = float(tol.get("rtol", 1e-3))
    try:
        torch.testing.assert_close(out_new, out_base, atol=atol, rtol=rtol)
    except AssertionError as e:
        return False, (
            f"PARITY_FAIL target={name} kind=numerical "
            f"detail={str(e).splitlines()[0][:200]}"
        )

    return True, f"PARITY_OK target={name}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument(
        "--search-path",
        type=Path,
        default=None,
        help="Where to resolve user-provided modules from. Defaults to the "
             "protocol's parent directory.",
    )
    args = parser.parse_args()

    targets = _parse_protocol(args.protocol)
    if not targets:
        print("PARITY_SKIP no kernel_targets in protocol", file=sys.stderr)
        return 0

    search_path = (args.search_path or args.protocol.parent).resolve()

    for target in targets:
        ok, msg = _check_target(target, search_path)
        if ok:
            print(msg)
        else:
            print(msg, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Make executable: `chmod +x skills/_ml-loop-primitives/kernel_parity.py`

- [ ] **Step 1.6: Run the smoke test to verify it passes**

Run: `bash tests/kernel-parity/run-test.sh`

Expected output:

```
Case: all-pass
  PASS: all-pass (exit=0)
Case: signature-bad
  PASS: signature-bad (exit=1)
    stderr: PARITY_FAIL target=fused_softmax kind=signature ...
Case: shape-bad
  PASS: shape-bad (exit=1)
    stderr: PARITY_FAIL target=fused_softmax kind=shape ...
Case: numerical-bad
  PASS: numerical-bad (exit=1)
    stderr: PARITY_FAIL target=fused_softmax kind=numerical ...
All cases passed.
```

If any case fails, fix the script (not the test) until all four exit codes match.

- [ ] **Step 1.7: Commit**

```bash
git add skills/_ml-loop-primitives/kernel_parity.py tests/kernel-parity/
git commit -m "feat(kernel-parity): carrier script with 4-case smoke test

Reads kernel_targets from a YAML or YAML-fenced markdown protocol; for each
target verifies inspect.signature equality, output pytree structure (shape +
dtype), and torch.testing.assert_close within declared atol/rtol. Exits 0 on
full pass, 1 on first failure with structured stderr (PARITY_FAIL target=...
kind=... detail=...).

Smoke harness exercises pass / signature-bad / shape-bad / numerical-bad."
```

---

## Task 2: Add `profile-first.md` primitive doc

**Files:**
- Create: `skills/_ml-loop-primitives/profile-first.md`

- [ ] **Step 2.1: Write the primitive doc**

Create `skills/_ml-loop-primitives/profile-first.md`:

```markdown
# Primitive — Profile-First Researcher Discipline

Pattern used by `autoresearch` and `ml-iteration` when the experiment's metric
category is `performance`. The Researcher subagent is instructed to begin every
round with profiling, so strategy decisions are grounded in measurement rather
than guesswork.

## When this applies

The Profile block in the protocol (`Profile.command`) is non-empty. Otherwise
the Researcher prompt is unchanged.

## Researcher prompt insertion

Insert this block at the top of the Researcher prompt, before `## Your role`:

\```
## Profile-first discipline (PERF MODE ACTIVE)
Before designing any strategy, you MUST:
1. Run `{profile_command}` and capture stdout.
2. Save raw profile to `profiles/round-{round}-before.md`.
3. Identify the top hotspot ops/kernels. Save your analysis to
   `profiles/round-{round}-analysis.md` (2-3 sentences: which op
   dominates, by how much, what you suspect).
4. The Strategy you append to experiences.md MUST cite a specific
   hotspot from the analysis. Skipping profile artifacts will be
   flagged by Supervisor and visible to next round's Researcher.

"It's obvious what's slow" is exactly the blind guessing this
discipline prevents.
\```

Then reorder `## Your task` so step 1 reads:
`Run profile_command, save profile + analysis (see Profile-first discipline above).`

## Supervisor advisory check

After Step 2 (Compliance), if perf-mode is active and either of
`profiles/round-{round}-before.md` or `profiles/round-{round}-analysis.md` is
missing, append a flag line to the current round's `experiences.md` Insight:

> `perf mode but no profile artifacts found — Researcher skipped discipline`

This is **advisory** — it does NOT trigger rollback. Its purpose is to make the
violation visible to the next round's Researcher (who reads experiences.md as
context) and to the human watching the loop.

## Trigger detection

In `autoresearch`: perf-mode is active iff the protocol contains a `### Profile`
block with non-empty `command`.

In `ml-iteration`: same — the Profile block is added to `iteration-protocol.md`
when `metric_category == performance` (or when the design doc's
`review_criteria.performance` is non-empty).
```

(Note: the inner triple-backticks above are escaped as `\``` for display in this plan; write the file with real triple-backticks.)

- [ ] **Step 2.2: Verify file rendered correctly**

Run: `head -20 skills/_ml-loop-primitives/profile-first.md`

Expected: file starts with `# Primitive — Profile-First Researcher Discipline` and the prompt block is wrapped in unescaped triple backticks.

- [ ] **Step 2.3: Commit**

```bash
git add skills/_ml-loop-primitives/profile-first.md
git commit -m "feat(primitives): profile-first researcher discipline doc

Template for Researcher prompt insertion when metric_category=performance;
Supervisor-side advisory check for missing profile artifacts (flag in
experiences.md Insight, not a rollback)."
```

---

## Task 3: Add `kernel-parity.md` primitive doc

**Files:**
- Create: `skills/_ml-loop-primitives/kernel-parity.md`

- [ ] **Step 3.1: Write the primitive doc**

Create `skills/_ml-loop-primitives/kernel-parity.md`:

```markdown
# Primitive — Kernel I/O Parity Guardrail

Pattern used by `autoresearch` and `ml-iteration` to verify, before training
each round, that any custom kernel introduced by the Researcher preserves the
baseline kernel's input/output contract.

## When this applies

The protocol contains a non-empty `kernel_targets` list. Each target declares:

- `name` — readable label (for logs and experiences.md insights).
- `new_module` — `module:callable` reference (must lie within Variable/focused files).
- `baseline_module` — `module:callable` reference (must lie within Fixed/locked files).
- `fixture` — `module:callable` (a zero-arg function returning `(args, kwargs)`).
- `tolerance` — `{atol, rtol}` for `torch.testing.assert_close`.

## Supervisor compliance sub-step (Step 2b)

After Step 2a (`git diff --name-only HEAD` shows no Fixed/locked file modified),
Supervisor invokes the parity script:

\```bash
# SKILL_DIR is the absolute path of this skill (autoresearch or ml-iteration).
# kernel_parity.py is a sibling primitive script.
python "$SKILL_DIR/../_ml-loop-primitives/kernel_parity.py" \
       --protocol "$EXPERIMENT_DIR/$PROTOCOL_FILE" \
       --search-path "$EXPERIMENT_DIR"
\```

Exit codes:
- `0` — all targets pass (or `kernel_targets` is empty).
- `1` — at least one target failed. First failure is on stderr:
  `PARITY_FAIL target=<name> kind=<signature|shape|numerical> detail=<...>`.

On failure: round is `not_improved (parity_violation)`. Skip training and
evaluation entirely — flow proceeds directly to Step 5 (Act on Result) which
performs rollback. The Insight field of the current round's experiences.md row
records the parity failure detail (next round's Researcher will see this).

## Task list S2 update

When `kernel_targets` is non-empty, the Step 0 task list S2 entry reads:
`"R{round} S2: Compliance — files + parity ({N} kernels)"`. On completion it
resolves to `"✅"` or `"❌ parity({kernel_name}): {kind} {detail snippet}"`.

## Handoff dry-run

At handoff time (before the iteration loop starts), the parity script is run
once on the baseline state. Because `new_module` is either the user's pre-coded
implementation that already matches baseline OR an auto-generated re-export
stub from baseline, the dry-run is expected to pass trivially. A failure at
this stage indicates a configuration error (wrong fixture, broken import path,
malformed tolerance) — handoff stops with a clear repair message.

## Why a script, not inline Python

Reused by two skills (`autoresearch`, `ml-iteration`). Failure diagnostics
require structured detail. Independent testability lets the smoke harness
regress on the four canonical failure modes (pass, signature-bad, shape-bad,
numerical-bad).
```

(Again — write file with real triple-backticks where this plan shows `\```.)

- [ ] **Step 3.2: Verify the file**

Run: `head -25 skills/_ml-loop-primitives/kernel-parity.md`

Expected: starts with `# Primitive — Kernel I/O Parity Guardrail`.

- [ ] **Step 3.3: Commit**

```bash
git add skills/_ml-loop-primitives/kernel-parity.md
git commit -m "feat(primitives): kernel parity guardrail doc

Describes Supervisor Step 2b sub-step, kernel_parity.py invocation contract
(--protocol + --search-path, exit codes 0/1, structured stderr), failure
semantics (round = parity_violation, rollback, insight records detail), and
the handoff dry-run rationale."
```

---

## Task 4: Extend `ml-brainstorming` — Q3.5 metric category + Q3.6 profile command

**Files:**
- Modify: `skills/ml-brainstorming/SKILL.md` (autoresearch question list, lines 171–177)

- [ ] **Step 4.1: Read current question list**

Run: `sed -n '169,182p' skills/ml-brainstorming/SKILL.md`

Verify questions 0–6 match what the spec assumes (Q4 Evaluation, then Q5 Termination, then Q6 Initial hints).

- [ ] **Step 4.2: Insert Q3.5 (metric category) and Q3.6 (profile command)**

The existing block is:

```markdown
4. **Evaluation** — "What metric determines success? We need a concrete, runnable eval script (e.g., `python eval.py --checkpoint best.pt`) that outputs the metric value. This script will be fixed before the loop starts — the agent cannot modify it. Do you have one, or do we need to build it?" (metric name, direction, eval script/command)
5. **Termination** — "When should the loop stop?" (max rounds, target metric value)
```

Insert two new numbered questions between Q4 and Q5 — and renumber Q5 → Q6, Q6 → Q7 (the initial-hints question shifts down by two).

Replace lines 175–177 with:

```markdown
4. **Evaluation** — "What metric determines success? We need a concrete, runnable eval script (e.g., `python eval.py --checkpoint best.pt`) that outputs the metric value. This script will be fixed before the loop starts — the agent cannot modify it. Do you have one, or do we need to build it?" (metric name, direction, eval script/command)
5. **Metric category** — "Does this metric measure **performance** (throughput / step time / latency / memory / MFU / TFLOPS / bandwidth), **accuracy** (loss / accuracy / BLEU / F1 / etc.), or **other**? This decides whether we enable profile-first discipline for the iteration loop." (maps to `metric_category`)
6. **Profile command** — only ask if Q5 = `performance`. "We need a `profile_command` that produces kernel/op-level timing to stdout each round, so the Researcher targets real hotspots instead of guessing. Provide a runnable command (e.g., `python profile.py --steps 50` running `torch.profiler` and printing a top-N table). If you don't have one yet, we'll record it as a VP-L1 build TODO — handoff will validate it before the loop starts." (maps to `profile_command`)
7. **Kernel R&D?** — "Will this research introduce a **custom kernel** replacing a baseline implementation (custom CUDA / Triton / fused op)? If yes, we'll add an I/O parity guardrail: every round, your new kernel will be compared to the baseline on the same fixture inputs for signature + numerical equivalence; out-of-tolerance auto-rollbacks the round before training spends time on it." (yes/no)
8. **Kernel targets** — only ask if Q7 = yes. Loop, collecting per target until user says "no more":
   - `name` — readable label.
   - `new_module` — `module:attr` import path. **Must be inside the Variable.files declared in Q3.** If not, push back: "this module lives outside the Variable file set — either add its file to Variable, or relocate the kernel."
   - `baseline_module` — `module:attr` import path. **Must be inside the Fixed.files declared in Q2.** If not, push back: "the baseline must be locked — add its file to Fixed, or this guardrail can't preserve the contract."
   - `fixture` — `module:attr` of a zero-arg callable returning `(args, kwargs)`.
   - `tolerance` — default `{atol: 1e-3, rtol: 1e-3}`; user can override.
9. **Termination** — "When should the loop stop?" (max rounds, target metric value)
10. **Initial hints（可选）** — "Any known experiences, constraints, or directions to try? (e.g., 'lr > 1e-3 causes gradient explosion', 'try cosine annealing')" — skip if none. Maps to R0 Note in experiences.md.
```

- [ ] **Step 4.3: Apply the edit**

Use `Edit` tool with `old_string` = lines 175–177 verbatim and `new_string` = the block above.

- [ ] **Step 4.4: Verify the edit**

Run: `sed -n '169,200p' skills/ml-brainstorming/SKILL.md`

Expected: Q4 Evaluation unchanged; Q5 metric category, Q6 profile command, Q7 kernel R&D, Q8 kernel targets, Q9 termination, Q10 initial hints — all 10 questions present in order.

- [ ] **Step 4.5: Commit**

```bash
git add skills/ml-brainstorming/SKILL.md
git commit -m "feat(ml-brainstorming): collect metric_category, profile_command, kernel_targets

Adds four new autoresearch protocol questions (metric category, profile
command, kernel R&D yes/no, kernel target details), renumbers termination
and initial-hints accordingly. Q6 (profile command) is conditional on
Q5=performance; Q8 (kernel targets) is conditional on Q7=yes."
```

---

## Task 5: Extend `ml-brainstorming` — design-doc Eval section

**Files:**
- Modify: `skills/ml-brainstorming/SKILL.md` (design-doc template section, around line 322)

- [ ] **Step 5.1: Read current Eval section in design-doc template**

Run: `sed -n '300,340p' skills/ml-brainstorming/SKILL.md`

Locate the `### Eval` section in the autoresearch design-doc template.

- [ ] **Step 5.2: Append new field declarations after `### Eval`**

After the existing `### Eval` block (`- metric:`, `- direction:`, `- command:`), append:

```markdown
- metric_category: <performance | accuracy | other — from Q5>

### Profile  (omit this block iff metric_category != performance)
- command: <profile_command — from Q6, or "TODO: build in VP L1" if user has none yet>
- expected_runtime: <human-readable, e.g., 30s — optional>

### Kernel Targets  (omit this block iff Q7 = no)
- name: <readable label>
  new_module: <module:attr>
  baseline_module: <module:attr>
  fixture: <module:attr>
  tolerance: { atol: <float>, rtol: <float> }
# repeat for each target
```

Use `Edit` with the existing `### Eval` 3-line snippet as anchor; append the new fields after it.

- [ ] **Step 5.3: Verify**

Run: `sed -n '320,345p' skills/ml-brainstorming/SKILL.md`

Expected: Eval section now includes `metric_category` line; Profile and Kernel Targets blocks appear with their omit-condition comments.

- [ ] **Step 5.4: Commit**

```bash
git add skills/ml-brainstorming/SKILL.md
git commit -m "feat(ml-brainstorming): emit Profile and Kernel Targets blocks in design doc

Design-doc autoresearch template gains metric_category field plus two
conditional blocks (Profile, Kernel Targets) that are emitted only when
the corresponding questions established them. handoff reads these blocks
verbatim into the protocol."
```

---

## Task 6: Extend `autoresearch` — Researcher prompt with perf section

**Files:**
- Modify: `skills/autoresearch/SKILL.md` (Step 1 prompt block, around lines 162–185)

- [ ] **Step 6.1: Read current Step 1 prompt**

Run: `sed -n '150,190p' skills/autoresearch/SKILL.md`

Verify the Researcher prompt template structure (`## Your role` → `## Constraints` → `## Recent experiences` → `## Your task`).

- [ ] **Step 6.2: Add perf-mode insertion**

Locate the existing block starting with `You are an ML researcher.` and ending with `Report "Code ready" as your final message.` (the entire prompt template at lines 162–185). Replace the block so that:

1. Just before `## Your role`, conditionally include the Profile-first section (only when protocol has a non-empty `Profile.command`):

```markdown
{# If protocol has Profile.command — include this block. Otherwise omit. #}
## Profile-first discipline (PERF MODE ACTIVE)
Before designing any strategy, you MUST:
1. Run `{profile_command}` and capture stdout.
2. Save raw profile to `profiles/round-{round}-before.md`.
3. Identify the top hotspot ops/kernels. Save your analysis to
   `profiles/round-{round}-analysis.md` (2-3 sentences: which op
   dominates, by how much, what you suspect).
4. The Strategy you append to experiences.md MUST cite a specific
   hotspot from the analysis. Skipping profile artifacts will be
   flagged by Supervisor and visible to next round's Researcher.

"It's obvious what's slow" is exactly the blind guessing this
discipline prevents.
{# end perf section #}
```

2. Update `## Your task` so that step 1 reads `Run profile_command, save profile + analysis (see Profile-first discipline above).` **when perf-mode is active**. Existing step 1 becomes step 2, etc.

Apply via `Edit` with full prompt block as `old_string`. Use the `{# ... #}` marker syntax (markdown comment) so Supervisor knows what to drop when perf-mode is inactive.

- [ ] **Step 6.3: Add Supervisor-side advisory check just after Step 2**

After `### Step 2: Compliance Check` (currently lines 187–195), insert an `### Step 2.5: Profile-Discipline Check (advisory)` section:

```markdown
### Step 2.5: Profile-Discipline Check (advisory)

Only when perf-mode is active (`Profile.command` non-empty in protocol):

```bash
ls profiles/round-{round}-before.md profiles/round-{round}-analysis.md
```

If either file is missing, append to this round's `experiences.md` Insight (do NOT rollback):

> `perf mode but no profile artifacts found — Researcher skipped discipline`

This flag is visible to next round's Researcher when they read the experiences table snippet. It is advisory, not a compliance failure.
```

Apply via `Edit`, inserting after the existing Step 2 block and before `### Step 3: Train`.

- [ ] **Step 6.4: Verify edits**

Run: `grep -n "Profile-first\|Profile-Discipline" skills/autoresearch/SKILL.md`

Expected: two matches — one inside the Step 1 prompt template, one as the new Step 2.5 heading.

- [ ] **Step 6.5: Commit**

```bash
git add skills/autoresearch/SKILL.md
git commit -m "feat(autoresearch): add profile-first prompt section and advisory check

Step 1 Researcher prompt gains a Profile-first discipline block (conditional
on Profile.command being declared in protocol); Your task step 1 reorders to
'run profile_command first'. New Step 2.5 logs an advisory insight to
experiences.md when profile artifacts are missing — flag, not rollback."
```

---

## Task 7: Extend `autoresearch` — Step 2 → 2a/2b with parity, plus Shared Patterns + task list

**Files:**
- Modify: `skills/autoresearch/SKILL.md` (Step 2 block, Step 0 task list, Shared Patterns list)

- [ ] **Step 7.1: Split Step 2 into 2a and 2b**

Replace the existing `### Step 2: Compliance Check` block (just the body — keep the heading) with:

```markdown
### Step 2: Compliance Check

**Step 2a — file boundary**

Supervisor runs directly:

\```bash
git diff --name-only HEAD
\```

Check that no Fixed.files were modified. New files are allowed. If any fixed file was modified → round is `not_improved`, skip training and evaluation, go directly to Step 5 (rollback).

**Step 2b — kernel parity (skip if protocol has no `kernel_targets`)**

Supervisor invokes the parity script:

\```bash
# SKILL_ROOT is the autoresearch skill's directory (where this SKILL.md lives).
# kernel_parity.py is a sibling primitive.
python "$SKILL_ROOT/../_ml-loop-primitives/kernel_parity.py" \
       --protocol "$EXPERIMENT_DIR/autoresearch-protocol.md" \
       --search-path "$EXPERIMENT_DIR"
\```

Exit 0 → continue. Exit 1 → round is `not_improved (parity_violation)`: read first stderr line (`PARITY_FAIL target=<name> kind=<...> detail=<...>`), record it in this round's `experiences.md` Insight, skip Step 3 and Step 4, go directly to Step 5 (rollback).
```

(Use `\``` placeholders → real triple-backticks in the actual file.)

Apply via `Edit` with the current Step 2 body as `old_string`.

- [ ] **Step 7.2: Update Step 0 task list — S2 string**

Locate the Step 0 block (around line 132) where S2 is created:

```
TaskCreate: "R{round} S2: Compliance — check {variable_files} only"
```

Update the line and the completion-format line below it:

```
TaskCreate: "R{round} S2: Compliance — files + parity ({N} kernels)" if protocol has kernel_targets,
            else "R{round} S2: Compliance — check {variable_files} only"
```

And update the completion-format example:

```
- S2 → `"R{round} S2: Compliance — ✅"` or
       `"❌ files: touched {file}"` or
       `"❌ parity({target_name}): {kind} {detail snippet}"`
```

Apply via `Edit`.

- [ ] **Step 7.3: Update Shared Patterns list**

Locate the Shared Patterns bullet list (around lines 26–32) and append two entries:

```markdown
- `profile-first.md` — Researcher-prompt template for perf-mode and Supervisor advisory check.
- `kernel-parity.md` — Supervisor Step 2b sub-step contract and `kernel_parity.py` invocation.
```

Apply via `Edit` inserting at the end of the existing 5-bullet list.

- [ ] **Step 7.4: Verify**

Run: `grep -n "Step 2a\|Step 2b\|parity\|profile-first" skills/autoresearch/SKILL.md`

Expected: at least 6 matches across the file (Step 2a heading, Step 2b heading, parity in body, parity in S2 string, profile-first in Shared Patterns, kernel-parity in Shared Patterns).

- [ ] **Step 7.5: Commit**

```bash
git add skills/autoresearch/SKILL.md
git commit -m "feat(autoresearch): split Step 2 into 2a/2b with kernel parity guardrail

Step 2b runs kernel_parity.py against protocol's kernel_targets; failure
rolls back the round before train+eval execute. Step 0 task list S2 string
reflects parity presence. Shared Patterns now references profile-first.md
and kernel-parity.md primitives."
```

---

## Task 8: Extend `ml-iteration` — Researcher prompt with perf section

**Files:**
- Modify: `skills/ml-iteration/SKILL.md` (Step 1 prompt block, around lines 121–164)

- [ ] **Step 8.1: Read current Step 1 prompt template**

Run: `sed -n '121,170p' skills/ml-iteration/SKILL.md`

- [ ] **Step 8.2: Add perf-mode insertion to the prompt**

Mirror Task 6.2 exactly — insert the same `## Profile-first discipline` block (conditional via `{# ... #}` markers) before `## Your role`. Reorder `## Task` so step 1 is the profile run when perf-mode is active.

Apply via `Edit` against the ml-iteration prompt block.

- [ ] **Step 8.3: Add advisory check as new Step 2.5**

Place the check right after Compliance (same slot as autoresearch in Task 6.3) — not after Eval. Reason: the advisory check inspects artifacts the Researcher produced in Step 1; placing it before Train means a discipline violation gets flagged before we spend train/eval cycles on it.

Insert a `### Step 2.5: Profile-Discipline Check (advisory)` heading with the same body as Task 6.3 (`ls profiles/round-{round}-*.md` → flag in `experiences.md` Insight; no rollback).

Apply via `Edit` inserting after Step 2 block and before Step 3.

- [ ] **Step 8.4: Verify edits**

Run: `grep -n "Profile-first\|Profile-Discipline" skills/ml-iteration/SKILL.md`

Expected: 2 matches.

- [ ] **Step 8.5: Commit**

```bash
git add skills/ml-iteration/SKILL.md
git commit -m "feat(ml-iteration): add profile-first prompt section and advisory check

Researcher prompt mirrors autoresearch perf-mode insertion. Advisory check
lives as Step 2.5 (after Compliance, before Train) so a discipline violation
is flagged before train/eval cycles are spent on it. Flag lands in
experiences.md Insight; no rollback."
```

---

## Task 9: Extend `ml-iteration` — Step 2 → 2a/2b with parity, plus Shared Patterns + task list

**Files:**
- Modify: `skills/ml-iteration/SKILL.md` (Step 2 block, Step 0 task list, Shared Patterns list)

- [ ] **Step 9.1: Split Step 2 into 2a and 2b**

Same structure as Task 7.1, adapted to ml-iteration terminology:

```markdown
### Step 2: Compliance Check

**Step 2a — file boundary**

\```bash
git diff --name-only HEAD
\```

Check that no `locked_files` were modified. If any was → this round is a violation: roll back, record in `experiences.md` Insight ("modified locked file {path}"), skip to Step 7.

Also grep newly-created files for eval-like function names (`evaluate`, `compute_metrics`, `accuracy`, `score`) and flag matches for user review; this is advisory, not a hard block.

**Step 2b — kernel parity (skip if protocol has no `kernel_targets`)**

\```bash
python "$SKILL_ROOT/../_ml-loop-primitives/kernel_parity.py" \
       --protocol "$EXPERIMENT_DIR/iteration-protocol.md" \
       --search-path "$EXPERIMENT_DIR"
\```

Exit 0 → continue. Exit 1 → round is `parity_violation`: read first stderr line, record in `experiences.md` Insight, skip Step 3, Step 4, Step 5; go directly to Step 6 with rollback verdict.
```

Note: in ml-iteration, rollback flows through Step 6 (Act on Verdict), not Step 5 — adjust the wording accordingly.

- [ ] **Step 9.2: Update Step 0 task list — S2 string**

Locate the S2 TaskCreate line (around line 110) and update the same way as Task 7.2.

- [ ] **Step 9.3: Update Shared Patterns list**

Append the two new primitive references (same wording as Task 7.3). Insert after the existing 5 bullets at the top of the file.

- [ ] **Step 9.4: Verify**

Run: `grep -n "Step 2a\|Step 2b\|parity\|profile-first" skills/ml-iteration/SKILL.md`

Expected: at least 6 matches.

- [ ] **Step 9.5: Commit**

```bash
git add skills/ml-iteration/SKILL.md
git commit -m "feat(ml-iteration): split Step 2 into 2a/2b with kernel parity guardrail

Mirrors autoresearch Step 2 split. Parity failure routes to Step 6 rollback
(ml-iteration's act-on-verdict step). S2 task string and Shared Patterns
list updated."
```

---

## Task 10: Extend `autoresearch-handoff` — Pre-check A (profile dry-run)

**Files:**
- Modify: `skills/autoresearch-handoff/SKILL.md` (insert new steps before existing Step 5)

- [ ] **Step 10.1: Add Pre-check A as new Step 4.5**

In `autoresearch-handoff/SKILL.md`, the checklist currently flows Step 1 (VP L1) → Step 2 (Baseline speed) → Step 3 (Pressure conditions) → Step 4 (Eval programmatic) → Step 5 (Generate protocol). Insert a new step between Step 4 and Step 5 — keep numbering by renaming the new step "Step 4.5" to avoid renumbering downstream references.

Insert this section after `## Step 4: Verify Eval Script is Programmatic` (after line 56):

```markdown
## Step 4.5: Profile Dry-Run (skip if `profile_command` empty in design doc)

The design doc may contain a `profile_command` (collected by ml-brainstorming Q6 when `metric_category == performance`). Validate it runs against the baseline before writing the protocol:

\```bash
cd <experiment_dir>
<profile_command>
\```

Requirements:
1. Exit code 0.
2. stdout is non-empty (at least one kernel/op timing line).
3. Save sample output to `<experiment_dir>/profiles/baseline.md` as the Researcher's starting reference.

If the design doc's `profile_command` was recorded as `TODO: build in VP L1` (user did not have one at brainstorming time), STOP the handoff and tell the user:

> "Build phase did not produce a `profile_command`. Add one to the design doc (and commit it to the experiment) before re-running handoff. We can't enter the iteration loop without a runnable profile command for perf-mode research."

If the command exists but exits non-zero, STOP and tell the user the exact error from stderr; ask them to fix `profile_command` and re-run handoff.
```

Apply via `Edit` inserting just before `## Step 5: Generate autoresearch-protocol.md`.

- [ ] **Step 10.2: Update checklist at top of file**

The numbered checklist (lines 22–29) needs an extra row. Update from:

```
1. **Verify VP L1** — record baseline metric
2. **Verify baseline speed** — first step/epoch prints fast enough
3. **Verify pressure condition termination** — time_limit / epoch_limit logic works
4. **Verify eval script is programmatic** — eval_command runs independently and produces deterministic output
5. **Generate autoresearch-protocol.md** — simplified format
6. **Initialize experiences.md** — table with baseline row
7. **Verify git state** — base code committed
8. **Present launch instructions**
```

To:

```
1. **Verify VP L1** — record baseline metric
2. **Verify baseline speed** — first step/epoch prints fast enough
3. **Verify pressure condition termination** — time_limit / epoch_limit logic works
4. **Verify eval script is programmatic** — eval_command runs independently and produces deterministic output
5. **Profile dry-run** — if perf-mode, validate `profile_command` runs (Step 4.5)
6. **Kernel parity dry-run** — if `kernel_targets` non-empty, validate parity machinery on baseline (Step 4.6)
7. **Generate autoresearch-protocol.md** — includes Profile + Kernel Targets blocks
8. **Initialize experiences.md** — table with baseline row
9. **Verify git state** — base code committed
10. **Present launch instructions**
```

- [ ] **Step 10.3: Verify**

Run: `grep -n "Step 4.5\|Step 4.6\|Profile dry-run\|Kernel parity dry-run" skills/autoresearch-handoff/SKILL.md`

Expected: at least one match (Step 4.5 inserted; 4.6 will be added in Task 11).

- [ ] **Step 10.4: Commit**

```bash
git add skills/autoresearch-handoff/SKILL.md
git commit -m "feat(autoresearch-handoff): add profile dry-run pre-check

Step 4.5 runs design-doc's profile_command on baseline and saves sample
output to profiles/baseline.md before the protocol is written. Failure
stops the handoff with a clear repair message."
```

---

## Task 11: Extend `autoresearch-handoff` — Pre-check B (parity dry-run) + protocol template

**Files:**
- Modify: `skills/autoresearch-handoff/SKILL.md`

- [ ] **Step 11.1: Add Pre-check B as new Step 4.6**

Insert after the new Step 4.5:

```markdown
## Step 4.6: Kernel Parity Dry-Run (skip if `kernel_targets` empty in design doc)

For each declared kernel target:

1. **Resolve `new_module`** — check the file exists in the experiment tree.
   - If absent, **auto-generate a re-export stub** that re-exports the baseline callable, so baseline-time parity is trivially passable. Write to the declared module path (e.g., `model/kernels/fused_softmax.py`):

   \```python
   # Auto-generated baseline stub. Researcher: replace with your custom
   # kernel. Must preserve the baseline signature and match within tolerance.
   from baseline.kernels.softmax_ref import softmax_ref as fused_softmax
   \```

   Use the user's declared baseline_module and the kernel's `name` to fill the import and re-export alias.

2. **Run parity dry-run:**

   \```bash
   python "$SKILL_ROOT/../_ml-loop-primitives/kernel_parity.py" \
          --protocol "$DRAFT_PROTOCOL" \
          --search-path "$EXPERIMENT_DIR"
   \```

   The draft protocol is a temporary file with the Kernel Targets block; `$EXPERIMENT_DIR` is the experiment root.

3. Exit code 0 → parity machinery is wired correctly; proceed. Exit code 1 → STOP and tell the user the exact `PARITY_FAIL` line from stderr. Typical causes: fixture returns wrong-shape inputs, import path typo, malformed tolerance.

This is intentionally a trivial pass — the new kernel is currently a re-export of baseline, so parity is mechanical. The dry-run's purpose is to surface **configuration** errors at handoff rather than at Round 1, where they would be misattributed to the Researcher.
```

- [ ] **Step 11.2: Update Step 5 protocol template to include Profile + Kernel Targets blocks**

Locate the markdown template in `## Step 5: Generate autoresearch-protocol.md` (around lines 60–82). Append after the existing `## Eval` section:

```markdown
{# Include only if design doc has profile_command #}
## Profile
- command: <profile_command from design doc>
- expected_runtime: <expected_runtime from design doc, or empty>
{# end Profile #}

{# Include only if design doc has kernel_targets #}
## Kernel Targets
- name: <readable name>
  new_module: <module:attr>
  baseline_module: <module:attr>
  fixture: <module:attr>
  tolerance: { atol: <float>, rtol: <float> }
{# repeat for each target. end Kernel Targets #}
```

- [ ] **Step 11.3: Verify**

Run: `grep -n "Step 4.6\|Kernel Parity Dry-Run\|## Profile\|## Kernel Targets" skills/autoresearch-handoff/SKILL.md`

Expected: 4 matches — Step 4.6 heading, dry-run body, Profile block in template, Kernel Targets block in template.

- [ ] **Step 11.4: Commit**

```bash
git add skills/autoresearch-handoff/SKILL.md
git commit -m "feat(autoresearch-handoff): add kernel parity dry-run + protocol template extension

Step 4.6 auto-generates a baseline-re-exporting stub for any new_module
that doesn't exist yet, then runs the parity script for a trivial-pass
sanity check. Failure stops handoff with the structured PARITY_FAIL
line. Protocol template now emits Profile and Kernel Targets blocks
when the design doc declared them."
```

---

## Task 12: Extend `training-handoff` (iteration branch) — Pre-check A (profile dry-run)

**Files:**
- Modify: `skills/training-handoff/SKILL.md` (Iteration Branch section, lines 219–258)

- [ ] **Step 12.1: Add new sub-step before "Generate iteration-protocol.md"**

The Iteration Branch checklist currently is 1 (training script readiness) → 2 (directory layout) → 3 (extract parameters) → 4 (generate iteration-protocol.md) → 5 (generate experiences.md) → 6 (generate iteration-prompt.md) → 7 (present launch).

Insert two new steps between 3 and 4 (renumbering downstream by +2):

After step 3 (`Extract iteration parameters`), add:

```markdown
3.5. **Profile dry-run** — if the design doc has `profile_command` (set when `metric_category == performance` or when `review_criteria.performance` is populated):

   \```bash
   cd <experiment_dir>
   <profile_command>
   \```

   Require exit 0 + non-empty stdout; save sample to `<experiment_dir>/profiles/baseline.md`. If the design doc recorded `profile_command: TODO`, STOP and ask the user to provide one before re-running handoff. Mirror the message from `autoresearch-handoff` Step 4.5.

3.6. **Kernel parity dry-run** — if the design doc has `kernel_targets` non-empty:

   For each target: auto-generate a baseline-re-exporting stub at the `new_module` path if it doesn't exist, then run:

   \```bash
   python "$SKILL_ROOT/../_ml-loop-primitives/kernel_parity.py" \
          --protocol <draft_iteration_protocol_path> \
          --search-path "$experiment_dir"
   \```

   Trivial-pass expected. Failure → STOP with the PARITY_FAIL line. Same semantics as `autoresearch-handoff` Step 4.6.
```

Renumber the existing steps 4–7 to 4–7 (no actual renumbering needed since we used 3.5 and 3.6).

- [ ] **Step 12.2: Verify**

Run: `grep -n "Profile dry-run\|Kernel parity dry-run\|3.5\|3.6" skills/training-handoff/SKILL.md`

Expected: at least 4 matches.

- [ ] **Step 12.3: Commit**

```bash
git add skills/training-handoff/SKILL.md
git commit -m "feat(training-handoff): add profile and parity dry-runs to iteration branch

Iteration branch gets steps 3.5 (profile dry-run) and 3.6 (kernel parity
dry-run) before iteration-protocol.md is written. Same semantics as the
autoresearch-handoff equivalents."
```

---

## Task 13: Extend `training-handoff` — iteration-protocol.md template

**Files:**
- Modify: `skills/training-handoff/SKILL.md` (iteration-protocol.md template, lines 236–258)

- [ ] **Step 13.1: Append Profile + Kernel Targets blocks to template**

The current template ends with `## Initial hints`. Append:

```markdown
{# Include only if design doc has profile_command #}
## Profile
- command: {profile_command from design doc}
- expected_runtime: {expected_runtime or empty}
{# end Profile #}

{# Include only if design doc has kernel_targets #}
## Kernel Targets
- name: {readable name}
  new_module: {module:attr}
  baseline_module: {module:attr}
  fixture: {module:attr}
  tolerance: { atol: {float}, rtol: {float} }
{# repeat per target. end Kernel Targets #}
```

Apply via `Edit` at the end of the existing template body, before the closing ` ``` `.

- [ ] **Step 13.2: Verify**

Run: `grep -n "## Profile\|## Kernel Targets" skills/training-handoff/SKILL.md`

Expected: 2 matches (the template now emits both blocks).

- [ ] **Step 13.3: Commit**

```bash
git add skills/training-handoff/SKILL.md
git commit -m "feat(training-handoff): emit Profile and Kernel Targets blocks in iteration-protocol

Template gains conditional Profile and Kernel Targets sections so the
iteration loop can read them via the same primitives autoresearch uses."
```

---

## Task 14: Version bump + release notes

**Files:**
- Modify: `.claude-plugin/plugin.json`
- Modify: `.cursor-plugin/plugin.json`
- Modify: `.claude-plugin/marketplace.json`
- Modify: `gemini-extension.json`
- Modify: `RELEASE-NOTES.md`

- [ ] **Step 14.1: Determine new version**

Current: `0.33.0`. This release adds two user-facing capabilities — minor bump per project convention (memory `feedback_versioning.md`): `0.33.0` → `0.34.0`.

- [ ] **Step 14.2: Bump versions via the project's bump script**

Run: `bash scripts/bump-version.sh 0.34.0`

Expected: all four files in `.version-bump.json` are updated; the script reports no drift.

If `scripts/bump-version.sh` is absent or fails, edit each file by hand:

```
.claude-plugin/plugin.json:        "version": "0.33.0"  →  "version": "0.34.0"
.cursor-plugin/plugin.json:        "version": "0.33.0"  →  "version": "0.34.0"
.claude-plugin/marketplace.json:   "plugins.0.version": "0.33.0"  →  "0.34.0"
gemini-extension.json:             "version": "0.33.0"  →  "0.34.0"
```

- [ ] **Step 14.3: Write release notes entry**

Prepend a new entry to `RELEASE-NOTES.md` (above the existing `## v0.33.0` heading). Use today's date:

```markdown
## v0.34.0 (2026-05-18)

### Added

**Profile-first discipline for perf-mode research.** When `ml-brainstorming` records `metric_category: performance`, the design doc and protocol carry a `profile_command`; the Researcher prompt in `autoresearch` and `ml-iteration` mandates running it each round and writing analysis to `profiles/round-N-analysis.md` before designing strategy. Supervisor logs an advisory flag to `experiences.md` when artifacts are missing.

**Kernel I/O parity guardrail.** When `kernel_targets` is declared in the protocol, Supervisor Step 2 (Compliance) runs `_ml-loop-primitives/kernel_parity.py` to compare each new kernel's `inspect.signature`, output pytree structure, and numerics (within user-declared atol/rtol) against the baseline. Mismatch auto-rollbacks the round before training spends time on it.

**Handoff pre-checks.** `autoresearch-handoff` and `training-handoff` validate `profile_command` (runs on baseline) and the parity machinery (trivial-pass dry-run on baseline-re-exporting stub) before writing the protocol — configuration errors are caught at handoff rather than Round 1.

**New primitives.** `skills/_ml-loop-primitives/profile-first.md`, `kernel-parity.md`, and `kernel_parity.py` (carrier script) — referenced by both `autoresearch` and `ml-iteration`.

### Changed

- `ml-brainstorming` autoresearch flow gains four questions (metric category, profile command, kernel R&D yes/no, kernel target details) before the existing termination question. Renumbered accordingly.
- `autoresearch/SKILL.md` Step 2 splits into 2a (file boundary) + 2b (parity).
- `ml-iteration/SKILL.md` Step 2 splits the same way; advisory profile-discipline check lives as Step 2.5 (mirrors autoresearch — flag before train/eval).
- `autoresearch-handoff` adds steps 4.5 (profile dry-run) and 4.6 (parity dry-run); protocol template emits Profile and Kernel Targets blocks.
- `training-handoff` adds steps 3.5 and 3.6 on the iteration branch with the same semantics.
```

- [ ] **Step 14.4: Verify**

Run:
```bash
grep "0.34.0" .claude-plugin/plugin.json .cursor-plugin/plugin.json .claude-plugin/marketplace.json gemini-extension.json
head -5 RELEASE-NOTES.md
```

Expected: all four version files contain `0.34.0`; `RELEASE-NOTES.md` starts with `## v0.34.0 (2026-05-18)`.

- [ ] **Step 14.5: Commit**

```bash
git add .claude-plugin/plugin.json .cursor-plugin/plugin.json .claude-plugin/marketplace.json gemini-extension.json RELEASE-NOTES.md
git commit -m "chore: bump version to 0.34.0 and release notes

Profile-first discipline + kernel I/O parity guardrail (orthogonal switches)
added to autoresearch and ml-iteration; ml-brainstorming collects metric
category, profile_command, and kernel_targets; both handoffs validate the
new fields on baseline before writing the protocol."
```

---

## Task 15: Sync to plugin cache + run smoke test

**Files:** (no edits — sync + verify)

- [ ] **Step 15.1: Re-run the kernel-parity smoke test in the source tree**

Run: `bash tests/kernel-parity/run-test.sh`

Expected: all four cases pass (as Task 1.6).

- [ ] **Step 15.2: Sync source skills to plugin cache**

From memory: "Changes to source (~/.claude/plugins/spml/) must also be synced to cache (~/.claude/plugins/cache/spml-dev/spml/0.1.0/), or reinstall the plugin."

Reinstall is the safest sync — it picks up the new version too:

```bash
# From the project root
ls ~/.claude/plugins/cache/spml-dev/spml/ 2>/dev/null
# If the cache exists, re-copy:
rsync -a --delete \
  ./skills/ \
  ~/.claude/plugins/cache/spml-dev/spml/0.1.0/skills/
# Sanity check:
diff -r ./skills/ ~/.claude/plugins/cache/spml-dev/spml/0.1.0/skills/ | head -20
```

Expected: no diffs reported after rsync.

If the cache layout has changed (different version dir), use:

```bash
find ~/.claude/plugins/cache -maxdepth 4 -name "kernel_parity.py" 2>/dev/null
```

to locate the install target and adjust the rsync target path.

- [ ] **Step 15.3: Smoke-test the existing autoresearch / ml-iteration harnesses still pass**

```bash
bash tests/autoresearch/run-test.sh
bash tests/ml-iteration/run-test.sh
```

Expected: both report success. (These exercise the train + eval dry path that our edits did not change. If they break, our markdown edits inadvertently broke a Supervisor instruction — investigate before proceeding.)

- [ ] **Step 15.4: Verify the new primitive scripts are reachable from skills**

Run:
```bash
SKILL_ROOT="$(pwd)/skills/autoresearch"
ls "$SKILL_ROOT/../_ml-loop-primitives/kernel_parity.py"
python "$SKILL_ROOT/../_ml-loop-primitives/kernel_parity.py" --help
```

Expected: file exists; `--help` prints argparse usage with `--protocol` and `--search-path`.

- [ ] **Step 15.5: Final task list cleanup**

Mark this plan as complete. No commit needed for the sync — it operates on artifacts outside the repo.

---

## Self-Review Notes

After completing all tasks, verify against the spec:

| Spec requirement | Implementing task(s) |
|---|---|
| Profile block in protocol | Tasks 5, 11, 13 (template edits) |
| Kernel Targets block in protocol | Tasks 5, 11, 13 |
| ml-brainstorming Q3.5 metric_category | Task 4 |
| ml-brainstorming Q3.6 profile_command | Task 4 |
| ml-brainstorming Q4 kernel R&D + Q4.x targets | Task 4 (with Fixed/Variable validation push-back) |
| Researcher prompt perf-mode block (autoresearch) | Task 6 |
| Researcher prompt perf-mode block (ml-iteration) | Task 8 |
| Supervisor advisory check for missing profile artifacts (autoresearch) | Task 6 (Step 2.5) |
| Supervisor advisory check for missing profile artifacts (ml-iteration) | Task 8 (Step 2.5) |
| Step 2 → 2a/2b split with parity (autoresearch) | Task 7 |
| Step 2 → 2a/2b split with parity (ml-iteration) | Task 9 |
| Task list S2 string update | Tasks 7, 9 |
| Shared Patterns list updates | Tasks 7, 9 |
| New primitive: profile-first.md | Task 2 |
| New primitive: kernel-parity.md | Task 3 |
| New script: kernel_parity.py | Task 1 |
| autoresearch-handoff Pre-check A | Task 10 |
| autoresearch-handoff Pre-check B + protocol template | Task 11 |
| training-handoff Pre-check A + B (iteration branch) | Task 12 |
| training-handoff template extension | Task 13 |
| Out-of-scope items (no implementation needed) | — |
