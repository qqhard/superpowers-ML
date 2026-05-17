# Profile-First Perf Research + Kernel Parity Guardrail — Design

## Problem

Two distinct failure modes show up in the current `autoresearch` and `ml-iteration` loops when applied to model performance optimization or kernel R&D:

1. **Blind guessing in perf optimization.** When the research metric is performance-class (throughput, step time, latency, MFU, memory peak), the Researcher subagent currently designs strategies from prior-experience text alone. With no fresh profile each round, it cannot tell which op/kernel dominates and routinely proposes generic interventions ("optimize attention", "fuse the MLP") that miss the real hotspot. Rounds get wasted on plausible-sounding changes to non-bottleneck code.

2. **Silent correctness drift in kernel R&D.** When a round replaces a baseline kernel with a custom implementation (CUDA / Triton / fused op), there is no mechanism verifying that the new kernel preserves the baseline's input/output contract. A signature mismatch, a wrong-shape output, or a numerically broken kernel currently flows through to training. The metric the loop watches may improve (speed up) while the model is silently doing the wrong math — corrupting subsequent rounds' starting state until someone notices manually.

Both issues compound in the typical "perf research with kernel replacement" combo, which is the canonical reason teams run autoresearch in the first place.

## Design Goals

1. **Profile-first as default for perf research.** When the metric is perf-class, force every round to begin with profiling so strategy is grounded in measurement, not pattern-matching.
2. **Kernel I/O parity as a hard pre-train compliance gate.** When the protocol declares kernel replacements, mismatched signature or out-of-tolerance numerics must rollback the round *before* training spends time on it.
3. **Two independent switches.** Perf-mode and kernel-parity are orthogonal — accuracy research can use kernels; perf research can be data/scheduler-only — so they activate independently.
4. **Apply to both loops.** `autoresearch` (single-metric Pareto) and `ml-iteration` (compound LLM-judged) both get the same guardrails, since both can target perf and both can host kernel work.
5. **Validate the guardrails themselves at baseline time.** Profile commands and parity fixtures are user-provided and easy to misconfigure. Validate them once at handoff, before the autonomous loop starts.

## Architecture Overview

```
ml-brainstorming  →  experiment-planning  →  ml-subagent-dev  →  *-handoff  →  *-run
   (Q3.5/3.6/4/4.x)    (plan unchanged)      (VP unchanged)    (pre-checks)   (loop body)
```

Two new orthogonal capabilities flow through the pipeline:

| Capability | Trigger | Carrier in protocol | Enforced in loop by | Validated at handoff by |
|---|---|---|---|---|
| **A. Profile-first** | metric category = performance | `Profile.command` | Researcher prompt (prompt-level discipline) | running `profile_command` once on baseline |
| **B. Kernel parity** | `kernel_targets` non-empty | `kernel_targets[]` | Supervisor Compliance Step (2b) | trivial parity run (new ≡ baseline) |

The two are signalled independently. Either, both, or neither may be active for a given experiment.

## Protocol Schema Extension

Both `autoresearch-protocol.md` and `iteration-protocol.md` gain two optional blocks.

### Profile block (perf-mode carrier)

```yaml
### Profile  (optional — present iff metric is perf-class)
- command: <profile_command, e.g., python profile.py --steps 50>
- expected_runtime: <human-readable, e.g., 30s>
```

Field presence = perf-mode active. Absence = ordinary accuracy research; loop behaves as before.

### Kernel Targets block (parity guardrail carrier)

```yaml
### Kernel Targets  (optional — non-empty enables parity guardrail)
- name: fused_softmax
  new_module: model.kernels.fused_softmax           # in Variable.files / focused_files
  baseline_module: baseline.kernels.softmax_ref     # in Fixed.files / locked_files
  fixture: tests.fixtures.softmax_inputs            # callable () -> (args, kwargs)
  tolerance: { atol: 1e-3, rtol: 1e-3 }
- name: ...
```

`baseline_module` must lie within the immutable file set (`Fixed.files` for autoresearch; `locked_files` for ml-iteration). `ml-brainstorming` enforces this at collection time.

`fixture` is the import path of a zero-arg function returning the `(args, kwargs)` tuple to feed both kernels.

## ml-brainstorming Collection Flow

Two new questions plus one follow-up loop slot into the existing autoresearch flow:

```
Q0  Experiment directory
Q1  Fixed (files + conditions)
Q2  Variable (files + conditions)
Q3  Evaluation (metric name + direction + eval_command)
Q3.5 NEW  Metric category — performance | accuracy | other
Q3.6 NEW  Profile command            (only if Q3.5 = performance)
Q4   NEW  Kernel R&D? (yes/no)
Q4.x NEW  kernel_targets entries    (loop while user has more)
Q5  Termination
```

### Q3.5 — Metric category

Asked directly, not inferred from metric name:

> "This metric measures model **performance** (speed/throughput/latency/memory/MFU/TFLOPS) or model **accuracy** (loss/accuracy/BLEU/F1) — or something else?"

Stored as `metric_category` in the design doc.

### Q3.6 — Profile command (perf only)

> "We need a `profile_command` that produces kernel/op-level timing breakdown to stdout each round, so the Researcher targets real hotspots instead of guessing. Provide a runnable command (e.g., `python profile.py --steps 50` running `torch.profiler` and printing a top-N table). If you don't have one yet, we'll mark it as a VP-L1 build TODO."

If the user has none, the design doc records a build-phase TODO; `*-handoff` enforces the dry-run later.

### Q4 — Kernel R&D? (always asked)

> "Will this research introduce a custom kernel replacing a baseline implementation (custom CUDA / Triton / fused op)? If yes, we'll add a parity guardrail: every round, your new kernel will be compared to the baseline on the same fixture inputs for signature + numerical equivalence; out-of-tolerance auto-rollbacks the round."

### Q4.x — kernel_targets entries

For each target, collect:

1. Readable `name` + `new_module` import path (must be in Variable / focused files)
2. `baseline_module` import path (must be in Fixed / locked files — brainstorming **rejects** otherwise and prompts user to lock it)
3. `fixture` import path
4. `tolerance` — default `{atol: 1e-3, rtol: 1e-3}`, user-overridable

Then ask "another kernel?" until done.

### ml-iteration mode

`ml-iteration` uses `review_criteria` instead of a single metric. Mapping:

- **Profile**: asked iff `review_criteria.performance` is populated (user cares about a performance dimension).
- **Kernel R&D**: asked unconditionally, same as autoresearch.

### Design-doc additions

```yaml
metric_category: performance | accuracy | other
profile_command: <string or null>
kernel_targets:
  - name: ...
    new_module: ...
    baseline_module: ...
    fixture: ...
    tolerance: { atol: ..., rtol: ... }
```

`experiment-planning` and `*-handoff` pass these through verbatim to protocol.

## Loop Body Changes

Both `autoresearch` and `ml-iteration` receive the same two insertions, scaled to each loop's conventions.

### Researcher prompt — Profile-first discipline (perf-mode only)

When `profile_command` is set, the Step 1 dispatch prompt prepends:

```
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
```

The `## Your task` step list is reordered so "Run profile_command, save profile + analysis" becomes step 1; the rest shift down.

**Supervisor-side advisory check** (not a hard gate): after Step 2, if perf-mode is active and `profiles/round-{round}-before.md` or `profiles/round-{round}-analysis.md` is missing, append a flag to `experiences.md` Insight ("perf mode but no profile artifacts — Researcher skipped discipline"). Does **not** trigger rollback; serves as a visible negative signal that accumulates in experiences for the next round's Researcher to see.

### Supervisor Compliance — Step 2b kernel parity

Step 2 splits into two sub-steps:

```
Step 2: Compliance Check
  2a. git diff --name-only HEAD
      → any Fixed/locked file modified → fail
  2b. NEW — kernel parity  (skip if kernel_targets empty)
      → for each kernel_target:
         - import new_module and baseline_module
         - compare inspect.signature → mismatch = fail
         - call fixture() → get (args, kwargs)
         - call both kernels
         - compare output pytree shape/dtype → mismatch = fail
         - torch.testing.assert_close(out_new, out_base,
                                      atol=t.atol, rtol=t.rtol) → fail
```

Any failure in 2a or 2b: round is `not_improved (parity_violation)`, **train + eval are skipped entirely**, flow jumps to Step 5 rollback, `experiences.md` Insight records the specific kernel and the failure detail (e.g., `parity violation: fused_softmax — max_diff=2.3e-2 exceeds atol=1e-3 at index [0,12,4]`).

#### Carrier script: `kernel_parity.py`

The parity check lives in `_ml-loop-primitives/kernel_parity.py` (new). Single-purpose tool, invoked by Supervisor via Bash:

```bash
python -m _ml_loop_primitives.kernel_parity --protocol <path>
#   exit 0 = all targets pass
#   exit 1 = at least one failed (structured details on stderr)
```

Shipping it as a script rather than inline Python: (a) reused by two skills, (b) failure diagnostics need structured detail (which target, which dim, max diff), (c) testable independently.

### Task list strings

Step 0 task list S2 entry updates when `kernel_targets` is non-empty:

```
old:  "R{round} S2: Compliance — check {variable_files} only"
new:  "R{round} S2: Compliance — files + parity ({N} kernels)"
```

On completion, S2 resolves to `"✅"` or `"❌ parity({kernel_name}): max_diff=…"`.

### New primitive documents

Two additions to `_ml-loop-primitives/`:

- `profile-first.md` — perf-mode Researcher-prompt template and trigger condition.
- `kernel-parity.md` — Supervisor compliance sub-step protocol and `kernel_parity.py` interface.

Both `autoresearch/SKILL.md` and `ml-iteration/SKILL.md` add these to their "Shared Patterns" list.

## Handoff Pre-Checks

Real executability of `profile_command` and `kernel_targets` is validated at handoff (after VP, before the iteration session is launched). `ml-brainstorming` only enforces protocol-level structure; runtime validation happens here because baseline code now exists.

### Pre-check A — Profile dry-run

Only when `profile_command` is set:

1. Run `profile_command` against the baseline build.
2. Require exit code 0 and non-empty stdout.
3. Persist sample output to `profiles/baseline.md` as the Researcher's starting reference.
4. Failure → handoff stops with a clear "fix profile_command before re-running handoff" message.

### Pre-check B — Kernel parity dry-run

Only when `kernel_targets` is non-empty:

1. For each target: verify the `new_module` file exists. If absent, handoff generates a re-export stub so baseline-time parity is trivially passable:

   ```python
   # model/kernels/fused_softmax.py  (auto-generated baseline stub)
   # Researcher: replace with your custom kernel. Must preserve the
   # baseline signature and match within tolerance.
   from baseline.kernels.softmax_ref import softmax_ref as fused_softmax
   ```

2. Run `python -m _ml_loop_primitives.kernel_parity --protocol /tmp/draft-protocol.md`.
3. Since `new ≡ baseline` at baseline time, parity must pass trivially.
4. Failure → handoff stops. Typical causes: fixture returns wrong-shape inputs, import path typo, malformed tolerance value.

### Why baseline-time parity matters even though it's trivial

The trivial pre-check exists to surface **configuration** errors in the parity machinery itself — fixture wiring, import resolution, tolerance schema — at handoff time rather than during Round 1, where they would be misattributed to the Researcher's actual kernel work.

After pre-checks pass, handoff writes the final protocol and emits the run-mode entry prompt as usual.

## Relationship to Existing VP Layers

VP layers (L0/L1/L2/L3) are not changed. The handoff pre-checks are **not** VP — they validate iteration-loop *inputs*, not baseline correctness. They run after VP is fully green, as a "smoke test for the iteration's guardrail wiring."

## Files Touched (summary)

| File | Change |
|---|---|
| `skills/ml-brainstorming/SKILL.md` | Add Q3.5 / Q3.6 / Q4 / Q4.x; design-doc field additions |
| `skills/autoresearch/SKILL.md` | Step 1 prompt addition (perf section); Step 2 → 2a/2b split; Shared Patterns list updates; Task list S2 string update |
| `skills/ml-iteration/SKILL.md` | Same changes as autoresearch, adapted to ml-iteration conventions |
| `skills/autoresearch-handoff/SKILL.md` | Add Pre-check A + Pre-check B before final protocol write |
| `skills/training-handoff/SKILL.md` | Same pre-checks on the iteration-protocol branch |
| `skills/_ml-loop-primitives/profile-first.md` | NEW primitive doc |
| `skills/_ml-loop-primitives/kernel-parity.md` | NEW primitive doc |
| `skills/_ml-loop-primitives/kernel_parity.py` | NEW carrier script (torch + yaml only) |

## Out of Scope

- Choosing a specific profiler (`torch.profiler` vs `nsys` vs custom). The user supplies `profile_command`; we don't prescribe the tool.
- Auto-generated profile analysis on the Supervisor side. Profile reading is the Researcher's job, by design.
- Fancier parity checks (gradient parity, distributed-equivalence). Out of scope for v1; add only if a real experiment needs them.
- Profile-aware verdict logic (rewarding hotspot-targeted strategies in review). The advisory flag in `experiences.md` is the v1 mechanism; richer scoring can come later.
