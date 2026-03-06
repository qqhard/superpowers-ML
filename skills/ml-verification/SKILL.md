---
name: ml-verification
description: Use when all ML experiment subtasks are complete - verifies Validation Pyramid passed, compiles conclusions, and presents summary with recommendations
---

# ML Verification

## Overview

Verify that the ML experiment is truly complete and compile conclusions. This is the final gate before claiming an experiment is done.

**Core principle:** Evidence before claims. In ML, "tests pass" is necessary but not sufficient — the Validation Pyramid must pass AND conclusions must be supported by metric evidence.

<HARD-GATE>
Do NOT claim an experiment is complete without running through this entire checklist.
</HARD-GATE>

## When to Use

- All subtasks in an experiment plan are marked complete
- Before claiming experiment results to the user
- Before deciding next steps (new experiment, deployment, etc.)

## The Verification Checklist

You MUST check each item. Create a TodoWrite task for each.

### 1. Validation Pyramid Completion

For each subtask, verify:

- [ ] All enabled VP layers were executed (not skipped)
- [ ] All enabled VP layers passed (check actual numbers, not claims)
- [ ] Metric values are within expected ranges from the plan
- [ ] No layers were added or removed without user approval

```
Subtask 1: [name]
  L0 Engineering: ✅ MFU=0.45 (target ≥0.40)
  L1 Process:     ✅ No NaN/Inf, gradient norm stable
  L2 Overfit:     ✅ Loss < 0.01 after 8 epochs
  L3 E2E:         ⏭️ Skipped (user decision in brainstorm)

Subtask 2: [name]
  ...
```

### 2. Conclusion Validity

For each subtask:

- [ ] Conclusion (effective/ineffective/inconclusive) is stated
- [ ] Conclusion is supported by metric evidence (not just "it works")
- [ ] Control variables were truly controlled (no confounds)
- [ ] Independent variable was the only thing that changed

### 3. Code Separation Verified

- [ ] Core code (model, training, data) has NO imports from test/validation/toolkit
- [ ] Validation scripts are external observers only
- [ ] Core code can be extracted for production as-is

Quick verification:
```bash
# In the core code directory, check for toolkit/test imports
grep -r "from toolkit\|import toolkit\|from test\|from validation" src/ model/ --include="*.py"
# Expected: no output
```

### 4. Reproducibility

- [ ] Random seeds are set and documented
- [ ] Key hyperparameters are recorded
- [ ] Data version/snapshot is identified
- [ ] Environment (GPU type, PyTorch version) is noted

### 5. Anomaly Review

- [ ] All anomalies noted during subtasks are reviewed
- [ ] No unresolved anomalies that could invalidate conclusions
- [ ] Anomalies that were resolved have documented fixes

## Summary Report Template

After completing the checklist, present this summary to the user:

```markdown
# Experiment Summary: [Experiment Name]

## Overall Result: [effective / partially effective / ineffective / inconclusive]

## Hypothesis
[Original hypothesis from brainstorm]

## Subtask Results

| Subtask | Hypothesis | Result | Key Metric | VP Status |
|---------|-----------|--------|------------|-----------|
| 1. [name] | [hypothesis] | ✅ effective | [metric=value] | All passed |
| 2. [name] | [hypothesis] | ❌ ineffective | [metric=value] | L2 failed |
| 3. [name] | [hypothesis] | ⚠️ inconclusive | [metric=value] | All passed |

## Key Findings
- [Finding 1 with metric evidence]
- [Finding 2 with metric evidence]

## Anomalies
- [Any unresolved or notable anomalies]

## Recommendations

**Option A:** [Next experiment / further investigation]
**Option B:** [Deploy / integrate the effective subtasks]
**Option C:** [Abandon this direction, try alternative approach]

## Environment
- GPU: [type]
- PyTorch: [version]
- Data: [version/snapshot]
- Seeds: [values used]
```

## Decision Handoff

After presenting the summary, ask the user:

**"Experiment complete. Three options:**

**1. Finish** — Effective subtasks are ready. Invoke `mlsp:finishing-a-development-branch` to wrap up.

**2. Add subtasks** — Continue this experiment with additional subtasks. I'll add them to the existing plan.

**3. New brainstorm** — Start a new experiment direction. Invoke `mlsp:ml-brainstorming`."

## Red Flags

**Never:**
- Claim "experiment complete" without this checklist
- Report "effective" without VP metric evidence
- Skip anomaly review
- Forget to check code separation
- Present conclusions without reproducibility info

**If checklist reveals issues:**
- Missing VP execution -> run the missing checks now
- Conclusion not supported -> downgrade to "inconclusive"
- Code separation violated -> fix before claiming complete
- Unresolved anomaly -> flag to user, let them decide

## Integration

- **mlsp:ml-subagent-dev** — Produces the subtask results this skill verifies
- **mlsp:validation-pyramid** — VP checks referenced in verification
- **mlsp:ml-brainstorming** — Next step if new experiment needed
- **mlsp:finishing-a-development-branch** — Next step if experiment is done
