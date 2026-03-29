> **Partially superseded:** The 6-item completion gate described here has been reduced to 5 items (L2 merged into L1). See `docs/superpowers/specs/2026-03-29-vp-l1-l2-merge-design.md`.

# Fix: ml-subagent-dev Subtask Completion Gate

## Problem

During a Gumbel-STE toy experiment, the AI completed subtask implementations (TDD red->green) but skipped all subsequent steps: Spec Review, Quality Review, VP L0/L1/L2, Record Conclusion, and Training Handoff.

**Root cause:** The AI read the skill, understood the flow, but self-judged it as "overkill" for a toy experiment and rationalized skipping. This is not an information gap — it's a discipline failure that the skill text did not sufficiently prevent.

## Design

### Change 1: Reorder Per-Subtask Flow — TDD = VP

**Current (broken) order:**
```
Implement(TDD) -> Spec Review -> Quality Review -> L0 -> L1 -> L2 -> Record Conclusion
```

**New order:**
```
Implement(TDD = VP) -> Spec Review -> Quality Review -> Record Conclusion
```

Expanded:
```
1. Write unit tests (TDD red)
2. Implement core code (TDD green)
3. L0: VP Static Checks (static correctness)
4. L1: ML Runtime Validator (5-min training)
5. L2: ML E2E Validator (end-to-end pipeline)
   ---- TDD/VP complete ----
6. Spec Review (experiment design compliance)
7. Quality Review (code quality)
8. Completion Gate (all items checked)
9. Record Conclusion
```

**Rationale:** ML experiments always involve a training-evaluation pipeline. Even "loss is decreasing" is evaluation. TDD for ML pipelines IS the Validation Pyramid. Reviews come after VP because reviewers can reference VP results for more informed reviews.

### Change 2: HARD-GATE at Skill Top

Add a `<HARD-GATE>` block at the very top of the skill (before the process flow), declaring:

- Every subtask MUST complete ALL steps before marking complete
- No exceptions for "simple" or "toy" experiments
- Explicit checklist: L0 passed, L1 passed, L2 passed, Spec Review passed, Quality Review passed, Conclusion recorded
- If any item unchecked, subtask is NOT complete

### Change 3: Anti-Pattern Table

Add an Anti-Pattern section after HARD-GATE, before the flow diagram. Lists forbidden rationalizations:

| Thought | Reality |
|---------|---------|
| "This is just a toy experiment" | Toy experiments with wrong gradients waste days of debugging |
| "The model code is simple" | Simple code with silent shape bugs produces plausible but wrong results |
| "Unit tests already passed" | Unit tests check deterministic logic. VP checks training dynamics. Different things. |
| "L1/L2 is overkill for this subtask" | If this subtask is part of an ML experiment, it WILL be trained and evaluated. VP validates that. |
| "I'll run VP at the end" | VP per subtask catches bugs early. VP at the end means debugging the entire codebase at once. |
| "The user wants speed" | Skipping VP and debugging silent failures later is SLOWER. |

Core principle: ML experiments ALWAYS involve a training-evaluation pipeline. If there is a pipeline, there is a VP. No exceptions.

### Change 4: Completion Gate Node in Flow Diagram

Add a red-colored `Completion Gate` diamond node between Quality Review passing and Record Conclusion. This gate checks all 6 items are confirmed before allowing the subtask to proceed.

### Change 5: Reviewer Prompt Context Updates

**Spec Reviewer:** Add note that VP (L0/L1/L2) has already passed, so reviewer can reference VP results when checking experiment design compliance.

**Quality Reviewer:** Add note that VP and Spec Review have already passed, so focus is purely code quality.

**Implementer:** Update note to reflect new order — VP runs after implementation as part of TDD, then Reviews follow.

### Change 6: Forced Handoff Decision Point

**Current (broken):** "Needs long-running training?" is a decision the AI makes silently. It can skip or forget.

**New:** After all subtasks complete, the orchestrator MUST pause and ask the user explicitly:

```
All subtasks complete. VP passed. Next step:

1. **Train** — needs long-running training (hours/days). I will invoke
   spml:training-handoff to generate experiment-context.md +
   watchdog-prompt.md for a new monitoring session.
2. **Done** — experiment is already complete within this session.
   I will invoke spml:verification.

Which one?
```

This is a HARD-GATE — the orchestrator cannot proceed to either training-handoff or verification without the user's explicit choice. The AI does not decide this itself.

**Rationale:** The previous bug was that the AI skipped the entire handoff step because it judged the experiment "too simple" for a separate training session. Making this a mandatory user-facing question removes the AI's ability to skip it.

## Scope

Only `skills/ml-subagent-dev/SKILL.md` is modified. No changes to:
- VP skills (L0/L1/L2) — their internal logic is unchanged
- training-handoff — unchanged
- verification — unchanged
- experiment-planning — unchanged

## Files Changed

- `skills/ml-subagent-dev/SKILL.md` — all changes in this single file
