# Step-Based Evaluation Subtask Design

**Date:** 2026-03-18
**Status:** Proposed
**Scope:** Move evaluation requirements upstream into planning and code structure so evaluation is a first-class subtask instead of a final-step afterthought

## Problem

Current training habits can still produce runs where evaluation happens only once at the very end. This causes two practical problems:

1. Long-running training may make no validation-visible progress until the final epoch, which delays feedback and hides failures.
2. When evaluation finally runs, the terminal may appear blank or stalled if evaluation has no explicit progress output.

This is not just a training-loop detail. It is an upstream design problem: plans and generated code do not currently require evaluation cadence, evaluation entrypoints, or evaluation observability to be designed as first-class concerns.

## Design Decisions

- **Evaluation becomes a dedicated subtask** in planning, not an implementation detail buried inside the trainer
- **Cadence expression is step-based by default**: plans must express evaluation cadence as `every N steps`, not implicit "at the end"
- **Default evaluation scope is full validation** unless the plan explicitly says otherwise
- **Evaluator is a standalone component** shared by both training-triggered evaluation and checkpoint-triggered evaluation
- **Two evaluation entry modes are required**:
  - from checkpoint on disk
  - from in-memory model state during training
- **Long-running evaluation must always show progress**; blank terminal periods during evaluation are treated as a design failure
- **Evaluation efficiency and observability are validation targets**, not optional niceties

## Target Architecture

The training stack is split into three explicit responsibilities:

### 1. Model Core

The model core is responsible only for model behavior:

- forward pass
- outputs needed by loss or metrics
- serializable state for checkpoint save/load

It does **not** own:

- evaluation cadence
- progress bars
- checkpoint loading policy
- metric aggregation orchestration

### 2. Trainer

The trainer owns the training process:

- training steps
- optimizer and scheduler updates
- checkpoint save policy
- cadence decisions for when evaluation should run

The trainer may trigger evaluation, but it must not own a separate evaluation implementation. Its responsibility is **when** to call evaluation, not **how** evaluation works.

### 3. Evaluator

The evaluator is a first-class component with a single responsibility: execute evaluation and report what happened.

It owns:

- validation dataloader iteration
- metric accumulation and aggregation
- evaluation progress display
- phase start/end logging
- evaluation timing and throughput summary
- normalization across different evaluation entry modes

The evaluator must support two state sources through one shared execution path:

- **Checkpoint mode:** load model state from checkpoint, then evaluate
- **In-memory mode:** receive current live model state from the trainer, then evaluate without forcing a round-trip through disk

The implementation may wrap the two sources differently at the boundary, but once model state is prepared, both paths must enter the same evaluator core. Separate duplicate logic for "standalone evaluation" and "training-time evaluation" is explicitly disallowed.

## Planning Changes

Planning must treat evaluation as its own subtask. The minimum expected subtask structure becomes:

- model core subtask
- trainer subtask
- evaluation subtask

The evaluation subtask includes both runtime forms:

- **Standalone evaluation** initiated from checkpoint
- **In-training evaluation** initiated by the trainer from in-memory state

An integration subtask is not required as a separate item if the evaluation subtask already covers both entry modes and the trainer's trigger contract.

### Required Plan Fields

Any training plan that includes a meaningful evaluation phase must explicitly state:

- **Evaluation cadence:** step-based, e.g. `every 500 steps`
- **Evaluation scope:** default `full validation`, or an explicit override
- **Evaluation entry modes:** checkpoint-based, in-memory, or both
- **Expected observability:** progress bar, phase messages, and summary output
- **Failure handling expectations:** what happens if checkpoint load, metric aggregation, or dataloader setup fails

If a plan says only "run validation" without cadence, entry mode, or observability requirements, the plan is incomplete.

## Evaluation Subtask Contract

The evaluation subtask must deliver one evaluator capability that works in both modes.

### Mode A: Standalone Evaluation

This path is used when evaluation starts from a saved checkpoint:

1. Locate checkpoint
2. Restore model state
3. Construct evaluator inputs
4. Run evaluator core
5. Emit metrics and efficiency summary

### Mode B: In-Training Evaluation

This path is used when evaluation is triggered during training:

1. Trainer reaches configured step cadence
2. Trainer passes the current in-memory model state into the evaluator boundary
3. Evaluator runs without mandatory checkpoint serialization
4. Emit metrics and efficiency summary
5. Return results to trainer for logging, scheduling, or early-stop logic

The difference between the modes is **state source only**. The evaluation logic itself must remain shared.

## Observability Requirements

Evaluation is long-running work. Users must be able to tell that the process is alive and what stage it is in.

The following are hard requirements for long-running evaluation:

- A clear phase-start message before evaluation begins
- An explicit indication of which mode is running:
  - checkpoint-based evaluation
  - in-training evaluation
- A dedicated evaluation progress bar
- The progress bar must show total evaluation batches and current progress
- A clear phase-end message when evaluation finishes
- A result summary after completion
- An efficiency summary after completion

Examples of acceptable efficiency summary fields:

- total evaluation duration
- batches per second
- samples per second
- checkpoint load time, when checkpoint mode is used

Silent evaluation that leaves the terminal blank for a long period is considered a design defect even if the code eventually completes.

## Efficiency Validation Requirements

Evaluation should not only be correct; it should also be visible and operationally healthy.

The evaluation subtask must define validation checks for:

- checkpoint load latency, when starting from checkpoint
- time-to-first-progress-update
- total evaluation runtime
- throughput reasonableness
- absence of long silent gaps in output

This is intentionally broader than pure metric correctness. Evaluation is treated as an engineering workflow that must be inspectable while it runs.

## Error Handling

The design must treat the following as explicit evaluation error scenarios:

- checkpoint missing or unreadable
- checkpoint restore failure
- validation dataloader empty or misconfigured
- metric accumulator failure
- non-finite metrics
- evaluation loop stalls or produces no progress output for too long

Errors must surface at the evaluation boundary with mode-aware context. For example, checkpoint failures should identify the checkpoint path; in-training failures should identify the triggering global step.

## Validation and Review Implications

This design changes what downstream review should enforce.

### Planning Review

Plans should fail review if they:

- omit an evaluation subtask
- omit step-based cadence
- hide evaluation inside "trainer implementation"
- define only final-epoch evaluation by default
- do not specify evaluation progress visibility

### Code Review / Static Checks

Training code should be flagged if:

- evaluation logic is duplicated between standalone and in-training paths
- trainer directly implements evaluation internals instead of calling an evaluator
- evaluation has no dedicated progress indicator
- evaluation can run for long periods without visible output

### Runtime Validation

Runtime validation should treat evaluation as a first-class monitored phase:

- verify evaluation actually triggers at the planned step cadence
- verify evaluation progress output appears while evaluation is running
- verify checkpoint-based evaluation reports checkpoint load behavior
- verify in-training evaluation reports that it is using in-memory state

## Out of Scope

- Defining a specific framework API for the evaluator
- Mandating a specific progress-bar library
- Redesigning model checkpoint format
- Choosing a universal metric schema across all experiment types

## Summary

The central change is upstream, not cosmetic: evaluation becomes a planned, reviewable, and testable subtask with one shared evaluator core and two entry modes. Training may trigger evaluation, but evaluation is no longer treated as a hidden tail step. Long-running evaluation must always be observable, and its efficiency must be checked whether it starts from checkpoint or from live in-memory model state.
