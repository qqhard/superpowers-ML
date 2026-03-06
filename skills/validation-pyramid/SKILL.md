---
name: validation-pyramid
description: Use when validating ML training code correctness - orchestrates layered checks from engineering efficiency through end-to-end pipeline, replacing traditional TDD for ML workflows
---

# Validation Pyramid

## Overview

The Validation Pyramid replaces traditional TDD for ML workflows. Instead of "write test, watch fail, write code, watch pass," it runs layered checks from cheap/fast (L0) to expensive/slow (L3), catching implementation errors before they waste GPU hours.

**Core principle:** In ML, code running without errors does NOT mean it's correct. The Validation Pyramid ensures the process is correct, so you can trust that "not working" means the strategy is ineffective — not that the implementation is wrong.

**This is a RIGID skill.** Follow exactly. Don't skip layers. Don't adapt away discipline.

## When to Use

- After implementing any ML code (model, training loop, data pipeline, custom layer)
- During each subtask in an experiment plan
- When ml-diagnostics identifies an issue and you need to re-validate after fixing

## Orchestration Logic

```dot
digraph validation_pyramid {
    "Read validation scope from design doc" [shape=box];
    "L0: Engineering Efficiency" [shape=box];
    "L0 passed?" [shape=diamond];
    "L1: Process Metrics" [shape=box];
    "L1 passed?" [shape=diamond];
    "L2: Overfitting Test" [shape=box];
    "L2 passed?" [shape=diamond];
    "L3: End-to-End Pipeline" [shape=box];
    "L3 passed?" [shape=diamond];
    "All enabled layers passed" [shape=doublecircle];
    "Trigger ml-diagnostics" [shape=box, style=filled, fillcolor="#ffcccc"];

    "Read validation scope from design doc" -> "L0: Engineering Efficiency";
    "L0: Engineering Efficiency" -> "L0 passed?";
    "L0 passed?" -> "L1: Process Metrics" [label="yes"];
    "L0 passed?" -> "Trigger ml-diagnostics" [label="no"];
    "L1: Process Metrics" -> "L1 passed?";
    "L1 passed?" -> "L2: Overfitting Test" [label="yes"];
    "L1 passed?" -> "Trigger ml-diagnostics" [label="no"];
    "L2: Overfitting Test" -> "L2 passed?";
    "L2 passed?" -> "L3: End-to-End Pipeline" [label="yes"];
    "L2 passed?" -> "Trigger ml-diagnostics" [label="no"];
    "L3: End-to-End Pipeline" -> "L3 passed?";
    "L3 passed?" -> "All enabled layers passed" [label="yes"];
    "L3 passed?" -> "Trigger ml-diagnostics" [label="no"];
}
```

**Rules:**
1. Execute layers in order: L0 -> L1 -> L2 -> L3
2. Skip layers marked as "skip" in design doc
3. Each layer must pass before proceeding to next
4. Failure at any layer -> trigger **mlsp:ml-diagnostics**
5. After diagnostics fix -> re-run from the failed layer, not from L0

## How to Use

1. Read the validation scope from the brainstorm design doc
2. For each enabled layer, invoke the corresponding vp-* skill:
   - L0: **mlsp:vp-engineering-efficiency**
   - L1: **mlsp:vp-process-metrics**
   - L2: **mlsp:vp-overfitting-test**
   - L3: **mlsp:vp-e2e-pipeline**
3. The vp-* skill tells you what to check, what tools to use, how to interpret results
4. Record pass/fail for each layer

## Dynamic Selection Within Layers

Each layer has universal checks and architecture-specific checks. See `decision-tree.md` for which sub-checks to load based on architecture type.

## Hierarchical Decomposition on Failure

When a layer fails:
```
Overall metric not meeting target
    -> Decompose into substructures (defined in brainstorm)
    -> Validate each substructure with mock data
    -> Locate bottleneck substructure
    -> Drill to operator level if needed
```

## Three Granularity Levels

| Granularity | When | Method |
|-------------|------|--------|
| Function/operator | Custom loss, custom layer, single operator | Traditional unit test, deterministic |
| Module/layer | Network substructure efficiency | Mock data, per-segment profiling |
| Experiment | Full L0-L3 pyramid | Training process metrics |

## Quick Reference

See `layer-overview.md` for a compact table of all layers, metrics, and thresholds.

## Red Flags

- Skipping a layer because "it's probably fine"
- Running L2 before L0/L1 pass
- Ignoring a failed layer and proceeding
- Not re-running after a fix
- "I'll validate later" — validate NOW

## Integration

- **mlsp:ml-brainstorming** — Defines validation scope
- **mlsp:ml-diagnostics** — Triggered on failure
- **mlsp:ml-experiment-planning** — Each subtask specifies which layers apply
