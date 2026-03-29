# ML Static Analysis Checklist

## How to Use

For each check: evaluate the **Condition** column against the code being reviewed. If the condition is met, verify the code satisfies **What to verify**. If not met, skip the check.

## Mandatory (Critical) — checks 1-6, 19-20, 24

Failure blocks progress. Implementer must fix before proceeding.

| # | Check | Condition | What to verify |
|---|-------|-----------|---------------|
| 1 | Device consistency | Uses CUDA | Model, data, loss on same device |
| 2 | Precision config | Has mixed precision / bf16 / fp16 | param.dtype matches expectation, autocast correct |
| 3 | FlashAttention | Has Attention layers | FA available and enabled |
| 4 | Optimizer coverage | Always | optimizer param_groups covers all trainable params |
| 5 | LR scheduler | Has lr_scheduler | Correctly linked to optimizer |
| 6 | DataLoader config | Has DataLoader | num_workers, pin_memory reasonable |
| 19 | Loss file output | Always | Code writes loss values to a file (not only stdout) — file-writing patterns associated with loss |
| 20 | Step speed / throughput file output | Always | Code writes step time or throughput to a file |
| 24 | Visualization tool correctness | Enabled in experiment design doc | Selected tool has init + log calls + frequency control; skip if not enabled |

## Advisory (Warning) — checks 7-18, 21-23, 25

Do not block progress. Report as warnings. Implementer may fix or acknowledge and proceed.

| # | Check | Condition | What to verify |
|---|-------|-----------|---------------|
| 7 | Data loading method | Dataset declared as large in brainstorming, or file size > 10GB | Uses mmap loading |
| 8 | Padding waste | Variable-length sequences | Excessive padding or tail-wave waste |
| 9 | Random seeds | Always | torch/np/random seeds set |
| 10 | Gradient accumulation consistency | Has gradient accumulation | accumulation_steps × micro_batch = global_batch |
| 11 | Loss reduction | Has gradient accumulation | mean vs sum matches accumulation strategy |
| 12 | Vocab/Embedding match | Has Embedding layer | tokenizer vocab_size == embedding dim |
| 13 | Frozen layers | Fine-tuning | Frozen layers match expectations |
| 14 | FLOPs estimate | Always | Estimate from model architecture (param dims, known FLOPs-per-op), order-of-magnitude check. NOT FlopCounterMode — that requires runtime, used in L1 |
| 15 | GPU hardware info | Uses CUDA | Review code for target GPU assumptions; actual hardware detection happens in L1 |
| 16 | Memory estimate | Always | Estimate from param count + dtype + expected activations; check if theoretical footprint fits target GPU capacity |
| 17 | MoE backend | MoE architecture | Expert parallel, routing optimization, aux loss |
| 18 | CUDA kernel selection | Uses CUDA | Optimized kernels, not fallback |
| 21 | Data loading duration log | Has DataLoader | Code records data loading start/end/duration |
| 22 | Output frequency control | Has file logging or console output | **Console**: tqdm (preferred) or print has minute-level frequency control — tqdm via `mininterval`, print via `if step % N == 0` with time-based gating. **File**: log output has interval control; not every step |
| 23 | Console metrics display | Always | Console output (tqdm or print) carries key runtime metrics (at least loss) — via tqdm `set_postfix` or formatted print string |
| 25 | Checkpoint interval configurability | Has checkpoint saving | Checkpoint save interval is configurable (via argument/config, not hardcoded) and default value is reasonable |

## Adding New Checks

When a new common ML agent mistake is identified:
1. Add it to the appropriate tier (Mandatory or Advisory)
2. Define a clear applicability condition
3. Describe what to verify in specific, actionable terms
4. Update the agent definition in `agents/ml-static-checks.md` to match
