# Validation Pyramid Decision Tree

## How to select checks within each layer

Based on the ML context collected during brainstorm, load the appropriate sub-checks.

### L0: Engineering Efficiency

| Context | Load |
|---------|------|
| Always | `backend-checks.md` (core backend verification) |
| Always | `gpu-utilization.md` (MFU/TCA/memory) |
| Multi-GPU or multi-node | `distributed-training.md` (NCCL, communication overhead) |

### L1: Process Metrics

**Universal (always load):**
- `gradient-checks.md` — NaN/Inf, distribution, vanishing/exploding
- `parameter-drift.md` — drift from initialization, loss spike

**Architecture-specific:**

| Architecture | Load |
|-------------|------|
| Transformer | `activation-checks.md` (attention distribution, entropy) |
| Residual networks | `residual-stream.md` (write ratio) |
| MoE | `moe-checks.md` (entropy, load balance) |
| RecSys | `embedding-checks.md` (norm stability, negative sampling) |
| LLM | `token-loss-checks.md` (per-token loss), `kv-cache-checks.md` |

Multiple can apply (e.g., Transformer + MoE + LLM).

### L2: Overfitting Test

Always the same procedure. Task-specific thresholds:

| Task | Success Criteria |
|------|-----------------|
| RecSys | NDCG@10 / Recall@10 near 1.0 |
| LLM | Perplexity < 1.1 or loss < 0.01 |
| General | Training loss steadily decreasing (前半 avg > 后半 avg, 下降比例 >= 60%) |

### L3: End-to-End Pipeline

Always the same: full flow on tiny data. No sub-selection needed.
