# KV Cache Checks (LLM)

## Memory Growth

```python
# Verify KV cache memory grows linearly with sequence length
def check_kv_cache_growth(model, tokenizer, prompt="Hello", max_steps=10):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
    memory_per_step = []

    past_key_values = None
    for step in range(max_steps):
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            outputs = model(
                input_ids if step == 0 else next_token,
                past_key_values=past_key_values,
                use_cache=True
            )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        memory_per_step.append(torch.cuda.max_memory_allocated() / 1e6)

    # Check growth is roughly linear
    if len(memory_per_step) >= 3:
        diffs = [memory_per_step[i+1] - memory_per_step[i] for i in range(1, len(memory_per_step)-1)]
        avg_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)

        print(f"KV cache growth per step: avg={avg_diff:.1f}MB, max={max_diff:.1f}MB")
        # Flag if any step grows much more than average (memory leak or unexpected allocation)
        assert max_diff < avg_diff * 3, f"Non-linear KV cache growth detected: max={max_diff:.1f}MB vs avg={avg_diff:.1f}MB"
```
