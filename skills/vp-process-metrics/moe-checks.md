# MoE Checks

## Expert Load Balance

```python
# Track token routing to each expert
expert_counts = {}

def router_hook(name):
    def hook(module, input, output):
        # output typically includes routing decisions
        # Adjust based on your MoE implementation
        if hasattr(output, 'router_logits'):
            routing = output.router_logits.argmax(dim=-1)
            counts = torch.bincount(routing.flatten(), minlength=module.num_experts)
            expert_counts[name] = counts.float()
    return hook

# After collecting:
for name, counts in expert_counts.items():
    total = counts.sum()
    probs = counts / total
    # Entropy of routing distribution
    entropy = -(probs * (probs + 1e-10).log()).sum()
    max_entropy = torch.log(torch.tensor(len(probs), dtype=torch.float))
    balance = (entropy / max_entropy).item()

    print(f"{name}: balance={balance:.3f} (1.0 = perfect)")
    # Flag if balance < 0.5 (experts very unevenly loaded)
    assert balance > 0.5, f"Poor expert balance in {name}: {balance:.3f}"
```

## Auxiliary Loss

```python
# Verify auxiliary balancing loss is active and reasonable
# This is typically a load-balancing loss added to the main loss
# Check it's not zero (disabled) and not dominating (too large)
if hasattr(model, 'aux_loss'):
    aux = model.aux_loss.item()
    main = loss.item()
    ratio = aux / (main + 1e-10)
    print(f"Aux loss ratio: {ratio:.3f}")
    assert ratio < 0.1, f"Aux loss too large relative to main loss: {ratio:.3f}"
    assert aux > 0, "Aux loss is zero — load balancing may be disabled"
```
