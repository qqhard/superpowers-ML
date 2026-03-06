# Gradient Checks

## NaN/Inf Detection

```python
# Check every step during initial validation
for name, param in model.named_parameters():
    if param.grad is not None:
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
        assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
```

## Gradient Norm Monitoring

```python
# Track gradient norms per layer over steps
def compute_grad_norms(model):
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()
    return norms

# After each step:
step_norms = compute_grad_norms(model)
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))

# Thresholds (adjust based on model):
# - Total norm < 1e-7: likely vanishing
# - Total norm > 1e3: likely exploding
# - Sudden 10x jump: gradient spike
```

## Gradient Distribution

```python
# Check gradient histogram for degenerate distributions
for name, param in model.named_parameters():
    if param.grad is not None:
        g = param.grad.flatten()
        # Not all zeros
        assert g.abs().max() > 0, f"All-zero gradient in {name}"
        # Not all same value
        assert g.std() > 0, f"Constant gradient in {name}"
        # Reasonable kurtosis (not too peaked)
```
