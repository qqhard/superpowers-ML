# Parameter Drift

## Track Drift from Initialization

```python
import copy

# Before training: snapshot initial parameters
init_params = {name: param.data.clone() for name, param in model.named_parameters()}

# After N steps: compute drift
def compute_drift(model, init_params):
    drift = {}
    for name, param in model.named_parameters():
        if name in init_params:
            diff = (param.data - init_params[name]).norm()
            init_norm = init_params[name].norm()
            drift[name] = {
                'absolute': diff.item(),
                'relative': (diff / (init_norm + 1e-10)).item()
            }
    return drift

drift = compute_drift(model, init_params)
for name, d in drift.items():
    # Parameters should be changing (training is happening)
    assert d['absolute'] > 0, f"{name} hasn't changed from init — not being trained?"
    # But not too fast (explosion)
    assert d['relative'] < 10.0, f"{name} drifted {d['relative']:.1f}x from init — too fast"
```

## Loss Spike Detection

```python
# Track loss over steps
loss_history = []

# After each step:
loss_history.append(loss.item())

# Check for spikes
if len(loss_history) > 10:
    window = loss_history[-10:]
    moving_avg = sum(window) / len(window)
    current = loss_history[-1]

    if current > moving_avg * 10:
        print(f"WARNING: Loss spike detected! Current={current:.4f}, Moving avg={moving_avg:.4f}")
```

## Loss Trend (Supervised Tasks)

```python
# For supervised learning, loss should decrease in early steps
if len(loss_history) >= 20:
    first_half = sum(loss_history[:10]) / 10
    second_half = sum(loss_history[10:20]) / 10
    assert second_half < first_half, \
        f"Loss not decreasing: first 10 steps avg={first_half:.4f}, next 10 avg={second_half:.4f}"
```
