# Token Loss Checks (LLM)

## Per-Token Loss Distribution

```python
# Check that loss is not concentrated on few tokens
def check_token_loss_distribution(model, input_ids, labels):
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        # Get per-token loss
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

    # Reshape and analyze
    valid = shift_labels.view(-1) != -100  # ignore padding
    valid_losses = per_token_loss[valid]

    print(f"Per-token loss: mean={valid_losses.mean():.3f}, std={valid_losses.std():.3f}")
    print(f"Max token loss: {valid_losses.max():.3f}")
    print(f"Fraction > 2*mean: {(valid_losses > 2 * valid_losses.mean()).float().mean():.3f}")

    # Flag if too concentrated (most tokens trivially predicted, few very hard)
    cv = valid_losses.std() / (valid_losses.mean() + 1e-10)
    assert cv < 5.0, f"Token loss too concentrated (CV={cv:.1f}) — model may be ignoring hard tokens"
```

## Generation Diversity

```python
# Check that generated text is not degenerate (all same token, repetitive)
def check_generation_diversity(model, tokenizer, prompt, num_samples=5, max_length=50):
    generations = []
    for _ in range(num_samples):
        input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
        output = model.generate(input_ids, max_length=max_length, do_sample=True, top_k=50)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        generations.append(text)

    # Check uniqueness
    unique = len(set(generations))
    print(f"Unique generations: {unique}/{num_samples}")
    assert unique > 1, "All generations identical — sampling may be broken"

    # Check for degenerate repetition within each generation
    for gen in generations:
        tokens = gen.split()
        if len(tokens) > 5:
            unique_tokens = len(set(tokens))
            ratio = unique_tokens / len(tokens)
            assert ratio > 0.2, f"Very repetitive generation (unique ratio={ratio:.2f}): {gen[:100]}"
```
