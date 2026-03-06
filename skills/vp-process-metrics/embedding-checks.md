# Embedding Checks (RecSys)

## Embedding Norm Stability

```python
# Track L2 norm of user/item embeddings over training steps
def check_embedding_norms(model, step):
    for name, param in model.named_parameters():
        if 'embed' in name.lower():
            norms = param.data.norm(dim=-1)
            print(f"Step {step} | {name}: mean_norm={norms.mean():.3f}, max={norms.max():.3f}, min={norms.min():.3f}")

            # Norms should be stable, not exploding
            assert norms.max() < 100, f"Embedding norm exploding in {name}: max={norms.max():.1f}"
            # Norms shouldn't collapse to zero
            assert norms.min() > 1e-4, f"Embedding norm collapsed in {name}: min={norms.min():.6f}"
```

## Negative Sampling Quality

```python
# Verify negative samples are actually harder than random
# positive score should be > negative score on average
def check_negative_sampling(model, pos_pairs, neg_pairs):
    with torch.no_grad():
        pos_scores = model.score(pos_pairs)
        neg_scores = model.score(neg_pairs)

    print(f"Pos score mean: {pos_scores.mean():.4f}")
    print(f"Neg score mean: {neg_scores.mean():.4f}")

    assert pos_scores.mean() > neg_scores.mean(), \
        "Negative samples scoring higher than positives — check sampling logic"
```

## Popularity Bias

```python
# Check that popular item embeddings aren't collapsing to be too similar
def check_popularity_bias(item_embeddings, popular_indices, threshold=0.95):
    popular_embeds = item_embeddings[popular_indices]
    # Cosine similarity between popular items
    normed = F.normalize(popular_embeds, dim=-1)
    sim_matrix = normed @ normed.T
    # Exclude diagonal
    mask = ~torch.eye(len(popular_indices), dtype=torch.bool, device=sim_matrix.device)
    avg_sim = sim_matrix[mask].mean().item()

    print(f"Avg cosine similarity between popular items: {avg_sim:.3f}")
    assert avg_sim < threshold, f"Popular items too similar ({avg_sim:.3f}) — embedding collapse"
```
