## v0.3: Mechanism confirmed active on natural language

### What is new

- **Mechanism activation confirmed on natural language.** fast_state_norm grows from step 1 on FineWeb-Edu, stabilises near 11.2, and holds. Baseline fast_state_norm is 0.0. Fast weights contribute roughly twice as much as slow weights by mid-training (around step 6,000). The model learned to use the fast buffer.
- **Row_topk consolidation improves BPB at 2 of 3 seeds.** ~0.009 BPB improvement over baseline on FineWeb-Edu. Third seed did not reproduce it.
- **bs=1 constraint removed.** Per-example fast state [batch, memory_size, d_model] enables batch_size=48. 140x more tokens per step vs v0.2.

### Mechanism activation

| Metric | With write-back | Baseline |
|---|---|---|
| fast_state_norm (plateau) | ~11.2 | 0.0 |
| fast_contrib_norm (mid-training) | ~6.9 | 0.0 |
| slow_contrib_norm (mid-training) | ~3.5 | ~3.5 |

fast_state_norm is non-zero from step 1. An early peak appears near step 41 (norm ~4.1), then a retreat as slow weights adapt, then steady growth to plateau. The baseline shows 0.0 throughout. The mechanism does not activate without write-back.

### BPB results

5 runs on FineWeb-Edu, raw byte tokenisation, 3-hour budget, RTX 5070 Ti, context_len=192.

| Run | Seed | Best val BPB | Best step | Tokens |
|---|---|---|---|---|
| Baseline (no fast weights) | 2026 | 1.6774 | 12000 | 111M |
| Fast weights, no consolidation | 2026 | 1.6768 | 12000 | 111M |
| Fast weights + row_topk consolidation | 1337 | 1.6690 | 11400 | 105M |
| Fast weights + row_topk consolidation | 2026 | **1.6685** | 12000 | 113M |
| Fast weights + row_topk consolidation | 2027 | 1.6832 | 12300 | 116M |

Fast weights alone match baseline within noise. Row_topk consolidation improves BPB by ~0.009 at seeds 1337 and 2026. Seed 2027 did not reproduce it. No matched seed 2027 baseline was run.

Raw byte BPB is not comparable to tokenised models or other architectures.

### bs=1 fix

v0.2 required bs=1. The shared fast buffer caused batch contamination when sequences were mixed. v0.3 uses per-example fast state:

```python
logits, next_fast_state = model(tokens, fast_state=fast_state)
```

At bs=48, each step processes 9,216 tokens vs 66 in v0.2 (140x more per step at similar step cost).

Throughput cost: fast weight training runs at ~10,600 tokens/s. Baseline: ~65,000 tokens/s. The per-token Hebbian update is not yet parallelised within a sequence.

### Limitations

- Throughput: ~10,600 tokens/s vs ~65,000 tokens/s baseline. Not yet parallelised within a sequence.
- BPB improvement reproduced at 2 of 3 seeds. Third seed (2027) did not reproduce it. 3-hour budget, small model, raw bytes.
- Not validated on downstream NLP tasks.
- Raw byte BPB is not directly comparable to tokenised models or other architectures.

### Reproducing v0.3

3 conditions reproduce the full table: baseline (no fast weights), fast weights with no consolidation, fast weights with row_topk consolidation. Each run takes approximately 3 hours on a consumer GPU with 11GB VRAM. All result logs in results/ are append-only.
