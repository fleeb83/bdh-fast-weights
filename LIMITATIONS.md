# Limitations

This is a mechanism proof, not a product. These limitations are known and documented honestly.

## What was proven

The Hebbian fast-weight mechanism activates on natural language. `fast_state_norm` grows from step 1 on FineWeb-Edu and holds near 11.2 across the full run. The baseline `fast_state_norm` is 0.0 throughout. By mid-training, around step 6,000, fast weights contribute roughly twice as much as slow weights: `fast_contrib_norm` is approximately 6.9 and `slow_contrib_norm` is approximately 3.5. The model learned to use the fast buffer. Row_topk consolidation improves BPB by approximately 0.009 at 2 of 3 seeds on FineWeb-Edu.

## Training budget

All v0.3 runs were stopped by the 3-hour wall-clock limit, not by convergence. Reported scores are best-within-budget. The true performance ceiling has not been found.

Mechanism activation is confirmed and is not budget-limited. `fast_state_norm` grows from step 1 and stabilises at approximately 11.2 within the 3-hour window. These observations are from 3-hour runs on a single small model.

The BPB improvement from consolidation (row_topk) reproduced at 2 of 3 seeds (1337 and 2026). A third seed (2027) completed at 1.6832, worse than the 2-seed result and worse than the seed 2026 baseline (1.6774). No seed 2027 baseline was run, so a paired comparison is not possible. Whether the BPB effect scales with longer training is unknown.

## Known limitations

- **Throughput cost is large.** Fast weight training runs at approximately 10,600 tokens/s. The same architecture without fast weights runs at approximately 65,000 tokens/s. That is roughly a 6x slowdown. The cause is the per-token Hebbian update: it is applied sequentially within each sequence and is not yet parallelised. This is an implementation constraint, not a fundamental one, but it has not been solved yet.

- **Single model size tested.** All results are from a small BDH model. Scaling behaviour to larger parameter counts is unknown.

- **Not validated on downstream NLP tasks.** Evaluation is on language modelling BPB only, plus earlier synthetic associative recall benchmarks. Whether the mechanism improves task accuracy has not been tested.

- **Raw byte BPB is not comparable to tokenised models.** v0.3 uses raw byte tokenisation (vocab=256) on FineWeb-Edu. Numbers from tokenised models or different architectures cannot be compared directly.

## Resolved in v0.3

These items were listed as limitations in earlier releases and are now resolved:

- **Batch size constraint: resolved.** v0.1/v0.2 required batch size 1 because the fast state buffer could not be shared across sequences. v0.3 introduces per-example fast state with shape [batch, memory_size, d_model], enabling batch size 48.

- **Natural language validation: complete.** Earlier releases were validated on synthetic benchmarks only. v0.3 results are on FineWeb-Edu with raw byte tokenisation.

- **Task-specific tokenizer: removed.** v0.1/v0.2 used a BPE vocabulary built for the synthetic benchmark. v0.3 uses raw bytes (vocab=256), with no task-specific tokenizer.

## Eval caveats

- **Eval is teacher-forced.** All reported BPB scores use teacher-forced next-byte prediction. Autoregressive generation quality has not been measured.

- **Context length is 192 tokens.** v0.3 extended context from 128 tokens (v0.1/v0.2) to 192 tokens. Behaviour on longer sequences has not been tested.

## Open questions

- Does the consolidation benefit hold at longer training runs past 3 hours?
- What is the fast buffer capacity -- how many associations can it hold before interference?
- Does the mechanism improve accuracy on downstream NLP tasks, or only language modelling BPB?
- Can the per-token Hebbian update be parallelised to close the throughput gap?
- Does the mechanism transfer across domains and data distributions?
- What happens to fast buffer quality over sequences longer than 192 tokens?

## What this does not claim

This work does not claim to reproduce Pathway's internal BDH implementation. The 97.4% Sudoku Extreme result reported by Pathway uses their proprietary build. This is an independent implementation of the Hebbian write-back mechanism described in the paper, arrived at through a separate experimental path.

The 0.009 BPB improvement is reproducible at 2 seeds (1337 and 2026). A third seed (2027) did not reproduce it. It is not a large number. It is evidence the mechanism can work on natural language, not evidence that it is ready for use.
