# Limitations

This is a mechanism proof on synthetic benchmarks. The following limitations are known and documented honestly.

## Training budget

All Hebbian runs in the BPE campaign were stopped by the 2-hour wall-clock limit, not by convergence. The baselines converged early (~6000 steps, ~2 minutes) because there is nothing to learn without write-back. The Hebbian runs were still improving at cutoff — step counts of 10855–10896 (base width) and 5979–5981 (wide, slower per step) with no plateau in the eval history. Reported scores are best-within-budget. The true performance ceiling has not been found.

## Proven limitations

- **Batch size 1 is a hard architectural constraint.** The fast buffer cannot be shared across sequences without destructive interference. This limits training throughput and inference parallelism.

- **Not validated on natural language.** All results are on synthetic n-back associative recall. Performance on real language modelling tasks is unknown.

- **BPE tokenizer is task-specific.** The BPE-192 vocabulary was built for the synthetic benchmark. A different tokenizer would be needed for natural language tasks and the mechanism's behaviour may differ.

- **Single model size tested.** Results are from a small BDH model. Scaling behaviour to larger parameter counts is unknown.

## Eval caveats

- **Counter-benchmarks were initially broken.** The original counter-benchmark eval used greedy autoregressive generation while the clean eval used teacher forcing. This made counter-benchmarks appear to collapse. The corrected eval (teacher-forced, same as clean eval) shows strong generalisation. Both the original and corrected logs are preserved for audit.

- **Eval is teacher-forced, not autoregressive.** The reported accuracies reflect teacher-forced exact-span scoring. Autoregressive generation accuracy would be lower, particularly for multi-token BPE answer spans.

## Open questions

- Does the mechanism improve language modelling perplexity, or only associative recall?
- What is the capacity of the fast buffer — how many facts can it store before interference?
- Can the addressing mechanism be learned rather than hardcoded to t-1?
- Does the mechanism transfer across domains and data distributions?
- What happens to fast buffer quality over very long sequences (1000+ tokens)?

## What this does not claim

This work does not claim to reproduce Pathway's internal BDH implementation. The 97.4% Sudoku Extreme result reported by Pathway uses their proprietary build. This is an independent implementation of the Hebbian write-back mechanism described in the paper, arriving at a working solution through a different experimental path.
