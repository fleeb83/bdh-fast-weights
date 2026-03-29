# BDH Fast Weights v.2

## Working title

Selective fast-to-slow consolidation preserves the core signal

## Announcement

v.2 extends this repo from proving Hebbian fast weights work inside an episode to testing whether those fast updates can be selectively consolidated into slow weights without destroying the memory effect. v.2 verifies that selective fast-to-slow consolidation is viable in this benchmark: `rowtop10` reproduces on an independent H100 run, and in the batch1 comparison it remains clearly stronger than dense writeback.

## Release notes

- The original fast-weight result still stands: the public repo already shows strong BPE-192 associative recall with confirmation runs reaching `97.5%` and `92.0%` on `n8`.
- v.2 adds the first public selective fast-to-slow consolidation result to the repo narrative.
- On the matched batch-1 comparison, no-consolidation control reached `97.2 / 95.5 / 97.4` on `n2 / n4 / n8`.
- Dense consolidation at `1e-4` degraded the same benchmark materially, falling to `75.4 / 68.1 / 89.8`.
- `rowtop10`, which writes back only the top 10% of rows by episode-local fast-row activity, reached `97.5 / 97.1 / 96.2`.
- The independent H100 verification package reproduced the result on seed 2026: matched control reached `72.7 / 80.6 / 85.2`, while `verify-rowtop10` reached `92.5 / 93.9 / 91.2`.
- The implication is narrow but important: selective fast-to-slow consolidation appears viable in this system, while dense writeback is too destructive.
- Verification counter-benchmarks for `verify-rowtop10` also stayed strong: `vargap1 = 93.0%`, `vargap2 = 92.6%`, `repeated8 = 91.6%`, and `n16 = 95.0%`.

## Why this matters

The architectural significance is not just that fast weights work, but that some of their episode-local information may be written back into persistent slow weights without erasing the learned behavior. That is the bridge from temporary associative memory toward longer-lived parameter change. The current evidence supports that bridge as feasible, but not yet as fully solved or globally optimal.

## Before push

- Re-read README and this file once more for wording now that the verification data is filled in.
- Decide whether to mention the matched verification control explicitly in the final public summary or keep that detail in the tables/results logs only.
