# bdh-fast-weights

The first working open-source implementation of Hebbian fast weight write-back for the BDH (Dragon Hatchling) architecture. BDH is described in [arxiv:2509.26507](https://arxiv.org/abs/2509.26507) (Kosowski et al., 2025). The paper describes a Hebbian synaptic plasticity mechanism where weights update during inference — but the [released code](https://github.com/pathwaycom/bdh) computes the co-activation product and discards it. This repo implements write-back and shows it works, first on synthetic benchmarks and then on natural language.

This is a mechanism proof, not a product.

---

## Results: v0.3 — Natural language and architectural improvements

### What changed

- Raw byte tokenisation on FineWeb-Edu: first natural language training results
- Per-example fast state: bs=1 constraint eliminated, 140x throughput improvement
- Consolidation (`row_topk` selective writeback) validated on natural language across 2 seeds

### Natural language results

![Validation BPB over training](docs/img/val_bpb_comparison.png)

All four runs completed. 3-hour budget on RTX 5070 Ti, context_len=192, d_model=768, batch_size=48.

| Run | Seed | Best val BPB | Best step | Tokens |
|---|---|---|---|---|
| Baseline (no fast weights) | 2026 | 1.6774 | 12000 | 111M |
| Fast weights, no consolidation | 2026 | 1.6768 | 12000 | 111M |
| Fast weights + row_topk consolidation | 1337 | 1.6690 | 11400 | 105M |
| Fast weights + row_topk consolidation | 2026 | **1.6685** | 12000 | 113M |
| Fast weights + row_topk consolidation | 2027 | 1.6832 | 12300 | 116M |

Seed 2027 did not improve over the seed 2026 baseline; no seed 2027 baseline was run, so a paired comparison is not possible.

Fast weights alone match baseline (1.6768 vs 1.6774 — within noise). Row_topk consolidation improves over baseline by ~0.009 BPB at seeds 1337 and 2026. A third seed (2027) produced 1.6832 — worse than the 2-seed result, but without a matched seed 2027 baseline the comparison is not clean.

Raw byte tokenisation is a deliberate choice. BDH's addressing mechanism uses token identity codes, and byte-level tokens provide stable, consistent addresses across all input. These are 3-hour runs on a small model. The BPB numbers are not comparable to tokenised models or larger architectures.

### Training dynamics

![Fast weight activation — early training](docs/img/fast_state_norm_bootstrap.png)

`fast_state_norm` is non-zero from step 1 and grows throughout training, confirming Hebbian updates are accumulating in the fast buffer immediately. An early peak appears around step 41 (norm ~4.1), followed by a retreat as slow weights adapt, then steady growth to a sustained plateau.

![Fast state norm — full training run](docs/img/fast_norm_sustained.png)

`fast_state_norm` stabilises near 11.2 and holds there across the full training run. Baseline `fast_state_norm` is 0.0 throughout — the mechanism is not active without write-back.

By mid-training, fast weights contribute roughly twice as much as slow weights to the read output: `fast_contrib_norm` ≈ 6.9, `slow_contrib_norm` ≈ 3.5. This is not forced by the architecture — the model learned to use the fast buffer.

### bs=1 constraint removed

![Training throughput comparison](docs/img/throughput_comparison.png)

v0.2 required bs=1: the shared fast buffer caused batch contamination when sequences were mixed. v0.3 uses per-example fast state with shape `[batch, memory_size, d_model]`, with an explicit forward API:

```python
logits, next_fast_state = model(tokens, fast_state=fast_state)
```

At bs=48, each training step processes 9,216 tokens vs 66 in v0.2 — 140x more per step at similar step cost.

Honest cost: fast weight training runs at ~10.6k tokens/s. The baseline runs at ~65k tokens/s. This is the cost of the per-token Hebbian update. It is not yet parallelised within a sequence.

---

## Results: v0.2 — Consolidation and confirmation

v0.2 asked whether fast weights can consolidate into slow weights between sequences without destroying the associative memory signal. Dense writeback did destroy it. Selective writeback (`row_topk`) preserved it.

| Run | n2 | n4 | n8 | Interpretation |
|---|---|---|---|---|
| batch1-control-bpe192-opt3e-3-hebb1e-2-consol0-s1337 | 97.2% | 95.5% | 97.4% | no fast-to-slow consolidation |
| batch1-dense-bpe192-opt3e-3-hebb1e-2-consol1e-4-s1337 | 75.4% | 68.1% | 89.8% | dense writeback degrades the signal |
| batch1-rowtop10-bpe192-opt3e-3-hebb1e-2-consol1e-4-s1337 | 97.5% | 97.1% | 96.2% | selective writeback preserves most of the control signal |

`rowtop10`: after each episode, only the top 10% of decoder rows by episode-local fast-row activity are written from fast weights into slow weights.

Independent H100 verification (seed 2026):

| Run | n2 | n4 | n8 |
|---|---|---|---|
| verify-control-bpe192-s2026 | 72.7% | 80.6% | 85.2% |
| verify-rowtop10-bpe192-s2026 | 92.5% | 93.9% | 91.2% |

Counter-benchmarks for verify-rowtop10: 93.0% vargap1, 92.6% vargap2, 91.6% repeated8, 95.0% n16.

Confirmation runs at 2 independent seeds:

| Run | n2 | n4 | n8 |
|---|---|---|---|
| confirm-bpe-hebb-bpe192-lr3e-3-s1337 | 99.0% | 98.0% | 97.5% |
| confirm-bpe-hebb-bpe192-lr3e-3-s2026 | 88.7% | 88.8% | 92.0% |

Counter-benchmarks on confirmation runs:

| Run | vargap1 | vargap2 | repeated8 | n16 |
|---|---|---|---|---|
| confirm-bpe-hebb-bpe192-lr3e-3-s1337 | 95.8% | 93.0% | 97.2% | 96.8% |
| confirm-bpe-hebb-bpe192-lr3e-3-s2026 | 87.8% | 87.4% | 84.8% | 94.8% |

Five implementation bugs were found and fixed during development. See [BUGS.md](BUGS.md).

Note on eval methodology: an earlier version of the counter-benchmarks used autoregressive generation and showed apparent collapse — that was a scorer mismatch, not a mechanism failure. Both logs are preserved in `results/` for audit.

---

## Results: v0.1 — Mechanism proof (synthetic benchmarks)

v0.1 established that Hebbian fast-weight write-back works on synthetic n-back associative recall. The only path to above-chance performance is correctly writing and reading associations — no language modelling shortcuts are available. Baseline (no write-back): 1.0% n8 (random chance for vocab size 64).

| Run | n2 | n4 | n8 | Stopped |
|---|---|---|---|---|
| baseline-bpe192-nohebb | 0.0% | 0.0% | 1.0% | converged (~2 min) |
| bpe-hebb-bpe192-lr3e-3 | 89.4% | 87.2% | 95.1% | time limit (2h) |
| bpe-hebb-bpe192-lr1e-2 | 25.8% | 37.2% | 91.3% | time limit (2h) |
| bpe-hebb-bpe192-lr3e-2 | 18.0% | 22.2% | 72.2% | time limit (2h) |

All Hebbian runs were still improving at the 2-hour cutoff.

---

## The mechanism

At each token step the model projects the input through a ReLU-gated sparse encoder to produce an address code. The same token always produces the same code regardless of position. An outer product of this code with the decoder input is accumulated into a fast weight buffer using a Hebbian learning rate.

At read time, slow and fast weights combine:

```
output = xy @ decoder + x_sparse @ decoder_fast
```

Slow weights (`decoder`) hold general knowledge learned by gradient descent. Fast weights (`decoder_fast`) hold associations from the current sequence, written by Hebbian update. The fast buffer is zeroed before each sequence — no cross-sequence memory accumulation.

---

## Architecture constraints

**Batch size.** Resolved in v0.3. v0.2 required bs=1 — shared fast buffer. v0.3 uses per-example fast state, enabling batch_size=48.

**Throughput.** ~10.6k tokens/s with fast weights vs ~65k tokens/s baseline. Cost of the per-token Hebbian update. Not yet parallelised within a sequence.

**Episodic reset is mandatory.** The fast buffer is zeroed before each sequence. No cross-sequence memory accumulation.

**Context window.** v0.1/v0.2 used context_len=128. v0.3 uses context_len=192. v0.4 will investigate the minimum viable floor.

---

## How to reproduce

```bash
pip install torch numpy flask    # flask only needed for dashboard
python prepare.py                # generates benchmarks (one-time)
```

Reproduce the baseline (no write-back):
```bash
HEBBIAN_TOKEN_MODE=bpe HEBBIAN_TOKENIZER_LABEL=bpe-192 HEBBIAN_HEBBIAN_LR=0 \
  HEBBIAN_EXPERIMENT_LABEL=baseline-bpe192-nohebb python train.py
```

Reproduce the best v0.1/v0.2 run (lr=3e-3):
```bash
HEBBIAN_TOKEN_MODE=bpe HEBBIAN_TOKENIZER_LABEL=bpe-192 HEBBIAN_HEBBIAN_LR=3e-3 \
  HEBBIAN_EXPERIMENT_LABEL=bpe-hebb-bpe192-lr3e-3 python train.py
```

Run counter-benchmarks against a saved checkpoint:
```bash
python eval_counter.py results/checkpoints/<checkpoint>.pt
```

`prepare.py` contains the fixed eval and is never modified — this prevents reward hacking. All result logs are append-only. Nothing is overwritten.

### Hardware

- RTX 5070 Ti Laptop GPU (11GB VRAM, CUDA 12.0)
- RunPod cloud GPU (24GB) for overnight sweeps

### Dashboard

```bash
python dashboard.py              # http://localhost:5000
```

### Results files

- `results/experiment_log.jsonl` — all BPE campaign runs with final scores
- `results/counter_log.jsonl` — counter-benchmark results (corrected teacher-forced eval)
- `results/counter_log_original_eval.jsonl` — original autoregressive eval results, preserved for audit
- `results/eval_history.jsonl` — per-checkpoint eval scores across training
- `results/char_level_experiment_log.jsonl` — all 16 character-level experiments from the initial research phase
- `snapshots/exp11_xsprev_shared_fastbuf_lr1e2_bs1.py` — frozen standalone script for the first working experiment (char-level, n8=64%), independently runnable

---

## What this is

A mechanism proof. Hebbian fast-weight write-back works on synthetic associative recall benchmarks and activates on natural language text. Row_topk consolidation shows a ~0.009 BPB improvement at 2 of 3 seeds. A third seed (2027) did not reproduce the improvement; a matched baseline for that seed was not run. Results are reproducible across independent seeds on both synthetic and natural language benchmarks.

## What this is not

Not yet validated for general language modelling tasks. The 0.009 BPB consolidation improvement is real and reproducible, but it is small and comes from 3-hour runs on a small model. Not a replacement for RAG or fine-tuning. Not the full BDH internal implementation — Pathway has not released that, and this is not it.

Eval is teacher-forced, not autoregressive. Reported accuracies reflect teacher-forced exact-span scoring.

---

## Future work

- v0.4: minimum viable context window experiment — find the floor below which fast-weight addressing degrades
- v0.5: compression layer for MCP serving
- Longer training runs — current results are time-limited
- Natural language benchmark performance — not mechanism activation, but demonstrable downstream improvement
- Batched evaluation (now unblocked by per-example fast state)

---

## Compute and support

This project runs on a single consumer GPU (RTX 5070 Ti Laptop, 11GB VRAM). Longer runs and larger models would strengthen these results.

If you want to help:
- **GitHub Sponsors**: [github.com/sponsors/fleeb83](https://github.com/sponsors/fleeb83)
- **Checkpoints**: Not currently hosted due to storage limits. Hugging Face hosting would make this practical. If you can help, open an issue.
- **Collaboration**: Issues and pull requests are open.
- **Compute**: If you have GPU time and want to run validation experiments, get in touch.

No obligation. Apache 2.0.

Contact: open a GitHub issue or find Russell Thomas on Reddit (u/fleeb83).

---

## Prior art note

The only other known open-source attempt at BDH Hebbian write-back is [`adamskrodzki/bdh`](https://github.com/adamskrodzki/bdh). A full audit found the forward pass is byte-for-byte identical to upstream — a single unused `lm_gate` parameter was added but never wired in. The mechanism was not implemented.

---

## Acknowledgements

- **BDH architecture**: Kosowski, Uznański, Chorowski, Stamirowska, Bartoszkiewicz — [arxiv:2509.26507](https://arxiv.org/abs/2509.26507), [github.com/pathwaycom/bdh](https://github.com/pathwaycom/bdh)
- **Fast Weight Programmers**: Schlag, Irie, Schmidhuber — [arxiv:2102.11174](https://arxiv.org/abs/2102.11174)
- **In-context learning as gradient descent**: von Oswald et al. — [arxiv:2212.07677](https://arxiv.org/abs/2212.07677)
- **Autoresearch framework**: Andrej Karpathy — [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)

---

## Author

**Russell Thomas** — Independent researcher, Kaniva, Victoria, Australia.

No university affiliation. No corporate lab.

## License

Apache 2.0 — see [LICENSE](LICENSE).
