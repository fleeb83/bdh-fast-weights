# bdh-fast-weights

**The first working open-source implementation of Hebbian fast weight write-back for the BDH (Dragon Hatchling) architecture.**

BDH is a biologically inspired language model architecture described in [arxiv:2509.26507](https://arxiv.org/abs/2509.26507) (Kosowski et al., 2025). The paper describes a Hebbian synaptic plasticity mechanism (Round 4l+1) where weights update during inference — but the [released code](https://github.com/pathwaycom/bdh) computes the co-activation product and discards it. Without this write-back the weights remain static throughout inference and the model cannot form new associations. The write-back was never implemented publicly.

This repo implements it, demonstrates it works, and documents the five bugs that had to be solved to get there.

---

## What this does

The model modifies its own weights during inference. When it sees a key-value association, it writes it into a fast weight buffer using the token's sparse activation code (a ReLU-gated projection through the BDH encoder — same token always produces the same code) as an address. Same token → same code → same address, regardless of position. The model can then retrieve that association from anywhere in the sequence.

This is not RAG. It's not fine-tuning. The model physically rewires itself as it processes each token.

## Results

Trained and evaluated on a synthetic n-back associative recall benchmark. The model sees key-value pairs and must retrieve the correct value for a queried key from n positions back. N-back recall isolates the memory mechanism cleanly — the only path to above-chance performance is correctly writing and retrieving associations, with no language modelling shortcuts.

Baseline (no write-back): **1.0% n8** (random chance for vocab size 64).

### Clean benchmark (BPE-192 tokenizer)

| Run | n2 | n4 | n8 | Stopped |
|---|---|---|---|---|
| baseline-bpe192-nohebb | 0.0% | 0.0% | 1.0% | converged (~2 min) |
| bpe-hebb-bpe192-lr3e-3 ★ | 89.4% | 87.2% | 95.1% | time limit (2h) |
| bpe-hebb-bpe192-lr1e-2 | 25.8% | 37.2% | 91.3% | time limit (2h) |
| bpe-hebb-bpe192-lr3e-2 | 18.0% | 22.2% | 72.2% | time limit (2h) |

All Hebbian runs were still improving when the 2-hour budget expired. These scores are best-within-budget, not a performance ceiling. The baseline converged quickly because without write-back there is nothing to learn beyond the slow weights.

### Confirmation runs (independent seeds)

The best configuration (lr=3e-3) was re-run from scratch at two independent seeds to verify reproducibility.

| Run | n2 | n4 | n8 | Stopped |
|---|---|---|---|---|
| confirm-baseline-bpe192-nohebb-s1337 | 0.0% | 0.0% | 1.6% | converged |
| confirm-bpe-hebb-bpe192-lr3e-3-s1337 ★ | 99.0% | 98.0% | 97.5% | time limit (2h) |
| confirm-bpe-hebb-bpe192-lr3e-3-s2026 | 88.7% | 88.8% | 92.0% | time limit (2h) |

Counter-benchmarks on confirmation runs:

| Run | vargap1 | vargap2 | repeated8 | n16 |
|---|---|---|---|---|
| confirm-bpe-hebb-bpe192-lr3e-3-s1337 ★ | 95.8% | 93.0% | 97.2% | 96.8% |
| confirm-bpe-hebb-bpe192-lr3e-3-s2026 | 87.8% | 87.4% | 84.8% | 94.8% |

The confirmation runs exceed the original overnight results — consistent with those runs also hitting the time limit before peaking. The mechanism holds across seeds.

### Counter-benchmarks (teacher-forced eval, same scorer as clean benchmark)

These test whether the mechanism generalises beyond clean adjacent key-value pairs. An earlier version of these benchmarks used autoregressive generation and showed apparent collapse — that was a scorer mismatch, not a mechanism failure. Both logs are preserved in `results/` for audit.

| Run | vargap1 | vargap2 | repeated8 | n16 |
|---|---|---|---|---|
| bpe-hebb-bpe192-lr3e-3 ★ | 85.2% | 84.2% | 86.4% | 88.8% |
| bpe-hebb-bpe192-lr1e-2 | 24.2% | 23.0% | 29.8% | 34.8% |
| bpe-hebb-bpe192-wide384-lr3e-3 | 21.4% | 23.6% | 24.0% | 27.8% |

`wide384` is a model ablation with `n_embd=384` instead of 256. The wider model achieves comparable clean n8 accuracy (93.2% vs 95.1%) but generalises dramatically worse on counter-benchmarks (21–28% vs 84–89%). More parameters did not help — and appear to hurt generalisation, suggesting the narrow model learns more transferable representations at this scale.

- **vargap1/2**: 1 or 2 distractor tokens between key and value
- **repeated8**: duplicate keys (last-write-wins)
- **n16**: 16 pairs per sequence (double standard length)

The lr3e-3 configuration handles all of these at 84-89%. This is a general associative memory, not an adjacency trick.

The learning rate comparison reveals why lr=3e-3 is the winner beyond raw accuracy: lr=1e-2 achieves 91.3% clean n8 but collapses to 34.8% on n16 — a 57-point drop. lr=3e-3 holds at 88.8% on n16, a 6-point drop. The lower rate learns the mechanism; the higher rate overfits to benchmark structure.

## Five bugs that had to be solved

Each one was independently sufficient to prevent the mechanism from working. They were found and fixed sequentially across 14+ experiments.

1. **Wrong target matrix** — writing back into the encoder changes how tokens are perceived, but doesn't create storage. The decoder is the correct target.

2. **Batch contamination** — batch-averaging mixes unrelated sequences' associations. The fast buffer must operate at batch size 1.

3. **Missing episodic reset** — the eval was accumulating writes from 1000 sequences into a single weight matrix. 1000 sequences of conflicting key-value pairs turning memory into noise. The fast buffer must be zeroed before each sequence.

4. **Attention-dependent addressing** — using xy_sparse or y_sparse as addresses degrades with distance because attention quality drops. Token-identity addressing (x_sparse of the token itself) produces the same code regardless of position.

5. **Eval mismatch** — BPE answer spans are 3 tokens. The counter-benchmark eval was using greedy autoregressive generation while the clean eval used teacher forcing. This made counter-benchmarks appear to collapse when the mechanism was actually working. Fixed by aligning both evals to teacher-forced scoring.

## Model size

| Model | Trainable params | Fast buffer | Total |
|---|---|---|---|
| Base (n_embd=256) | 25.3M | 8.4M | 33.7M |
| Wide (n_embd=384) | 56.8M | 18.9M | 75.6M |

The fast buffer (`decoder_fast`) is the same shape as the slow decoder — an exact parallel matrix, zeroed at the start of each sequence and written by Hebbian updates instead of gradient descent. At inference the model reads from both simultaneously: `output = xy @ decoder + x_sparse @ decoder_fast`. The slow decoder holds general knowledge; the fast decoder holds associations from this sequence only.

## Key architectural findings

- **bs=1 is a hard architectural constraint** — fast buffer cannot be batched without destructive interference; this limits training throughput and inference parallelism (see Limitations)
- **lr=3e-3 is the sweet spot for BPE-192** — lower starves the mechanism, higher overfits and hurts generalisation
- **Episodic reset is mandatory** — fast buffer must zero before each sequence; this means no cross-sequence memory accumulation (see Limitations)
- **Token-identity addressing works** — x_sparse of the same token produces the same code regardless of position
- **Slow weights + fast buffer is the right decomposition** — backprop learns representations, Hebbian stores associations
- **BPE dramatically improves convergence** — meaningful token-level addresses outperform character-level (59.7% → 95.1% on n8)
- **Write-back overhead is per-token sequential** — the Hebbian update loop runs token-by-token and cannot be parallelised within a sequence; inference is slower than the static baseline by roughly the sequence length factor

## How to reproduce

```bash
pip install torch numpy flask    # flask only needed for dashboard
python prepare.py                # generates benchmarks (one-time)
```

To reproduce the baseline (no write-back):
```bash
HEBBIAN_TOKEN_MODE=bpe HEBBIAN_TOKENIZER_LABEL=bpe-192 HEBBIAN_HEBBIAN_LR=0 \
  HEBBIAN_EXPERIMENT_LABEL=baseline-bpe192-nohebb python train.py
```

To reproduce the best run (lr=3e-3):
```bash
HEBBIAN_TOKEN_MODE=bpe HEBBIAN_TOKENIZER_LABEL=bpe-192 HEBBIAN_HEBBIAN_LR=3e-3 \
  HEBBIAN_EXPERIMENT_LABEL=bpe-hebb-bpe192-lr3e-3 python train.py
```

`prepare.py` contains the fixed eval and is never modified — this prevents reward hacking. Results are logged to `results/experiment_log.jsonl`.

To run counter-benchmarks against a saved checkpoint:
```bash
python eval_counter.py results/checkpoints/<checkpoint>.pt
```

### Hardware

Developed and tested on:
- RTX 5070 Ti Laptop GPU (11GB VRAM, CUDA 12.0)
- RunPod cloud GPU (24GB) for overnight sweeps

### Dashboard

```bash
python dashboard.py              # http://localhost:5000
```

Live experiment progress, best result, all results table.

## Results files

- `results/experiment_log.jsonl` — all BPE campaign runs with final scores
- `results/counter_log.jsonl` — counter-benchmark results (corrected teacher-forced eval)
- `results/counter_log_original_eval.jsonl` — original counter-benchmark results using autoregressive generation, preserved for audit (see Limitations)
- `results/eval_history.jsonl` — per-checkpoint eval scores across training
- `results/char_level_experiment_log.jsonl` — all 16 character-level experiments from the initial research phase; documents the bug-finding journey before BPE was introduced
- `snapshots/exp11_xsprev_shared_fastbuf_lr1e2_bs1.py` — frozen standalone script for the first working experiment (char-level, n8=64%), independently runnable

## What this is

A mechanism proof on synthetic benchmarks, reproduced and counter-tested. The first working open-source implementation of BDH's missing Hebbian fast weight write-back. A working demonstration that sub-1B models can learn new facts at inference time by modifying their own weights.

## What this is not

Not proven on natural language yet. Not a product. Not a replacement for RAG or fine-tuning today. Not the full BDH internal implementation (which Pathway has not released).

## Future work

- Longer training runs beyond the 2-hour budget to find the true performance ceiling — all Hebbian runs were still improving at cutoff
- Natural language validation (tinyStories or similar corpus)
- Capacity scaling experiments (how many facts per fast buffer?)
- Attention-guided write addressing (learn where to write, not hardcoded t-1)

## Prior art note

The only other known open-source attempt at BDH Hebbian write-back is [`adamskrodzki/bdh`](https://github.com/adamskrodzki/bdh). A full audit found the forward pass is byte-for-byte identical to upstream — a single unused `lm_gate` parameter was added but never wired in. The mechanism was not implemented.

## Acknowledgements

- **BDH architecture**: Kosowski, Uznański, Chorowski, Stamirowska, Bartoszkiewicz — [arxiv:2509.26507](https://arxiv.org/abs/2509.26507), [github.com/pathwaycom/bdh](https://github.com/pathwaycom/bdh)
- **Fast Weight Programmers**: Schlag, Irie, Schmidhuber — [arxiv:2102.11174](https://arxiv.org/abs/2102.11174)
- **In-context learning as gradient descent**: von Oswald et al. — [arxiv:2212.07677](https://arxiv.org/abs/2212.07677)
- **Autoresearch framework**: Andrej Karpathy — [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)

## Author

**Russell Thomas** — Independent researcher, Kaniva, Victoria, Australia.

No university affiliation. No corporate lab. Independent researcher with a GPU and a question worth answering.

## License

Apache 2.0 — see [LICENSE](LICENSE).
