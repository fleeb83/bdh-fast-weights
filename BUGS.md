# BUGS.md

Five implementation bugs were found and fixed during development (v0.1–v0.3). Each was independently sufficient to prevent the mechanism from working. They are documented here for reproducibility and for anyone implementing similar systems.

Full per-experiment log: `results/char_level_experiment_log.jsonl` (all 16 character-level experiments from the initial research phase).

---

## Bug 1: Wrong target matrix

**Symptom:** Fast weights never accumulated meaningful signal.

**Cause:** The Hebbian outer product was being written into the encoder weight matrix instead of the decoder weight matrix. The encoder produces the address code; the decoder is what gets read at inference time. Writing to the wrong matrix means the read path is never affected.

**Fix:** Target the decoder (`decoder_fast`) in the write-back step.

---

## Bug 2: Batch contamination

**Symptom:** Training appeared to work at bs=1 but degraded or behaved inconsistently at higher batch sizes.

**Cause:** The fast weight buffer was shared across all examples in a batch. When the model processed multiple sequences simultaneously, their Hebbian updates mixed in the same buffer — associations from sequence A polluted reads for sequence B.

**Fix:** Per-example fast state (implemented in v0.3). v0.1/v0.2 worked around this with bs=1.

---

## Bug 3: Missing episodic reset

**Symptom:** Performance on n-back recall was poor and noisy.

**Cause:** The fast buffer was not zeroed between sequences. Associations from previous training examples leaked into the current sequence, creating interference. The model could not learn clean within-sequence associations because the buffer always contained residual signal from prior episodes.

**Fix:** Zero the fast buffer at the start of each sequence (before the first token).

---

## Bug 4: Attention-dependent addressing

**Symptom:** Recall accuracy degraded with distance — close pairs recalled correctly, far pairs failed.

**Cause:** The address code was derived from the attention output rather than the token embedding directly. Attention outputs are context-dependent and vary with sequence position, so the same token produced different address codes at different positions. This broke the core addressing property: same token should always produce same address.

**Fix:** Address from the token embedding (pre-attention), not the attention output. Same token, same code, regardless of position.

---

## Bug 5: Eval mismatch (teacher-forcing vs autoregressive)

**Symptom:** Counter-benchmarks (vargap, repeated8, n16) showed near-zero accuracy even when clean n-back recall worked at 90%+. Appeared to be a mechanism failure.

**Cause:** The model was trained with teacher-forcing (each token is predicted from the true previous token). The counter-benchmark eval used autoregressive generation (each token is predicted from the model's own previous output). For multi-token answer spans, one wrong token early in the answer causes all subsequent tokens to be wrong, collapsing accuracy to near zero. This was scorer failure, not mechanism failure.

**Fix:** Switch counter-benchmark eval to teacher-forced scoring. Both the original (autoregressive) and corrected (teacher-forced) logs are preserved in `results/` for audit.
