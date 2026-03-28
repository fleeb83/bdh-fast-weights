"""
eval_counter.py - Adversarial counter-benchmarks for the BDH Hebbian mechanism.

These tests break patterns that the clean n-back eval might let a model exploit:

  1. vargap1 - 1 random distractor token between each key and value.
               The current write rule (store x[t] at x_sparse[t-1]) should
               fail here: it would store x[VALUE] at x_sparse[DISTRACTOR],
               not at x_sparse[KEY].

  2. repeated - Keys reused within a sequence. Query is always a repeated key;
                correct answer is the most recent value.

  3. n16 - 16 key-value pairs, query among the last 16.

Usage:
    python eval_counter.py results/checkpoints/repro-exp11-xsprev-addr-lr1e-2-bs1.pt
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from prepare import (
    DATA_DIR,
    KEY_RANGE,
    MODEL_VOCAB,
    RESULTS_DIR,
    SEP_TOKEN,
    VAL_RANGE,
    eval_autocast_context,
)
from tokenization import (
    ANSWER_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    SEP_TOKEN_TEXT,
    EncodingConfig,
    load_bpe_tokenizer,
)

COUNTER_EVAL_SEED = 77
N_COUNTER = 500
COUNTER_BENCHMARKS = ("vargap1", "vargap2", "repeated8", "n16")


def _gen_vargap(n_pairs, count, seed, gap=1):
    rng = random.Random(seed)
    seqs = []
    n_back = n_pairs // 2
    for _ in range(count):
        pairs = [(rng.randint(*KEY_RANGE), rng.randint(*VAL_RANGE)) for _ in range(n_pairs)]
        qi = rng.randint(n_pairs - n_back, n_pairs - 1)
        qk, qv = pairs[qi]
        tokens = []
        for k, v in pairs:
            tokens.append(k)
            tokens.extend(rng.randint(0, MODEL_VOCAB - 3) for _ in range(gap))
            tokens.append(v)
        tokens += [SEP_TOKEN, qk]
        seqs.append({"tokens": tokens, "correct_value": qv})
    return seqs


def _gen_repeated_keys(n_pairs, count, seed):
    rng = random.Random(seed)
    seqs = []
    n_unique = n_pairs // 2
    for _ in range(count):
        all_keys = list(range(*KEY_RANGE))
        rng.shuffle(all_keys)
        keys = all_keys[:n_unique]
        first_vals = [rng.randint(*VAL_RANGE) for _ in range(n_unique)]
        second_vals = [rng.randint(*VAL_RANGE) for _ in range(n_unique)]
        positions = list(range(n_pairs))
        rng.shuffle(positions)
        slot = {}
        for i, k in enumerate(keys):
            slot[positions[i]] = (k, first_vals[i])
            slot[positions[n_unique + i]] = (k, second_vals[i])
        pairs = [slot[p] for p in range(n_pairs)]
        qi = rng.randint(0, n_unique - 1)
        qk = keys[qi]
        last_val = None
        for k, v in pairs:
            if k == qk:
                last_val = v
        tokens = []
        for k, v in pairs:
            tokens += [k, v]
        tokens += [SEP_TOKEN, qk]
        seqs.append({"tokens": tokens, "correct_value": last_val})
    return seqs


def _gen_n16(count, seed):
    rng = random.Random(seed)
    n_pairs = 16
    n_back = 16
    seqs = []
    for _ in range(count):
        pairs = [(rng.randint(*KEY_RANGE), rng.randint(*VAL_RANGE)) for _ in range(n_pairs)]
        qi = rng.randint(n_pairs - n_back, n_pairs - 1)
        qk, qv = pairs[qi]
        tokens = []
        for k, v in pairs:
            tokens += [k, v]
        tokens += [SEP_TOKEN, qk]
        seqs.append({"tokens": tokens, "correct_value": qv})
    return seqs


def prepare_counter_data():
    counter_dir = DATA_DIR / "counter"
    counter_dir.mkdir(exist_ok=True)

    specs = [
        ("vargap1", _gen_vargap(8, N_COUNTER, COUNTER_EVAL_SEED, gap=1)),
        ("vargap2", _gen_vargap(8, N_COUNTER, COUNTER_EVAL_SEED + 1, gap=2)),
        ("repeated8", _gen_repeated_keys(8, N_COUNTER, COUNTER_EVAL_SEED + 2)),
        ("n16", _gen_n16(N_COUNTER, COUNTER_EVAL_SEED + 3)),
    ]
    for name, seqs in specs:
        path = counter_dir / f"{name}.jsonl"
        if not path.exists():
            path.write_text("\n".join(json.dumps(s) for s in seqs))
            print(f"Generated {path}")
    return counter_dir


class CounterDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        tokens = row["tokens"]
        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(tokens[1:] + [row["correct_value"]], dtype=torch.long)
        return x, y


def _load_counter_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _split_counter_row(row: dict) -> tuple[list[int], int, int]:
    tokens = list(row["tokens"])
    sep_idx = len(tokens) - 2
    if sep_idx < 0 or tokens[sep_idx] != SEP_TOKEN:
        raise ValueError("Counter row does not end with the expected separator/query layout")
    return tokens[:sep_idx], int(tokens[sep_idx + 1]), int(row["correct_value"])


def _symbolic_gap_token_text(token_id: int) -> str:
    if token_id < VAL_RANGE[0]:
        return f"K{token_id:03d}"
    return f"V{token_id:03d}"


def _serialize_counter_row_text(name: str, row: dict) -> tuple[str, str]:
    context_tokens, query_key, correct_value = _split_counter_row(row)
    if name.startswith("vargap"):
        gap = 1 if name == "vargap1" else 2
        step = gap + 2
        if len(context_tokens) % step != 0:
            raise ValueError(f"{name} row has invalid context length")

        parts: list[str] = []
        for offset in range(0, len(context_tokens), step):
            key_token = context_tokens[offset]
            gap_tokens = context_tokens[offset + 1 : offset + 1 + gap]
            value_token = context_tokens[offset + gap + 1]
            parts.append(f"K{key_token:03d}")
            parts.extend(_symbolic_gap_token_text(token) for token in gap_tokens)
            parts.append(f"V{value_token:03d}")
    else:
        if len(context_tokens) % 2 != 0:
            raise ValueError(f"{name} row has invalid pair layout")
        parts = []
        for offset in range(0, len(context_tokens), 2):
            parts.append(f"K{context_tokens[offset]:03d}")
            parts.append(f"V{context_tokens[offset + 1]:03d}")

    context_text = " ".join(parts)
    prompt_text = f"{BOS_TOKEN} {context_text} {SEP_TOKEN_TEXT} K{query_key:03d} {ANSWER_TOKEN}"
    full_text = f"{prompt_text} V{correct_value:03d} {EOS_TOKEN}"
    return prompt_text, full_text


def _collate_bpe_counter_rows(rows: list[dict], pad_token_id: int) -> tuple[torch.Tensor, list[tuple[int, list[int]]]]:
    max_len = max(len(row["input_ids"]) for row in rows)
    batch = torch.full((len(rows), max_len), pad_token_id, dtype=torch.long)
    spans = []
    for row_idx, row in enumerate(rows):
        ids = row["input_ids"]
        batch[row_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        answer_ids = list(row["answer_ids"])
        answer_start = len(row["prompt_ids"]) - 1
        spans.append((answer_start, answer_ids))
    return batch, spans


@torch.no_grad()
def _eval_counter_file_int(model, path, device):
    model.eval()
    ds = CounterDataset(path)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    has_fast = hasattr(model, "decoder_fast")
    correct = 0
    total = 0
    for x, y in loader:
        if has_fast:
            model.decoder_fast.zero_()
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        pred = logits[:, -1, :].argmax(dim=-1)
        correct += (pred == y[:, -1]).sum().item()
        total += 1
    return correct / total


@torch.no_grad()
def _eval_counter_file_bpe(model, path, device, name: str, encoding: EncodingConfig):
    model.eval()
    tokenizer = load_bpe_tokenizer(encoding)
    rows = _load_counter_rows(path)
    default_batch_size = int(os.environ.get("HEBBIAN_COUNTER_EVAL_BATCH_SIZE_BPE", "16"))
    if getattr(getattr(model, "config", None), "hebbian_lr", 0.0) > 0.0:
        default_batch_size = 1
    correct = 0
    total = 0
    encoded_rows = []
    for row in rows:
        prompt_text, full_text = _serialize_counter_row_text(name, row)
        prompt_ids = tokenizer.encode(prompt_text)
        full_ids = tokenizer.encode(full_text)
        answer_ids = full_ids[len(prompt_ids) :]
        encoded_rows.append(
            {
                "input_ids": full_ids[:-1],
                "prompt_ids": prompt_ids,
                "answer_ids": answer_ids,
            }
        )

    for start_idx in range(0, len(encoded_rows), default_batch_size):
        batch_rows = encoded_rows[start_idx : start_idx + default_batch_size]
        x, spans = _collate_bpe_counter_rows(batch_rows, encoding.pad_token_id)
        x = x.to(device)
        if hasattr(model, "decoder_fast"):
            model.decoder_fast.zero_()
        with eval_autocast_context(device):
            logits, _ = model(x)
        pred_ids = logits.argmax(dim=-1).cpu()
        for row_idx, (answer_start, answer_ids) in enumerate(spans):
            answer_end = answer_start + len(answer_ids)
            pred_span = pred_ids[row_idx, answer_start:answer_end].tolist()
            correct += int(pred_span == answer_ids)
            total += 1
    return correct / total


def evaluate_counter(model, device, encoding: EncodingConfig | None = None):
    counter_dir = prepare_counter_data()
    results = {}
    for name in COUNTER_BENCHMARKS:
        path = counter_dir / f"{name}.jsonl"
        if encoding is not None and encoding.mode == "bpe":
            results[name] = _eval_counter_file_bpe(model, path, device, name, encoding)
        else:
            results[name] = _eval_counter_file_int(model, path, device)
    return results


def _encoding_from_checkpoint(ckpt: dict) -> EncodingConfig:
    raw = ckpt.get("encoding")
    if not raw:
        return EncodingConfig(
            mode="int",
            label="int",
            vocab_size=MODEL_VOCAB,
            pad_token_id=MODEL_VOCAB - 1,
            bos_token_id=None,
            eos_token_id=None,
            tokenizer_path=None,
        )
    return EncodingConfig(**raw)


def _load_model_from_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    encoding = _encoding_from_checkpoint(ckpt)

    train_path = Path(__file__).parent / "train.py"
    module_name = "hebbian_train_for_eval_counter"
    spec = importlib.util.spec_from_file_location(module_name, train_path)
    train_mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = train_mod
    spec.loader.exec_module(train_mod)

    config = train_mod.BDHConfig(**ckpt["config"])
    model = train_mod.BDH(config).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    return model, ckpt, encoding


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_counter.py <checkpoint.pt>")
        sys.exit(1)

    ckpt_path = Path(sys.argv[1])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt, encoding = _load_model_from_checkpoint(ckpt_path, device)

    label = ckpt.get("label", ckpt_path.stem)
    print(f"\nLoaded: {label}")
    print(f"Train scores: n2={ckpt['scores']['n2']:.1%}  n4={ckpt['scores']['n4']:.1%}  n8={ckpt['scores']['n8']:.1%}")
    print(f"Encoding: {encoding.label} ({encoding.mode})\n")

    print("Running counter-benchmarks...")
    results = evaluate_counter(model, device, encoding)

    print(f"\n{'Benchmark':<15} {'Accuracy':>10}  Notes")
    print("-" * 55)
    notes = {
        "vargap1": "1 distractor between k,v - tests adjacency assumption",
        "vargap2": "2 distractors between k,v - harder",
        "repeated8": "repeated keys, correct=most recent value",
        "n16": "n-back-16, 16 pairs - capacity test",
    }
    for name, acc in results.items():
        print(f"{name:<15} {acc:>9.1%}  {notes[name]}")

    entry = {
        "label": label,
        "counter": results,
        "ckpt": str(ckpt_path),
        "encoding": asdict(encoding),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    counter_log = RESULTS_DIR / "counter_log.jsonl"
    with open(counter_log, "a") as handle:
        handle.write(json.dumps(entry) + "\n")
    print(f"\nLogged to {counter_log}")
