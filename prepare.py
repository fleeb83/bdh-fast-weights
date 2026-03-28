"""
prepare.py - locked data generation and evaluation for Hebbian experiments.

This file defines the benchmark, immutable eval path, and result logging.
"""

from __future__ import annotations

import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch

from tokenization import (
    ensure_tokenized_split,
    load_bpe_tokenizer,
    load_encoding_config,
    tokenized_dataset_path,
)

# Integer benchmark constants retained for the canonical baseline.
VOCAB_SIZE = 128
KEY_RANGE = (0, 64)
VAL_RANGE = (64, 128)
SEP_TOKEN = 128
PAD_TOKEN = 129
MODEL_VOCAB = 130

TRAIN_SEED = 42
EVAL_SEED = 99
N_TRAIN = 8000
N_EVAL = 1000

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path(os.environ.get("HEBBIAN_RESULTS_DIR", str(Path(__file__).parent / "results")))
RESULTS_DIR.mkdir(exist_ok=True)

EXPERIMENT_LOG = RESULTS_DIR / "experiment_log.jsonl"


def _generate_sequences(n_back: int, seq_len: int, count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    sequences = []
    for _ in range(count):
        keys = [rng.randint(*KEY_RANGE) for _ in range(seq_len)]
        values = [rng.randint(*VAL_RANGE) for _ in range(seq_len)]
        pairs = list(zip(keys, values))
        query_idx = rng.randint(seq_len - n_back, seq_len - 1)
        query_key, correct_value = pairs[query_idx]
        tokens = []
        for key, value in pairs:
            tokens += [key, value]
        tokens += [SEP_TOKEN, query_key]
        sequences.append(
            {
                "tokens": tokens,
                "correct_value": correct_value,
                "n_back": n_back,
                "seq_len": seq_len,
            }
        )
    return sequences


def raw_dataset_path(n_back: int, split: str) -> Path:
    return DATA_DIR / f"{split}_n{n_back}.jsonl"


def prepare_raw_data() -> None:
    configs = [
        {"n_back": 2, "seq_len": 8},
        {"n_back": 4, "seq_len": 16},
        {"n_back": 8, "seq_len": 32},
    ]
    for cfg in configs:
        n_back = cfg["n_back"]
        train_path = raw_dataset_path(n_back, "train")
        eval_path = raw_dataset_path(n_back, "eval")
        if not train_path.exists():
            seqs = _generate_sequences(n_back, cfg["seq_len"], N_TRAIN, TRAIN_SEED)
            train_path.write_text("\n".join(json.dumps(seq) for seq in seqs))
            print(f"Generated {train_path}")
        if not eval_path.exists():
            seqs = _generate_sequences(n_back, cfg["seq_len"], N_EVAL, EVAL_SEED)
            eval_path.write_text("\n".join(json.dumps(seq) for seq in seqs))
            print(f"Generated {eval_path}")


def prepare_data() -> None:
    prepare_raw_data()
    encoding = load_encoding_config()
    if encoding.mode == "int":
        return

    tokenizer = load_bpe_tokenizer(encoding)
    for n_back in (2, 4, 8):
        for split in ("train", "eval"):
            ensure_tokenized_split(
                raw_path=raw_dataset_path(n_back, split),
                encoded_path=tokenized_dataset_path(DATA_DIR, split, n_back, encoding),
                sep_token_id=SEP_TOKEN,
                tokenizer=tokenizer,
            )


class NBackDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path):
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        if "input_ids" in row:
            x = torch.tensor(row["input_ids"], dtype=torch.long)
            y = torch.tensor(row["target_ids"], dtype=torch.long)
            return x, y

        tokens = row["tokens"]
        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(tokens[1:] + [row["correct_value"]], dtype=torch.long)
        return x, y


def get_dataset_path(n_back: int, split: str) -> Path:
    encoding = load_encoding_config()
    return tokenized_dataset_path(DATA_DIR, split, n_back, encoding)


def get_dataloader(n_back: int, split: str, batch_size: int, shuffle: bool = True):
    ds = NBackDataset(get_dataset_path(n_back, split))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def eval_autocast_context(device: str):
    if device != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _parse_eval_nbacks(final: bool) -> list[int]:
    if final:
        raw = os.environ.get("HEBBIAN_FINAL_EVAL_NBACKS", "2,4,8")
    else:
        raw = os.environ.get("HEBBIAN_TRAIN_EVAL_NBACKS", "2,4,8")
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values or [8]


@torch.no_grad()
def _evaluate_int(
    model,
    n_back: int,
    device: str,
    batch_size: int,
    progress_callback: Callable[..., None] | None = None,
) -> float:
    model.eval()
    loader = get_dataloader(n_back, "eval", batch_size, shuffle=False)
    correct = 0
    total = 0
    dataset_total = len(loader.dataset)
    for x, y in loader:
        if hasattr(model, "decoder_fast"):
            model.decoder_fast.zero_()
        x, y = x.to(device), y.to(device)
        with eval_autocast_context(device):
            logits, _ = model(x)
        pred = logits[:, -1, :].argmax(dim=-1)
        target = y[:, -1]
        correct += (pred == target).sum().item()
        total += target.shape[0]
        if progress_callback and (total == dataset_total or total % 128 == 0):
            progress_callback(
                n_back=n_back,
                done=total,
                total=dataset_total,
                accuracy=(correct / total) if total else None,
            )
    return correct / total


def _collate_bpe_eval_rows(rows: list[dict], pad_token_id: int) -> tuple[torch.Tensor, list[tuple[int, list[int]]]]:
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
def _evaluate_bpe(
    model,
    n_back: int,
    device: str,
    batch_size: int,
    limit: int | None = None,
    progress_callback: Callable[..., None] | None = None,
) -> float:
    model.eval()
    encoding = load_encoding_config()
    ds = NBackDataset(get_dataset_path(n_back, "eval"))
    correct = 0
    total = 0
    rows = ds.rows[:limit] if limit is not None else ds.rows
    rows_total = len(rows)
    for start_idx in range(0, len(rows), batch_size):
        batch_rows = rows[start_idx : start_idx + batch_size]
        x, spans = _collate_bpe_eval_rows(batch_rows, encoding.pad_token_id)
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
        if progress_callback and (total == rows_total or total % max(batch_size * 25, 25) == 0):
            progress_callback(
                n_back=n_back,
                done=total,
                total=rows_total,
                accuracy=(correct / total) if total else None,
            )
    return correct / total


@torch.no_grad()
def evaluate(
    model,
    n_back: int,
    device: str,
    batch_size: int = 128,
    progress_callback: Callable[..., None] | None = None,
) -> float:
    """
    Returns accuracy on the held-out eval set (EVAL_SEED=99).

    Integer mode uses final-token accuracy.
    BPE mode uses exact answer-span accuracy under a single teacher-forced pass.
    """
    encoding = load_encoding_config()
    if encoding.mode == "bpe":
        default_bpe_batch_size = int(os.environ.get("HEBBIAN_EVAL_BATCH_SIZE_BPE", "16"))
        if getattr(getattr(model, "config", None), "hebbian_lr", 0.0) > 0.0:
            default_bpe_batch_size = 1
        bpe_batch_size = default_bpe_batch_size
        limit_raw = os.environ.get("HEBBIAN_EVAL_LIMIT_BPE")
        limit = int(limit_raw) if limit_raw and limit_raw.strip() else None
        return _evaluate_bpe(
            model,
            n_back,
            device,
            batch_size=bpe_batch_size,
            limit=limit,
            progress_callback=progress_callback,
        )

    has_fast = hasattr(model, "decoder_fast")
    if has_fast:
        batch_size = 1
    return _evaluate_int(model, n_back, device, batch_size, progress_callback=progress_callback)


def evaluate_all(model, device: str, *, final: bool = False, progress_callback: Callable[..., None] | None = None) -> dict:
    metrics = {}
    for n_back in _parse_eval_nbacks(final):
        if progress_callback:
            progress_callback(n_back=n_back, done=0, total=None, accuracy=None)
        metrics[f"n{n_back}"] = evaluate(
            model,
            n_back=n_back,
            device=device,
            progress_callback=progress_callback,
        )
    if final:
        for n_back in (2, 4, 8):
            if f"n{n_back}" in metrics:
                continue
            if progress_callback:
                progress_callback(n_back=n_back, done=0, total=None, accuracy=None)
            metrics.setdefault(
                f"n{n_back}",
                evaluate(model, n_back=n_back, device=device, progress_callback=progress_callback),
            )
    return metrics


def log_result(label: str, scores: dict, notes: str = "") -> dict:
    import time

    entry = {
        "label": label,
        "n2": scores["n2"],
        "n4": scores["n4"],
        "n8": scores["n8"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": notes,
    }
    with open(EXPERIMENT_LOG, "a") as handle:
        handle.write(json.dumps(entry) + "\n")
    print(f"[{label}]  n2={scores['n2']:.1%}  n4={scores['n4']:.1%}  n8={scores['n8']:.1%}")
    return entry


if __name__ == "__main__":
    print("Preparing data...")
    prepare_data()
    print("Done. Run train.py to start experiments.")
