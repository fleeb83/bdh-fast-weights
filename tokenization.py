"""
Tokenizer and symbolic-sequence helpers for the Hebbian experiments.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from bpe_tokenizer import BpeTokenizer

TOKENIZERS_DIR = Path(__file__).parent / "tokenizers"

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SEP_TOKEN_TEXT = "<sep>"
ANSWER_TOKEN = "<answer>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    SEP_TOKEN_TEXT,
    ANSWER_TOKEN,
]


@dataclass
class EncodingConfig:
    mode: str
    label: str
    vocab_size: int
    pad_token_id: int
    bos_token_id: int | None
    eos_token_id: int | None
    tokenizer_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _resolve_tokenizer_path() -> Path | None:
    explicit = os.environ.get("HEBBIAN_TOKENIZER_PATH")
    if explicit:
        return Path(explicit)

    label = os.environ.get("HEBBIAN_TOKENIZER_LABEL")
    if label:
        return TOKENIZERS_DIR / label / "tokenizer.json"

    return None


def load_encoding_config() -> EncodingConfig:
    mode = os.environ.get("HEBBIAN_TOKEN_MODE", "int").strip().lower()
    if mode == "int":
        return EncodingConfig(
            mode="int",
            label="int",
            vocab_size=130,
            pad_token_id=129,
            bos_token_id=None,
            eos_token_id=None,
            tokenizer_path=None,
        )

    if mode != "bpe":
        raise ValueError(f"Unsupported HEBBIAN_TOKEN_MODE={mode!r}")

    tokenizer_path = _resolve_tokenizer_path()
    if tokenizer_path is None or not tokenizer_path.exists():
        raise FileNotFoundError(
            "BPE mode requires a tokenizer artifact. Run build_bpe_tokenizer.py and set "
            "HEBBIAN_TOKENIZER_LABEL or HEBBIAN_TOKENIZER_PATH."
        )

    tokenizer = BpeTokenizer.load(tokenizer_path)
    label = tokenizer_path.parent.name
    return EncodingConfig(
        mode="bpe",
        label=label,
        vocab_size=len(tokenizer.id_to_token),
        pad_token_id=tokenizer.token_to_id[PAD_TOKEN],
        bos_token_id=tokenizer.token_to_id[BOS_TOKEN],
        eos_token_id=tokenizer.token_to_id[EOS_TOKEN],
        tokenizer_path=str(tokenizer_path),
    )


def load_bpe_tokenizer(config: EncodingConfig) -> BpeTokenizer:
    if config.mode != "bpe":
        raise ValueError("BPE tokenizer requested for non-BPE config")
    candidates = []
    if config.tokenizer_path:
        candidates.append(Path(config.tokenizer_path))
    candidates.append(TOKENIZERS_DIR / config.label / "tokenizer.json")
    for path in candidates:
        if path.exists():
            return BpeTokenizer.load(path)
    raise FileNotFoundError(
        f"Could not resolve tokenizer artifact for label={config.label!r}. "
        f"Tried: {', '.join(str(path) for path in candidates)}"
    )


def parse_raw_row(raw_row: dict, sep_token_id: int) -> tuple[list[tuple[int, int]], int, int]:
    tokens = list(raw_row["tokens"])
    sep_idx = len(tokens) - 2
    if sep_idx < 0 or tokens[sep_idx] != sep_token_id:
        raise ValueError("Raw row does not end with the expected separator/query layout")
    pair_tokens = tokens[:sep_idx]
    query_key = tokens[sep_idx + 1]
    pairs = []
    for idx in range(0, len(pair_tokens), 2):
        pairs.append((pair_tokens[idx], pair_tokens[idx + 1]))
    return pairs, query_key, int(raw_row["correct_value"])


def serialize_example_text(
    pairs: list[tuple[int, int]],
    query_key: int,
    correct_value: int | None = None,
) -> tuple[str, str | None]:
    context = " ".join(f"K{key:03d} V{value:03d}" for key, value in pairs)
    prompt = f"{BOS_TOKEN} {context} {SEP_TOKEN_TEXT} K{query_key:03d} {ANSWER_TOKEN}"
    if correct_value is None:
        return prompt, None
    full = f"{prompt} V{correct_value:03d} {EOS_TOKEN}"
    return prompt, full


def convert_raw_row_to_bpe_row(raw_row: dict, *, sep_token_id: int, tokenizer: BpeTokenizer) -> dict:
    pairs, query_key, correct_value = parse_raw_row(raw_row, sep_token_id)
    prompt_text, full_text = serialize_example_text(pairs, query_key, correct_value)
    assert full_text is not None

    prompt_ids = tokenizer.encode(prompt_text)
    full_ids = tokenizer.encode(full_text)
    answer_ids = full_ids[len(prompt_ids):]

    return {
        "input_ids": full_ids[:-1],
        "target_ids": full_ids[1:],
        "prompt_ids": prompt_ids,
        "answer_ids": answer_ids,
        "prompt_text": prompt_text,
        "answer_text": f"V{correct_value:03d}",
        "full_text": full_text,
        "n_back": raw_row["n_back"],
        "seq_len": raw_row["seq_len"],
    }


def tokenized_dataset_path(data_dir: Path, split: str, n_back: int, config: EncodingConfig) -> Path:
    if config.mode == "int":
        return data_dir / f"{split}_n{n_back}.jsonl"
    return data_dir / f"{split}_n{n_back}.{config.label}.jsonl"


def ensure_tokenized_split(
    *,
    raw_path: Path,
    encoded_path: Path,
    sep_token_id: int,
    tokenizer: BpeTokenizer,
) -> None:
    if encoded_path.exists():
        return

    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    encoded_rows = [
        convert_raw_row_to_bpe_row(row, sep_token_id=sep_token_id, tokenizer=tokenizer)
        for row in rows
    ]
    encoded_path.write_text("\n".join(json.dumps(row) for row in encoded_rows))
