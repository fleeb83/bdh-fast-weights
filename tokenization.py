"""
Optional raw-byte tokenizer artifact helpers for v3.

The training path consumes packed uint8 shard files directly, but this module
keeps a small serialized tokenizer artifact so the byte-level contract is
explicit and reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    from .address_map import ensure_address_map, load_address_map
    from .prepare import DATA_DIR, DEFAULT_ADDRESS_MAP_LABEL, ROOT, file_sha256
except ImportError:  # pragma: no cover - direct execution fallback
    from address_map import ensure_address_map, load_address_map
    from prepare import DATA_DIR, DEFAULT_ADDRESS_MAP_LABEL, ROOT, file_sha256


RAW_BYTE_TOKENIZER_LABEL = "raw_byte_256"
RAWBYTE_VOCAB_SIZE = 256
ADDRESS_MAP_ROWS = 256
ADDRESS_MAP_K = 16
DEFAULT_ADDRESS_SPACE = 4096
DEFAULT_ADDRESS_MAP_SEED = 1337
TOKENIZERS_DIR = DATA_DIR / "tokenizers"


@dataclass
class RawByteTokenizerArtifacts:
    label: str = RAW_BYTE_TOKENIZER_LABEL
    token_mode: str = RAW_BYTE_TOKENIZER_LABEL
    vocab_size: int = RAWBYTE_VOCAB_SIZE
    token_dtype: str = "uint8"
    bytes_per_token: float = 1.0
    address_map_label: str = DEFAULT_ADDRESS_MAP_LABEL
    address_map_rows: int = ADDRESS_MAP_ROWS
    address_map_k: int = ADDRESS_MAP_K
    address_space: int = DEFAULT_ADDRESS_SPACE
    address_map_seed: int = DEFAULT_ADDRESS_MAP_SEED
    address_map_path: str | None = None
    address_map_sha256: str | None = None
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    notes: str = "UTF-8 raw byte tokenizer with static SAC lookup."

    def to_dict(self) -> dict:
        return asdict(self)


def _artifact_path(label: str) -> Path:
    return TOKENIZERS_DIR / f"{label}.json"


def _repo_rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def ensure_raw_byte_tokenizer_artifacts(
    label: str = RAW_BYTE_TOKENIZER_LABEL,
    *,
    address_map_label: str = DEFAULT_ADDRESS_MAP_LABEL,
    address_space: int = DEFAULT_ADDRESS_SPACE,
    seed: int = DEFAULT_ADDRESS_MAP_SEED,
) -> RawByteTokenizerArtifacts:
    TOKENIZERS_DIR.mkdir(parents=True, exist_ok=True)
    address_map_path = ensure_address_map(address_map_label, k=ADDRESS_MAP_K, address_space=address_space, seed=seed)
    artifact_path = _artifact_path(label)
    payload = RawByteTokenizerArtifacts(
        label=label,
        token_mode=RAW_BYTE_TOKENIZER_LABEL,
        vocab_size=RAWBYTE_VOCAB_SIZE,
        token_dtype="uint8",
        bytes_per_token=1.0,
        address_map_label=address_map_label,
        address_map_rows=ADDRESS_MAP_ROWS,
        address_map_k=ADDRESS_MAP_K,
        address_space=address_space,
        address_map_seed=seed,
        address_map_path=_repo_rel(address_map_path),
        address_map_sha256=file_sha256(address_map_path),
        artifact_path=_repo_rel(artifact_path),
        notes="UTF-8 raw byte tokenizer with static 256x16 SAC address map.",
    )
    artifact_path.write_text(json.dumps(payload.to_dict(), indent=2), encoding="utf-8")
    payload.artifact_sha256 = file_sha256(artifact_path)
    artifact_path.write_text(json.dumps(payload.to_dict(), indent=2), encoding="utf-8")
    payload.artifact_sha256 = file_sha256(artifact_path)
    return payload


def load_raw_byte_tokenizer_artifacts(label: str = RAW_BYTE_TOKENIZER_LABEL) -> RawByteTokenizerArtifacts:
    artifact_path = _artifact_path(label)
    if not artifact_path.exists():
        return ensure_raw_byte_tokenizer_artifacts(label)
    return RawByteTokenizerArtifacts(**json.loads(artifact_path.read_text(encoding="utf-8")))


def encode_text_to_raw_bytes(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)


def load_address_map_array(address_map_label: str = DEFAULT_ADDRESS_MAP_LABEL) -> np.ndarray:
    return load_address_map(address_map_label)


__all__ = [
    "ADDRESS_MAP_K",
    "ADDRESS_MAP_ROWS",
    "DEFAULT_ADDRESS_SPACE",
    "RAWBYTE_VOCAB_SIZE",
    "RAW_BYTE_TOKENIZER_LABEL",
    "RawByteTokenizerArtifacts",
    "encode_text_to_raw_bytes",
    "ensure_raw_byte_tokenizer_artifacts",
    "load_address_map_array",
    "load_raw_byte_tokenizer_artifacts",
]
