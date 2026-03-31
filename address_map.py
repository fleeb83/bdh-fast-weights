"""
Address-map artifact helpers for the v3 raw-byte sparse front-end.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from .prepare import (
        ADDRESS_MAPS_DIR,
        AddressMapArtifact,
        DEFAULT_ADDRESS_MAP_LABEL,
        DEFAULT_ADDRESS_SPACE,
        DEFAULT_SAC_K,
        RAWBYTE_VOCAB_SIZE,
        address_map_paths,
        file_sha256,
        upsert_address_map_manifest_entry,
    )
except ImportError:
    from prepare import (
        ADDRESS_MAPS_DIR,
        AddressMapArtifact,
        DEFAULT_ADDRESS_MAP_LABEL,
        DEFAULT_ADDRESS_SPACE,
        DEFAULT_SAC_K,
        RAWBYTE_VOCAB_SIZE,
        address_map_paths,
        file_sha256,
        upsert_address_map_manifest_entry,
    )


@dataclass
class AddressMapSpec:
    label: str = DEFAULT_ADDRESS_MAP_LABEL
    vocab_size: int = RAWBYTE_VOCAB_SIZE
    k: int = DEFAULT_SAC_K
    address_space: int = DEFAULT_ADDRESS_SPACE
    seed: int = 1337
    notes: str = "Static SAC lookup for raw-byte v3."


def build_address_map(spec: AddressMapSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    rows = np.empty((spec.vocab_size, spec.k), dtype=np.int32)
    for token_id in range(spec.vocab_size):
        rows[token_id] = rng.choice(spec.address_space, size=spec.k, replace=False)
    return rows


def save_address_map(spec: AddressMapSpec, rows: np.ndarray) -> Path:
    npy_path, json_path = address_map_paths(spec.label)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, rows)
    sha = file_sha256(npy_path)
    json_path.write_text(
        json.dumps(
            {
                **asdict(spec),
                "path": npy_path.relative_to(ADDRESS_MAPS_DIR.parent.parent).as_posix(),
                "sha256": sha,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    upsert_address_map_manifest_entry(
        AddressMapArtifact(
            label=spec.label,
            vocab_size=spec.vocab_size,
            k=spec.k,
            address_space=spec.address_space,
            seed=spec.seed,
            path=npy_path.relative_to(ADDRESS_MAPS_DIR.parent.parent).as_posix(),
            sha256=sha,
            notes=spec.notes,
        )
    )
    return npy_path


def ensure_address_map(
    label: str = DEFAULT_ADDRESS_MAP_LABEL,
    *,
    k: int = DEFAULT_SAC_K,
    address_space: int = DEFAULT_ADDRESS_SPACE,
    seed: int = 1337,
) -> Path:
    npy_path, _json_path = address_map_paths(label)
    if npy_path.exists():
        return npy_path
    spec = AddressMapSpec(label=label, k=k, address_space=address_space, seed=seed)
    rows = build_address_map(spec)
    return save_address_map(spec, rows)


def load_address_map(label: str = DEFAULT_ADDRESS_MAP_LABEL) -> np.ndarray:
    npy_path, _json_path = address_map_paths(label)
    if not npy_path.exists():
        ensure_address_map(label)
    return np.load(npy_path)


def load_address_map_tensor(label: str = DEFAULT_ADDRESS_MAP_LABEL, device: torch.device | str | None = None) -> torch.Tensor:
    rows = load_address_map(label)
    tensor = torch.from_numpy(rows.astype(np.int64, copy=False))
    if device is not None:
        tensor = tensor.to(device=device, non_blocking=True)
    return tensor
