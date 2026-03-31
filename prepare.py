"""
Fixed data/export contract for v3 (raw-byte FineWeb-Edu).

v3 uses a raw UTF-8 byte tokenizer (vocab 256) plus a serialized static
address-map artifact that maps each byte ID to k active synaptic indices.

Directory layout (all overridable via env vars):

  BDH_DATA_DIR    -- root for downloaded corpus, packed shards, address maps
                     default: ./data
  BDH_RESULTS_DIR -- root for experiment logs and checkpoints
                     default: ./results

Run fetch_corpus.py first to populate BDH_DATA_DIR, then prepare.py to
pack shards. train.py reads from BDH_DATA_DIR and writes to BDH_RESULTS_DIR.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


ROOT = Path(__file__).parent
DATA_DIR = Path(os.environ.get("BDH_DATA_DIR", str(ROOT / "data")))
DATASETS_DIR = DATA_DIR / "datasets"
ADDRESS_MAPS_DIR = DATA_DIR / "address_maps"
TOKENIZERS_DIR = DATA_DIR / "tokenizers"
DOCS_SELECTED_PATH = DATA_DIR / "docs_selected.jsonl"
DOCS_SOURCE_MANIFEST_PATH = DATA_DIR / "docs_selected.source_manifest.json"
DATA_MANIFEST_PATH = DATA_DIR / "manifest.json"
RESULTS_DIR = Path(os.environ.get("BDH_RESULTS_DIR", str(ROOT / "results")))
EXPERIMENT_LOG = RESULTS_DIR / "experiment_log.jsonl"

RAWBYTE_VOCAB_SIZE = 256
DEFAULT_SAC_K = 16
DEFAULT_ADDRESS_SPACE = 4096
DEFAULT_ADDRESS_MAP_LABEL = "rawbyte256_sac16_default"
DEFAULT_DATASET_NAME = "fineweb_edu_rawbyte256_sac16"


@dataclass
class AddressMapArtifact:
    label: str
    vocab_size: int
    k: int
    address_space: int
    seed: int
    path: str
    sha256: str
    notes: str = ""


@dataclass
class DatasetExport:
    name: str
    tokenizer_mode: str
    vocab_size: int
    token_dtype: str
    address_map_label: str
    address_map_sha256: str
    train_shards: int
    val_shards: int
    train_docs: int | None = None
    val_docs: int | None = None
    total_train_tokens: int | None = None
    total_val_tokens: int | None = None
    export_manifest_path: str | None = None
    split_contract: str = ""
    docs_source_manifest_sha256: str | None = None
    source: str = "local"
    files_train: list[str] = field(default_factory=list)
    files_val: list[str] = field(default_factory=list)
    sha256_train: list[str] = field(default_factory=list)
    sha256_val: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class DataManifest:
    datasets: list[DatasetExport]
    address_maps: list[AddressMapArtifact]
    docs_selected_path: str | None = None
    docs_selected_sha256: str | None = None
    docs_source_manifest_path: str | None = None
    docs_source_manifest_sha256: str | None = None

    def to_dict(self) -> dict:
        return {
            "datasets": [asdict(item) for item in self.datasets],
            "address_maps": [asdict(item) for item in self.address_maps],
            "docs_selected_path": self.docs_selected_path,
            "docs_selected_sha256": self.docs_selected_sha256,
            "docs_source_manifest_path": self.docs_source_manifest_path,
            "docs_source_manifest_sha256": self.docs_source_manifest_sha256,
        }


def _coerce_dataset_entry(row: dict) -> DatasetExport:
    payload = {
        "name": row["name"],
        "tokenizer_mode": row.get("tokenizer_mode", "raw_byte_256"),
        "vocab_size": row.get("vocab_size", row.get("byte_vocab_size", RAWBYTE_VOCAB_SIZE)),
        "token_dtype": row.get("token_dtype", row.get("tokens_dtype", "uint8")),
        "address_map_label": row.get("address_map_label", row.get("tokenizer_name", DEFAULT_ADDRESS_MAP_LABEL)),
        "address_map_sha256": row.get("address_map_sha256", ""),
        "train_shards": row.get("train_shards", len(row.get("files_train", []))),
        "val_shards": row.get("val_shards", len(row.get("files_val", []))),
        "train_docs": row.get("train_docs"),
        "val_docs": row.get("val_docs"),
        "total_train_tokens": row.get("total_train_tokens"),
        "total_val_tokens": row.get("total_val_tokens"),
        "export_manifest_path": row.get("export_manifest_path"),
        "split_contract": row.get("split_contract", ""),
        "docs_source_manifest_sha256": row.get("docs_source_manifest_sha256"),
        "source": row.get("source", "local"),
        "files_train": row.get("files_train", []),
        "files_val": row.get("files_val", []),
        "sha256_train": row.get("sha256_train", []),
        "sha256_val": row.get("sha256_val", []),
        "notes": row.get("notes", ""),
    }
    return DatasetExport(**payload)


def _coerce_address_map_entry(row: dict) -> AddressMapArtifact:
    payload = {
        "label": row["label"],
        "vocab_size": row.get("vocab_size", row.get("address_map_rows", RAWBYTE_VOCAB_SIZE)),
        "k": row.get("k", row.get("address_map_k", DEFAULT_SAC_K)),
        "address_space": row.get("address_space", DEFAULT_ADDRESS_SPACE),
        "seed": row.get("seed", 1337),
        "path": row.get("path", row.get("address_map_path", "")),
        "sha256": row.get("sha256", row.get("address_map_sha256", "")),
        "notes": row.get("notes", ""),
    }
    return AddressMapArtifact(**payload)


def ensure_layout() -> None:
    for path in (
        DATASETS_DIR,
        ADDRESS_MAPS_DIR,
        TOKENIZERS_DIR,
        RESULTS_DIR,
        RESULTS_DIR / "checkpoints",
        RESULTS_DIR / "manifests",
    ):
        path.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_dir(name: str) -> Path:
    return DATASETS_DIR / name


def address_map_paths(label: str) -> tuple[Path, Path]:
    folder = ADDRESS_MAPS_DIR / label
    return folder / "address_map.npy", folder / "address_map.json"


def export_bin_path(dataset_name: str, split: str, shard_idx: int) -> Path:
    return dataset_dir(dataset_name) / f"pretokenized_{split}_{shard_idx:06d}.bin"


def default_dataset_name() -> str:
    return DEFAULT_DATASET_NAME


def load_data_manifest() -> dict | None:
    if not DATA_MANIFEST_PATH.exists():
        return None
    return json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))


def write_data_manifest(address_maps: list[AddressMapArtifact], datasets: list[DatasetExport]) -> Path:
    ensure_layout()
    payload = DataManifest(
        datasets=datasets,
        address_maps=address_maps,
        docs_selected_path=DOCS_SELECTED_PATH.relative_to(ROOT).as_posix() if DOCS_SELECTED_PATH.exists() else None,
        docs_selected_sha256=file_sha256(DOCS_SELECTED_PATH) if DOCS_SELECTED_PATH.exists() else None,
        docs_source_manifest_path=(
            DOCS_SOURCE_MANIFEST_PATH.relative_to(ROOT).as_posix() if DOCS_SOURCE_MANIFEST_PATH.exists() else None
        ),
        docs_source_manifest_sha256=(
            file_sha256(DOCS_SOURCE_MANIFEST_PATH) if DOCS_SOURCE_MANIFEST_PATH.exists() else None
        ),
    )
    DATA_MANIFEST_PATH.write_text(json.dumps(payload.to_dict(), indent=2), encoding="utf-8")
    return DATA_MANIFEST_PATH


def upsert_address_map_manifest_entry(entry: AddressMapArtifact) -> Path:
    manifest = load_data_manifest() or {"datasets": [], "address_maps": []}
    kept = [row for row in manifest.get("address_maps", []) if row.get("label") != entry.label]
    address_maps = [_coerce_address_map_entry(row) for row in kept]
    address_maps.append(entry)
    datasets = [_coerce_dataset_entry(row) for row in manifest.get("datasets", []) if "name" in row]
    return write_data_manifest(address_maps=address_maps, datasets=datasets)


def upsert_dataset_manifest_entry(entry: DatasetExport) -> Path:
    manifest = load_data_manifest() or {"datasets": [], "address_maps": []}
    kept = [row for row in manifest.get("datasets", []) if row.get("name") != entry.name]
    datasets = [_coerce_dataset_entry(row) for row in kept]
    datasets.append(entry)
    address_maps = [_coerce_address_map_entry(row) for row in manifest.get("address_maps", []) if "label" in row]
    return write_data_manifest(address_maps=address_maps, datasets=datasets)


def require_frozen_docs_cache() -> tuple[Path, Path]:
    if not DOCS_SELECTED_PATH.exists():
        raise FileNotFoundError(f"Missing frozen docs cache: {DOCS_SELECTED_PATH}")
    if not DOCS_SOURCE_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing docs source manifest: {DOCS_SOURCE_MANIFEST_PATH}")
    return DOCS_SELECTED_PATH, DOCS_SOURCE_MANIFEST_PATH


def validate_docs_cache_against_manifest() -> None:
    manifest = load_data_manifest()
    if manifest is None or not DOCS_SELECTED_PATH.exists():
        return
    expected = manifest.get("docs_selected_sha256")
    if expected and file_sha256(DOCS_SELECTED_PATH) != expected:
        raise ValueError("docs_selected.jsonl hash does not match data manifest")
    expected_source = manifest.get("docs_source_manifest_sha256")
    if expected_source and DOCS_SOURCE_MANIFEST_PATH.exists() and file_sha256(DOCS_SOURCE_MANIFEST_PATH) != expected_source:
        raise ValueError("docs_selected.source_manifest.json hash does not match data manifest")


def validate_dataset_export(name: str, *, require_full_val: bool = True, required_train_prefix: int | None = None) -> None:
    manifest = load_data_manifest()
    if manifest is None:
        raise FileNotFoundError("Missing data manifest")
    entry = next((row for row in manifest.get("datasets", []) if row.get("name") == name), None)
    if entry is None:
        raise KeyError(f"Unknown dataset export: {name}")

    files_train = entry.get("files_train", [])
    files_val = entry.get("files_val", [])
    if required_train_prefix is not None and len(files_train) < required_train_prefix:
        raise ValueError(f"Dataset {name} only has {len(files_train)} train shards, need {required_train_prefix}")
    if require_full_val and len(files_val) != int(entry.get("val_shards", 0)):
        raise ValueError(f"Dataset {name} val split is incomplete")

    file_rows = list(zip(files_train, entry.get("sha256_train", []), strict=False)) + list(
        zip(files_val, entry.get("sha256_val", []), strict=False)
    )
    for rel, expected_sha in file_rows:
        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset shard: {path}")
        if expected_sha and file_sha256(path) != expected_sha:
            raise ValueError(f"Dataset shard hash mismatch: {path}")


def register_published_export(
    *,
    dataset_name: str,
    address_map_label: str,
    address_map_sha256: str,
    train_files: list[Path],
    val_files: list[Path],
    train_docs: int | None = None,
    val_docs: int | None = None,
    total_train_tokens: int | None = None,
    total_val_tokens: int | None = None,
    export_manifest_path: Path | None = None,
    split_contract: str = "",
    docs_source_manifest_sha256: str | None = None,
    source: str = "published",
    notes: str = "",
) -> Path:
    train_rel = [path.relative_to(ROOT).as_posix() for path in train_files]
    val_rel = [path.relative_to(ROOT).as_posix() for path in val_files]
    entry = DatasetExport(
        name=dataset_name,
        tokenizer_mode="raw_byte_256",
        vocab_size=RAWBYTE_VOCAB_SIZE,
        token_dtype="uint8",
        address_map_label=address_map_label,
        address_map_sha256=address_map_sha256,
        train_shards=len(train_files),
        val_shards=len(val_files),
        train_docs=train_docs,
        val_docs=val_docs,
        total_train_tokens=total_train_tokens,
        total_val_tokens=total_val_tokens,
        export_manifest_path=(
            export_manifest_path.relative_to(ROOT).as_posix() if export_manifest_path is not None else None
        ),
        split_contract=split_contract,
        docs_source_manifest_sha256=docs_source_manifest_sha256,
        source=source,
        files_train=train_rel,
        files_val=val_rel,
        sha256_train=[file_sha256(path) for path in train_files],
        sha256_val=[file_sha256(path) for path in val_files],
        notes=notes,
    )
    return upsert_dataset_manifest_entry(entry)


def evaluate_all(dataset_name: str | None = None) -> dict:
    target = dataset_name or DEFAULT_DATASET_NAME
    validate_dataset_export(target)
    manifest = load_data_manifest()
    if manifest is None:
        raise FileNotFoundError("Missing data manifest")
    entry = next((row for row in manifest.get("datasets", []) if row.get("name") == target), None)
    if entry is None:
        raise KeyError(f"Unknown dataset export: {target}")
    return {
        "dataset_name": target,
        "tokenizer_mode": entry.get("tokenizer_mode"),
        "vocab_size": entry.get("vocab_size", entry.get("byte_vocab_size", RAWBYTE_VOCAB_SIZE)),
        "token_dtype": entry.get("token_dtype", entry.get("tokens_dtype", "uint8")),
        "address_map_label": entry.get("address_map_label"),
        "address_map_sha256": entry.get("address_map_sha256"),
        "train_shards": entry.get("train_shards"),
        "val_shards": entry.get("val_shards"),
        "train_docs": entry.get("train_docs"),
        "val_docs": entry.get("val_docs"),
        "total_train_tokens": entry.get("total_train_tokens"),
        "total_val_tokens": entry.get("total_val_tokens"),
    }


def prepare_data() -> None:
    ensure_layout()
    try:
        from .tokenization import ensure_raw_byte_tokenizer_artifacts
    except ImportError:
        from tokenization import ensure_raw_byte_tokenizer_artifacts

    ensure_raw_byte_tokenizer_artifacts()


def log_result(label: str, metrics: dict, notes: str = "") -> dict:
    ensure_layout()
    entry = {"label": label, **metrics, "notes": notes}
    with open(EXPERIMENT_LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return entry


if __name__ == "__main__":
    prepare_data()
    print("v3 data/export contract ready")
