"""
Stream FineWeb-Edu docs into data/docs_selected.jsonl for v3 raw-byte export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

try:
    from .prepare import DATA_DIR, DOCS_SELECTED_PATH, DOCS_SOURCE_MANIFEST_PATH
except ImportError:
    from prepare import DATA_DIR, DOCS_SELECTED_PATH, DOCS_SOURCE_MANIFEST_PATH


DATASET_REPO = "HuggingFaceFW/fineweb-edu"
DATASET_NAME = "sample-10BT"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stream_docs(split: str, n_docs: int) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: install datasets to fetch FineWeb-Edu", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Streaming {n_docs:,} FineWeb-Edu docs (split={split}) ...")
    t0 = time.time()
    ds = load_dataset(
        DATASET_REPO,
        name=DATASET_NAME,
        split=split,
        streaming=True,
        trust_remote_code=False,
    )

    written = 0
    with DOCS_SELECTED_PATH.open("w", encoding="utf-8") as fh:
        for record in ds:
            if written >= n_docs:
                break
            text = record.get("text", "")
            if not text or not text.strip():
                continue
            row = {"text": text}
            for key in ("id", "url", "score", "dump", "file_path"):
                if key in record and record[key] is not None:
                    row[key] = record[key]
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    DOCS_SOURCE_MANIFEST_PATH.write_text(
        json.dumps(
            {
                "dataset_repo": DATASET_REPO,
                "dataset_name": DATASET_NAME,
                "split": split,
                "n_docs_requested": n_docs,
                "n_docs_written": written,
                "docs_selected_sha256": file_sha256(DOCS_SELECTED_PATH),
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Done in {time.time() - t0:.0f}s -> {DOCS_SELECTED_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-docs", type=int, default=100_000)
    parser.add_argument("--split", default="train")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_docs = 2_000
    if not args.overwrite and DOCS_SELECTED_PATH.exists() and DOCS_SOURCE_MANIFEST_PATH.exists():
        print(f"Corpus already exists: {DOCS_SELECTED_PATH}")
        return
    stream_docs(split=args.split, n_docs=args.n_docs)


if __name__ == "__main__":
    main()
