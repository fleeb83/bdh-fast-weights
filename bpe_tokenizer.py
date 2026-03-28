"""
Minimal special-token-aware character BPE tokenizer for the Hebbian experiments.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


def _split_special_aware(text: str, special_tokens: list[str]) -> list[str]:
    parts: list[str] = []
    i = 0
    while i < len(text):
        matched = None
        for token in special_tokens:
            if text.startswith(token, i):
                matched = token
                break
        if matched is not None:
            parts.append(matched)
            i += len(matched)
        else:
            parts.append(text[i])
            i += 1
    return parts


def _merge_once(tokens: list[str], pair: tuple[str, str]) -> list[str]:
    merged: list[str] = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            merged.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


@dataclass
class BpeTokenizer:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    merges: list[tuple[str, str]]
    special_tokens: list[str]

    @classmethod
    def train(
        cls,
        corpus: list[str],
        vocab_size: int,
        *,
        special_tokens: list[str],
        min_pair_count: int = 2,
    ) -> "BpeTokenizer":
        sequences = [_split_special_aware(text, special_tokens) for text in corpus]

        base_tokens = set(special_tokens)
        for seq in sequences:
            for token in seq:
                if token not in special_tokens:
                    base_tokens.add(token)

        token_to_id = {tok: idx for idx, tok in enumerate(special_tokens)}
        id_to_token = list(special_tokens)
        for token in sorted(base_tokens):
            if token in token_to_id:
                continue
            token_to_id[token] = len(id_to_token)
            id_to_token.append(token)

        merges: list[tuple[str, str]] = []

        while len(id_to_token) < vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for seq in sequences:
                for left, right in zip(seq, seq[1:]):
                    if left in special_tokens or right in special_tokens:
                        continue
                    pair_counts[(left, right)] += 1

            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_pair_count:
                break

            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in token_to_id:
                token_to_id[merged_token] = len(id_to_token)
                id_to_token.append(merged_token)
            merges.append(best_pair)
            sequences = [_merge_once(seq, best_pair) for seq in sequences]

        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            merges=merges,
            special_tokens=special_tokens,
        )

    def encode(self, text: str) -> list[int]:
        tokens = _split_special_aware(text, self.special_tokens)
        for pair in self.merges:
            tokens = _merge_once(tokens, pair)
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_token[idx] for idx in ids)

    def save(self, path: Path) -> None:
        payload = {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "merges": [[left, right] for left, right in self.merges],
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "BpeTokenizer":
        payload = json.loads(path.read_text())
        return cls(
            token_to_id={k: int(v) for k, v in payload["token_to_id"].items()},
            id_to_token=list(payload["id_to_token"]),
            merges=[(left, right) for left, right in payload["merges"]],
            special_tokens=list(payload["special_tokens"]),
        )
