"""
Run mechanism ablation evals on saved v3 checkpoints.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

try:
    from .prepare import RESULTS_DIR
    from .train import EvalControls, evaluate_checkpoint
except ImportError:  # pragma: no cover
    from prepare import RESULTS_DIR
    from train import EvalControls, evaluate_checkpoint


DEFAULT_MODES = [
    "normal",
    "fast_read_off",
    "fast_write_off",
    "fast_off",
    "slow_read_off",
    "fast_rows_shuffled",
    "carry_fast_state",
]


def _controls_for(mode: str, *, context_len: int | None, shuffle_seed: int) -> EvalControls:
    mode = mode.lower()
    if mode == "normal":
        return EvalControls(context_len_override=context_len)
    if mode == "fast_read_off":
        return EvalControls(disable_fast_read=True, context_len_override=context_len)
    if mode == "fast_write_off":
        return EvalControls(disable_fast_write=True, context_len_override=context_len)
    if mode == "fast_off":
        return EvalControls(disable_fast_read=True, disable_fast_write=True, context_len_override=context_len)
    if mode == "slow_read_off":
        return EvalControls(disable_slow_read=True, context_len_override=context_len)
    if mode == "fast_rows_shuffled":
        return EvalControls(shuffle_fast_rows=True, shuffle_seed=shuffle_seed, context_len_override=context_len)
    if mode == "carry_fast_state":
        return EvalControls(carry_fast_state=True, context_len_override=context_len)
    raise ValueError(f"Unknown mechanism eval mode: {mode}")


def _targets(labels: list[str], checkpoints: list[str]) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for label in labels:
        rows.append((label, RESULTS_DIR / "checkpoints" / f"{label}_best.pt"))
    for raw in checkpoints:
        path = Path(raw)
        rows.append((path.stem.removesuffix("_best"), path))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mechanism ablation evals on v3 checkpoints")
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--modes", nargs="*", default=DEFAULT_MODES)
    parser.add_argument("--context-lens", nargs="*", type=int, default=[])
    parser.add_argument("--eval-tokens", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=str(RESULTS_DIR / "mechanism_eval_results.jsonl"))
    args = parser.parse_args()

    targets = _targets(args.labels, args.checkpoints)
    if not targets:
        raise SystemExit("No checkpoint targets supplied. Use --labels or --checkpoints.")

    context_lens = args.context_lens or [None]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    rows: list[dict] = []
    for label, ckpt_path in targets:
        for context_len in context_lens:
            for mode in args.modes:
                row = {
                    "label": label,
                    "checkpoint": str(ckpt_path),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "mode": mode,
                    "context_len_override": context_len,
                    "eval_tokens": args.eval_tokens,
                }
                if not ckpt_path.exists():
                    row["status"] = "missing_checkpoint"
                    rows.append(row)
                    continue
                controls = _controls_for(mode, context_len=context_len, shuffle_seed=args.shuffle_seed)
                metrics = evaluate_checkpoint(ckpt_path, args.eval_tokens, device, controls=controls)
                row.update(status="ok", **metrics)
                rows.append(row)

    with out_path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
