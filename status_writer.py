"""
Status helpers for experiments/v3.

This module keeps the monitoring schema broad so the dashboard can evolve
without requiring trainer-side refactors for every new telemetry field.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parent
RESULTS_DIR = Path(
    os.environ.get(
        "V3_RESULTS_DIR",
        os.environ.get("PG_RESULTS_DIR", str(ROOT / "runs" / "fineweb-edu-v3")),
    )
)
STATUS_PATH = RESULTS_DIR / "status.json"
AGENT_STATUS_PATH = RESULTS_DIR / "agent_status.json"

# Fields written to agent_status.json — lean subset for agent decision-making.
_AGENT_STATUS_FIELDS = (
    "label", "step", "phase", "phase_detail",
    "train_loss", "elapsed_s", "tokens_per_s",
    "best_val_bpb", "val_bpb",
    "nan_detected", "inf_detected", "timestamp",
)
TRAIN_HISTORY_PATH = RESULTS_DIR / "train_history.jsonl"
EVAL_HISTORY_PATH = RESULTS_DIR / "eval_history.jsonl"
PERF_HISTORY_PATH = RESULTS_DIR / "perf_history.jsonl"
CONSOLIDATION_HISTORY_PATH = RESULTS_DIR / "consolidation_history.jsonl"
EXPERIMENT_LOG_PATH = RESULTS_DIR / "experiment_log.jsonl"


def _prune_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def write_status(
    *,
    label: str,
    step: int,
    elapsed_s: float,
    phase: str,
    train_loss: float | None = None,
    loss: float | None = None,
    phase_detail: str | None = None,
    tokenizer_label: str | None = None,
    dataset_name: str | None = None,
    address_map_hash: str | None = None,
    address_map_loaded: bool | None = None,
    device_name: str | None = None,
    backend: str | None = None,
    dtype: str | None = None,
    compile_enabled: bool | None = None,
    triton_enabled: bool | None = None,
    stream_overlap_enabled: bool | None = None,
    bf16_enabled: bool | None = None,
    tf32_enabled: bool | None = None,
    gpu_mem_allocated_gb: float | None = None,
    gpu_mem_reserved_gb: float | None = None,
    gpu_mem_peak_gb: float | None = None,
    gpu_mem_total_gb: float | None = None,
    tokens_seen: int | None = None,
    bytes_seen: int | None = None,
    tokens_per_s: float | None = None,
    bytes_per_s: float | None = None,
    step_ms: float | None = None,
    forward_ms: float | None = None,
    backward_ms: float | None = None,
    optimizer_ms: float | None = None,
    data_wait_ms: float | None = None,
    grad_norm: float | None = None,
    clipped_grad_fraction: float | None = None,
    sac_k: int | None = None,
    unique_synapses: int | None = None,
    address_rows_touched: int | None = None,
    address_collision_rate: float | None = None,
    byte_histogram: dict | None = None,
    byte_topk: list | None = None,
    hebb_lr: float | None = None,
    hebb_decay: float | None = None,
    update_cadence_tokens: int | None = None,
    fast_state_norm: float | None = None,
    fast_state_max_abs: float | None = None,
    update_norm: float | None = None,
    read_norm: float | None = None,
    read_write_ratio: float | None = None,
    slow_contrib_norm: float | None = None,
    fast_contrib_norm: float | None = None,
    backend_fallback_reason: str | None = None,
    consolidation_enabled: bool | None = None,
    consolidation_mode: str | None = None,
    consolidation_interval: int | None = None,
    consolidation_topk_frac: float | None = None,
    rows_selected: int | None = None,
    write_norm: float | None = None,
    clip_fraction: float | None = None,
    write_applied: bool | None = None,
    eval_scores: dict | None = None,
    eval_step: int | None = None,
    best_val_bpb: float | None = None,
    best_step: int | None = None,
    delta_vs_baseline: float | None = None,
    eval_kind: str | None = None,
    eval_tokens: int | None = None,
    eval_ms: float | None = None,
    val_loss: float | None = None,
    val_bpb: float | None = None,
    light_eval: dict | None = None,
    final_eval: dict | None = None,
    checkpoint_source: str | None = None,
    nan_detected: bool | None = None,
    inf_detected: bool | None = None,
    notes: str | None = None,
    **extra: Any,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = _prune_none(
        {
            "label": label,
            "step": step,
            "elapsed_s": float(elapsed_s),
            "phase": phase,
            "train_loss": float(train_loss) if train_loss is not None else None,
            "loss": float(loss) if loss is not None else None,
            "phase_detail": phase_detail,
            "tokenizer_label": tokenizer_label,
            "dataset_name": dataset_name,
            "address_map_hash": address_map_hash,
            "address_map_loaded": address_map_loaded,
            "device_name": device_name,
            "backend": backend,
            "dtype": dtype,
            "compile_enabled": compile_enabled,
            "triton_enabled": triton_enabled,
            "stream_overlap_enabled": stream_overlap_enabled,
            "bf16_enabled": bf16_enabled,
            "tf32_enabled": tf32_enabled,
            "gpu_mem_allocated_gb": gpu_mem_allocated_gb,
            "gpu_mem_reserved_gb": gpu_mem_reserved_gb,
            "gpu_mem_peak_gb": gpu_mem_peak_gb,
            "gpu_mem_total_gb": gpu_mem_total_gb,
            "tokens_seen": tokens_seen,
            "bytes_seen": bytes_seen,
            "tokens_per_s": tokens_per_s,
            "bytes_per_s": bytes_per_s,
            "step_ms": step_ms,
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "optimizer_ms": optimizer_ms,
            "data_wait_ms": data_wait_ms,
            "grad_norm": grad_norm,
            "clipped_grad_fraction": clipped_grad_fraction,
            "sac_k": sac_k,
            "unique_synapses": unique_synapses,
            "address_rows_touched": address_rows_touched,
            "address_collision_rate": address_collision_rate,
            "byte_histogram": byte_histogram,
            "byte_topk": byte_topk,
            "hebb_lr": hebb_lr,
            "hebb_decay": hebb_decay,
            "update_cadence_tokens": update_cadence_tokens,
            "fast_state_norm": fast_state_norm,
            "fast_state_max_abs": fast_state_max_abs,
            "update_norm": update_norm,
            "read_norm": read_norm,
            "read_write_ratio": read_write_ratio,
            "slow_contrib_norm": slow_contrib_norm,
            "fast_contrib_norm": fast_contrib_norm,
            "backend_fallback_reason": backend_fallback_reason,
            "consolidation_enabled": consolidation_enabled,
            "consolidation_mode": consolidation_mode,
            "consolidation_interval": consolidation_interval,
            "consolidation_topk_frac": consolidation_topk_frac,
            "rows_selected": rows_selected,
            "write_norm": write_norm,
            "clip_fraction": clip_fraction,
            "write_applied": write_applied,
            "eval_scores": eval_scores,
            "eval_step": eval_step,
            "best_val_bpb": best_val_bpb,
            "best_step": best_step,
            "delta_vs_baseline": delta_vs_baseline,
            "eval_kind": eval_kind,
            "eval_tokens": eval_tokens,
            "eval_ms": eval_ms,
            "val_loss": val_loss,
            "val_bpb": val_bpb,
            "light_eval": light_eval,
            "final_eval": final_eval,
            "checkpoint_source": checkpoint_source,
            "nan_detected": nan_detected,
            "inf_detected": inf_detected,
            "notes": notes,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    payload.update(_prune_none(extra))
    body = json.dumps(payload)

    # Write lean agent_status.json alongside the full status.json.
    agent_body = json.dumps({k: payload[k] for k in _AGENT_STATUS_FIELDS if k in payload})
    try:
        AGENT_STATUS_PATH.write_text(agent_body, encoding="utf-8")
    except OSError:
        pass

    tmp = STATUS_PATH.with_name(f"{STATUS_PATH.stem}.{os.getpid()}.tmp")
    last_error: Exception | None = None
    for delay_s in (0.0, 0.05, 0.1, 0.25, 0.5):
        if delay_s:
            time.sleep(delay_s)
        try:
            tmp.write_text(body, encoding="utf-8")
            os.replace(tmp, STATUS_PATH)
            return
        except PermissionError as exc:
            last_error = exc
        finally:
            tmp.unlink(missing_ok=True)
    try:
        STATUS_PATH.write_text(body, encoding="utf-8")
    except PermissionError:
        if last_error is not None:
            return


def append_eval_history(
    label: str,
    step: int,
    loss: float,
    elapsed_s: float,
    scores: dict,
    *,
    eval_kind: str = "quick",
    eval_ms: float | None = None,
    tokens_seen: int | None = None,
    bytes_seen: int | None = None,
    backend: str | None = None,
    phase: str | None = None,
    **extra: Any,
) -> None:
    """Append one eval checkpoint row to eval_history.jsonl."""
    entry = _prune_none(
        {
            "label": label,
            "step": step,
            "loss": float(loss),
            "elapsed_s": float(elapsed_s),
            "val_loss": float(scores["val_loss"]) if scores.get("val_loss") is not None else None,
            "val_bpb": float(scores["val_bpb"]) if scores.get("val_bpb") is not None else None,
            "val_bpt": float(scores["val_bpt"]) if scores.get("val_bpt") is not None else None,
            "val_tokens": int(scores["val_tokens"]) if scores.get("val_tokens") is not None else None,
            "best_val_bpb": float(scores["best_val_bpb"]) if scores.get("best_val_bpb") is not None else None,
            "eval_kind": eval_kind,
            "eval_ms": eval_ms,
            "tokens_seen": tokens_seen,
            "bytes_seen": bytes_seen,
            "backend": backend,
            "phase": phase,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    entry.update(_prune_none(extra))
    _append_jsonl(EVAL_HISTORY_PATH, entry)


def append_train_history(
    label: str,
    step: int,
    loss: float,
    elapsed_s: float,
    *,
    lr: float | None = None,
    phase: str = "train",
    tokens_seen: int | None = None,
    bytes_seen: int | None = None,
    tokens_per_s: float | None = None,
    bytes_per_s: float | None = None,
    step_ms: float | None = None,
    forward_ms: float | None = None,
    backward_ms: float | None = None,
    optimizer_ms: float | None = None,
    data_wait_ms: float | None = None,
    grad_norm: float | None = None,
    backend: str | None = None,
    **extra: Any,
) -> None:
    """Append one live training checkpoint row to train_history.jsonl."""
    entry = _prune_none(
        {
            "label": label,
            "step": step,
            "loss": float(loss),
            "elapsed_s": float(elapsed_s),
            "lr": float(lr) if lr is not None else None,
            "phase": phase,
            "tokens_seen": tokens_seen,
            "bytes_seen": bytes_seen,
            "tokens_per_s": tokens_per_s,
            "bytes_per_s": bytes_per_s,
            "step_ms": step_ms,
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "optimizer_ms": optimizer_ms,
            "data_wait_ms": data_wait_ms,
            "grad_norm": grad_norm,
            "backend": backend,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    entry.update(_prune_none(extra))
    _append_jsonl(TRAIN_HISTORY_PATH, entry)


def append_perf_history(
    label: str,
    step: int,
    elapsed_s: float,
    stats: dict,
) -> None:
    """Append one perf-oriented checkpoint row to perf_history.jsonl."""
    entry = _prune_none(
        {
            "label": label,
            "step": step,
            "elapsed_s": float(elapsed_s),
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            **stats,
        }
    )
    _append_jsonl(PERF_HISTORY_PATH, entry)


def append_consolidation_history(
    label: str,
    step: int,
    elapsed_s: float,
    stats: dict,
) -> None:
    """Append one consolidation writeback row to consolidation_history.jsonl."""
    entry = _prune_none(
        {
            "label": label,
            "step": step,
            "elapsed_s": float(elapsed_s),
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            **stats,
        }
    )
    _append_jsonl(CONSOLIDATION_HISTORY_PATH, entry)
