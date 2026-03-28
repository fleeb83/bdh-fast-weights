"""
status_writer.py — shared path constants + write_status helper.
Imported by train.py (to write) and dashboard.py (to read paths).
"""

import json
import os
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("HEBBIAN_RESULTS_DIR", str(Path(__file__).parent / "results")))
STATUS_PATH = RESULTS_DIR / "status.json"
EVAL_HISTORY_PATH = RESULTS_DIR / "eval_history.jsonl"
TRAIN_HISTORY_PATH = RESULTS_DIR / "train_history.jsonl"


def write_status(
    label: str,
    step: int,
    loss: float,
    elapsed_s: float,
    *,
    eval_scores: dict = None,
    eval_step: int = None,
    best_n8: float = None,
    best_step: int = None,
    max_elapsed_s: float = 7200.0,
    convergence_delta: float = None,
    converged: bool = False,
    phase: str = "train",
    phase_detail: str | None = None,
    lr: float | None = None,
    encoding_label: str | None = None,
    scoring_mode: str | None = None,
) -> None:
    """Write current experiment state to status.json atomically."""
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = json.dumps({
        "label":             label,
        "step":              step,
        "loss":              float(loss),
        "elapsed_s":         float(elapsed_s),
        "max_elapsed_s":     float(max_elapsed_s),
        "eval":              eval_scores,
        "eval_step":         eval_step,
        "best_n8":           float(best_n8) if best_n8 is not None else None,
        "best_step":         best_step,
        "convergence_delta": float(convergence_delta) if convergence_delta is not None else None,
        "converged":         converged,
        "phase":             phase,
        "phase_detail":      phase_detail,
        "lr":                float(lr) if lr is not None else None,
        "encoding_label":    encoding_label,
        "scoring_mode":      scoring_mode,
        "timestamp":         datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    })
    tmp = STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(payload)
    os.replace(tmp, STATUS_PATH)


def append_eval_history(label: str, step: int, loss: float, elapsed_s: float, scores: dict) -> None:
    """Append one eval checkpoint row to eval_history.jsonl."""
    RESULTS_DIR.mkdir(exist_ok=True)
    entry = {
        "label":     label,
        "step":      step,
        "loss":      float(loss),
        "elapsed_s": float(elapsed_s),
        "n2":        float(scores["n2"]) if scores.get("n2") is not None else None,
        "n4":        float(scores["n4"]) if scores.get("n4") is not None else None,
        "n8":        float(scores["n8"]) if scores.get("n8") is not None else None,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(EVAL_HISTORY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def append_train_history(
    label: str,
    step: int,
    loss: float,
    elapsed_s: float,
    *,
    lr: float | None = None,
    phase: str = "train",
) -> None:
    """Append one live training checkpoint row to train_history.jsonl."""
    RESULTS_DIR.mkdir(exist_ok=True)
    entry = {
        "label": label,
        "step": step,
        "loss": float(loss),
        "elapsed_s": float(elapsed_s),
        "lr": float(lr) if lr is not None else None,
        "phase": phase,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(TRAIN_HISTORY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
