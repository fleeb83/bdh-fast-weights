"""
dashboard.py - experiments/v3 monitor.

Tracks raw-byte export state, SAC health, Hebbian state, consolidation,
hardware, throughput, eval provenance, and failure conditions.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime

from flask import Flask

try:
    from .status_writer import (
        CONSOLIDATION_HISTORY_PATH,
        EVAL_HISTORY_PATH,
        EXPERIMENT_LOG_PATH,
        PERF_HISTORY_PATH,
        RESULTS_DIR,
        STATUS_PATH,
        TRAIN_HISTORY_PATH,
    )
except ImportError:
    from status_writer import (
        CONSOLIDATION_HISTORY_PATH,
        EVAL_HISTORY_PATH,
        EXPERIMENT_LOG_PATH,
        PERF_HISTORY_PATH,
        RESULTS_DIR,
        STATUS_PATH,
        TRAIN_HISTORY_PATH,
    )

app = Flask(__name__)

STALE_AFTER_S = int(
    os.environ.get(
        "V3_DASHBOARD_STALE_AFTER_S",
        os.environ.get("PG_DASHBOARD_STALE_AFTER_S", "7200"),
    )
)


def load_status():
    if not STATUS_PATH.exists():
        return None
    try:
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        ts = status.get("timestamp")
        if ts:
            age_s = (datetime.now() - datetime.fromisoformat(ts)).total_seconds()
            if 0 <= age_s < STALE_AFTER_S:
                return status
        if time.time() - STATUS_PATH.stat().st_mtime < STALE_AFTER_S:
            return status
    except Exception:
        pass
    return None


def _load_jsonl(path, label=None):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if label is None or row.get("label") == label:
                rows.append(row)
    return rows


def _numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _score(row):
    for key in ("val_bpb", "loss"):
        v = row.get(key)
        if _numeric(v):
            return float(v)
    return float("inf")


def format_value(value, key=""):
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "--"
        key = (key or "").lower()
        if "lr" in key:
            return f"{value:.2e}"
        if "ms" in key or "elapsed" in key:
            return f"{value:.1f} ms"
        if "gb" in key:
            return f"{value:.2f} GB"
        if any(part in key for part in ("bpb", "loss", "norm", "ratio", "delta", "fraction")):
            return f"{value:.4f}"
        if "tokens_per_s" in key or "bytes_per_s" in key:
            return f"{value:.1f}"
        return f"{value:.4g}"
    if isinstance(value, dict):
        items = list(value.items())[:4]
        return " | ".join(f"{k}:{format_value(v, str(k))}" for k, v in items) if items else "{}"
    if isinstance(value, list):
        return " | ".join(format_value(v) for v in value[:6]) if value else "[]"
    return str(value)


def summarize_hist(value):
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: kv[1], reverse=True)
        return " | ".join(f"{k}:{v}" for k, v in items[:6]) if items else "--"
    if isinstance(value, list):
        return " | ".join(format_value(v) for v in value[:6]) if value else "--"
    return format_value(value)


def load_results():
    return sorted(_load_jsonl(EXPERIMENT_LOG_PATH), key=_score)


def load_eval_history(label=None):
    return sorted(_load_jsonl(EVAL_HISTORY_PATH, label=label), key=lambda r: r.get("step", 0))


def load_train_history(label=None):
    return sorted(_load_jsonl(TRAIN_HISTORY_PATH, label=label), key=lambda r: r.get("step", 0))


def load_perf_history(label=None):
    return sorted(_load_jsonl(PERF_HISTORY_PATH, label=label), key=lambda r: r.get("step", 0))


def load_consolidation_history(label=None):
    return sorted(_load_jsonl(CONSOLIDATION_HISTORY_PATH, label=label), key=lambda r: r.get("step", 0))


def is_baseline_label(label):
    text = (label or "").lower()
    return any(token in text for token in ("baseline", "nohebb", "no-hebb", "static"))


def get_baseline_result(results):
    baselines = [r for r in results if is_baseline_label(r.get("label", ""))]
    return min(baselines, key=_score) if baselines else None


def card(title, fields, subtitle=None, tone=""):
    items = []
    for label, value, key in fields:
        items.append(
            f'<div class="fact"><div class="fact-label">{label}</div><div class="fact-value">{format_value(value, key)}</div></div>'
        )
    subtitle_html = f'<div class="card-subtitle">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="card {tone}">
      <div class="card-header"><div class="card-label">{title}</div></div>
      {subtitle_html}
      <div class="fact-grid">{''.join(items)}</div>
    </div>"""


def chart(rows, key, title, color, higher_is_better=False, suffix=""):
    pts = []
    for row in rows:
        value = row.get(key)
        if _numeric(value):
            pts.append((int(row.get("step", len(pts))), float(value)))
    if len(pts) < 2:
        return f'<div class="chart-card"><div class="chart-title">{title}</div><div class="chart-empty">Waiting for {title} history...</div></div>'

    pad_l, pad_r, pad_t, pad_b = 44, 16, 10, 24
    width, height = 640, 130
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    steps = [s for s, _ in pts]
    values = [v for _, v in pts]
    min_s, max_s = min(steps), max(steps)
    min_v, max_v = min(values), max(values)
    if min_v == max_v:
        max_v = min_v + 1.0
    s_range = max_s - min_s or 1
    v_range = max_v - min_v or 1

    def sx(step):
        return pad_l + (step - min_s) / s_range * inner_w

    def sy(value):
        return pad_t + inner_h - (value - min_v) / v_range * inner_h

    ordered = sorted(pts, key=lambda p: p[0])
    d = "M " + " L ".join(f"{sx(s):.1f},{sy(v):.1f}" for s, v in ordered)
    best_v = max(values) if higher_is_better else min(values)
    best_s = [s for s, v in ordered if abs(v - best_v) < 1e-9][-1]
    last_s, last_v = ordered[-1]

    grid = ""
    for frac in (0.0, 0.5, 1.0):
        y = pad_t + inner_h - frac * inner_h
        label = min_v + frac * v_range
        grid += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + inner_w}" y2="{y:.1f}" stroke="#21262d" stroke-width="1"/>'
        grid += f'<text x="{pad_l - 4}" y="{y + 4:.1f}" text-anchor="end" fill="#93a1ad" font-size="10">{format_value(label, key)}{suffix}</text>'

    return f"""
    <div class="chart-card">
      <div class="chart-title">{title}</div>
      <svg width="640" height="130" xmlns="http://www.w3.org/2000/svg" style="display:block;max-width:100%">
        {grid}
        <path d="{d}" stroke="{color}" stroke-width="2" fill="none" stroke-linejoin="round"/>
        <circle cx="{sx(last_s):.1f}" cy="{sy(last_v):.1f}" r="3" fill="{color}" stroke="#0f1216" stroke-width="1"/>
        <circle cx="{sx(best_s):.1f}" cy="{sy(best_v):.1f}" r="6" fill="none" stroke="#ffcf5a" stroke-width="2"/>
      </svg>
    </div>"""


def alert_banner(status):
    if not status:
        return ""
    alerts = []
    for key, label in (
        ("backend_fallback_reason", "backend fallback"),
        ("address_map_mismatch", "address-map mismatch"),
        ("eval_accounting_mismatch", "eval accounting mismatch"),
        ("nan_detected", "NaN detected"),
        ("inf_detected", "Inf detected"),
    ):
        value = status.get(key)
        if value:
            alerts.append(f"{label}: {format_value(value, key)}")
    for key, label in (
        ("portable_compile_delta", "portable vs compile delta"),
        ("portable_triton_delta", "portable vs Triton delta"),
        ("parity_delta", "parity delta"),
    ):
        value = status.get(key)
        if _numeric(value):
            alerts.append(f"{label}: {format_value(value, key)}")
    if not alerts:
        return ""
    return f'<div class="banner warning"><div class="banner-title">Attention</div><div class="banner-body">{" | ".join(alerts)}</div></div>'


def run_card(status):
    if not status:
        return """
        <div class="card idle">
          <div class="card-label">IDLE</div>
          <div class="card-title">No experiment running</div>
          <p class="idle-hint">Start the queue to begin a v3 run.</p>
        </div>"""

    elapsed = float(status.get("elapsed_s", 0.0) or 0.0)
    max_s = float(status.get("max_elapsed_s", 7200.0) or 7200.0)
    pct = min(100, int(elapsed / max(max_s, 1.0) * 100))
    phase = status.get("phase", "train")
    phase_label = {"train": "RUNNING", "eval": "EVALUATING", "final_eval": "FINAL EVAL", "done": "DONE"}.get(phase, str(phase).upper())
    loss = status.get("train_loss", status.get("loss"))
    eval_data = status.get("final_eval") or status.get("light_eval") or status.get("eval") or {}

    fields = [
        ("step", status.get("step"), "step"),
        ("elapsed", elapsed, "elapsed_s"),
        ("budget", max_s, "elapsed_s"),
        ("train loss", loss, "loss"),
        ("val bpb", eval_data.get("val_bpb", status.get("val_bpb")), "val_bpb"),
        ("best bpb", status.get("best_val_bpb"), "best_val_bpb"),
    ]
    meta = [
        ("backend", status.get("backend"), "backend"),
        ("device", status.get("device_name"), "device_name"),
        ("dtype", status.get("dtype"), "dtype"),
        ("tokenizer", status.get("tokenizer_label"), "tokenizer_label"),
        ("dataset", status.get("dataset_name"), "dataset_name"),
        ("address map", status.get("address_map_hash"), "address_map_hash"),
    ]

    return f"""
    <div class="card live-card">
      <div class="card-header"><div class="card-label running">{phase_label}</div><div class="card-title">{status.get('label', '')}</div></div>
      <div class="progress-bar"><div class="progress-fill" style="width:{pct}%"></div></div>
      <div class="progress-label">{format_value(elapsed, 'elapsed_s')} / {format_value(max_s, 'elapsed_s')} &middot; step {status.get('step', 0)} &middot; loss {format_value(loss, 'loss')}</div>
      <div class="run-meta">{status.get('phase_detail') or 'active'}</div>
      <div class="run-meta">last update {status.get('timestamp', '').replace('T', ' ')}</div>
      <div class="mini-section-title">Run snapshot</div>
      <div class="fact-grid compact">{''.join(f'<div class="fact"><div class="fact-label">{label}</div><div class="fact-value">{format_value(value, key)}</div></div>' for label, value, key in fields)}</div>
      <div class="mini-section-title">Provenance</div>
      <div class="fact-grid compact">{''.join(f'<div class="fact"><div class="fact-label">{label}</div><div class="fact-value">{format_value(value, key)}</div></div>' for label, value, key in meta)}</div>
      <div class="mini-section-title">Eval snapshot</div>
      <div class="fact-grid compact">
        <div class="fact"><div class="fact-label">kind</div><div class="fact-value">{format_value(status.get('eval_kind'), 'eval_kind')}</div></div>
        <div class="fact"><div class="fact-label">val loss</div><div class="fact-value">{format_value(eval_data.get('val_loss', status.get('val_loss')), 'val_loss')}</div></div>
        <div class="fact"><div class="fact-label">checkpoint</div><div class="fact-value">{format_value(status.get('checkpoint_source'), 'checkpoint_source')}</div></div>
        <div class="fact"><div class="fact-label">delta</div><div class="fact-value">{format_value(status.get('delta_vs_baseline'), 'delta_vs_baseline')}</div></div>
      </div>
      {alert_banner(status)}
    </div>"""


def best_card(best, baseline):
    if not best:
        return """
        <div class="card idle">
          <div class="card-label">BEST</div>
          <div class="card-title">No completed runs yet</div>
          <p class="idle-hint">Finish a run to populate this card.</p>
        </div>"""

    best_score = best.get("val_bpb", best.get("loss"))
    baseline_score = None
    if baseline:
        baseline_score = baseline.get("val_bpb", baseline.get("loss"))
    delta = None
    if _numeric(best_score) and _numeric(baseline_score):
        delta = float(baseline_score) - float(best_score)

    fields = [
        ("score", best_score, "val_bpb"),
        ("baseline", baseline_score, "val_bpb"),
        ("delta", delta, "delta_vs_baseline"),
        ("backend", best.get("backend"), "backend"),
        ("dtype", best.get("dtype"), "dtype"),
        ("tokenizer", best.get("tokenizer_label"), "tokenizer_label"),
    ]
    return card(f"best run: {best.get('label', 'unknown')}", fields, subtitle=best.get("notes") or best.get("phase_detail"), tone="best-card")


def table(title, headers, rows_html):
    return f"""
    <div class="section-block">
      <div class="section-title">{title}</div>
      <table>
        <thead><tr>{''.join(f'<th>{h}</th>' for h in headers)}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""


def eval_table(rows, best_val_bpb=None):
    if not rows:
        return "<p class='no-results'>No eval checkpoints yet.</p>"
    body = ""
    for row in reversed(rows[-20:]):
        row_class = "best-row" if best_val_bpb is not None and _numeric(row.get("val_bpb")) and abs(row["val_bpb"] - best_val_bpb) < 1e-9 else ""
        body = f"""
        <tr class="{row_class}">
          <td>{row.get('step', '')}</td>
          <td>{format_value(row.get('loss'), 'loss')}</td>
          <td>{format_value(row.get('val_loss'), 'val_loss')}</td>
          <td>{format_value(row.get('val_bpb'), 'val_bpb')}</td>
          <td>{format_value(row.get('eval_kind'), 'eval_kind')}</td>
          <td class="ts">{row.get('timestamp', '')[:16].replace('T', ' ')}</td>
        </tr>""" + body
    return table("Eval checkpoints - current run", ["step", "loss", "val loss", "val bpb", "kind", "time"], body)


def results_table(results, baseline):
    if not results:
        return "<p class='no-results'>No completed experiments yet.</p>"
    baseline_score = _score(baseline) if baseline else None
    body = ""
    for idx, row in enumerate(results):
        score = _score(row)
        delta = None
        if _numeric(score) and _numeric(baseline_score):
            delta = baseline_score - score
        body += f"""
        <tr class="{ 'best-row' if idx == 0 else '' }">
          <td>{'* ' if idx == 0 else ''}{row.get('label', '')}</td>
          <td>{format_value(row.get('val_bpb'), 'val_bpb')}</td>
          <td>{format_value(row.get('loss'), 'loss')}</td>
          <td>{format_value(delta, 'delta_vs_baseline')}</td>
          <td>{format_value(row.get('backend'), 'backend')}</td>
          <td class="ts">{row.get('timestamp', '')[:16].replace('T', ' ')}</td>
        </tr>"""
    return table("All completed runs", ["label", "val bpb", "loss", "delta vs baseline", "backend", "time"], body)


@app.route("/")
def index():
    status = load_status()
    results = load_results()
    best = results[0] if results else None
    baseline = get_baseline_result(results)
    current_label = status["label"] if status else None

    train_rows = load_train_history(current_label)
    eval_rows = load_eval_history(current_label)
    perf_rows = load_perf_history(current_label)
    consolidation_rows = load_consolidation_history(current_label)
    run_count = len(results)

    current_eval = (status or {}).get("final_eval") or (status or {}).get("light_eval") or (status or {}).get("eval") or {}

    summary_cards = [
        card(
            "run overview",
            [
                ("label", status.get("label") if status else None, "label"),
                ("phase", status.get("phase") if status else None, "phase"),
                ("step", status.get("step") if status else None, "step"),
                ("elapsed", status.get("elapsed_s") if status else None, "elapsed_s"),
                ("backend", status.get("backend") if status else None, "backend"),
                ("device", status.get("device_name") if status else None, "device_name"),
                ("dtype", status.get("dtype") if status else None, "dtype"),
                ("phase detail", status.get("phase_detail") if status else None, "phase_detail"),
            ],
        ),
        card(
            "data and SAC",
            [
                ("dataset", status.get("dataset_name") if status else None, "dataset_name"),
                ("tokenizer", status.get("tokenizer_label") if status else None, "tokenizer_label"),
                ("vocab", status.get("vocab_size") if status else 256, "vocab_size"),
                ("k", status.get("sac_k") if status else 16, "sac_k"),
                ("address map", status.get("address_map_hash") if status else None, "address_map_hash"),
                ("loaded", status.get("address_map_loaded") if status else None, "address_map_loaded"),
                ("rows touched", status.get("address_rows_touched") if status else None, "address_rows_touched"),
                ("byte topk", status.get("byte_topk") if status else None, "byte_topk"),
            ],
            subtitle=summarize_hist(status.get("byte_histogram") if status else None),
        ),
        card(
            "throughput and perf",
            [
                ("tokens/s", status.get("tokens_per_s") if status else None, "tokens_per_s"),
                ("bytes/s", status.get("bytes_per_s") if status else None, "bytes_per_s"),
                ("step ms", status.get("step_ms") if status else None, "step_ms"),
                ("forward ms", status.get("forward_ms") if status else None, "forward_ms"),
                ("backward ms", status.get("backward_ms") if status else None, "backward_ms"),
                ("optimizer ms", status.get("optimizer_ms") if status else None, "optimizer_ms"),
                ("data wait ms", status.get("data_wait_ms") if status else None, "data_wait_ms"),
                ("grad norm", status.get("grad_norm") if status else None, "grad_norm"),
            ],
        ),
        card(
            "hardware and backend",
            [
                ("bf16", status.get("bf16_enabled") if status else None, "bf16_enabled"),
                ("tf32", status.get("tf32_enabled") if status else None, "tf32_enabled"),
                ("compile", status.get("compile_enabled") if status else None, "compile_enabled"),
                ("triton", status.get("triton_enabled") if status else None, "triton_enabled"),
                ("streams", status.get("stream_overlap_enabled") if status else None, "stream_overlap_enabled"),
                ("mem alloc", status.get("gpu_mem_allocated_gb") if status else None, "gpu_mem_allocated_gb"),
                ("mem peak", status.get("gpu_mem_peak_gb") if status else None, "gpu_mem_peak_gb"),
                ("mem total", status.get("gpu_mem_total_gb") if status else None, "gpu_mem_total_gb"),
            ],
            subtitle=status.get("backend_fallback_reason") if status else None,
        ),
        card(
            "hebbian state",
            [
                ("hebb lr", status.get("hebb_lr") if status else None, "hebb_lr"),
                ("hebb decay", status.get("hebb_decay") if status else None, "hebb_decay"),
                ("cadence", status.get("update_cadence_tokens") if status else 1, "update_cadence_tokens"),
                ("fast norm", status.get("fast_state_norm") if status else None, "fast_state_norm"),
                ("fast max", status.get("fast_state_max_abs") if status else None, "fast_state_max_abs"),
                ("update norm", status.get("update_norm") if status else None, "update_norm"),
                ("read norm", status.get("read_norm") if status else None, "read_norm"),
                ("ratio", status.get("read_write_ratio") if status else None, "read_write_ratio"),
            ],
            subtitle=f"slow={format_value(status.get('slow_contrib_norm') if status else None, 'slow_contrib_norm')} fast={format_value(status.get('fast_contrib_norm') if status else None, 'fast_contrib_norm')}",
        ),
        card(
            "consolidation",
            [
                ("enabled", status.get("consolidation_enabled") if status else None, "consolidation_enabled"),
                ("mode", status.get("consolidation_mode") if status else None, "consolidation_mode"),
                ("interval", status.get("consolidation_interval") if status else None, "consolidation_interval"),
                ("topk frac", status.get("consolidation_topk_frac") if status else None, "consolidation_topk_frac"),
                ("rows selected", status.get("rows_selected") if status else None, "rows_selected"),
                ("write norm", status.get("write_norm") if status else None, "write_norm"),
                ("clip frac", status.get("clip_fraction") if status else None, "clip_fraction"),
                ("applied", status.get("write_applied") if status else None, "write_applied"),
            ],
        ),
        card(
            "evaluation",
            [
                ("kind", status.get("eval_kind") if status else None, "eval_kind"),
                ("eval tokens", status.get("eval_tokens") if status else None, "eval_tokens"),
                ("eval ms", status.get("eval_ms") if status else None, "eval_ms"),
                ("val loss", current_eval.get("val_loss") if current_eval else status.get("val_loss") if status else None, "val_loss"),
                ("val bpb", current_eval.get("val_bpb") if current_eval else status.get("val_bpb") if status else None, "val_bpb"),
                ("best bpb", status.get("best_val_bpb") if status else None, "best_val_bpb"),
                ("delta", status.get("delta_vs_baseline") if status else None, "delta_vs_baseline"),
                ("checkpoint", status.get("checkpoint_source") if status else None, "checkpoint_source"),
            ],
        ),
        card(
            "provenance",
            [
                ("commit", status.get("git_commit") if status else None, "git_commit"),
                ("config", status.get("config_signature") if status else None, "config_signature"),
                ("dataset hash", status.get("dataset_hash") if status else None, "dataset_hash"),
                ("address hash", status.get("address_map_hash") if status else None, "address_map_hash"),
                ("token hist", status.get("byte_histogram") if status else None, "byte_histogram"),
            ],
        ),
    ]

    charts = [
        chart(train_rows or perf_rows, "loss", "train loss", "#ff7b72"),
        chart(eval_rows or train_rows, "val_bpb", "val bpb", "#58a6ff"),
        chart(train_rows or perf_rows, "lr", "learning rate", "#79c0ff"),
        chart(perf_rows or train_rows, "tokens_per_s", "tokens per second", "#3fb950", higher_is_better=True),
        chart(perf_rows or train_rows, "bytes_per_s", "bytes per second", "#3fb950", higher_is_better=True),
        chart(perf_rows or train_rows, "step_ms", "step time", "#ffcf5a"),
        chart(perf_rows or train_rows, "gpu_mem_allocated_gb", "GPU memory allocated", "#db8422"),
        chart(perf_rows or train_rows, "fast_state_norm", "fast state norm", "#5ee28a"),
        chart(consolidation_rows or perf_rows, "write_norm", "consolidation write norm", "#ffcf5a"),
    ]

    live = run_card(status)
    best_block = best_card(best, baseline)
    eval_block = eval_table(eval_rows, best_val_bpb=status.get("best_val_bpb") if status else None)
    results_block = results_table(results, baseline)

    refresh = 5 if status else 60
    note = f"{run_count} completed run{'s' if run_count != 1 else ''} &middot; raw-byte 256 + static SAC + per-token Hebbian"

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{refresh}">
  <title>v3 monitor</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 24px; background: #0f1216; color: #e6edf3; font-family: ui-monospace, 'Cascadia Code', monospace; font-size: 14px; }}
    h1 {{ margin: 0 0 4px; font-size: 18px; color: #e6edf3; }}
    .meta {{ color: #93a1ad; font-size: 12px; margin-bottom: 18px; }}
    .top-grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin-bottom: 16px; }}
    .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 16px; }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    @media (max-width: 980px) {{ .top-grid {{ grid-template-columns: 1fr; }} }}
    .card {{ background: linear-gradient(180deg, #161b22 0%, #13171d 100%); border: 1px solid #30363d; border-radius: 10px; padding: 16px; }}
    .card.idle {{ opacity: 0.55; }}
    .card-header {{ display: flex; align-items: baseline; gap: 10px; }}
    .card-label {{ font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #93a1ad; }}
    .running {{ color: #5ee28a; }}
    .card-title {{ font-size: 14px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .card-subtitle, .run-meta {{ color: #93a1ad; font-size: 11px; line-height: 1.4; margin-top: 2px; }}
    .progress-bar {{ background: #29313a; border-radius: 999px; height: 6px; overflow: hidden; margin: 10px 0 6px; }}
    .progress-fill {{ background: linear-gradient(90deg, #58a6ff 0%, #79c0ff 100%); height: 6px; border-radius: 999px; transition: width .5s; }}
    .progress-label {{ font-size: 11px; color: #93a1ad; }}
    .fact-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 8px; }}
    .fact-grid.compact {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .fact {{ padding: 8px; border: 1px solid #21262d; border-radius: 8px; background: #0f141a; min-height: 52px; }}
    .fact-label {{ color: #93a1ad; font-size: 10px; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 4px; }}
    .fact-value {{ color: #e6edf3; font-size: 13px; line-height: 1.3; word-break: break-word; }}
    .mini-section-title {{ font-size: 11px; color: #93a1ad; letter-spacing: 0.06em; text-transform: uppercase; margin-top: 10px; }}
    .chart-card {{ background: linear-gradient(180deg, #161b22 0%, #13171d 100%); border: 1px solid #30363d; border-radius: 10px; padding: 12px; }}
    .chart-title {{ font-size: 11px; color: #93a1ad; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 6px; }}
    .chart-empty {{ color: #93a1ad; font-size: 12px; margin: 8px 0; }}
    .banner.warning {{ margin-bottom: 16px; border: 1px solid #4d3520; background: #2a1c10; border-radius: 10px; padding: 12px 14px; }}
    .banner-title {{ color: #ffcf5a; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px; }}
    .banner-body {{ color: #f2d7b3; font-size: 12px; line-height: 1.4; }}
    .section-block {{ margin-top: 24px; }}
    .section-title {{ font-size: 12px; color: #93a1ad; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 10px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{ text-align: left; font-size: 11px; color: #93a1ad; padding: 6px 8px; border-bottom: 1px solid #30363d; letter-spacing: 0.06em; text-transform: uppercase; }}
    td {{ padding: 7px 8px; border-bottom: 1px solid #21262d; font-size: 13px; }}
    tr:last-child td {{ border-bottom: none; }}
    .best-row td {{ color: #58a6ff; background: #0d1520; }}
    .ts {{ color: #93a1ad; font-size: 12px; white-space: nowrap; }}
    .no-results {{ color: #93a1ad; padding: 16px 0; }}
    .idle-hint {{ color: #93a1ad; font-size: 12px; margin: 4px 0 0; }}
  </style>
</head>
<body>
  <h1>v3 monitor</h1>
  <div class="meta">{note}</div>
  {alert_banner(status)}
  <div class="top-grid">{live}{best_block}</div>
  <div class="card-grid">{''.join(summary_cards)}</div>
  <div class="chart-grid">{''.join(charts)}</div>
  {eval_block}
  {results_block}
</body>
</html>"""


if __name__ == "__main__":
    host = os.environ.get("V3_DASHBOARD_HOST", os.environ.get("PG_DASHBOARD_HOST", "127.0.0.1"))
    port = int(os.environ.get("V3_DASHBOARD_PORT", os.environ.get("PG_DASHBOARD_PORT", "5062")))
    print(f"Dashboard: http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
