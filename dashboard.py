"""
dashboard.py - autoresearch-hebbian mechanism monitor.
Usage: python dashboard.py   (then open http://localhost:5000)
"""

import json
import os
from datetime import datetime

from flask import Flask

from status_writer import EVAL_HISTORY_PATH, RESULTS_DIR, STATUS_PATH, TRAIN_HISTORY_PATH

app = Flask(__name__)

STALE_AFTER_S = int(os.environ.get("HEBBIAN_DASHBOARD_STALE_AFTER_S", "7200"))


def load_status():
    if not STATUS_PATH.exists():
        return None
    try:
        status = json.loads(STATUS_PATH.read_text())
        age_s = (datetime.now() - datetime.fromisoformat(status["timestamp"])).total_seconds()
        return status if age_s < STALE_AFTER_S else None
    except Exception:
        return None


def _load_jsonl(path, label=None):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            if label is None or row.get("label") == label:
                rows.append(row)
        except Exception:
            pass
    return rows


def load_results():
    rows = _load_jsonl(RESULTS_DIR / "experiment_log.jsonl")
    return sorted(rows, key=lambda r: r.get("n8", 0.0), reverse=True)


def load_eval_history(label=None):
    rows = _load_jsonl(EVAL_HISTORY_PATH, label=label)
    return sorted(rows, key=lambda r: r.get("step", 0))


def load_train_history(label=None):
    rows = _load_jsonl(TRAIN_HISTORY_PATH, label=label)
    return sorted(rows, key=lambda r: r.get("step", 0))


def render_eval_chart(eval_rows, best_n8=None, width=640, height=130):
    if len(eval_rows) < 2:
        return '<p class="chart-empty">Waiting for first eval checkpoint...</p>'

    pad_l, pad_r, pad_t, pad_b = 42, 16, 10, 24
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b

    steps = [r["step"] for r in eval_rows]
    min_s, max_s = min(steps), max(steps)
    s_range = max_s - min_s or 1

    def sx(step):
        return pad_l + (step - min_s) / s_range * inner_w

    def sy(val):
        return pad_t + inner_h - val * inner_h

    grid = ""
    for val in (0.25, 0.50, 0.75, 1.00):
        y = sy(val)
        grid += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + inner_w}" y2="{y:.1f}" stroke="#21262d" stroke-width="1"/>'
        grid += f'<text x="{pad_l - 4}" y="{y + 4:.1f}" text-anchor="end" fill="#93a1ad" font-size="10">{val:.0%}</text>'

    axis = ""
    n_ticks = min(5, len(steps))
    tick_steps = [steps[int(i * (len(steps) - 1) / max(n_ticks - 1, 1))] for i in range(n_ticks)]
    for ts in tick_steps:
        x = sx(ts)
        axis += f'<line x1="{x:.1f}" y1="{pad_t + inner_h}" x2="{x:.1f}" y2="{pad_t + inner_h + 4}" stroke="#30363d" stroke-width="1"/>'
        axis += f'<text x="{x:.1f}" y="{pad_t + inner_h + 16}" text-anchor="middle" fill="#93a1ad" font-size="10">{ts}</text>'

    def make_path(key, color):
        pts = [(sx(r["step"]), sy(r[key])) for r in eval_rows if r.get(key) is not None]
        if len(pts) < 2:
            return ""
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        return f'<path d="{d}" stroke="{color}" stroke-width="2" fill="none" stroke-linejoin="round"/>'

    paths = make_path("n2", "#58a6ff") + make_path("n4", "#db8422") + make_path("n8", "#3fb950")

    dots = ""
    for r in eval_rows:
        if r.get("n8") is None:
            continue
        cx, cy = sx(r["step"]), sy(r["n8"])
        dots += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="#3fb950" stroke="#0f1216" stroke-width="1.5"/>'

    if best_n8 is not None:
        best_rows = [r for r in eval_rows if r.get("n8") is not None and abs(r["n8"] - best_n8) < 0.0001]
        if best_rows:
            br = best_rows[-1]
            bx, by = sx(br["step"]), sy(br["n8"])
            dots += f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="6" fill="none" stroke="#ffcf5a" stroke-width="2"/>'
            dots += f'<text x="{bx + 8:.1f}" y="{by - 4:.1f}" fill="#ffcf5a" font-size="10">best {best_n8:.1%}</text>'

    legend = (
        f'<rect x="{pad_l}" y="{height - 8}" width="10" height="4" fill="#58a6ff" rx="1"/>'
        f'<text x="{pad_l + 13}" y="{height - 4}" fill="#93a1ad" font-size="10">n2</text>'
        f'<rect x="{pad_l + 35}" y="{height - 8}" width="10" height="4" fill="#db8422" rx="1"/>'
        f'<text x="{pad_l + 48}" y="{height - 4}" fill="#93a1ad" font-size="10">n4</text>'
        f'<rect x="{pad_l + 70}" y="{height - 8}" width="10" height="4" fill="#3fb950" rx="1"/>'
        f'<text x="{pad_l + 83}" y="{height - 4}" fill="#93a1ad" font-size="10">n8</text>'
    )

    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="display:block;max-width:100%">'
        f"{grid}{axis}{paths}{dots}{legend}</svg>"
    )


def render_loss_chart(train_rows, width=640, height=130):
    if len(train_rows) < 2:
        return '<p class="chart-empty">Waiting for live loss history...</p>'

    rows = train_rows[-80:]
    pad_l, pad_r, pad_t, pad_b = 42, 16, 10, 24
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b

    steps = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]
    min_s, max_s = min(steps), max(steps)
    min_l, max_l = min(losses), max(losses)
    s_range = max_s - min_s or 1
    l_range = max_l - min_l or 1

    def sx(step):
        return pad_l + (step - min_s) / s_range * inner_w

    def sy(loss):
        return pad_t + inner_h - (loss - min_l) / l_range * inner_h

    grid = ""
    for frac in (0.0, 0.5, 1.0):
        y = pad_t + inner_h - frac * inner_h
        label = min_l + frac * l_range
        grid += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + inner_w}" y2="{y:.1f}" stroke="#21262d" stroke-width="1"/>'
        grid += f'<text x="{pad_l - 4}" y="{y + 4:.1f}" text-anchor="end" fill="#93a1ad" font-size="10">{label:.2f}</text>'

    pts = [(sx(r["step"]), sy(r["loss"])) for r in rows]
    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    line = f'<path d="{d}" stroke="#ff7b72" stroke-width="2" fill="none" stroke-linejoin="round"/>'
    dots = ""
    for r in rows[-8:]:
        cx, cy = sx(r["step"]), sy(r["loss"])
        dots += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.5" fill="#ff7b72" stroke="#0f1216" stroke-width="1"/>'

    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="display:block;max-width:100%">'
        f"{grid}{line}{dots}</svg>"
    )


def render_lr_chart(train_rows, width=640, height=90):
    rows = [r for r in train_rows[-80:] if r.get("lr") is not None]
    if len(rows) < 2:
        return '<p class="chart-empty">Waiting for lr history...</p>'

    pad_l, pad_r, pad_t, pad_b = 42, 16, 10, 24
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b

    steps = [r["step"] for r in rows]
    lrs = [r["lr"] for r in rows]
    min_s, max_s = min(steps), max(steps)
    min_lr, max_lr = min(lrs), max(lrs)
    s_range = max_s - min_s or 1
    lr_range = max_lr - min_lr or max(max_lr, 1e-12)

    def sx(step):
        return pad_l + (step - min_s) / s_range * inner_w

    def sy(lr):
        return pad_t + inner_h - (lr - min_lr) / lr_range * inner_h

    grid = ""
    for frac in (0.0, 0.5, 1.0):
        y = pad_t + inner_h - frac * inner_h
        label = min_lr + frac * lr_range
        grid += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + inner_w}" y2="{y:.1f}" stroke="#21262d" stroke-width="1"/>'
        grid += f'<text x="{pad_l - 4}" y="{y + 4:.1f}" text-anchor="end" fill="#93a1ad" font-size="10">{label:.1e}</text>'

    pts = [(sx(r["step"]), sy(r["lr"])) for r in rows]
    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    line = f'<path d="{d}" stroke="#79c0ff" stroke-width="2" fill="none" stroke-linejoin="round"/>'
    last_x, last_y = pts[-1]
    dots = f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3" fill="#79c0ff" stroke="#0f1216" stroke-width="1"/>'

    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="display:block;max-width:100%">'
        f"{grid}{line}{dots}</svg>"
    )


def render_live_card(status, eval_rows, train_rows):
    if not status:
        return """
        <div class="card idle">
          <div class="card-label">IDLE</div>
          <div class="card-title">No experiment running</div>
          <p class="idle-hint">Start train.py to begin an experiment.</p>
        </div>"""

    elapsed = status.get("elapsed_s", 0)
    max_time = status.get("max_elapsed_s", 7200)
    pct = min(100, int(elapsed / max_time * 100))
    elapsed_h = int(elapsed // 3600)
    elapsed_m = int((elapsed % 3600) // 60)
    elapsed_sec = int(elapsed % 60)
    max_h = int(max_time // 3600)
    elapsed_str = f"{elapsed_h}h {elapsed_m:02d}m {elapsed_sec:02d}s" if elapsed_h else f"{elapsed_m}m {elapsed_sec:02d}s"
    max_str = f"{max_h}h 00m" if max_h else f"{int(max_time // 60)}m"

    phase = status.get("phase", "train")
    phase_detail = status.get("phase_detail") or ""
    phase_label = {
        "train": "RUNNING",
        "eval": "EVALUATING",
        "final_eval": "FINAL EVAL",
        "done": "DONE",
    }.get(phase, phase.upper())
    current_lr = status.get("lr")
    lr_text = f"{current_lr:.2e}" if current_lr is not None else "--"
    scoring_mode = status.get("scoring_mode") or "--"
    encoding_label = status.get("encoding_label") or "--"
    timestamp = status.get("timestamp", "").replace("T", " ")

    ev = status.get("eval")
    if ev:
        best_n8 = status.get("best_n8", ev.get("n8", 0.0))
        best_step = status.get("best_step", "")
        eval_html = f"""
        <div class="eval-row">
          <div class="eval-metric"><span class="em-label">n2</span><span class="em-val">{ev['n2']:.1%}</span></div>
          <div class="eval-metric"><span class="em-label">n4</span><span class="em-val">{ev['n4']:.1%}</span></div>
          <div class="eval-metric n8-metric"><span class="em-label">n8</span><span class="em-val n8-val">{ev['n8']:.1%}</span></div>
          <div class="eval-metric"><span class="em-label">best n8</span><span class="em-val best-val">{best_n8:.1%} <span class="step-hint">@{best_step}</span></span></div>
        </div>"""
    elif phase in {"eval", "final_eval"}:
        eval_html = f'<div class="eval-pending">{phase_detail or "Held-out evaluation in progress..."}</div>'
    else:
        eval_html = '<div class="eval-pending">Waiting for next eval checkpoint...</div>'

    conv_delta = status.get("convergence_delta")
    conv_html = ""
    if conv_delta is not None:
        conv_color = "#3fb950" if conv_delta < 0.005 else "#93a1ad"
        conv_html = f'<div class="conv-status" style="color:{conv_color}">convergence delta {conv_delta:.3%} (need &lt;0.5%)</div>'

    eval_chart = render_eval_chart(eval_rows, best_n8=status.get("best_n8"))
    loss_chart = render_loss_chart(train_rows)
    lr_chart = render_lr_chart(train_rows)

    return f"""
    <div class="card live-card">
      <div class="card-header">
        <div class="card-label running">{phase_label}</div>
        <div class="card-title">{status['label']}</div>
      </div>
      <div class="progress-bar"><div class="progress-fill" style="width:{pct}%"></div></div>
      <div class="progress-label">{elapsed_str} / {max_str} &middot; step {status['step']} &middot; loss {status['loss']:.4f}</div>
      <div class="run-meta">lr {lr_text} &middot; {encoding_label} &middot; {scoring_mode}</div>
      <div class="run-meta">{phase_detail or "active"}</div>
      <div class="run-meta">last update {timestamp}</div>
      {eval_html}
      {conv_html}
      <div class="mini-section-title">Recent training loss</div>
      <div class="chart-wrap">{loss_chart}</div>
      <div class="mini-section-title">Recent lr</div>
      <div class="chart-wrap">{lr_chart}</div>
      <div class="mini-section-title">Held-out eval trajectory</div>
      <div class="chart-wrap">{eval_chart}</div>
    </div>"""


def is_baseline_label(label):
    return any(s in (label or "").lower() for s in ("baseline", "no-hebbian", "nohebbian"))


def get_baseline_result(results):
    baselines = [r for r in results if is_baseline_label(r.get("label", ""))]
    return max(baselines, key=lambda r: r.get("n8", 0.0)) if baselines else None


def uplift_pp(row, baseline):
    if not row or not baseline:
        return None
    return (row.get("n8", 0.0) - baseline.get("n8", 0.0)) * 100.0


def materially_regressed(row, baseline):
    if not row or not baseline:
        return False
    return row.get("n2", 0.0) < baseline.get("n2", 0.0) or row.get("n4", 0.0) < baseline.get("n4", 0.0)


def proof_verdict(best, baseline):
    if not baseline:
        return "Need baseline", "neutral"
    if not best:
        return "No evidence yet", "neutral"
    uplift = uplift_pp(best, baseline)
    if uplift is None:
        return "No evidence yet", "neutral"
    if uplift >= 10.0 and not materially_regressed(best, baseline):
        return "Target hit", "hit"
    if uplift > 0.0:
        return "Improving", "improving"
    return "No evidence yet", "neutral"


def render_proof_card(best, baseline):
    verdict_text, verdict_class = proof_verdict(best, baseline)
    baseline_n8 = f"{baseline['n8']:.1%}" if baseline else "--"
    best_n8_str = f"{best['n8']:.1%}" if best else "--"
    uplift = uplift_pp(best, baseline)
    uplift_text = f"{uplift:+.1f} pp" if uplift is not None else "--"
    best_label = best["label"] if best else "No completed runs yet"

    return f"""
    <div class="card proof-card">
      <div class="card-label proof {verdict_class}">MECHANISM CHECK</div>
      <div class="card-title">{best_label}</div>
      <div class="proof-grid">
        <div class="proof-item">
          <div class="score-label">baseline n8</div>
          <div class="score-val">{baseline_n8}</div>
        </div>
        <div class="proof-item">
          <div class="score-label">best n8</div>
          <div class="score-val n8">{best_n8_str}</div>
        </div>
        <div class="proof-item">
          <div class="score-label">uplift</div>
          <div class="score-val uplift">{uplift_text}</div>
        </div>
      </div>
      <div class="verdict {verdict_class}">{verdict_text}</div>
    </div>"""


def render_eval_checkpoints_table(eval_rows, best_n8=None):
    if not eval_rows:
        return ""
    rows_html = ""
    prev_n8 = None
    for r in reversed(eval_rows[-20:]):
        delta_html = ""
        if prev_n8 is not None and r.get("n8") is not None:
            delta = r["n8"] - prev_n8
            col = "#3fb950" if delta > 0 else "#f85149" if delta < -0.001 else "#93a1ad"
            delta_html = f'<span style="color:{col}">{delta:+.1%}</span>'
        is_best = best_n8 is not None and r.get("n8") is not None and abs(r["n8"] - best_n8) < 0.0001
        row_cls = "best-row" if is_best else ""
        star = "*" if is_best else ""
        ts = r.get("timestamp", "")[:16].replace("T", " ")
        rows_html = f"""
        <tr class="{row_cls}">
          <td>{star} {r['step']}</td>
          <td>{r['loss']:.4f}</td>
          <td>{r['n2']:.1%}</td>
          <td>{r['n4']:.1%}</td>
          <td>{r['n8']:.1%}</td>
          <td>{delta_html}</td>
          <td class="ts">{ts}</td>
        </tr>""" + rows_html
        prev_n8 = r.get("n8")

    return f"""
    <h3 class="section-title">Eval checkpoints - current run</h3>
    <table>
      <thead><tr>
        <th>step</th><th>loss</th><th>n2</th><th>n4</th><th>n8</th><th>delta n8</th><th>time</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>"""


def render_results_table(results, baseline):
    if not results:
        return "<p class='no-results'>No completed experiments yet.</p>"

    rows_html = ""
    for idx, row in enumerate(results):
        row_class = "best-row" if idx == 0 else ""
        prefix = "* " if idx == 0 else ""
        timestamp = row.get("timestamp", "")[:16].replace("T", " ")
        uplift = uplift_pp(row, baseline)
        uplift_text = f"{uplift:+.1f} pp" if uplift is not None else "--"
        rows_html += f"""
        <tr class="{row_class}">
          <td>{prefix}{row['label']}</td>
          <td>{row['n2']:.1%}</td>
          <td>{row['n4']:.1%}</td>
          <td>{row['n8']:.1%}</td>
          <td>{uplift_text}</td>
          <td class="ts">{timestamp}</td>
        </tr>"""

    return f"""
    <h3 class="section-title">All completed runs</h3>
    <table>
      <thead><tr>
        <th>label</th><th>n2</th><th>n4</th><th>n8</th><th>uplift</th><th>time</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>"""


@app.route("/")
def index():
    status = load_status()
    results = load_results()
    best = results[0] if results else None
    baseline = get_baseline_result(results)

    current_label = status["label"] if status else None
    eval_rows = load_eval_history(current_label)
    train_rows = load_train_history(current_label)
    run_count = len(results)

    live_card = render_live_card(status, eval_rows, train_rows)
    proof_card = render_proof_card(best, baseline)
    ckpt_table = render_eval_checkpoints_table(eval_rows, best_n8=status.get("best_n8") if status else None)
    results_table = render_results_table(results, baseline)

    refresh = 5 if status else 60

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{refresh}">
  <title>autoresearch-hebbian</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 24px; background: #0f1216; color: #e6edf3;
           font-family: ui-monospace, 'Cascadia Code', monospace; font-size: 14px; }}
    h1 {{ margin: 0 0 4px; font-size: 18px; color: #e6edf3; }}
    .meta {{ color: #93a1ad; font-size: 12px; margin-bottom: 20px; }}
    .top-grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin-bottom: 24px; }}
    @media (max-width: 860px) {{ .top-grid {{ grid-template-columns: 1fr; }} }}
    .card {{ background: linear-gradient(180deg, #161b22 0%, #13171d 100%);
             border: 1px solid #30363d; border-radius: 10px; padding: 16px; }}
    .card.idle {{ opacity: 0.55; }}
    .live-card {{ display: flex; flex-direction: column; gap: 10px; }}
    .proof-card {{ display: flex; flex-direction: column; gap: 10px; }}
    .card-header {{ display: flex; align-items: baseline; gap: 10px; }}
    .card-label {{ font-size: 11px; font-weight: 700; letter-spacing: 0.08em; }}
    .card-label.running {{ color: #5ee28a; }}
    .card-label.proof.hit {{ color: #ffcf5a; }}
    .card-label.proof.improving {{ color: #58a6ff; }}
    .card-label.proof.neutral {{ color: #93a1ad; }}
    .card-title {{ font-size: 14px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .progress-bar {{ background: #29313a; border-radius: 999px; height: 6px; overflow: hidden; }}
    .progress-fill {{ background: linear-gradient(90deg, #3fb950 0%, #5ee28a 100%);
                      height: 6px; border-radius: 999px; transition: width .5s; }}
    .progress-label {{ font-size: 11px; color: #93a1ad; }}
    .run-meta {{ font-size: 11px; color: #93a1ad; }}
    .eval-row {{ display: flex; gap: 20px; flex-wrap: wrap; padding: 8px 0;
                 border-top: 1px solid #21262d; border-bottom: 1px solid #21262d; }}
    .eval-metric {{ display: flex; flex-direction: column; }}
    .em-label {{ font-size: 10px; color: #93a1ad; letter-spacing: 0.06em; }}
    .em-val {{ font-size: 22px; font-weight: 700; color: #e6edf3; }}
    .n8-metric .n8-val {{ color: #3fb950; }}
    .best-val {{ color: #ffcf5a; }}
    .step-hint {{ font-size: 12px; color: #93a1ad; font-weight: 400; }}
    .eval-pending {{ color: #93a1ad; font-size: 12px; padding: 4px 0; }}
    .conv-status {{ font-size: 12px; padding: 2px 0; }}
    .mini-section-title {{ font-size: 11px; color: #93a1ad; letter-spacing: 0.06em; text-transform: uppercase; margin-top: 6px; }}
    .chart-wrap {{ margin-top: 4px; overflow: hidden; }}
    .chart-empty {{ color: #93a1ad; font-size: 12px; margin: 8px 0; }}
    .idle-hint {{ color: #93a1ad; font-size: 12px; margin: 4px 0 0; }}
    .proof-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
    .proof-item {{ text-align: center; }}
    .score-label {{ font-size: 11px; color: #93a1ad; margin-bottom: 4px; }}
    .score-val {{ font-size: 24px; font-weight: 700; color: #e6edf3; }}
    .score-val.n8 {{ color: #5ee28a; }}
    .score-val.uplift {{ color: #58a6ff; }}
    .verdict {{ display: inline-block; padding: 5px 8px; border-radius: 999px; font-size: 11px;
                font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }}
    .verdict.hit {{ background: #3a2c09; color: #ffcf5a; }}
    .verdict.improving {{ background: #10243d; color: #58a6ff; }}
    .verdict.neutral {{ background: #20262d; color: #93a1ad; }}
    .section-title {{ font-size: 12px; color: #93a1ad; letter-spacing: 0.06em; text-transform: uppercase;
                      margin: 24px 0 10px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{ text-align: left; font-size: 11px; color: #93a1ad; padding: 6px 8px;
          border-bottom: 1px solid #30363d; letter-spacing: 0.06em; }}
    td {{ padding: 7px 8px; border-bottom: 1px solid #21262d; font-size: 13px; }}
    tr:last-child td {{ border-bottom: none; }}
    .best-row td {{ color: #5ee28a; background: #0d1f10; }}
    .ts {{ color: #93a1ad; font-size: 12px; white-space: nowrap; }}
    .no-results {{ color: #93a1ad; padding: 16px 0; }}
  </style>
</head>
<body>
  <h1>autoresearch-hebbian</h1>
  <div class="meta">{run_count} completed run{"s" if run_count != 1 else ""} &middot; convergence-based stopping (max 2h) &middot; primary metric: n8</div>
  <div class="top-grid">
    {live_card}
    {proof_card}
  </div>
  {ckpt_table}
  {results_table}
</body>
</html>"""


if __name__ == "__main__":
    host = os.environ.get("HEBBIAN_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("HEBBIAN_DASHBOARD_PORT", "5000"))
    print(f"Dashboard: http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
