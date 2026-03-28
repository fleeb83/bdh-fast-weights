"""
train.py - mutable experiment file for Hebbian BDH runs.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from prepare import evaluate_all, get_dataloader, log_result, prepare_data
from status_writer import RESULTS_DIR, append_eval_history, append_train_history, write_status
from tokenization import load_encoding_config


def configure_torch(device: str):
    if device != "cuda":
        return None
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(device: str, amp_dtype):
    if device == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 130
    hebbian_lr: float = 0.01
    hebbian_decay: float = 0.9999


def get_freqs(n: int, theta: int, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        nh = config.n_head
        d_model = config.n_embd
        latent_dim = config.mlp_internal_dim_multiplier * d_model // nh
        self.register_buffer(
            "freqs",
            get_freqs(latent_dim, theta=2**16, dtype=torch.float32).view(1, 1, 1, latent_dim),
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, value):
        value_rot = torch.stack((-value[..., 1::2], value[..., ::2]), dim=-1).view(*value.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (value * phases_cos).to(value.dtype) + (value_rot * phases_sin).to(value.dtype)

    def forward(self, q, k, v):
        assert self.freqs.dtype == torch.float32
        assert k is q
        _, _, seq_len, _ = q.size()
        rope_phases = (
            torch.arange(0, seq_len, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)
        ) * self.freqs
        q_rot = self.rope(rope_phases, q)
        scores = (q_rot @ q_rot.mT).tril(diagonal=-1)
        return scores @ v


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        d_model = config.n_embd
        latent_dim = config.mlp_internal_dim_multiplier * d_model // nh

        self.decoder = nn.Parameter(torch.zeros((nh * latent_dim, d_model)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, d_model, latent_dim)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, d_model, latent_dim)).normal_(std=0.02))
        self.attn = Attention(config)
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, d_model)
        self.drop = nn.Dropout(config.dropout)
        self.lm_head = nn.Parameter(torch.zeros((d_model, config.vocab_size)).normal_(std=0.02))
        self.register_buffer("decoder_fast", torch.zeros(nh * latent_dim, d_model))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        cfg = self.config
        batch_size, seq_len = idx.size()
        d_model = cfg.n_embd
        n_head = cfg.n_head
        latent_dim = d_model * cfg.mlp_internal_dim_multiplier // n_head

        x = self.ln(self.embed(idx).unsqueeze(1))

        for _ in range(cfg.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)

            y_kv = self.ln(self.attn(q=x_sparse, k=x_sparse, v=x))
            y_sparse = F.relu(y_kv @ self.encoder_v)
            xy_sparse = x_sparse * y_sparse

            xy_dropped = self.drop(xy_sparse)
            if cfg.hebbian_lr > 0.0:
                x_sq = x.squeeze(1)
                y_mlp_steps = []
                for t in range(seq_len):
                    xy_t = xy_dropped[:, :, t, :].reshape(batch_size, n_head * latent_dim)
                    xs_t = x_sparse[:, :, t, :].reshape(batch_size, n_head * latent_dim)
                    y_t = xy_t @ self.decoder + xs_t @ self.decoder_fast
                    y_mlp_steps.append(y_t)
                    if t > 0:
                        with torch.no_grad():
                            xs_prev = x_sparse[:, :, t - 1, :].reshape(batch_size, n_head * latent_dim).float()
                            x_t = x_sq[:, t, :].float()
                            delta = torch.einsum("bi,bd->id", xs_prev, x_t)
                            self.decoder_fast.add_(cfg.hebbian_lr * delta.to(self.decoder_fast.dtype))
                y_mlp = torch.stack(y_mlp_steps, dim=1).unsqueeze(1)
            else:
                y_mlp = (
                    xy_dropped.transpose(1, 2).reshape(batch_size, 1, seq_len, latent_dim * n_head)
                    @ self.decoder
                )

            x = self.ln(x + self.ln(y_mlp))

        logits = x.view(batch_size, seq_len, d_model) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        out = idx
        for _ in range(max_new_tokens):
            if hasattr(self, "decoder_fast"):
                self.decoder_fast.zero_()
            logits, _ = self(out)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out = torch.cat([out, next_id], dim=1)
        return out


MAX_HOURS = 2.0
EVAL_INTERVAL = 500
CONVERGENCE_WINDOW = 3
CONVERGENCE_THRESHOLD = 0.005
OVERFIT_THRESHOLD = 0.03

BATCH_SIZE = 1
LR = 1e-3
N_BACK_TRAIN = 8

EXPERIMENT_LABEL = "convergence-exp11-xsprev-addr-lr1e-2-bs1"

_GIT_ROOT = Path(__file__).parent.parent.parent


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def set_reproducibility(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _save_checkpoint(model, config, step, scores, label, *, encoding, best=False):
    ckpt_dir = RESULTS_DIR / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    suffix = "_best" if best else f"_step{step:07d}"
    ckpt_path = ckpt_dir / f"{label}{suffix}.pt"
    ckpt = {
        "label": label,
        "config": dataclasses.asdict(config),
        "encoding": encoding.to_dict(),
        "step": step,
        "scores": scores,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("decoder_fast")},
    }
    torch.save(ckpt, ckpt_path)
    return ckpt_path


def _git_commit(label, scores, step):
    try:
        files = [
            str(RESULTS_DIR / "experiment_log.jsonl"),
            str(RESULTS_DIR / "eval_history.jsonl"),
            str(Path(__file__)),
        ]
        subprocess.run(["git", "add"] + files, cwd=str(_GIT_ROOT), check=True, capture_output=True)
        msg = (
            f"run: {label}  n8={scores['n8']:.1%}  n4={scores['n4']:.1%}  n2={scores['n2']:.1%}  step={step}\n\n"
            "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
        )
        subprocess.run(
            ["git", "-c", "user.email=russell@local", "-c", "user.name=Russell", "commit", "-m", msg],
            cwd=str(_GIT_ROOT),
            check=True,
            capture_output=True,
        )
        print(f"Git commit: {label} n8={scores['n8']:.1%}")
    except subprocess.CalledProcessError as exc:
        print(f"[git commit skipped: {exc}]")


def _git_rev():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_GIT_ROOT),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return None


def _env_snapshot():
    keys = sorted(k for k in os.environ if k.startswith("HEBBIAN_"))
    return {k: os.environ[k] for k in keys}


def _write_manifest(label, payload):
    manifest_dir = RESULTS_DIR / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    path = manifest_dir / f"{label}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = env_int("HEBBIAN_SEED", 1337)
    deterministic = env_bool("HEBBIAN_DETERMINISTIC", False)
    set_reproducibility(seed, deterministic=deterministic)
    prepare_data()
    encoding = load_encoding_config()
    amp_dtype = configure_torch(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and amp_dtype == torch.float16))

    max_hours = env_float("HEBBIAN_MAX_HOURS", MAX_HOURS)
    default_eval_interval = 2000 if encoding.mode == "bpe" else EVAL_INTERVAL
    eval_interval = env_int("HEBBIAN_EVAL_INTERVAL", default_eval_interval)
    batch_size = env_int("HEBBIAN_BATCH_SIZE", BATCH_SIZE)
    lr = env_float("HEBBIAN_LR", LR)
    n_back_train = env_int("HEBBIAN_N_BACK_TRAIN", N_BACK_TRAIN)
    experiment_label = env_str("HEBBIAN_EXPERIMENT_LABEL", EXPERIMENT_LABEL)
    hebbian_lr = env_float("HEBBIAN_HEBBIAN_LR", BDHConfig.hebbian_lr)
    if encoding.mode == "bpe" and batch_size != 1:
        print(f"Forcing batch_size=1 for BPE mode (requested {batch_size})")
        batch_size = 1

    config = BDHConfig(
        n_layer=env_int("HEBBIAN_N_LAYER", BDHConfig.n_layer),
        n_embd=env_int("HEBBIAN_N_EMBD", BDHConfig.n_embd),
        dropout=env_float("HEBBIAN_DROPOUT", BDHConfig.dropout),
        n_head=env_int("HEBBIAN_N_HEAD", BDHConfig.n_head),
        mlp_internal_dim_multiplier=env_int(
            "HEBBIAN_MLP_INTERNAL_DIM_MULTIPLIER", BDHConfig.mlp_internal_dim_multiplier
        ),
        vocab_size=encoding.vocab_size,
        hebbian_lr=hebbian_lr,
        hebbian_decay=env_float("HEBBIAN_HEBBIAN_DECAY", BDHConfig.hebbian_decay),
    )
    model = BDH(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    loader = get_dataloader(n_back_train, "train", batch_size, shuffle=True)
    manifest = {
        "label": experiment_label,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_rev(),
        "seed": seed,
        "deterministic": deterministic,
        "encoding": encoding.to_dict(),
        "config": dataclasses.asdict(config),
        "env": _env_snapshot(),
        "n_back_train": n_back_train,
        "batch_size": batch_size,
        "lr": lr,
        "max_hours": max_hours,
        "eval_interval": eval_interval,
        "status": "started",
    }
    manifest_path = _write_manifest(experiment_label, manifest)

    start_time = time.time()
    deadline = start_time + max_hours * 3600
    step = 0
    it = iter(loader)

    eval_n8_history = []
    best_n8 = 0.0
    best_step = 0
    best_ckpt_path = None
    last_eval_scores = None
    last_eval_step = None
    stop_reason = "max_time"
    scoring_mode = "bpe-teacher-forced-span" if encoding.mode == "bpe" else "int-final-token"

    if device == "cuda":
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(
            f"CUDA device: {torch.cuda.get_device_name(0)}  total_memory={total_mem_gb:.2f} GB  "
            f"amp={amp_dtype}  encoding={encoding.label}"
        )

    while time.time() < deadline:
        model.train()
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)

        x, y = x.to(device), y.to(device)
        model.decoder_fast.zero_()
        with autocast_context(device, amp_dtype):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        elapsed = time.time() - start_time

        if step % 100 == 0:
            print(f"step {step}  loss {loss.item():.4f}  elapsed {elapsed:.0f}s")
            current_lr = opt.param_groups[0]["lr"]
            append_train_history(experiment_label, step, loss.item(), elapsed, lr=current_lr, phase="train")
            write_status(
                experiment_label,
                step,
                loss.item(),
                elapsed,
                eval_scores=last_eval_scores,
                eval_step=last_eval_step,
                best_n8=best_n8 if best_n8 > 0 else None,
                best_step=best_step,
                max_elapsed_s=max_hours * 3600,
                phase="train",
                lr=current_lr,
                encoding_label=encoding.label,
                scoring_mode=scoring_mode,
            )

        if step % eval_interval == 0:
            print(f"\n--- Eval at step {step} ---")
            current_lr = opt.param_groups[0]["lr"]
            write_status(
                experiment_label,
                step,
                loss.item(),
                elapsed,
                eval_scores=last_eval_scores,
                eval_step=last_eval_step,
                best_n8=best_n8 if best_n8 > 0 else None,
                best_step=best_step,
                max_elapsed_s=max_hours * 3600,
                phase="eval",
                phase_detail="Evaluating held-out n2,n4,n8",
                lr=current_lr,
                encoding_label=encoding.label,
                scoring_mode=scoring_mode,
            )
            if device == "cuda":
                torch.cuda.empty_cache()

            def eval_progress(n_back: int, done: int, total: int | None, accuracy: float | None) -> None:
                if total:
                    pct = 100.0 * done / total
                    partial = f" partial {accuracy:.1%}" if accuracy is not None and done > 0 else ""
                    detail = f"Evaluating held-out n{n_back}: {done}/{total} ({pct:.1f}%){partial}"
                else:
                    detail = f"Evaluating held-out n{n_back}: starting"
                write_status(
                    experiment_label,
                    step,
                    loss.item(),
                    time.time() - start_time,
                    eval_scores=last_eval_scores,
                    eval_step=last_eval_step,
                    best_n8=best_n8 if best_n8 > 0 else None,
                    best_step=best_step,
                    max_elapsed_s=max_hours * 3600,
                    phase="eval",
                    phase_detail=detail,
                    lr=current_lr,
                    encoding_label=encoding.label,
                    scoring_mode=scoring_mode,
                )

            scores = evaluate_all(model, device, progress_callback=eval_progress)
            last_eval_scores = scores
            last_eval_step = step

            append_eval_history(experiment_label, step, loss.item(), elapsed, scores)
            print(f"  n2={scores['n2']:.1%}  n4={scores['n4']:.1%}  n8={scores['n8']:.1%}")

            _save_checkpoint(model, config, step, scores, experiment_label, encoding=encoding)

            if scores["n8"] > best_n8:
                best_n8 = scores["n8"]
                best_step = step
                best_ckpt_path = _save_checkpoint(
                    model, config, step, scores, experiment_label, encoding=encoding, best=True
                )
                print(f"  ** New best n8: {best_n8:.1%} at step {step}")

            eval_n8_history.append(scores["n8"])
            if len(eval_n8_history) >= CONVERGENCE_WINDOW:
                recent = eval_n8_history[-CONVERGENCE_WINDOW:]
                delta = max(recent) - min(recent)
                write_status(
                    experiment_label,
                    step,
                    loss.item(),
                    elapsed,
                    eval_scores=scores,
                    eval_step=step,
                    best_n8=best_n8,
                    best_step=best_step,
                    max_elapsed_s=max_hours * 3600,
                    convergence_delta=delta,
                    phase="train",
                    lr=current_lr,
                    encoding_label=encoding.label,
                    scoring_mode=scoring_mode,
                )
                if delta < CONVERGENCE_THRESHOLD:
                    stop_reason = "converged"
                    break
            else:
                write_status(
                    experiment_label,
                    step,
                    loss.item(),
                    elapsed,
                    eval_scores=scores,
                    eval_step=step,
                    best_n8=best_n8,
                    best_step=best_step,
                    max_elapsed_s=max_hours * 3600,
                    phase="train",
                    lr=current_lr,
                    encoding_label=encoding.label,
                    scoring_mode=scoring_mode,
                )

            if scores["n8"] < best_n8 - OVERFIT_THRESHOLD:
                stop_reason = "overfit"
                break

            model.train()

    print(f"\nTraining complete ({step} steps, reason={stop_reason}). Final eval...")
    current_lr = opt.param_groups[0]["lr"]
    write_status(
        experiment_label,
        step,
        loss.item(),
        time.time() - start_time,
        eval_scores=last_eval_scores,
        eval_step=last_eval_step,
        best_n8=best_n8 if best_n8 > 0 else None,
        best_step=best_step,
        max_elapsed_s=max_hours * 3600,
        phase="final_eval",
        phase_detail="Running final held-out eval",
        lr=current_lr,
        encoding_label=encoding.label,
        scoring_mode=scoring_mode,
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    def final_eval_progress(n_back: int, done: int, total: int | None, accuracy: float | None) -> None:
        if total:
            pct = 100.0 * done / total
            partial = f" partial {accuracy:.1%}" if accuracy is not None and done > 0 else ""
            detail = f"Final held-out n{n_back}: {done}/{total} ({pct:.1f}%){partial}"
        else:
            detail = f"Final held-out n{n_back}: starting"
        write_status(
            experiment_label,
            step,
            loss.item(),
            time.time() - start_time,
            eval_scores=last_eval_scores,
            eval_step=last_eval_step,
            best_n8=best_n8 if best_n8 > 0 else None,
            best_step=best_step,
            max_elapsed_s=max_hours * 3600,
            phase="final_eval",
            phase_detail=detail,
            lr=current_lr,
            encoding_label=encoding.label,
            scoring_mode=scoring_mode,
        )

    scores = evaluate_all(model, device, final=True, progress_callback=final_eval_progress)
    log_result(experiment_label, scores, notes=f"encoding={encoding.label}")

    elapsed = time.time() - start_time
    append_eval_history(experiment_label, step, loss.item(), elapsed, scores)
    _save_checkpoint(model, config, step, scores, experiment_label, encoding=encoding)
    if scores["n8"] > best_n8:
        best_n8 = scores["n8"]
        best_step = step
        best_ckpt_path = _save_checkpoint(
            model, config, step, scores, experiment_label, encoding=encoding, best=True
        )

    if best_ckpt_path:
        print(f"Best checkpoint: {best_ckpt_path}")

    manifest.update(
        {
            "status": "completed",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stop_reason": stop_reason,
            "final_step": step,
            "final_scores": scores,
            "best_n8": best_n8,
            "best_step": best_step,
            "best_checkpoint": str(best_ckpt_path) if best_ckpt_path else None,
            "manifest_path": str(manifest_path),
        }
    )
    _write_manifest(experiment_label, manifest)

    write_status(
        experiment_label,
        step,
        loss.item(),
        elapsed,
        eval_scores=scores,
        eval_step=step,
        best_n8=best_n8,
        best_step=best_step,
        max_elapsed_s=max_hours * 3600,
        converged=(stop_reason == "converged"),
        phase="done",
        phase_detail=f"Finished ({stop_reason})",
        lr=current_lr,
        encoding_label=encoding.label,
        scoring_mode=scoring_mode,
    )

    _git_commit(experiment_label, scores, step)


if __name__ == "__main__":
    train()
