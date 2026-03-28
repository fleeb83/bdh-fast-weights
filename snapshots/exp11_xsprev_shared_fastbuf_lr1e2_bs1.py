"""
Snapshot: Exp 11 canonical configuration.

Result (original run): n2=64.0%  n4=65.4%  n8=58.6%
Hardware: RTX 5070 Ti Laptop, 12GB VRAM, bfloat16
Training: 5 min wall clock, 811 steps, final loss ~1.96

Mechanism: shared decoder_fast buffer updated Hebbianly per-token using
x_sparse[t-1] as write address and x[t] as value. All 6 layers read/write
the same buffer — cross-layer cascade is essential (per-layer buffers fail).

This file is a frozen snapshot. Do not modify. To reproduce, run:
    python snapshots/exp11_xsprev_shared_fastbuf_lr1e2_bs1.py
"""

import dataclasses
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from status_writer import write_status
from prepare import (
    MODEL_VOCAB,
    evaluate,
    evaluate_all,
    get_dataloader,
    log_result,
    prepare_data,
)


def configure_torch(device):
    if device != "cuda":
        return None

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(device, amp_dtype):
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
    vocab_size: int = MODEL_VOCAB
    hebbian_lr: float = 0.01
    hebbian_decay: float = 0.9999


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        nh = config.n_head
        d_model = config.n_embd
        latent_dim = config.mlp_internal_dim_multiplier * d_model // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(latent_dim, theta=2**16, dtype=torch.float32).view(1, 1, 1, latent_dim)
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
        self.register_buffer('decoder_fast', torch.zeros(nh * latent_dim, d_model))
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
                            xs_prev = x_sparse[:, :, t-1, :].reshape(batch_size, n_head * latent_dim).float()
                            x_t     = x_sq[:, t, :].float()
                            delta   = torch.einsum('bi,bd->id', xs_prev, x_t)
                            self.decoder_fast.add_(cfg.hebbian_lr * delta.to(self.decoder_fast.dtype))
                y_mlp = torch.stack(y_mlp_steps, dim=1).unsqueeze(1)
            else:
                y_mlp = xy_dropped.transpose(1, 2).reshape(batch_size, 1, seq_len, latent_dim * n_head) @ self.decoder
            x = self.ln(x + self.ln(y_mlp))

        logits = x.view(batch_size, seq_len, d_model) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


TRAIN_MINUTES = 5
BATCH_SIZE = 1
LR = 1e-3
N_BACK_TRAIN = 8

EXPERIMENT_LABEL = "repro-exp11-xsprev-addr-lr1e-2-bs1"


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prepare_data()
    amp_dtype = configure_torch(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and amp_dtype == torch.float16))

    config = BDHConfig()
    model = BDH(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    loader = get_dataloader(N_BACK_TRAIN, "train", BATCH_SIZE, shuffle=True)

    start_time = time.time()
    deadline = start_time + TRAIN_MINUTES * 60
    step = 0
    it = iter(loader)

    if device == "cuda":
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"CUDA device: {torch.cuda.get_device_name(0)}  total_memory={total_mem_gb:.2f} GB  amp={amp_dtype}")

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

        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"step {step}  loss {loss.item():.4f}  elapsed {elapsed:.0f}s")
            write_status(EXPERIMENT_LABEL, step, loss.item(), elapsed)

    print(f"\nTraining complete ({step} steps). Evaluating...")
    if device == "cuda":
        torch.cuda.empty_cache()
    scores = evaluate_all(model, device)
    log_result(EXPERIMENT_LABEL, scores)

    from prepare import RESULTS_DIR
    ckpt_dir = RESULTS_DIR / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt = {
        "label":  EXPERIMENT_LABEL,
        "config": dataclasses.asdict(config),
        "steps":  step,
        "scores": scores,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()
                        if not k.startswith("decoder_fast")},
    }
    ckpt_path = ckpt_dir / f"{EXPERIMENT_LABEL}.pt"
    torch.save(ckpt, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    train()
