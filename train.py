"""
train.py - raw-byte FineWeb-Edu trainer for experiments/v3.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .address_map import ensure_address_map, load_address_map_tensor
    from .hebbian_backend import make_backend
    from .prepare import DEFAULT_ADDRESS_MAP_LABEL, DEFAULT_DATASET_NAME, RESULTS_DIR, dataset_dir, log_result, prepare_data
    from .status_writer import append_consolidation_history, append_eval_history, append_perf_history, append_train_history, write_status
except ImportError:
    from address_map import ensure_address_map, load_address_map_tensor
    from hebbian_backend import make_backend
    from prepare import DEFAULT_ADDRESS_MAP_LABEL, DEFAULT_DATASET_NAME, RESULTS_DIR, dataset_dir, log_result, prepare_data
    from status_writer import append_consolidation_history, append_eval_history, append_perf_history, append_train_history, write_status


@dataclass
class RunConfig:
    label: str = "v3-baseline"
    dataset_name: str = DEFAULT_DATASET_NAME
    address_map_label: str = DEFAULT_ADDRESS_MAP_LABEL
    d_model: int = 256
    n_head: int = 4
    n_layer: int = 4
    seed: int = 1337
    context_len: int = 256
    ffn_mult: int = 4
    dropout: float = 0.0
    batch_size: int = 32
    lr: float = 3e-4
    warmup_steps: int = 100
    lr_decay_steps: int | None = None
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    train_minutes: float = 10.0
    light_eval_every: int = 500
    light_eval_tokens: int = 1 << 20
    full_eval_at_end: bool = True
    final_eval_tokens: int | None = None
    save_best: bool = True
    save_last: bool = True
    save_eval_checkpoints: bool = False
    prune_checkpoints_to_best: bool = False
    early_stop_patience_evals: int = 0
    early_stop_min_improve_bpb: float = 0.0
    early_stop_min_steps: int = 0
    early_stop_loss_threshold: float | None = None
    early_stop_fast_state_norm_threshold: float | None = None
    notes: str = ""
    dtype: str = "bf16"
    fast_state_mode: str = "per_example"
    hebb_backend: str = "portable"
    enable_triton: bool = False
    compile_model: bool = False
    enable_stream_overlap: bool = False
    hebb_lr: float = 1e-3
    hebb_decay: float = 0.995
    memory_size: int = 64
    sac_k: int = 16
    sac_address_space: int = 4096
    enable_consolidation: bool = False
    consolidation_mode: str = "off"
    consolidation_interval: int = 100
    consolidation_topk_frac: float = 0.1
    consolidation_lr: float = 1e-4

    @classmethod
    def from_env(cls) -> "RunConfig":
        cfg = cls()
        config_path = os.environ.get("V3_CONFIG")
        if config_path:
            payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
            for key, value in payload.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        e = os.environ
        overrides = {
            "label": ("V3_RUN_LABEL", str),
            "dataset_name": ("V3_DATASET_NAME", str),
            "address_map_label": ("V3_ADDRESS_MAP_LABEL", str),
            "d_model": ("V3_D_MODEL", int),
            "n_head": ("V3_N_HEAD", int),
            "n_layer": ("V3_N_LAYER", int),
            "seed": ("V3_SEED", int),
            "context_len": ("V3_CONTEXT_LEN", int),
            "ffn_mult": ("V3_FFN_MULT", int),
            "dropout": ("V3_DROPOUT", float),
            "batch_size": ("V3_BATCH_SIZE", int),
            "lr": ("V3_LR", float),
            "warmup_steps": ("V3_WARMUP_STEPS", int),
            "lr_decay_steps": ("V3_LR_DECAY_STEPS", int),
            "grad_clip": ("V3_GRAD_CLIP", float),
            "weight_decay": ("V3_WEIGHT_DECAY", float),
            "train_minutes": ("V3_TRAIN_MINUTES", float),
            "light_eval_every": ("V3_LIGHT_EVAL_EVERY", int),
            "light_eval_tokens": ("V3_LIGHT_EVAL_TOKENS", int),
            "final_eval_tokens": ("V3_FINAL_EVAL_TOKENS", int),
            "early_stop_patience_evals": ("V3_EARLY_STOP_PATIENCE_EVALS", int),
            "early_stop_min_improve_bpb": ("V3_EARLY_STOP_MIN_IMPROVE_BPB", float),
            "early_stop_min_steps": ("V3_EARLY_STOP_MIN_STEPS", int),
            "early_stop_loss_threshold": ("V3_EARLY_STOP_LOSS_THRESHOLD", float),
            "early_stop_fast_state_norm_threshold": ("V3_EARLY_STOP_FAST_STATE_NORM_THRESHOLD", float),
            "notes": ("V3_NOTES", str),
            "dtype": ("V3_DTYPE", str),
            "fast_state_mode": ("V3_FAST_STATE_MODE", str),
            "hebb_backend": ("V3_HEBB_BACKEND", str),
            "hebb_lr": ("V3_HEBB_LR", float),
            "hebb_decay": ("V3_HEBB_DECAY", float),
            "memory_size": ("V3_MEMORY_SIZE", int),
            "sac_k": ("V3_SAC_K", int),
            "sac_address_space": ("V3_SAC_ADDRESS_SPACE", int),
            "consolidation_mode": ("V3_CONSOLIDATION_MODE", str),
            "consolidation_interval": ("V3_CONSOLIDATION_INTERVAL", int),
            "consolidation_topk_frac": ("V3_CONSOLIDATION_TOPK_FRAC", float),
            "consolidation_lr": ("V3_CONSOLIDATION_LR", float),
        }
        for field, (env_key, caster) in overrides.items():
            if env_key in e:
                setattr(cfg, field, caster(e[env_key]))
        cfg.full_eval_at_end = e.get("V3_FULL_EVAL_AT_END", "1") == "1"
        cfg.save_eval_checkpoints = e.get("V3_SAVE_EVAL_CHECKPOINTS", "1" if cfg.save_eval_checkpoints else "0") == "1"
        cfg.prune_checkpoints_to_best = e.get("V3_PRUNE_CHECKPOINTS_TO_BEST", "1" if cfg.prune_checkpoints_to_best else "0") == "1"
        cfg.enable_triton = e.get("V3_ENABLE_TRITON", "1" if cfg.enable_triton else "0") == "1"
        cfg.compile_model = e.get("V3_COMPILE_MODEL", "1" if cfg.compile_model else "0") == "1"
        cfg.enable_stream_overlap = e.get("V3_ENABLE_STREAM_OVERLAP", "1" if cfg.enable_stream_overlap else "0") == "1"
        cfg.enable_consolidation = e.get("V3_ENABLE_CONSOLIDATION", "1" if cfg.enable_consolidation else "0") == "1"
        return cfg


@dataclass(frozen=True)
class EvalControls:
    disable_fast_read: bool = False
    disable_fast_write: bool = False
    disable_slow_read: bool = False
    shuffle_fast_rows: bool = False
    shuffle_seed: int = 0
    carry_fast_state: bool = False
    context_len_override: int | None = None

    def effective_context_len(self, cfg: RunConfig) -> int:
        if self.context_len_override is None:
            return cfg.context_len
        return max(1, min(int(self.context_len_override), cfg.context_len))

    def to_dict(self) -> dict[str, int | bool | None]:
        return {
            "disable_fast_read": self.disable_fast_read,
            "disable_fast_write": self.disable_fast_write,
            "disable_slow_read": self.disable_slow_read,
            "shuffle_fast_rows": self.shuffle_fast_rows,
            "shuffle_seed": self.shuffle_seed,
            "carry_fast_state": self.carry_fast_state,
            "context_len_override": self.context_len_override,
        }


def _purge_checkpoints(ckpt_dir: Path, label: str, keep_names: set[str]) -> None:
    for path in ckpt_dir.glob(f"{label}_*.pt"):
        if path.name not in keep_names:
            path.unlink(missing_ok=True)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def _config_signature(cfg: RunConfig) -> str:
    return hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _shard_paths(dataset_name: str, split: str) -> list[Path]:
    manifest = json.loads((dataset_dir(dataset_name) / "export_manifest.json").read_text(encoding="utf-8"))
    return [dataset_dir(dataset_name) / fn for fn in manifest[f"{split}_files"]]


class ByteShardLoader:
    def __init__(self, paths: list[Path], context_len: int, batch_size: int, seed: int = 42):
        self.paths = paths
        self.context_len = context_len
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)
        self._order: list[int] = []
        self._tokens: np.ndarray | None = None
        self._pos = 0
        self._advance()

    def _advance(self) -> None:
        if not self._order:
            self._order = self._rng.permutation(len(self.paths)).tolist()
        self._tokens = np.memmap(self.paths[self._order.pop(0)], dtype=np.uint8, mode="r")
        self._pos = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        need = self.batch_size * (self.context_len + 1)
        assert self._tokens is not None
        if self._pos + need > len(self._tokens):
            self._advance()
        chunk = np.array(self._tokens[self._pos : self._pos + need], dtype=np.int64)
        self._pos += need
        chunk2d = chunk.reshape(self.batch_size, self.context_len + 1)
        x = torch.from_numpy(np.ascontiguousarray(chunk2d[:, :-1]))
        y = torch.from_numpy(np.ascontiguousarray(chunk2d[:, 1:]))
        return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time_steps, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=2)
        d_head = channels // self.n_head

        def reshape(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(bsz, time_steps, self.n_head, d_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            reshape(q),
            reshape(k),
            reshape(v),
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(y.transpose(1, 2).contiguous().view(bsz, time_steps, channels))


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * ffn_mult, bias=False)
        self.fc2 = nn.Linear(d_model * ffn_mult, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))


class SACEmbedding(nn.Module):
    def __init__(self, address_map: torch.Tensor, address_space: int, d_model: int, context_len: int, dropout: float):
        super().__init__()
        self.register_buffer("address_map", address_map.long(), persistent=False)
        self.synapse_emb = nn.Embedding(address_space, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.synapse_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        syn = self.address_map[idx.long()]
        emb = self.synapse_emb(syn).mean(dim=2)
        pos = torch.arange(idx.size(1), device=idx.device)
        return self.drop(emb + self.pos_emb(pos)), syn


class V3Model(nn.Module):
    def __init__(self, cfg: RunConfig, address_map: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.embedding = SACEmbedding(address_map, cfg.sac_address_space, cfg.d_model, cfg.context_len, cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_head, cfg.ffn_mult, cfg.dropout) for _ in range(cfg.n_layer)])
        self.query_projs = nn.ModuleList([nn.Linear(cfg.d_model, cfg.memory_size, bias=False) for _ in range(cfg.n_layer)])
        self.memory_norms = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layer)])
        self.memory_gates = nn.Parameter(torch.zeros(cfg.n_layer))
        self.slow_memory = nn.Parameter(torch.zeros(cfg.n_layer, cfg.memory_size, cfg.d_model).normal_(std=0.02))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, 256, bias=False)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_fast_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.cfg.n_layer, batch_size, self.cfg.memory_size, self.cfg.d_model, device=device, dtype=torch.float32)

    def _slow_only_layer(self, layer_idx: int, x: torch.Tensor, controls: EvalControls | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        controls = controls or EvalControls()
        queries = F.relu(self.query_projs[layer_idx](x)).float()
        queries = queries / queries.sum(dim=-1, keepdim=True).clamp(min=1.0)
        read_slow = torch.einsum("btm,md->btd", queries, self.slow_memory[layer_idx].float())
        if controls.disable_slow_read:
            read_slow = torch.zeros_like(read_slow)
        gated = torch.tanh(self.memory_gates[layer_idx]).to(x.dtype) * read_slow.to(x.dtype)
        x = self.memory_norms[layer_idx](x + gated)
        stats = {
            "update_norm": 0.0,
            "read_norm": float(read_slow.norm().item()),
            "fast_state_norm": 0.0,
            "fast_state_max_abs": 0.0,
            "slow_contrib_norm": float(read_slow.norm().item()),
            "fast_contrib_norm": 0.0,
        }
        return x, stats

    def _hebbian_sequence_layer(
        self,
        layer_idx: int,
        x: torch.Tensor,
        backend,
        layer_state: torch.Tensor,
        controls: EvalControls | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        controls = controls or EvalControls()
        queries = F.relu(self.query_projs[layer_idx](x))
        read_total, next_state, stats = backend.sequence(
            queries,
            x,
            self.slow_memory[layer_idx],
            layer_state,
            self.cfg.hebb_lr,
            self.cfg.hebb_decay,
            disable_fast_read=controls.disable_fast_read,
            disable_fast_write=controls.disable_fast_write,
            disable_slow_read=controls.disable_slow_read,
            shuffle_fast_rows=controls.shuffle_fast_rows,
            shuffle_seed=controls.shuffle_seed + layer_idx,
        )
        gated = torch.tanh(self.memory_gates[layer_idx]).to(x.dtype) * read_total.to(x.dtype)
        x = self.memory_norms[layer_idx](x + gated)
        return x, next_state.to(layer_state.dtype), stats

    def forward(
        self,
        idx: torch.Tensor,
        backend,
        fast_state: torch.Tensor | None = None,
        controls: EvalControls | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        controls = controls or EvalControls()
        batch_size, time_steps = idx.shape
        x, syn = self.embedding(idx)
        if fast_state is None:
            fast_state = self.init_fast_state(batch_size, idx.device)
        stats = {"update_norm": 0.0, "read_norm": 0.0, "fast_state_norm": 0.0, "fast_state_max_abs": 0.0, "slow_contrib_norm": 0.0, "fast_contrib_norm": 0.0}
        for layer_idx, block in enumerate(self.blocks):
            x = block(x)
            if self.cfg.hebb_lr <= 0.0:
                x, layer_stats = self._slow_only_layer(layer_idx, x, controls=controls)
                for key in stats:
                    if key == "fast_state_max_abs":
                        stats[key] = max(stats[key], layer_stats[key])
                    else:
                        stats[key] += layer_stats[key]
                continue
            layer_state = fast_state[layer_idx]
            x, layer_state, layer_stats = self._hebbian_sequence_layer(layer_idx, x, backend, layer_state, controls=controls)
            fast_state[layer_idx] = layer_state
            for key in stats:
                if key == "fast_state_max_abs":
                    stats[key] = max(stats[key], layer_stats[key])
                else:
                    stats[key] += layer_stats[key]
        denom = max(self.cfg.n_layer * time_steps, 1)
        for key in stats:
            if key != "fast_state_max_abs":
                stats[key] /= denom
        syn_flat = syn.reshape(-1, syn.size(-1))
        unique_synapses = int(torch.unique(syn_flat).numel())
        touched_rows = int(torch.unique(idx).numel())
        total_slots = max(int(syn_flat.numel()), 1)
        stats["unique_synapses"] = unique_synapses
        stats["address_rows_touched"] = touched_rows
        stats["address_collision_rate"] = 1.0 - min(unique_synapses / total_slots, 1.0)
        return self.lm_head(self.ln_f(x)), fast_state, stats


def consolidate(model: V3Model, fast_state: torch.Tensor, cfg: RunConfig) -> dict:
    if not cfg.enable_consolidation or cfg.consolidation_mode == "off":
        return {"rows_selected": 0, "write_norm": 0.0, "clip_fraction": 0.0, "write_applied": False}
    mean_state = fast_state.mean(dim=1)
    write_norm = 0.0
    rows_selected = 0
    with torch.no_grad():
        if cfg.consolidation_mode == "dense":
            model.slow_memory.add_(mean_state.to(model.slow_memory.dtype), alpha=cfg.consolidation_lr)
            rows_selected = mean_state.size(0) * mean_state.size(1)
            write_norm = float(mean_state.norm().item())
        else:
            per_layer_rows = max(1, int(cfg.consolidation_topk_frac * mean_state.size(1)))
            for layer_idx in range(mean_state.size(0)):
                top_rows = mean_state[layer_idx].norm(dim=1).topk(per_layer_rows).indices
                model.slow_memory[layer_idx, top_rows].add_(mean_state[layer_idx, top_rows].to(model.slow_memory.dtype), alpha=cfg.consolidation_lr)
                rows_selected += int(top_rows.numel())
                write_norm += float(mean_state[layer_idx, top_rows].norm().item())
    return {"rows_selected": rows_selected, "write_norm": write_norm, "clip_fraction": 0.0, "write_applied": rows_selected > 0}


@torch.no_grad()
def evaluate_model(
    model: V3Model,
    cfg: RunConfig,
    backend,
    val_paths: list[Path],
    n_tokens: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    controls: EvalControls | None = None,
) -> dict:
    controls = controls or EvalControls()
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    effective_context_len = controls.effective_context_len(cfg)
    need = cfg.batch_size * (effective_context_len + 1)
    stat_keys = ("update_norm", "read_norm", "fast_state_norm", "fast_state_max_abs", "slow_contrib_norm", "fast_contrib_norm", "unique_synapses", "address_rows_touched", "address_collision_rate")
    stat_sums = {key: 0.0 for key in stat_keys}
    fast_state = None
    for path in val_paths:
        if total_tokens >= n_tokens:
            break
        if not controls.carry_fast_state:
            fast_state = None
        tokens = np.memmap(path, dtype=np.uint8, mode="r")
        pos = 0
        while pos + need <= len(tokens) and total_tokens < n_tokens:
            chunk = np.array(tokens[pos : pos + need], dtype=np.int64)
            pos += need
            chunk2d = chunk.reshape(cfg.batch_size, effective_context_len + 1)
            x = torch.from_numpy(np.ascontiguousarray(chunk2d[:, :-1])).to(device)
            y = torch.from_numpy(np.ascontiguousarray(chunk2d[:, 1:])).to(device)
            with _autocast_context(device, amp_dtype):
                logits, fast_state, stats = model(x, backend, fast_state=fast_state, controls=controls)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            if not controls.carry_fast_state:
                fast_state = None
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
            for key in stat_keys:
                stat_sums[key] += float(stats.get(key, 0.0)) * x.numel()
    model.train()
    if total_tokens == 0:
        return {
            "val_loss": float("inf"),
            "val_bpt": float("inf"),
            "val_bpb": float("inf"),
            "val_tokens": 0,
            "effective_context_len": effective_context_len,
            **controls.to_dict(),
        }
    val_loss = total_loss / total_tokens
    val_bpt = val_loss / math.log(2)
    out = {"val_loss": val_loss, "val_bpt": val_bpt, "val_bpb": val_bpt, "val_tokens": total_tokens, "effective_context_len": effective_context_len, **controls.to_dict()}
    for key, value in stat_sums.items():
        out[f"avg_{key}"] = value / total_tokens
    return out


def cosine_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * (step + 1) / max(warmup, 1)
    if step >= total:
        return lr_min
    t = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t))


def _byte_histogram(x: torch.Tensor) -> tuple[dict[str, int], list[list[int]]]:
    hist = torch.bincount(x.reshape(-1), minlength=256)
    top = torch.topk(hist, k=8)
    return ({str(i): int(hist[i].item()) for i in top.indices.tolist() if hist[i].item() > 0}, [[int(i), int(v)] for i, v in zip(top.indices.tolist(), top.values.tolist(), strict=True) if v > 0])


def _gpu_mem_stats(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"gpu_mem_allocated_gb": None, "gpu_mem_reserved_gb": None, "gpu_mem_peak_gb": None, "gpu_mem_total_gb": None}
    props = torch.cuda.get_device_properties(device)
    return {
        "gpu_mem_allocated_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
        "gpu_mem_reserved_gb": torch.cuda.memory_reserved(device) / (1024 ** 3),
        "gpu_mem_peak_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "gpu_mem_total_gb": props.total_memory / (1024 ** 3),
    }


def _autocast_context(device: torch.device, amp_dtype: torch.dtype):
    if amp_dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _torch_compile_support(device: torch.device) -> tuple[bool, str | None]:
    if device.type != "cuda":
        return False, "torch.compile disabled: CUDA device required for the current v3 fast path"
    try:
        import triton  # noqa: F401

        return True, None
    except Exception as exc:
        return False, f"torch.compile disabled: Triton/Inductor unavailable on this machine ({exc})"


def train() -> None:
    prepare_data()
    cfg = RunConfig.from_env()
    ensure_address_map(cfg.address_map_label, k=cfg.sac_k, address_space=cfg.sac_address_space)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(os.environ.get("V3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = torch.bfloat16 if cfg.dtype == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if cfg.compile_model or cfg.hebb_backend == "compile":
        triton_cache_dir = RESULTS_DIR / ".triton-cache"
        triton_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache_dir))
    compile_enabled = False
    compile_fallback_reason = None
    if cfg.compile_model or cfg.hebb_backend == "compile":
        compile_enabled, compile_fallback_reason = _torch_compile_support(device)
    effective_hebb_backend = cfg.hebb_backend if (cfg.hebb_backend != "compile" or compile_enabled) else "portable"
    train_paths, val_paths = _shard_paths(cfg.dataset_name, "train"), _shard_paths(cfg.dataset_name, "val")
    loader = ByteShardLoader(train_paths, cfg.context_len, cfg.batch_size, seed=cfg.seed)
    address_map = load_address_map_tensor(cfg.address_map_label, device=device)
    backend = make_backend(
        effective_hebb_backend,
        enable_compile=compile_enabled and (cfg.compile_model or effective_hebb_backend == "compile"),
        enable_triton=cfg.enable_triton,
    )
    if compile_fallback_reason:
        backend.state.fallback_reason = (
            f"{backend.state.fallback_reason}; {compile_fallback_reason}"
            if backend.state.fallback_reason
            else compile_fallback_reason
        )
    model = V3Model(cfg, address_map).to(device)
    if compile_enabled and cfg.compile_model:
        try:
            model = torch.compile(model, dynamic=False, mode="reduce-overhead")
        except Exception as exc:
            compile_enabled = False
            backend.state.fallback_reason = (
                f"{backend.state.fallback_reason}; model compile fallback: {exc}"
                if backend.state.fallback_reason
                else f"model compile fallback: {exc}"
            )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    ckpt_dir = RESULTS_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    total_steps_est = cfg.lr_decay_steps or max(1000, int(cfg.train_minutes * 60 * 8))
    deadline = time.time() + cfg.train_minutes * 60.0
    best_bpb, best_step, step = float("inf"), 0, 0
    best_ckpt_path = ckpt_dir / f"{cfg.label}_best.pt"
    no_improve_evals = 0
    tokens_seen = bytes_seen = 0
    t0, last_status_t = time.time(), time.time()
    stop_reason: str | None = None
    common = {
        "dataset_name": cfg.dataset_name,
        "tokenizer_label": "raw_byte_256",
        "address_map_hash": hashlib.sha256(address_map.cpu().numpy().tobytes()).hexdigest()[:16],
        "address_map_loaded": True,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "backend": backend.state.name,
        "dtype": cfg.dtype,
        "compile_enabled": compile_enabled,
        "triton_enabled": backend.state.triton_enabled,
        "stream_overlap_enabled": cfg.enable_stream_overlap,
        "bf16_enabled": amp_dtype == torch.bfloat16,
        "tf32_enabled": device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32,
        "sac_k": cfg.sac_k,
        "hebb_lr": cfg.hebb_lr,
        "hebb_decay": cfg.hebb_decay,
        "update_cadence_tokens": 1,
        "consolidation_enabled": cfg.enable_consolidation,
        "consolidation_mode": cfg.consolidation_mode,
        "consolidation_interval": cfg.consolidation_interval,
        "consolidation_topk_frac": cfg.consolidation_topk_frac,
        "git_commit": _git_commit(),
        "config_signature": _config_signature(cfg),
        "seed": cfg.seed,
        "backend_fallback_reason": backend.state.fallback_reason,
        "vocab_size": 256,
        "train_minutes": cfg.train_minutes,
        "lr_decay_steps": cfg.lr_decay_steps,
        "final_eval_tokens": cfg.final_eval_tokens,
        "early_stop_patience_evals": cfg.early_stop_patience_evals,
        "early_stop_min_improve_bpb": cfg.early_stop_min_improve_bpb,
    }
    last_loss = 0.0
    while time.time() < deadline:
        step_start = time.time()
        lr = cosine_lr(step, cfg.warmup_steps, total_steps_est, cfg.lr, cfg.lr * 0.1)
        for group in optimizer.param_groups:
            group["lr"] = lr
        data_t0 = time.time()
        x_cpu, y_cpu = loader.next_batch()
        data_wait_ms = (time.time() - data_t0) * 1000.0
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        byte_histogram, byte_topk = _byte_histogram(x_cpu)
        optimizer.zero_grad(set_to_none=True)
        fwd_t0 = time.time()
        with _autocast_context(device, amp_dtype):
            logits, fast_state, hebb_stats = model(x, backend, fast_state=None)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        forward_ms = (time.time() - fwd_t0) * 1000.0
        bwd_t0 = time.time()
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item())
        backward_ms = (time.time() - bwd_t0) * 1000.0
        opt_t0 = time.time()
        optimizer.step()
        optimizer_ms = (time.time() - opt_t0) * 1000.0
        cons = {"rows_selected": 0, "write_norm": 0.0, "clip_fraction": 0.0, "write_applied": False}
        if cfg.enable_consolidation and step % max(cfg.consolidation_interval, 1) == 0:
            cons = consolidate(model, fast_state.detach(), cfg)
            append_consolidation_history(cfg.label, step, time.time() - t0, {"consolidation_mode": cfg.consolidation_mode, "consolidation_topk_frac": cfg.consolidation_topk_frac, **cons})
        step_ms = (time.time() - step_start) * 1000.0
        batch_tokens = int(x.numel())
        tokens_seen += batch_tokens
        bytes_seen += batch_tokens
        elapsed = time.time() - t0
        tokens_per_s = tokens_seen / max(elapsed, 1e-6)
        bytes_per_s = bytes_seen / max(elapsed, 1e-6)
        step += 1
        last_loss = float(loss.item())
        mem = _gpu_mem_stats(device)
        payload = {
            **common, **mem, **hebb_stats, **cons,
            "label": cfg.label, "step": step, "elapsed_s": elapsed, "phase": "train", "phase_detail": f"lr={lr:.2e}",
            "train_loss": last_loss, "loss": last_loss, "tokens_seen": tokens_seen, "bytes_seen": bytes_seen,
            "tokens_per_s": tokens_per_s, "bytes_per_s": bytes_per_s, "step_ms": step_ms, "forward_ms": forward_ms,
            "backward_ms": backward_ms, "optimizer_ms": optimizer_ms, "data_wait_ms": data_wait_ms, "grad_norm": grad_norm,
            "byte_histogram": byte_histogram, "byte_topk": byte_topk, "nan_detected": bool(torch.isnan(loss).item()), "inf_detected": bool(torch.isinf(loss).item()),
        }
        if time.time() - last_status_t >= 2.0 or step <= 5:
            write_status(**payload)
            append_train_history(cfg.label, step, last_loss, elapsed, lr=lr, phase="train", tokens_seen=tokens_seen, bytes_seen=bytes_seen, tokens_per_s=tokens_per_s, bytes_per_s=bytes_per_s, step_ms=step_ms, forward_ms=forward_ms, backward_ms=backward_ms, optimizer_ms=optimizer_ms, data_wait_ms=data_wait_ms, grad_norm=grad_norm, backend=backend.state.name, fast_state_norm=hebb_stats["fast_state_norm"], update_norm=hebb_stats["update_norm"], read_norm=hebb_stats["read_norm"], gpu_mem_allocated_gb=mem["gpu_mem_allocated_gb"])
            append_perf_history(cfg.label, step, elapsed, {"tokens_per_s": tokens_per_s, "bytes_per_s": bytes_per_s, "step_ms": step_ms, "forward_ms": forward_ms, "backward_ms": backward_ms, "optimizer_ms": optimizer_ms, "data_wait_ms": data_wait_ms, "gpu_mem_allocated_gb": mem["gpu_mem_allocated_gb"], "fast_state_norm": hebb_stats["fast_state_norm"], "write_norm": cons["write_norm"]})
            last_status_t = time.time()
        if step % cfg.light_eval_every == 0:
            eval_t0 = time.time()
            eval_stats = evaluate_model(model, cfg, backend, val_paths, cfg.light_eval_tokens, device, amp_dtype)
            eval_ms = (time.time() - eval_t0) * 1000.0
            improved_for_patience = (
                best_bpb == float("inf")
                or eval_stats["val_bpb"] <= best_bpb - cfg.early_stop_min_improve_bpb
            )
            if eval_stats["val_bpb"] < best_bpb:
                best_bpb, best_step = float(eval_stats["val_bpb"]), step
                if cfg.save_best:
                    torch.save({"step": step, "config": asdict(cfg), "model_state": model.state_dict(), "metrics": eval_stats}, best_ckpt_path)
            if cfg.save_eval_checkpoints:
                torch.save(
                    {"step": step, "config": asdict(cfg), "model_state": model.state_dict(), "metrics": eval_stats},
                    ckpt_dir / f"{cfg.label}_step{step:06d}.pt",
                )
            if improved_for_patience:
                no_improve_evals = 0
            else:
                no_improve_evals += 1
            eval_payload = dict(payload)
            eval_payload.update(
                phase="eval",
                phase_detail="light eval",
                eval_kind="light",
                eval_tokens=cfg.light_eval_tokens,
                eval_ms=eval_ms,
                val_loss=eval_stats["val_loss"],
                val_bpb=eval_stats["val_bpb"],
                best_val_bpb=best_bpb,
                best_step=best_step,
                light_eval={**eval_stats, "best_val_bpb": best_bpb},
                checkpoint_source="best" if best_step == step else "last",
                no_improve_evals=no_improve_evals,
            )
            write_status(**eval_payload)
            append_eval_history(cfg.label, step, last_loss, elapsed, {**eval_stats, "best_val_bpb": best_bpb}, eval_kind="light", eval_ms=eval_ms, tokens_seen=tokens_seen, bytes_seen=bytes_seen, backend=backend.state.name, phase="eval")
            eligible_for_early_stop = (
                cfg.early_stop_patience_evals > 0
                and step >= cfg.early_stop_min_steps
                and no_improve_evals >= cfg.early_stop_patience_evals
            )
            if eligible_for_early_stop and cfg.early_stop_loss_threshold is not None:
                eligible_for_early_stop = last_loss <= cfg.early_stop_loss_threshold
            if eligible_for_early_stop and cfg.hebb_lr > 0.0 and cfg.early_stop_fast_state_norm_threshold is not None:
                eligible_for_early_stop = hebb_stats["fast_state_norm"] >= cfg.early_stop_fast_state_norm_threshold
            if eligible_for_early_stop:
                stop_reason = (
                    f"early_stop: no val_bpb improvement for {no_improve_evals} evals"
                    f" (min_delta={cfg.early_stop_min_improve_bpb:.4f})"
                )
                done_payload = dict(eval_payload)
                done_payload.update(phase="done", phase_detail=stop_reason, stop_reason=stop_reason)
                write_status(**done_payload)
                break
    final_eval = {}
    final_checkpoint_source = "best" if best_ckpt_path.exists() else "current"
    if cfg.full_eval_at_end:
        total_val_tokens = sum(len(np.memmap(path, dtype=np.uint8, mode="r")) for path in val_paths)
        if cfg.final_eval_tokens is not None:
            total_val_tokens = min(total_val_tokens, cfg.final_eval_tokens)
        if best_ckpt_path.exists():
            final_eval = evaluate_checkpoint(best_ckpt_path, total_val_tokens, device)
            final_checkpoint_source = "best"
        else:
            final_eval = evaluate_model(model, cfg, backend, val_paths, total_val_tokens, device, amp_dtype)
    if cfg.save_last:
        torch.save({"step": step, "config": asdict(cfg), "model_state": model.state_dict(), "metrics": final_eval}, ckpt_dir / f"{cfg.label}_last.pt")
    if cfg.prune_checkpoints_to_best:
        keep_names = {best_ckpt_path.name} if best_ckpt_path.exists() else set()
        _purge_checkpoints(ckpt_dir, cfg.label, keep_names)
    result = log_result(cfg.label, {"step": step, "tokens_seen": tokens_seen, "bytes_seen": bytes_seen, "loss": last_loss, "val_loss": final_eval.get("val_loss"), "val_bpb": final_eval.get("val_bpb", best_bpb), "best_val_bpb": best_bpb, "best_step": best_step, "backend": backend.state.name, "dtype": cfg.dtype, "tokenizer_label": "raw_byte_256", "dataset_name": cfg.dataset_name, "address_map_hash": common["address_map_hash"], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, notes=cfg.notes)
    done_detail = stop_reason or f"best={best_bpb:.4f}"
    write_status(**common, **_gpu_mem_stats(device), label=cfg.label, step=step, elapsed_s=time.time() - t0, phase="done", phase_detail=done_detail, train_loss=last_loss, loss=last_loss, tokens_seen=tokens_seen, bytes_seen=bytes_seen, best_val_bpb=best_bpb, best_step=best_step, final_eval=final_eval or None, checkpoint_source=final_checkpoint_source, stop_reason=stop_reason, no_improve_evals=no_improve_evals)
    print(f"Logged: {result}")


@torch.no_grad()
def evaluate_checkpoint(
    ckpt_path: Path,
    eval_tokens: int | None,
    device: torch.device,
    controls: EvalControls | None = None,
) -> dict:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = RunConfig(**payload["config"])
    address_map = load_address_map_tensor(cfg.address_map_label, device=device)
    backend = make_backend(cfg.hebb_backend, enable_compile=False, enable_triton=cfg.enable_triton)
    model = V3Model(cfg, address_map).to(device)
    model.load_state_dict(payload["model_state"])
    val_paths = _shard_paths(cfg.dataset_name, "val")
    amp_dtype = torch.bfloat16 if cfg.dtype == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    if eval_tokens is None:
        eval_tokens = sum(len(np.memmap(path, dtype=np.uint8, mode="r")) for path in val_paths)
    return evaluate_model(model, cfg, backend, val_paths, eval_tokens, device, amp_dtype, controls=controls)


if __name__ == "__main__":
    try:
        train()
    except Exception:
        results_dir = Path(os.environ.get("V3_RESULTS_DIR", str(RESULTS_DIR)))
        results_dir.mkdir(parents=True, exist_ok=True)
        label = os.environ.get("V3_RUN_LABEL", "unknown")
        fatal_path = results_dir / f"{label}_fatal_error.txt"
        fatal_path.write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc(file=sys.stderr)
        raise
