"""
Hebbian runtime backends for v3.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def portable_hebbian_step_tensors(
    query: torch.Tensor,
    value: torch.Tensor,
    slow_memory: torch.Tensor,
    fast_state: torch.Tensor,
    hebb_lr: float,
    hebb_decay: float,
    disable_fast_read: bool = False,
    disable_fast_write: bool = False,
    disable_slow_read: bool = False,
    shuffle_fast_rows: bool = False,
    shuffle_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = query.float()
    q = q / q.sum(dim=-1, keepdim=True).clamp(min=1.0)
    q_fast_read = q
    if shuffle_fast_rows:
        generator = torch.Generator(device=q.device)
        generator.manual_seed(int(shuffle_seed))
        perm = torch.randperm(q.size(-1), generator=generator, device=q.device)
        q_fast_read = q.index_select(-1, perm)
    read_slow = torch.zeros(value.shape, device=value.device, dtype=torch.float32) if disable_slow_read else q @ slow_memory.float()
    read_fast = torch.zeros_like(read_slow) if disable_fast_read else torch.einsum("bm,bmd->bd", q_fast_read, fast_state.float())
    update = torch.einsum("bm,bd->bmd", q, value.float())
    next_state = fast_state.float() * hebb_decay if disable_fast_write else fast_state.float() * hebb_decay + update * hebb_lr
    return (
        read_slow + read_fast,
        next_state,
        update.norm(),
        (read_slow + read_fast).norm(),
        next_state.norm(),
        next_state.abs().max(),
        read_slow.norm(),
        read_fast.norm(),
    )


def portable_hebbian_sequence_tensors(
    query: torch.Tensor,
    value: torch.Tensor,
    slow_memory: torch.Tensor,
    fast_state: torch.Tensor,
    hebb_lr: float,
    hebb_decay: float,
    disable_fast_read: bool = False,
    disable_fast_write: bool = False,
    disable_slow_read: bool = False,
    shuffle_fast_rows: bool = False,
    shuffle_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = query.float()
    q = q / q.sum(dim=-1, keepdim=True).clamp(min=1.0)
    q_fast_read = q
    if shuffle_fast_rows:
        generator = torch.Generator(device=q.device)
        generator.manual_seed(int(shuffle_seed))
        perm = torch.randperm(q.size(-1), generator=generator, device=q.device)
        q_fast_read = q.index_select(-1, perm)
    value_f = value.float()
    slow_f = slow_memory.float()
    state0 = fast_state.float()
    _, time_steps, _ = q.shape

    read_slow = torch.zeros_like(value_f) if disable_slow_read else torch.einsum("btm,md->btd", q, slow_f)
    decay_powers = torch.pow(
        torch.full((time_steps,), float(hebb_decay), device=q.device, dtype=torch.float32),
        torch.arange(time_steps, device=q.device, dtype=torch.float32),
    )
    if disable_fast_read:
        read_fast_init = torch.zeros_like(value_f)
    else:
        read_fast_init = torch.einsum("btm,bmd->btd", q_fast_read, state0) * decay_powers.view(1, time_steps, 1)

    time_idx = torch.arange(time_steps, device=q.device)
    deltas = (time_idx.view(-1, 1) - time_idx.view(1, -1) - 1).clamp(min=0).to(torch.float32)
    causal_mask = torch.tril(torch.ones(time_steps, time_steps, device=q.device, dtype=torch.float32), diagonal=-1)
    decay_kernel = torch.pow(
        torch.full((time_steps, time_steps), float(hebb_decay), device=q.device, dtype=torch.float32),
        deltas,
    ) * causal_mask
    pairwise = torch.einsum("btm,bsm->bts", q_fast_read, q)
    if disable_fast_read or disable_fast_write:
        read_fast_hist = torch.zeros_like(value_f)
    else:
        read_fast_hist = hebb_lr * torch.einsum("bts,bsd->btd", pairwise * decay_kernel.unsqueeze(0), value_f)
    read_fast = read_fast_init + read_fast_hist

    reverse_decay = torch.flip(decay_powers, dims=(0,))
    if disable_fast_write:
        next_state = state0 * decay_powers[-1] * hebb_decay
    else:
        next_state = state0 * decay_powers[-1]
        next_state = next_state * hebb_decay + hebb_lr * torch.einsum("btm,btd,t->bmd", q, value_f, reverse_decay)

    update_norm = torch.sqrt((q.square().sum(dim=-1) * value_f.square().sum(dim=-1)).sum(dim=0)).sum()
    read_norm = torch.linalg.vector_norm(read_slow + read_fast, dim=(0, 2)).sum()
    slow_norm = torch.linalg.vector_norm(read_slow, dim=(0, 2)).sum()
    fast_norm = torch.linalg.vector_norm(read_fast, dim=(0, 2)).sum()
    return (
        read_slow + read_fast,
        next_state,
        update_norm,
        read_norm,
        next_state.norm() * time_steps,
        next_state.abs().max(),
        slow_norm,
        fast_norm,
    )


@dataclass
class BackendState:
    name: str
    compile_enabled: bool
    triton_enabled: bool
    fallback_reason: str | None


class HebbianBackend:
    def __init__(self, *, name: str, compile_enabled: bool, triton_enabled: bool, fallback_reason: str | None = None):
        self.state = BackendState(
            name=name,
            compile_enabled=compile_enabled,
            triton_enabled=triton_enabled,
            fallback_reason=fallback_reason,
        )
        self._step = portable_hebbian_step_tensors
        self._sequence = portable_hebbian_sequence_tensors

    @staticmethod
    def _stats(
        update_norm: torch.Tensor,
        read_norm: torch.Tensor,
        fast_state_norm: torch.Tensor,
        fast_state_max_abs: torch.Tensor,
        slow_contrib_norm: torch.Tensor,
        fast_contrib_norm: torch.Tensor,
    ) -> dict[str, float]:
        return {
            "update_norm": float(update_norm.item()),
            "read_norm": float(read_norm.item()),
            "fast_state_norm": float(fast_state_norm.item()),
            "fast_state_max_abs": float(fast_state_max_abs.item()),
            "slow_contrib_norm": float(slow_contrib_norm.item()),
            "fast_contrib_norm": float(fast_contrib_norm.item()),
        }

    def step(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        slow_memory: torch.Tensor,
        fast_state: torch.Tensor,
        hebb_lr: float,
        hebb_decay: float,
        *,
        disable_fast_read: bool = False,
        disable_fast_write: bool = False,
        disable_slow_read: bool = False,
        shuffle_fast_rows: bool = False,
        shuffle_seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        try:
            read, next_state, update_norm, read_norm, fast_state_norm, fast_state_max_abs, slow_contrib_norm, fast_contrib_norm = self._step(
                query,
                value,
                slow_memory,
                fast_state,
                hebb_lr,
                hebb_decay,
                disable_fast_read,
                disable_fast_write,
                disable_slow_read,
                shuffle_fast_rows,
                shuffle_seed,
            )
            return read, next_state, self._stats(
                update_norm,
                read_norm,
                fast_state_norm,
                fast_state_max_abs,
                slow_contrib_norm,
                fast_contrib_norm,
            )
        except Exception as exc:  # pragma: no cover
            self._step = portable_hebbian_step_tensors
            self._sequence = portable_hebbian_sequence_tensors
            self.state.name = "portable"
            self.state.compile_enabled = False
            self.state.triton_enabled = False
            self.state.fallback_reason = f"runtime fallback: {exc}"
            return self.step(
                query,
                value,
                slow_memory,
                fast_state,
                hebb_lr,
                hebb_decay,
                disable_fast_read=disable_fast_read,
                disable_fast_write=disable_fast_write,
                disable_slow_read=disable_slow_read,
                shuffle_fast_rows=shuffle_fast_rows,
                shuffle_seed=shuffle_seed,
            )

    def sequence(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        slow_memory: torch.Tensor,
        fast_state: torch.Tensor,
        hebb_lr: float,
        hebb_decay: float,
        *,
        disable_fast_read: bool = False,
        disable_fast_write: bool = False,
        disable_slow_read: bool = False,
        shuffle_fast_rows: bool = False,
        shuffle_seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        try:
            read, next_state, update_norm, read_norm, fast_state_norm, fast_state_max_abs, slow_contrib_norm, fast_contrib_norm = self._sequence(
                query,
                value,
                slow_memory,
                fast_state,
                hebb_lr,
                hebb_decay,
                disable_fast_read,
                disable_fast_write,
                disable_slow_read,
                shuffle_fast_rows,
                shuffle_seed,
            )
            return read, next_state, self._stats(
                update_norm,
                read_norm,
                fast_state_norm,
                fast_state_max_abs,
                slow_contrib_norm,
                fast_contrib_norm,
            )
        except Exception as exc:  # pragma: no cover
            self._step = portable_hebbian_step_tensors
            self._sequence = portable_hebbian_sequence_tensors
            self.state.name = "portable"
            self.state.compile_enabled = False
            self.state.triton_enabled = False
            self.state.fallback_reason = f"runtime fallback: {exc}"
            return self.sequence(
                query,
                value,
                slow_memory,
                fast_state,
                hebb_lr,
                hebb_decay,
                disable_fast_read=disable_fast_read,
                disable_fast_write=disable_fast_write,
                disable_slow_read=disable_slow_read,
                shuffle_fast_rows=shuffle_fast_rows,
                shuffle_seed=shuffle_seed,
            )


def make_backend(name: str, *, enable_compile: bool, enable_triton: bool) -> HebbianBackend:
    requested = (name or "portable").lower()
    backend = HebbianBackend(name="portable", compile_enabled=False, triton_enabled=False)
    if requested == "compile" or enable_compile:
        try:
            backend._step = torch.compile(portable_hebbian_step_tensors, dynamic=False)
            backend._sequence = torch.compile(portable_hebbian_sequence_tensors, dynamic=False)
            backend.state.name = "compile"
            backend.state.compile_enabled = True
            return backend
        except Exception as exc:  # pragma: no cover
            backend.state.fallback_reason = f"compile fallback: {exc}"
            return backend
    if requested == "triton" or enable_triton:
        try:
            import triton  # noqa: F401

            backend.state.fallback_reason = "triton requested but no custom kernel is registered; using portable reference"
            return backend
        except Exception as exc:  # pragma: no cover
            backend.state.fallback_reason = f"triton fallback: {exc}"
            return backend
    return backend
