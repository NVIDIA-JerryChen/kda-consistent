"""Select a fixed state contract without changing the accepted kernel program."""

from dataclasses import dataclass, field

import torch


def _validate_contract(q, state_in, state_dtype, workspace_limit_bytes):
    if state_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("state_dtype must be torch.bfloat16 or torch.float32")
    if not isinstance(q, torch.Tensor) or not isinstance(state_in, torch.Tensor):
        raise TypeError("q and state_in must be torch.Tensor")
    if state_in.dtype != state_dtype:
        raise ValueError("state_in.dtype must equal state_dtype; no implicit conversion")
    if q.device.type != "cuda":
        raise ValueError("q must be a CUDA tensor")
    if not isinstance(workspace_limit_bytes, int) or workspace_limit_bytes < 0:
        raise ValueError("workspace_limit_bytes must be a nonnegative integer")
    with torch.cuda.device(q.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare must run outside CUDA Graph capture")
        if torch.cuda.get_device_capability(q.device) not in ((10, 0), (10, 3)):
            raise ValueError("this package targets SM100/SM103 Blackwell")


@dataclass(frozen=True)
class PrefillPlan:
    """Own bounded workspace; keep this plan and its buffers alive during execution."""

    _backend: object = field(repr=False)
    out: torch.Tensor = field(repr=False)
    state_out: torch.Tensor = field(repr=False)
    state_dtype: torch.dtype
    workspace_bytes: int

    def run(self):
        """Enqueue on the bound device's current stream; return output and state."""
        with torch.cuda.device(self.out.device):
            self._backend.run()
        return self.out, self.state_out


@dataclass(frozen=True)
class DecodePlan:
    """Q1 terminal state or Q8 per-token checkpoints under one fixed contract."""

    _backend: object = field(repr=False)
    out: torch.Tensor = field(repr=False)
    state_out: torch.Tensor = field(repr=False)
    checkpoint_states: torch.Tensor | None = field(repr=False)
    state_dtype: torch.dtype
    workspace_bytes: int

    def run(self):
        """Enqueue the complete decode; Q8 returns a view of its final checkpoint."""
        with torch.cuda.device(self.out.device):
            self._backend.run()
        return self.out, self.state_out


def prepare_prefill(
    q, k, v, g, beta, A_log, dt_bias, state_in, out, state_out, *,
    state_dtype, cu_seqlens=None, scale=128**-0.5, eps=1e-6,
    lower_bound=-5.0, workspace_limit_bytes=256 * 1024**2,
):
    """Prepare dense [B,L,H,128] or packed [1,T,H,128] no_grad forward.

    state_dtype is required and selects BF16 or FP32 continuation storage.
    Q/K/V/g/beta/output are BF16; A_log/dt_bias are FP32. Packed cu_seqlens
    is device Int32 [N+1]. Input/output states are [N,H,128,128], V then K.
    See docs/api.md for metadata, ownership, alignment and Graph constraints.
    """
    _validate_contract(q, state_in, state_dtype, workspace_limit_bytes)
    if state_dtype == torch.float32:
        from ._fp32_state.api import prepare_fp32_prefill as prepare
    else:
        from ._bf16_state.api import prepare_unified_prefill as prepare
    backend = prepare(
        q, k, v, g, beta, A_log, dt_bias, state_in, out, state_out,
        cu_seqlens=cu_seqlens, scale=scale, eps=eps, lower_bound=lower_bound,
        workspace_limit_bytes=workspace_limit_bytes,
    )
    return PrefillPlan(backend, out, state_out, state_dtype, backend.workspace_bytes)


def prepare_decode(
    q, k, v, g, beta, A_log, dt_bias, state_in, out, state_out=None, *,
    state_dtype, checkpoint_states=None, scale=128**-0.5, eps=1e-6,
    lower_bound=-5.0, workspace_limit_bytes=64 * 1024**2,
):
    """Prepare dense [B,Q,H,128] decode, Q=1 or Q=8.

    Q1 requires state_out and forbids checkpoint_states. Q8 requires
    checkpoint_states [B,8,H,128,128] and state_out=None; run returns the
    checkpoint_states[:, -1] view. Accept/reject selection is caller-owned.
    No input dtype conversion or device-to-host metadata read is performed.
    """
    _validate_contract(q, state_in, state_dtype, workspace_limit_bytes)
    if q.ndim != 4 or q.shape[1] not in (1, 8) or q.shape[2] < 1 or q.shape[3] != 128:
        raise ValueError("q must have shape [B,Q,H,128], Q=1 or 8, H>0")
    b, length, h, _ = q.shape
    if length == 1:
        if not isinstance(state_out, torch.Tensor) or checkpoint_states is not None:
            raise ValueError("Q1 requires state_out and no checkpoint_states")
    else:
        if not isinstance(checkpoint_states, torch.Tensor) or state_out is not None:
            raise ValueError("Q8 requires checkpoint_states and state_out=None")
        if (checkpoint_states.shape != (b, 8, h, 128, 128)
                or checkpoint_states.dtype != state_dtype
                or checkpoint_states.device != q.device
                or not checkpoint_states.is_contiguous()):
            raise ValueError("Q8 checkpoints must be contiguous [B,8,H,128,128] with matching device/state_dtype")
    args = (q, k, v, g, beta, A_log, dt_bias, state_in, out)
    options = dict(scale=scale, eps=eps, lower_bound=lower_bound)
    if state_dtype == torch.float32:
        from ._fp32_state.api import prepare_fp32_decode
        backend = prepare_fp32_decode(
            *args, state_out, checkpoint_states=checkpoint_states, **options,
        )
        workspace_bytes = backend.workspace_bytes
    else:
        from ._bf16_state.api import prepare_bf16_decode
        backend = prepare_bf16_decode(
            *args, state_out, checkpoint_states=checkpoint_states,
            workspace_limit_bytes=workspace_limit_bytes, **options,
        )
        workspace_bytes = backend.workspace_bytes
    terminal = checkpoint_states[:, -1] if length == 8 else state_out
    return DecodePlan(backend, out, terminal, checkpoint_states, state_dtype, workspace_bytes)
