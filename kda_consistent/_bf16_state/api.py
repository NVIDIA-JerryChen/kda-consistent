"""Buffer preparation and validation for the BF16-state backend."""

from dataclasses import dataclass, field, replace
import torch
from .._validation import _contract_scalars, _range, _pool_size


@dataclass(frozen=True)
class ConsistentDecodePlan:
    """Buffer-bound launch prepared by prepare_consistent_decode.

    Keep tensor addresses, metadata and scalar parameters unchanged. Tensor
    contents may change on the caller's stream. Recreate the plan to rebind
    buffers or shapes; compatible plans reuse the compiled kernel.
    """

    _kernel: object = field(repr=False)
    _arguments: tuple = field(repr=False)
    _requires_grad: bool = field(repr=False)
    out: torch.Tensor = field(repr=False)
    state_out: torch.Tensor = field(repr=False)
    workspace_bytes: int = field(default=0, init=False)

    def run(self):
        """Enqueue one token on the current stream; no allocation or copies."""
        if self._requires_grad and torch.is_grad_enabled():
            raise RuntimeError("consistent decode supports no_grad forward only")
        if self._kernel is not None:
            self._kernel(*self._arguments)
        return self.out, self.state_out


def prepare_consistent_decode(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    state_in,
    out,
    state_out,
    *,
    state_indices=None,
    scale=128**-0.5,
    eps=1e-6,
    lower_bound=-5.0,
):
    """Validate and bind one-token buffers, then compile outside Graph capture.

    BF16 q/k/v/g/out: [B,1,H,128]; beta: [B,1,H]; state: [B,H,128,128].
    FP32 A_log: [H]; dt_bias: [H,128]. All tensors must be contiguous CUDA
    tensors on one device, with 32-byte aligned data pointers. State input and
    output may be identical; all other overlap is forbidden. Buffers belong to
    the caller. Empty batches are no-ops. Prepare outside CUDA Graph capture.
    See docs/api.md for the complete contract.
    """
    tensors = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        state_in=state_in,
        out=out,
        state_out=state_out,
    )
    if not all(isinstance(t, torch.Tensor) for t in tensors.values()):
        raise TypeError("all inputs and output buffers must be torch.Tensor")
    if q.ndim != 4 or q.shape[1] != 1 or q.shape[2] < 1 or q.shape[3] != 128:
        raise ValueError("q must have shape [B,1,H,128], H > 0")
    b, _, h, _ = q.shape
    if b * h * 8 >= 2**31:
        raise ValueError("decode requires B * H * 8 < 2**31")
    pool_size = _pool_size(state_in, state_indices, b, h, q.device)
    if state_indices is not None:
        tensors["state_indices"] = state_indices
    shapes = dict(
        state_indices=state_indices.shape if state_indices is not None else (0,),
        q=q.shape,
        k=q.shape,
        v=q.shape,
        g=q.shape,
        out=q.shape,
        beta=(b, 1, h),
        A_log=(h,),
        dt_bias=(h, 128),
        state_in=(pool_size, h, 128, 128),
        state_out=(pool_size, h, 128, 128),
    )
    for name, t in tensors.items():
        dtype = (
            torch.int32
            if name == "state_indices"
            else torch.float32
            if name in ("A_log", "dt_bias")
            else torch.bfloat16
        )
        if t.device.type != "cuda" or t.device != q.device:
            raise ValueError(f"{name} must be on the same CUDA device as q")
        if t.dtype != dtype or tuple(t.shape) != tuple(shapes[name]):
            raise ValueError(f"{name} must have dtype {dtype} and shape {shapes[name]}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if t.numel() and t.data_ptr() % 32:
            raise ValueError(f"{name} data pointer must be 32-byte aligned")
        if torch.is_grad_enabled() and t.requires_grad:
            raise RuntimeError("consistent_decode_fwd supports no_grad forward only")
    _contract_scalars(scale, eps, lower_bound)
    ranges = [(name, *_range(t)) for name, t in tensors.items()]
    for i, (name, lo, hi) in enumerate(ranges):
        for other, ulo, uhi in ranges[i + 1 :]:
            if max(lo, ulo) < min(hi, uhi):
                if {name, other} == {"state_in", "state_out"} and (lo, hi) == (
                    ulo,
                    uhi,
                ):
                    continue
                raise ValueError(f"overlapping tensors are forbidden: {name}, {other}")
    if b == 0:
        return ConsistentDecodePlan(
            None, (), any(t.requires_grad for t in tensors.values()), out, state_out
        )
    from .decode_q1 import compiled_decode_kernel

    with torch.cuda.device(q.device):
        kernel = compiled_decode_kernel(
            q.device.index,
            torch.cuda.get_device_capability(q.device),
            state_indices is not None,
        )
        arguments = (
            q,
            k,
            v,
            g,
            beta,
            state_out,
            state_in,
            out,
            A_log,
            dt_bias.view(-1),
            float(scale),
            float(eps),
            float(lower_bound),
            state_indices,
        )
    return ConsistentDecodePlan(
        kernel,
        arguments,
        any(t.requires_grad for t in tensors.values()),
        out,
        state_out,
    )


@dataclass(frozen=True)
class UnifiedPrefillPlan:
    """Own workspace and bind caller buffers; prepare outside CUDA Graph capture."""

    _plan: object = field(repr=False)
    workspace_bytes: int
    out: torch.Tensor = field(repr=False)
    state_out: torch.Tensor = field(repr=False)
    cu_seqlens: torch.Tensor = field(repr=False)
    _requires_grad: bool = field(repr=False)

    def run(self):
        """Enqueue on this device's current stream without allocation or D2H."""
        if self._requires_grad and torch.is_grad_enabled():
            raise RuntimeError("unified prefill supports no_grad forward only")
        with torch.cuda.device(self.out.device):
            self._plan.run()
        return self.out, self.state_out


def prepare_unified_prefill(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    state_in,
    out,
    state_out,
    *,
    state_indices=None,
    cu_seqlens=None,
    scale=128**-0.5,
    eps=1e-6,
    lower_bound=-5.0,
    workspace_limit_bytes=256 * 1024 * 1024,
):
    """Bind BF16 D128 prefill using one compiled scan for all B/H/lengths.

    Dense inputs are [B,L,H,128]; packed inputs are [1,capacity,H,128] with
    device Int32 cu_seqlens [N+1]. State is BF16 [N,H,V,K] and is required.
    Valid metadata starts at zero, is nondecreasing, and ends at or below
    capacity. Its values remain device-resident; callers must maintain these
    preconditions, including during CUDA Graph replay. See docs/api.md.
    """
    tensors = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        state_in=state_in,
        out=out,
        state_out=state_out,
    )
    if cu_seqlens is not None:
        tensors["cu_seqlens"] = cu_seqlens
    if not all(isinstance(t, torch.Tensor) for t in tensors.values()):
        raise TypeError("all inputs and output buffers must be torch.Tensor")
    if q.ndim != 4 or q.shape[2] < 1 or q.shape[3] != 128:
        raise ValueError("q must have shape [B,L,H,128], H > 0")
    b, length, h, _ = q.shape
    total = b * length
    owns_cu = cu_seqlens is None
    if cu_seqlens is None:
        n = b
    else:
        if b != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 1:
            raise ValueError("packed input requires B=1 and cu_seqlens [N+1]")
        n = cu_seqlens.numel() - 1
    if total + 127 >= 2**31 or n * h * 8 >= 2**31 or (total + 127) // 128 + n >= 2**31:
        raise ValueError("token capacity, schedule capacity and grid must fit Int32")
    pool_size = _pool_size(state_in, state_indices, n, h, q.device)
    if state_indices is not None:
        tensors["state_indices"] = state_indices
    shapes = dict(
        state_indices=state_indices.shape if state_indices is not None else (0,),
        q=q.shape,
        k=q.shape,
        v=q.shape,
        g=q.shape,
        out=q.shape,
        beta=(b, length, h),
        A_log=(h,),
        dt_bias=(h, 128),
        state_in=(pool_size, h, 128, 128),
        state_out=(pool_size, h, 128, 128),
        cu_seqlens=(n + 1,),
    )
    requires_grad = False
    for name, tensor in tensors.items():
        dtype = (
            torch.int32
            if name in ("cu_seqlens", "state_indices")
            else (torch.float32 if name in ("A_log", "dt_bias") else torch.bfloat16)
        )
        if tensor.device.type != "cuda" or tensor.device != q.device:
            raise ValueError(f"{name} must be on the same CUDA device as q")
        if tensor.dtype != dtype or tuple(tensor.shape) != tuple(shapes[name]):
            raise ValueError(f"{name} must have dtype {dtype} and shape {shapes[name]}")
        if not tensor.is_contiguous() or (tensor.numel() and tensor.data_ptr() % 32):
            raise ValueError(f"{name} must be contiguous and 32-byte aligned")
        requires_grad |= tensor.requires_grad
    if requires_grad and torch.is_grad_enabled():
        raise RuntimeError("unified prefill supports no_grad forward only")
    _contract_scalars(scale, eps, lower_bound)
    if not isinstance(workspace_limit_bytes, int) or workspace_limit_bytes < 0:
        raise ValueError("workspace_limit_bytes must be a nonnegative integer")
    ranges = [(name, *_range(t)) for name, t in tensors.items()]
    for i, (name, lo, hi) in enumerate(ranges):
        for other, ulo, uhi in ranges[i + 1 :]:
            if max(lo, ulo) < min(hi, uhi):
                if {name, other} == {"state_in", "state_out"} and (lo, hi) == (
                    ulo,
                    uhi,
                ):
                    continue
                raise ValueError(f"overlapping tensors are forbidden: {name}, {other}")
    cu_bytes = (n + 1) * 4 if owns_cu else 0
    if workspace_limit_bytes < cu_bytes:
        raise ValueError("workspace cannot hold dense sequence metadata")
    with torch.cuda.device(q.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "prepare_unified_prefill must run outside CUDA Graph capture"
            )
        if cu_seqlens is None:
            cu_seqlens = (
                torch.arange(n + 1, device=q.device, dtype=torch.int32) * length
            )
        from .prefill.plan import prepare

        packed = [x.view(1, total, *x.shape[2:]) for x in (q, k, v, g, beta)]
        plan = prepare(
            *packed,
            A_log,
            dt_bias,
            state_in,
            cu_seqlens,
            out.view(1, total, h, 128),
            state_out,
            workspace_limit_bytes - cu_bytes,
            state_indices=state_indices,
            scale=scale,
            eps=eps,
            lower_bound=lower_bound,
        )
    workspace_bytes = plan.workspace_bytes + cu_bytes
    return UnifiedPrefillPlan(
        plan, workspace_bytes, out, state_out, cu_seqlens, requires_grad
    )


@dataclass(frozen=True)
class CheckpointDecodePlan:
    """Buffer-bound launch prepared by prepare_checkpoint_decode.

    Keep tensor addresses, metadata and scalar parameters unchanged. Tensor
    contents may change on the caller's stream. Recreate the plan to rebind
    buffers or shapes; compatible plans reuse the compiled kernel.
    """

    _calls: tuple = field(repr=False)
    workspace_bytes: int
    _requires_grad: bool = field(repr=False)
    out: torch.Tensor = field(repr=False)
    state_out: torch.Tensor | None = field(repr=False)
    checkpoint_states: torch.Tensor | None = field(default=None, repr=False)
    checkpoint_indices: torch.Tensor | None = field(default=None, repr=False)

    def run(self):
        """Enqueue the prepared sequence on the current stream without allocation."""
        if self._requires_grad and torch.is_grad_enabled():
            raise RuntimeError("Q8 decode supports no_grad forward only")
        for kernel, arguments in self._calls:
            kernel(*arguments)
        return self.out, self.state_out


def prepare_checkpoint_decode(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    state_in,
    out,
    state_out,
    *,
    state_indices=None,
    scale=128**-0.5,
    eps=1e-6,
    lower_bound=-5.0,
    workspace_limit_bytes=64 * 1024 * 1024,
    checkpoint_states=None,
    checkpoint_indices=None,
):
    """Bind Q8 buffers and unique checkpoint slots; prepare outside capture."""
    if state_out is not None:
        raise ValueError("Q8 uses checkpoints instead of a separate state_out")
    if checkpoint_states is None or checkpoint_indices is None:
        raise ValueError("Q8 requires checkpoint_states and checkpoint_indices")
    tensors = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        state_in=state_in,
        out=out,
    )
    tensors.update(
        checkpoint_states=checkpoint_states, checkpoint_indices=checkpoint_indices
    )
    if not all(isinstance(t, torch.Tensor) for t in tensors.values()):
        raise TypeError("all inputs and output buffers must be torch.Tensor")
    if q.ndim != 4 or q.shape[1] != 8 or q.shape[2] < 1 or q.shape[3] != 128:
        raise ValueError("q must have shape [B,8,H,128], H > 0")
    b, length, h, _ = q.shape
    if b * h * 64 >= 2**31:
        raise ValueError("Q8 decode requires B * H * 64 < 2**31")
    pool_size = _pool_size(state_in, state_indices, b, h, q.device)
    if state_indices is not None:
        tensors["state_indices"] = state_indices
    shapes = dict(
        state_indices=state_indices.shape if state_indices is not None else (0,),
        q=q.shape,
        k=q.shape,
        v=q.shape,
        g=q.shape,
        out=q.shape,
        beta=(b, length, h),
        A_log=(h,),
        dt_bias=(h, 128),
        state_in=(pool_size, h, 128, 128),
    )
    if checkpoint_states.ndim != 4 or tuple(checkpoint_states.shape[1:]) != (
        h,
        128,
        128,
    ):
        raise ValueError("checkpoint_states must have shape [P,H,128,128]")
    capacity = checkpoint_states.shape[0]
    if capacity < b * length or capacity >= 2**31:
        raise ValueError("checkpoint pool requires B * L <= P < 2**31")
    shapes.update(
        checkpoint_states=(capacity, h, 128, 128), checkpoint_indices=(b, length)
    )
    for name, t in tensors.items():
        dtype = (
            torch.int32
            if name in ("checkpoint_indices", "state_indices")
            else torch.float32
            if name in ("A_log", "dt_bias")
            else torch.bfloat16
        )
        if t.device.type != "cuda" or t.device != q.device:
            raise ValueError(f"{name} must be on the same CUDA device as q")
        if t.dtype != dtype or tuple(t.shape) != tuple(shapes[name]):
            raise ValueError(f"{name} must have dtype {dtype} and shape {shapes[name]}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if t.numel() and t.data_ptr() % 32:
            raise ValueError(f"{name} data pointer must be 32-byte aligned")
        if torch.is_grad_enabled() and t.requires_grad:
            raise RuntimeError("Q8 decode supports no_grad forward only")
    _contract_scalars(scale, eps, lower_bound)
    if not isinstance(workspace_limit_bytes, int) or workspace_limit_bytes < 0:
        raise ValueError("workspace_limit_bytes must be a nonnegative integer")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "prepare_checkpoint_decode must run outside CUDA Graph capture"
        )
    ranges = [(name, *_range(t)) for name, t in tensors.items()]
    for i, (name, lo, hi) in enumerate(ranges):
        for other, ulo, uhi in ranges[i + 1 :]:
            if max(lo, ulo) < min(hi, uhi):
                raise ValueError(f"overlapping tensors are forbidden: {name}, {other}")
    grad = any(t.requires_grad for t in tensors.values())
    if b == 0:
        return CheckpointDecodePlan(
            (), 0, grad, out, state_out, checkpoint_states, checkpoint_indices
        )
    calls = []
    used_workspace = 0
    final_buffer = state_in
    # Split V further only when the eight-row grid cannot fill the device.
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    tile = 2 if b * h * (128 // 8) <= sms else 8
    with torch.cuda.device(q.device):
        device = q.device.index
        capability = torch.cuda.get_device_capability(q.device)
        from .decode_q8 import (
            compiled_coeff_kernel,
            compiled_scan_checkpoints,
        )

        row_bytes = b * h * 392 * 4
        workspace_budget = workspace_limit_bytes
        window = min(length, workspace_budget // row_bytes)
        if window < 1:
            raise ValueError(
                "workspace limit cannot hold one token for the whole batch"
            )
        # Cap the number of rows consumed by the Int32 coefficient grid.
        window = min(window, (2**31 - 1) // (b * h))
        workspace = torch.empty(
            b * window * h * 392, device=q.device, dtype=torch.float32
        )
        used_workspace = workspace.numel() * workspace.element_size()
        coeff = compiled_coeff_kernel(device, capability)
        scan = compiled_scan_checkpoints(
            device, capability, tile, False, state_indices is not None
        )
        for start in range(0, length, window):
            count = min(window, length - start)
            calls.append(
                (
                    coeff,
                    (
                        q,
                        k,
                        g,
                        beta,
                        A_log,
                        dt_bias.view(-1),
                        workspace,
                        float(scale),
                        float(eps),
                        float(lower_bound),
                        start,
                        count,
                    ),
                )
            )
            scan_arguments = (
                v,
                workspace,
                final_buffer,
                state_in,
                out,
            )
            scan_arguments += (checkpoint_states, checkpoint_indices, state_indices)
            calls.append((scan, scan_arguments + (start, count)))
    return CheckpointDecodePlan(
        tuple(calls),
        used_workspace,
        grad,
        out,
        state_out,
        checkpoint_states,
        checkpoint_indices,
    )


def prepare_bf16_decode(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    state_in,
    out,
    state_out=None,
    *,
    state_indices=None,
    checkpoint_states=None,
    scale=128**-0.5,
    eps=1e-6,
    lower_bound=-5.0,
    workspace_limit_bytes=64 * 1024**2,
):
    """Own Q8 index preparation and account for its complete workspace."""
    args = (q, k, v, g, beta, A_log, dt_bias, state_in, out)
    options = dict(
        scale=scale, eps=eps, lower_bound=lower_bound, state_indices=state_indices
    )
    b, length, h, _ = q.shape
    if length == 1:
        return prepare_consistent_decode(*args, state_out, **options)
    metadata_bytes = b * 8 * 4
    if workspace_limit_bytes < metadata_bytes:
        raise ValueError("workspace cannot hold checkpoint indices")
    with torch.cuda.device(q.device):
        indices = torch.arange(b * 8, device=q.device, dtype=torch.int32).view(b, 8)
        backend = prepare_checkpoint_decode(
            *args,
            None,
            checkpoint_states=checkpoint_states.view(b * 8, h, 128, 128),
            checkpoint_indices=indices,
            workspace_limit_bytes=workspace_limit_bytes - metadata_bytes,
            **options,
        )
    return replace(backend, workspace_bytes=backend.workspace_bytes + metadata_bytes)
