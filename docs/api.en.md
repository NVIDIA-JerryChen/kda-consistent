# API and data contract

[简体中文](api.md) · [Home](../README.md)

Exports: `prepare_prefill`, `prepare_decode`, `PrefillPlan`, `DecodePlan`.
Both prepare functions start with `q,k,v,g,beta,A_log,dt_bias,state_in,out`.
The required keyword `state_dtype` must equal the state/checkpoint dtype; no implicit
conversion occurs. All tensors must be on the same CUDA device, contiguous, and
32-byte aligned when nonempty. D=128 and H>0; state axes are V,K.

| Argument | Prefill | Decode | dtype |
|---|---|---|---|
| q/k/v/g/out | dense `[B,L,H,128]` or packed `[1,T,H,128]` | `[B,Q,H,128]`, Q=1/8 | BF16 |
| beta | matching first three axes | `[B,Q,H]` | BF16 |
| A_log / dt_bias | `[H]` / `[H,128]` | same | FP32 |
| state_in | `[N,H,128,128]`, dense N=B | `[B,H,128,128]` | state_dtype |
| state_out | required, same as state_in | required for Q1; must be None for Q8 | state_dtype |
| checkpoint_states | unsupported | forbidden for Q1; required `[B,8,H,128,128]` for Q8 | state_dtype |
| cu_seqlens | packed Int32 `[N+1]`, omitted for dense | unused | CUDA Int32 |

Q/K are L2-normalized internally; g/beta are pre-activation values. Defaults:
`scale=128**-0.5`, `eps=1e-6`, `lower_bound=-5.0`. Eps must be positive in FP32,
lower_bound must be in `[-5,0]`, and scalars must be finite FP32 values.
`workspace_limit_bytes` defaults to 256 MiB for prefill and 64 MiB for decode,
including internal metadata. Insufficient space for a minimal window raises an error.
Q1 and FP32 Q8 need no internal device workspace; BF16 Q8 needs coefficients and indices.

Prefill supports up to 1023 requests, with Int32 capacity/grid checks. Empty batches
and requests are valid. Packed offsets start at zero, are nondecreasing, and end
at or below T; unused output tails are untouched. The caller must maintain valid
metadata contents; the API does not read them back to the host. Historical KV is
represented by state, without a KV length argument. Disjoint sequence fragments
require their own correct entry states.

## State and checkpoints

`run()` returns `(out,state_out)`. For Q8 the second value is the usually noncontiguous
view `checkpoint_states[:, -1]`. Use `.contiguous()` or copy it to a contiguous buffer
before passing it to a subsequent call. Accepting a tokens selects the original
entry state when a=0, otherwise checkpoint a−1. Selection, compaction and framework
state management belong to the caller and must be timed when measuring that scope.
Q1/prefill allow exact state_in/state_out in-place aliasing; other overlap is forbidden.
Q8 checkpoints must be independent of the entry state to preserve acceptance of zero tokens.

## Lifecycle, Graphs and compilation

Prepare validates, compiles, binds fixed capacities/addresses, and allocates bounded
workspace outside Graph capture. Run allocates no device memory, compiles nothing,
performs no D2H, and uses the bound device's current stream. Keep the plan and all
buffers alive until GPU completion. Update input contents and valid device metadata
in place; recreate the plan for new addresses/shapes/scalars. The caller synchronizes
cross-stream dependencies. Do not concurrently reuse a plan/workspace/output.
Graph replay with in-place state advances the state.

Prefill does not select compilation variants by per-request length; B/H/token counts
and metadata contents remain runtime. Decode uses Q and finite configurations,
without exact-batch or KV-length specialization. Invalid type/shape/dtype/device/
alignment/alias/scalars raise TypeError or ValueError. Prepare during capture and
autograd-enabled inputs requiring gradients raise RuntimeError. Valid inputs and
results must be finite; run does not scan for NaN/Inf. Only no_grad forward is supported.
