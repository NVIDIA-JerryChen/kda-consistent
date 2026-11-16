# KDA Consistent

[简体中文](README.zh-CN.md)

KDA forward kernels with **BF16 or FP32 continuation state** and bitwise
prefill/decode consistency. **Prefer FP32 state for accuracy**
(`state_dtype=torch.float32`) to avoid per-token BF16 state quantization.

## Installation

CuTe DSL 4.8.1+; SM100/SM103.

```bash
python -m pip install '.[test]'
python -m examples.forward --state-dtype bf16
python -m examples.forward --state-dtype fp32
```

The first `prepare` call compiles the kernels.

## API

```python
import torch
from kda_consistent import prepare_prefill, prepare_decode

# All tensors are caller-owned CUDA buffers; see the runnable example above.
plan = prepare_prefill(
    q, k, v, g, beta, A_log, dt_bias, state_in, out, state_out,
    state_dtype=torch.float32, cu_seqlens=cu_seqlens,
)
with torch.no_grad():
    output, terminal = plan.run()
```

`state_dtype` is required: `torch.bfloat16` or `torch.float32`. No implicit state
conversion is performed. Q/K/V/g/beta/output are BF16; gate parameters are FP32.
Q1 writes a terminal state. Q8 requires all eight checkpoints and returns a view
of the last one. Device `state_indices` supports state pools; Q8 `commit()` selects
the accepted prefix using device counts. Packed varlen prefill uses device Int32 `cu_seqlens`; decode has a fixed Q per request.

- [Complete API and ownership contract](docs/api.en.md)
- [Numerical contracts and validation](docs/numerics.en.md)
- [Tests](tests/README.md) and [benchmarks / historical data](benchmarks/README.md)
- [Source attribution](provenance/README.md)

## Consistency and CP scope

On the same GPU model and software stack, each state contract preserves bitwise
outputs, complete state and checkpoints across prefill, Q1/Q8, segmentation/PD,
batching/reordering, varlen and eager/CUDA Graph. BF16 and FP32 contracts need not
match each other.

Prefill and decode use the same per-token recurrence, arithmetic/reduction order
and rounding boundaries. BF16 state computes the output from the updated FP32
state, then rounds the state to BF16 for the next token. FP32 state remains FP32
between tokens and calls. Scheduling preserves this numerical program.

Supports complete-head sharding for Ulysses Parallel (UP); Ring CP is unsupported.
The caller handles communication and framework integration, partitioning inputs,
gate parameters and state by the same heads.

## Performance

231 cases: H=12/16/32, D=128, decode B=64, varlen forward.
Tables show geometric mean latency ratios; lower is better, with the baseline
specified in each table. Timing covers the complete prepared GPU API using
hot-address CUDA Graphs, one complete call per graph, including all Q8 checkpoints, accepted-prefix selection and state writeback. CF-BI Q8 uses eight Q1
calls plus full-state snapshots. Compilation, history preparation and acceptance decision generation are excluded.

### Reference implementations

B200 and B300 use the following reference versions.

| Implementation | Repository | Commit | Measured path |
|---|---|---|---|
| FlashInfer | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | [c3af0c13328fd23b256c53b9b4f94d9e160d118c](https://github.com/flashinfer-ai/flashinfer/commit/c3af0c13328fd23b256c53b9b4f94d9e160d118c) | Q1 single-token; native Q8 grouped checkpoints |
| FlashKDA | [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) | [d7e3b3a9f448500630fa60fb5200f4f9c9b1fc72](https://github.com/MoonshotAI/FlashKDA/commit/d7e3b3a9f448500630fa60fb5200f4f9c9b1fc72) | Original packed forward |
| CF-BI | [Inferact/zero-mismatch-rl](https://github.com/Inferact/zero-mismatch-rl) | [5735778f43f7d94108d432706dc5b17765bf723a](https://github.com/Inferact/zero-mismatch-rl/commit/5735778f43f7d94108d432706dc5b17765bf723a) | C16/v6 and C64/v5; branch `vllm/kimi-k3.1` |

FlashKDA uses the [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass) submodule at
[5c149f52a436782210263fb2f19b354443a61c6a](https://github.com/NVIDIA/cutlass/commit/5c149f52a436782210263fb2f19b354443a61c6a).

### B200

CuTe DSL 4.8.1 · CUDA 13.4 · PyTorch 2.14.0a0+b2c75dd062.nv26.09

[Data and environment](benchmarks/data/k3-runtime-20261116/b200/README.md).

| Workload | Reference = 1× | BF16 state | FP32 state | CF-BI C16 | CF-BI C64 |
|---|---|---:|---:|---:|---:|
| Decode Q1 | FlashInfer | 0.90× | 1.19× | 5.39× | 8.51× |
| Decode Q8 + commit | FlashInfer | 0.88× | 1.01× | 8.43× | 12.82× |
| Chunk prefill | FlashKDA | 2.70× | 2.37× | 8.80× | 7.15× |
| Prefill | FlashKDA | 3.50× | 2.83× | 9.09× | 5.25× |
| 192K forward | FlashKDA | 1.79× | 1.58× | 5.08× | 2.90× |

### B300

CuTe DSL 4.8.1 · CUDA 13.2 · PyTorch 2.12.0+cu132

[Data and environment](benchmarks/data/k3-runtime-20261116/b300/README.md).

| Workload | Reference = 1× | BF16 state | FP32 state | CF-BI C16 | CF-BI C64 |
|---|---|---:|---:|---:|---:|
| Decode Q1 | FlashInfer | 0.87× | 1.23× | 4.95× | 8.21× |
| Decode Q8 + commit | FlashInfer | 0.89× | 1.03× | 8.00× | 12.61× |
| Chunk prefill | FlashKDA | 2.73× | 2.38× | 8.40× | 6.63× |
| Prefill | FlashKDA | 3.54× | 2.86× | 9.09× | 5.27× |
| 192K forward | FlashKDA | 1.80× | 1.59× | 4.97× | 2.78× |

[Complete pool API and pre-change regression comparison](benchmarks/data/k3-runtime-20261116/README.md).

### Indexed-state decode versus FlashInfer

Both implementations index the state pool directly, with B=64 and H=12/16/32.
Q8 includes all checkpoints and accepted-state writeback. Both GPUs use
CuTe DSL 4.8.1 / CUDA 13.4 / PyTorch 2.14.0a0+b2c75dd062.nv26.09.
Geometric mean latency ratios relative to FlashInfer; lower is better.

| GPU | State writeback | Q1 BF16 | Q1 FP32 | Q8 BF16 | Q8 FP32 |
|---|---|---:|---:|---:|---:|
| B200 | Out of place | 0.90× | 1.22× | 0.79× | 1.00× |
| B200 | In place | 0.92× | 1.20× | 0.79× | 1.00× |
| B300 | Out of place | 0.92× | 1.25× | 0.79× | 1.03× |
| B300 | In place | 0.95× | 1.24× | 0.80× | 1.03× |

FlashInfer Q8 uses the same copy helper to write the accepted state into the
destination pool. Integrations that only update the next checkpoint index and
omit physical writeback are outside this comparison. FP32 Q1 with H=32 and
in-place writeback has ratios of 1.41×/1.47× on B200/B300.
[Timing protocol, per-H summaries and per-case data](benchmarks/data/k3-flashinfer-pool-20261116/README.md).

## Numerical accuracy

Maximum observed per-case errors against independent FP64 recurrence, from the
[B200 data](benchmarks/data/k3-runtime-20261116/b200/accuracy.csv)
([by gate](benchmarks/data/k3-runtime-20261116/b200/accuracy-by-gate.csv)).
H=3, D=128; lengths 1/8/64/256/4096/32768; normal/open/saturated gates;
shared nonzero BF16-representable initial state. Outputs are BF16; the reference
keeps FP64 state without BF16 rounding.

NRMSE = RMS(error) / RMS(reference); max normalized error =
max(abs(error)) / max(abs(reference)). Values are ratios, not percentages.

| Implementation | Output NRMSE | Output max normalized error | State NRMSE | State max normalized error |
|---|---:|---:|---:|---:|
| BF16 state | 2.197e-02 | 2.569e-02 | 2.195e-02 | 4.149e-02 |
| FP32 state | 1.739e-03 | 3.588e-03 | 3.575e-07 | 7.516e-07 |
| CF-BI C16 | 1.739e-03 | 3.588e-03 | 6.517e-07 | 1.254e-06 |
| CF-BI C64 | 1.739e-03 | 3.588e-03 | 1.626e-06 | 3.123e-06 |
| FlashInfer | 1.739e-03 | 2.636e-03 | 1.676e-03 | 3.202e-03 |
| FlashKDA | 8.029e-03 | 9.402e-03 | 7.440e-03 | 1.189e-02 |

FlashInfer covers six Q1/Q8 cases; other implementations cover 18 output cases.
CF-BI state errors cover 12 closed-chunk cases at lengths 64/256/4096/32768;
BF16 state, FP32 state and FlashKDA cover 18 state cases. Maxima with different
coverage are not a like-for-like ranking or error bounds. Metric details and
consistency checks: [numerical contract](docs/numerics.en.md).

## License

See [LICENSE](LICENSE) and [third-party licenses and notices](licenses/).
