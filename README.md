# KDA Consistent

[简体中文](README.zh-CN.md)

Standalone KDA no_grad forward kernels with **BF16 or FP32 continuation state**.
Each contract preserves bitwise outputs and complete continuation state across
prefill, Q1/Q8 decode, request batching/reordering, segmentation, and full-head sharding
on the same GPU model and software stack. The two state contracts are not bitwise
interchangeable. Backward, full-model logits, framework integration, and communication
are outside scope.

## Installation

Targets SM100/SM103 Blackwell. Validated with Python 3.12 and CuTe DSL 4.8.1 on
B300 (PyTorch 2.12.0+cu132, CUDA 13.2) and B200
(PyTorch 2.14.0a0+b2c75dd062.nv26.09, CUDA 13.4).
Other environments require independent validation. Install a matching CUDA PyTorch
build first, then, from this checkout:

```bash
python -m pip install '.[test]'
python -m examples.forward --state-dtype bf16
python -m examples.forward --state-dtype fp32
```

The Python package builds without a GPU; the first prepare compiles its
CuTe DSL kernels on the target GPU. Triton is used only by the independent FP64 diagnostic.

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
of the last one. The caller selects the accepted prefix. Packed varlen prefill
uses device Int32 `cu_seqlens`; decode has a fixed Q per request.

- [Complete API and ownership contract](docs/api.en.md)
- [Numerical contracts and validation](docs/numerics.en.md)
- [Tests](tests/README.md) and [benchmarks / historical data](benchmarks/README.md)
- [Source attribution](provenance/README.md)

## Performance

### B200

[B200 comparison and raw data](benchmarks/data/b200-20261116/README.md):
231 varlen cases, H=12/16/32, D=128; decode B=64. Values below are geometric
mean **latency ratios** to FlashInfer (decode) or FlashKDA (forward); lower is better.

| Workload | Reference = 1× | BF16 state | FP32 state | CF-BI C16 | CF-BI C64 |
|---|---|---:|---:|---:|---:|
| Decode Q1 | FlashInfer | 0.90× | 1.19× | 5.37× | 8.50× |
| Decode Q8 | FlashInfer | 0.93× | 0.98× | 8.84× | 12.18× |
| Chunk prefill | FlashKDA | 2.71× | 2.35× | 8.80× | 7.15× |
| Prefill | FlashKDA | 3.45× | 2.83× | 9.09× | 5.24× |
| 192K forward | FlashKDA | 1.73× | 1.52× | 5.09× | 2.90× |

Complete prepared GPU API, hot-address CUDA Graph; Q8 includes all eight checkpoints.
CF-BI Q8 uses eight public Q1 calls plus full-state snapshots. B200 measurements
use PyTorch 2.14.0a0 / CUDA 13.4 / CuTe DSL 4.8.1. See the data page for exact
versions, timing boundaries, per-case results, full-cycle decode and FP64 errors.

### B300 — historical comparisons

B300 SXM6 AC, PyTorch 2.12.0+cu132, CUDA 13.2, CuTe DSL 4.8.1;
H=12/16/32, D=128, decode B=64. These measurements use frozen source snapshots
from before production cleanup, not the current standalone package.

[External-reference comparisons](benchmarks/data/b300-external-20261114/README.md):
Q1/forward and the separate Q8 checkpoint run; geometric mean latency ratios,
with FlashInfer or FlashKDA as 1×.

| Workload | Reference = 1× | BF16 state | FP32 state |
|---|---|---:|---:|
| Decode Q1 | FlashInfer | 0.85× | 1.35× |
| Decode Q8 | FlashInfer | 0.97× | 1.05× |
| Chunk prefill | FlashKDA | 2.71× | 2.33× |
| Prefill | FlashKDA | 3.47× | 2.84× |
| 192K forward | FlashKDA | 1.73× | 1.50× |

[Separate four-way comparison](benchmarks/data/four-way-20261114/README.md):
231 cases, with **BF16 state as 1×**.

| Workload | BF16 state = 1× | FP32 state | CF-BI C16 | CF-BI C64 |
|---|---:|---:|---:|---:|
| Decode Q1 | 1.00× | 1.39× | 5.68× | 9.42× |
| Decode Q8 | 1.00× | 1.07× | 8.79× | 12.55× |
| Chunk prefill | 1.00× | 0.86× | 3.09× | 2.44× |
| Prefill | 1.00× | 0.82× | 2.62× | 1.52× |
| 192K forward | 1.00× | 0.87× | 2.86× | 1.60× |

Both sets time the complete prepared GPU API with hot-address CUDA Graphs and all
Q8 checkpoints. CF-BI Q8 uses eight public Q1 calls plus full-state snapshots.
External comparisons use 32 Q1 calls or eight Q8 calls per graph; the four-way
experiment uses one. Entry-state preparation also differs. Do not multiply ratios
across these tables. The current standalone package has not yet been remeasured
on B300 with the B200 comparison protocol; these are not hardware-only B200/B300 ratios.

## License

See [LICENSE](LICENSE) and [third-party licenses and notices](licenses/).
