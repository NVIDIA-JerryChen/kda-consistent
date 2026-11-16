# B200 comparison, 2026-11-16

NVIDIA B200 (SM100), driver 615.71.10, 1000 W power limit. Two allocated GPUs with
NV18 connectivity were verified. Timing ran serially on one GPU; this is not
multi-GPU throughput. No thermal throttling was observed on the measured GPU.

Python 3.12.3, PyTorch 2.14.0a0+b2c75dd062.nv26.09, CUDA 13.4,
CuTe DSL 4.8.1, apache-tvm-ffi 0.1.10, Triton 3.8.0+gitbd40f349.nv26.9.
The 154-test suite passed with no skips, including two-GPU full-head sharding.
Benchmark checks cover finite outputs, complete effective continuation state,
Graph replay, CF mixed/row agreement and every Q8 checkpoint against prefix execution.

## Implementations

- Current public `kda_consistent` APIs, BF16 and FP32 state contracts. Exact Python
  file hashes are in [package-sources.json](package-sources.json); the aggregate
  hash and frozen experiment digest are in [summary.json](summary.json).
- FlashInfer `c3af0c13328fd23b256c53b9b4f94d9e160d118c`: Q1 single-token and
  native Q8 grouped checkpoint paths, BF16 state/checkpoint storage.
- FlashKDA `d7e3b3a9f448500630fa60fb5200f4f9c9b1fc72`, CUTLASS
  `5c149f52a436782210263fb2f19b354443a61c6a`: original packed forward, BF16 state.
- Inferact/zero-mismatch-rl `vllm/kimi-k3.1`,
  `5735778f43f7d94108d432706dc5b17765bf723a`: CF-BI C16/v6 and C64/v5.
  Original CUDA kernels are unchanged, compiled for `sm_100a`.

## Inputs and timing

231 approved varlen cases: Q1 21, Q8 21, chunk prefill 84, prefill 9,
192K training forward 96; H=12/16/32, D=128. Decode B=64, Q=1/8,
KV limits 8K/16K/32K/64K/128K/256K/512K. Chunk prefill B=8/16;
prefill B=16, maximum lengths 8K/16K/32K. Training forward uses selected CP16
rank fragments totaling 12288 query tokens per case; no backward or communication.
Exact query/history lengths are in [cases.csv](cases.csv).

Q/K/V/g/beta/output are BF16, gate parameters FP32. State/checkpoint precision
follows the implementation's contract. Each of the four consistent implementations
prepares its own complete entry state from the same history, outside timing.
FlashInfer/FlashKDA receive the same BF16 entry matrix as BF16 recurrent; this
compares the current call, not their own entire historical trajectory.

Complete prepared GPU API timing includes entry-state preservation and CF copies,
packing, gather/scatter and open-chunk completion. Q8 includes all eight complete
checkpoints. CF Q8 uses eight public Q1 calls plus E/cache/position snapshots;
it is **not an optimized native Q8 checkpoint kernel**. Compilation, allocation,
host metadata construction, history preparation and acceptance selection are excluded.

Fixed-address hot-cache CUDA Graphs: one complete call per graph, three replays per
sample, rotating variant order, three warmup rounds and nine retained rounds.
No cold-L2, CPU overhead or framework end-to-end performance claim.
Large validation snapshots are held on CPU and checked in bounded blocks, outside timing.
`cases.csv` retains all nine samples and medians. `phase-cases.csv` adds
384 measurements: 64 starting positions × Q1/Q8 × H12/16/32.
Their full-cycle means are in `summary.json`.

## Performance

Geometric mean of paired per-case latency ratios, each case equally weighted;
lower is better. The reference is FlashInfer for decode and FlashKDA for forward.

| Workload | Reference = 1× | BF16 state | FP32 state | CF-BI C16 | CF-BI C64 |
|---|---|---:|---:|---:|---:|
| Decode Q1 | FlashInfer | 0.90× | 1.19× | 5.37× | 8.50× |
| Decode Q8 | FlashInfer | 0.93× | 0.98× | 8.84× | 12.18× |
| Chunk prefill | FlashKDA | 2.71× | 2.35× | 8.80× | 7.15× |
| Prefill | FlashKDA | 3.45× | 2.83× | 9.09× | 5.24× |
| 192K forward | FlashKDA | 1.73× | 1.52× | 5.09× | 2.90× |

Arithmetic means of per-case median latency in microseconds, grouped by H:

| Workload | H | BF16 state | FP32 state | CF-BI C16 | CF-BI C64 | FlashInfer / FlashKDA |
|---|---:|---:|---:|---:|---:|---:|
| Decode Q1 | 12 | 13.82 | 16.50 | 88.54 | 127.52 | 14.37 |
| Decode Q1 | 16 | 16.33 | 22.40 | 98.95 | 160.36 | 18.40 |
| Decode Q1 | 32 | 31.28 | 43.62 | 170.85 | 291.05 | 36.59 |
| Decode Q8 | 12 | 75.83 | 79.21 | 831.27 | 1009.16 | 82.20 |
| Decode Q8 | 16 | 96.29 | 103.09 | 920.02 | 1283.73 | 102.37 |
| Decode Q8 | 32 | 183.93 | 194.38 | 1507.43 | 2329.12 | 198.45 |
| Chunk prefill | 12 | 892.12 | 755.65 | 2855.07 | 2107.47 | 363.63 |
| Chunk prefill | 16 | 1143.83 | 953.27 | 3493.98 | 2607.10 | 396.39 |
| Chunk prefill | 32 | 2176.11 | 1833.98 | 6236.38 | 4718.80 | 672.73 |
| Prefill | 12 | 12079.03 | 9571.68 | 31574.58 | 18364.39 | 3821.34 |
| Prefill | 16 | 15247.48 | 12597.62 | 42098.94 | 23310.94 | 4284.02 |
| Prefill | 32 | 30699.13 | 25754.55 | 77859.31 | 46280.29 | 8391.39 |
| 192K forward | 12 | 765.95 | 683.86 | 2373.71 | 1267.18 | 579.39 |
| 192K forward | 16 | 977.29 | 862.86 | 2706.34 | 1624.53 | 602.36 |
| 192K forward | 32 | 1512.08 | 1297.73 | 4522.48 | 2606.29 | 681.01 |

The summary also retains per-case ratio extrema and a separate weighted aggregate:
for each measurement round, sum `case_weight × latency` within each phase, divide
by the reference sum, then report the median of nine ratios. Decode and prefill
are never combined. `sample_quality` records the maximum within-case relative
median absolute deviation and max/min sample spread for each variant and phase. Workload weights are recorded in the CSV. Observed speed
ratios are not universal performance guarantees, and differences between contracts
cannot be attributed solely to state storage dtype.

## FP64 diagnostics

96 records across six implementations: 18 common inputs for each of BF16, FP32,
CF C16, CF C64 and FlashKDA, plus six FlashInfer records for Q1/Q8.
H=3; lengths 1/8/64/256/4096/32768; normal/open/saturated gates; common nonzero,
BF16-representable initial state. The independent FP64 recurrence measures both
output and state error. CF matrix errors are reported only at closed chunk
boundaries; open-chunk caches are checked for consistency separately.

NRMSE = RMS(error)/RMS(reference); max_normalized = max(abs(error))/max(abs(reference)),
with denominators clamped at 1e-30. See [accuracy.csv](accuracy.csv) and
[accuracy-by-gate.csv](accuracy-by-gate.csv). These are diagnostics, not accepted
error bounds. BF16 output rounding remains present for all variants.

## Historical data and reproduction

[B300 historical measurements](../four-way-20261114/README.md) are preserved unchanged.
Their source snapshots and PyTorch/CUDA versions differ: do not interpret B200/B300
latency ratios as isolated hardware speedups. No historical value was substituted
for a missing B200 measurement.

The public `python -m benchmarks.bench` command exercises the current package but
does not reproduce this complete external-reference experiment. External reference
builds and the comparative runner are not shipped as kernel dependencies.
