# B200 comparison, 2026-11-16

NVIDIA B200, SM100; driver 615.71.10.
Two GPUs with NV18 connectivity; serial timing on one GPU.
Python 3.12.3 · CuTe DSL 4.8.1 · CUDA 13.4 · PyTorch 2.14.0a0+b2c75dd062.nv26.09.
224 tests passed with no skips, including dynamic metadata, all Q8 checkpoints,
accepted-prefix continuation, Graph replay and two-GPU complete-head sharding.

## Versions and workload

BF16/FP32 use the same current package on both GPUs: exact hashes in
[package-sources.json](package-sources.json), aggregate and experiment hashes in
[summary.json](summary.json). References:

- [FlashInfer c3af0c13328fd23b256c53b9b4f94d9e160d118c](https://github.com/flashinfer-ai/flashinfer/commit/c3af0c13328fd23b256c53b9b4f94d9e160d118c): Q1 and native Q8 checkpoints.
- [FlashKDA d7e3b3a9f448500630fa60fb5200f4f9c9b1fc72](https://github.com/MoonshotAI/FlashKDA/commit/d7e3b3a9f448500630fa60fb5200f4f9c9b1fc72), [CUTLASS 5c149f52a436782210263fb2f19b354443a61c6a](https://github.com/NVIDIA/cutlass/commit/5c149f52a436782210263fb2f19b354443a61c6a): packed forward.
- [CF-BI 5735778f43f7d94108d432706dc5b17765bf723a](https://github.com/Inferact/zero-mismatch-rl/commit/5735778f43f7d94108d432706dc5b17765bf723a): C16/v6 and C64/v5, branch `vllm/kimi-k3.1`; unchanged CUDA compiled for sm_100a.

231 cases: Q1 21, Q8 21, chunk prefill 84, prefill 9, 192K forward 96.
H=12/16/32, D=128. Decode B=64, Q=1/8, KV limits
8K/16K/32K/64K/128K/256K/512K. Chunk prefill B=8/16; prefill B=16,
maximum query lengths 8K/16K/32K. Training forward retains the selected CP16
rank fragments, 12288 query tokens per case. This workload involves no backward
or framework communication. Exact varlen lengths are in [cases.csv](cases.csv); input spans and per-H seeds
are in [inputs.json](../inputs.json). Main inputs use a CUDA Torch generator: draw
FP32 normal dt_bias, four BF16 normal Q/K/V/g arrays, then BF16 normal beta. A_log
and document initial states are zero. Seeds are fixed per case and H. The FP64
fixture uses seed `8841+length`, and decode cycle inputs use seeds 777/778.

Q/K/V/g/beta/output are BF16; gate parameters FP32. Each consistent contract
prepares its complete entry state from identical history inputs outside timing.
FlashInfer/FlashKDA receive the BF16 recurrent entry matrix. State/checkpoint
storage follows each implementation's contract.

## Timing

One complete call per fixed-address CUDA Graph, three replays per sample,
rotating implementation order, three warmup rounds and nine retained rounds.
All GPU operations required by the prepared API are included; compilation,
allocation, history preparation, capture and host overhead are excluded.

Q8 includes all eight outputs/checkpoints **and accepted-state selection/writeback**,
with device counts `request_index % 9`. BF16/FP32 use public `run()` + `commit()`.
FlashInfer uses native Q8 followed by the same dtype-preserving selection kernel.
CF-BI uses eight public Q1 calls, complete E/cache/position snapshots and PyTorch
device selection of all three state components. This is a fallback, not a native
optimized CF-BI Q8 kernel. Acceptance decision generation is excluded.

The main table uses compact state buffers. [Pool measurements](pool.csv) separately
compare indexed pools with the frozen compact API plus necessary state movement.
Previous Q8 data without selection are not directly comparable.

## Performance

Geometric mean of paired per-case latency ratios; lower is better.

| Workload | Reference = 1× | BF16 state | FP32 state | CF-BI C16 | CF-BI C64 |
|---|---|---:|---:|---:|---:|
| Decode Q1 | FlashInfer | 0.90× | 1.19× | 5.39× | 8.51× |
| Decode Q8 + commit | FlashInfer | 0.88× | 1.01× | 8.43× | 12.82× |
| Chunk prefill | FlashKDA | 2.70× | 2.37× | 8.80× | 7.15× |
| Prefill | FlashKDA | 3.50× | 2.83× | 9.09× | 5.25× |
| 192K forward | FlashKDA | 1.79× | 1.58× | 5.08× | 2.90× |

Arithmetic means of case medians, microseconds:

| Workload | H | BF16 state | FP32 state | CF-BI C16 | CF-BI C64 | Reference |
|---|---:|---:|---:|---:|---:|---:|
| Decode Q1 | 12 | 13.86 | 16.48 | 89.32 | 127.69 | 14.39 |
| Decode Q1 | 16 | 16.37 | 22.47 | 99.68 | 160.60 | 18.36 |
| Decode Q1 | 32 | 31.35 | 43.63 | 170.51 | 291.03 | 36.63 |
| Decode Q8 + commit | 12 | 86.25 | 97.47 | 934.79 | 1259.21 | 96.95 |
| Decode Q8 + commit | 16 | 109.63 | 127.04 | 1049.90 | 1614.41 | 124.73 |
| Decode Q8 + commit | 32 | 209.07 | 237.57 | 1744.35 | 2967.93 | 236.64 |
| Chunk prefill | 12 | 878.63 | 777.57 | 2854.26 | 2107.18 | 363.63 |
| Chunk prefill | 16 | 1144.32 | 953.17 | 3491.87 | 2605.03 | 395.93 |
| Chunk prefill | 32 | 2165.61 | 1823.74 | 6228.96 | 4713.80 | 672.00 |
| Prefill | 12 | 12411.87 | 9565.98 | 31565.26 | 18350.52 | 3812.89 |
| Prefill | 16 | 15411.99 | 12589.79 | 42051.26 | 23294.60 | 4277.42 |
| Prefill | 32 | 30642.44 | 25810.11 | 77770.18 | 46239.97 | 8385.28 |
| 192K forward | 12 | 792.20 | 702.07 | 2373.00 | 1266.88 | 580.11 |
| 192K forward | 16 | 1033.67 | 945.28 | 2707.30 | 1623.16 | 603.18 |
| 192K forward | 32 | 1514.05 | 1293.73 | 4520.09 | 2604.08 | 681.50 |

[summary.json](summary.json) also contains per-case extrema and a separate weighted
aggregate: sum `case_weight × latency` within each phase and measurement round,
divide by the reference sum, then take the median of nine ratios. No combined
decode/prefill score. All nine samples are preserved in CSV; sample-quality fields
report maximum relative MAD and sample spread.

## Indexed pools

462 pairs: all 231 cases × BF16/FP32. Same exact query lengths, nonzero random
logical initial states, pool capacity `2*B+1`, reversed odd slots, counts 0..8.
This supplemental experiment does not replay KV history. The frozen compact API
uses `index_select` before the call and `index_copy` afterwards (Q8 uses the same
selection helper); the new API addresses device slots directly. Both include
all required state movement and Q8 selection. Outputs, entire pools and all
checkpoints match bitwise before and after Graph replay.

| Workload | BF16 indexed / compact+movement | FP32 indexed / compact+movement |
|---|---:|---:|
| Decode Q1 | 0.226× | 0.250× |
| Decode Q8 + commit | 0.811× | 0.849× |
| Chunk prefill | 0.937× | 0.968× |
| Prefill | 1.012× | 1.000× |
| 192K forward | 1.016× | 1.026× |

## Decode positions and FP64 diagnostics

[phase-cases.csv](phase-cases.csv) covers all 64 starting positions × Q1/Q8 ×
H12/16/32 (384 cases), including chunk updates. `cycles` in the summary reports
arithmetic means across the complete 64-position cycle.

[accuracy.csv](accuracy.csv) and [accuracy-by-gate.csv](accuracy-by-gate.csv)
retain independent FP64 errors. H=3, D=128; lengths 1/8/64/256/4096/32768;
normal/open/saturated gates; shared nonzero BF16-representable initial states.
FP64 state never rounds to BF16. NRMSE=RMS(error)/RMS(reference);
max normalized error=max(abs(error))/max(abs(reference)).

Each implementation has 18 output records except FlashInfer, which has six Q1/Q8
records. CF state errors cover 12 closed-chunk cases; other state diagnostics
cover all their records. Unequal coverage is not an accuracy ranking. Errors are
diagnostics, not certified bounds. NaN/Inf and same-contract bitwise mismatches fail.
