# B300 comparison, 2026-11-16

NVIDIA B300, SM103; driver 615.71.10.
Two GPUs with NV18 connectivity; serial timing on one GPU.
Python 3.12.3 · CuTe DSL 4.8.1 · CUDA 13.2 · PyTorch 2.12.0+cu132.
224 tests passed with no skips, including dynamic metadata, all Q8 checkpoints,
accepted-prefix continuation, Graph replay and two-GPU complete-head sharding.

## Versions and workload

BF16/FP32 use the same current package on both GPUs: exact hashes in
[package-sources.json](package-sources.json), aggregate and experiment hashes in
[summary.json](summary.json). References:

- [FlashInfer c3af0c13328fd23b256c53b9b4f94d9e160d118c](https://github.com/flashinfer-ai/flashinfer/commit/c3af0c13328fd23b256c53b9b4f94d9e160d118c): Q1 and native Q8 checkpoints.
- [FlashKDA d7e3b3a9f448500630fa60fb5200f4f9c9b1fc72](https://github.com/MoonshotAI/FlashKDA/commit/d7e3b3a9f448500630fa60fb5200f4f9c9b1fc72), [CUTLASS 5c149f52a436782210263fb2f19b354443a61c6a](https://github.com/NVIDIA/cutlass/commit/5c149f52a436782210263fb2f19b354443a61c6a): packed forward.
- [CF-BI 5735778f43f7d94108d432706dc5b17765bf723a](https://github.com/Inferact/zero-mismatch-rl/commit/5735778f43f7d94108d432706dc5b17765bf723a): C16/v6 and C64/v5, branch `vllm/kimi-k3.1`; unchanged CUDA compiled for sm_103a.

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
| Decode Q1 | FlashInfer | 0.87× | 1.23× | 4.95× | 8.21× |
| Decode Q8 + commit | FlashInfer | 0.89× | 1.03× | 8.00× | 12.61× |
| Chunk prefill | FlashKDA | 2.73× | 2.38× | 8.40× | 6.63× |
| Prefill | FlashKDA | 3.54× | 2.86× | 9.09× | 5.27× |
| 192K forward | FlashKDA | 1.80× | 1.59× | 4.97× | 2.78× |

Arithmetic means of case medians, microseconds:

| Workload | H | BF16 state | FP32 state | CF-BI C16 | CF-BI C64 | Reference |
|---|---:|---:|---:|---:|---:|---:|
| Decode Q1 | 12 | 12.24 | 16.52 | 78.33 | 118.51 | 14.33 |
| Decode Q1 | 16 | 15.40 | 22.07 | 88.26 | 149.11 | 16.92 |
| Decode Q1 | 32 | 29.67 | 43.72 | 151.58 | 269.38 | 35.52 |
| Decode Q8 + commit | 12 | 82.82 | 95.10 | 839.55 | 1184.74 | 92.28 |
| Decode Q8 + commit | 16 | 106.14 | 124.05 | 956.65 | 1514.52 | 118.92 |
| Decode Q8 + commit | 32 | 202.20 | 234.00 | 1589.36 | 2787.66 | 227.08 |
| Chunk prefill | 12 | 850.33 | 746.74 | 2612.94 | 1878.25 | 347.54 |
| Chunk prefill | 16 | 1104.22 | 916.76 | 3221.45 | 2339.40 | 378.28 |
| Chunk prefill | 32 | 2089.91 | 1756.67 | 5834.43 | 4274.10 | 640.76 |
| Prefill | 12 | 11991.48 | 9232.99 | 30082.06 | 17561.27 | 3638.39 |
| Prefill | 16 | 14847.76 | 12147.50 | 40079.54 | 22303.91 | 4081.26 |
| Prefill | 32 | 29548.29 | 24674.76 | 74188.40 | 44278.93 | 7993.52 |
| 192K forward | 12 | 764.48 | 669.99 | 2210.05 | 1140.58 | 553.55 |
| 192K forward | 16 | 994.84 | 906.00 | 2523.81 | 1483.23 | 575.39 |
| 192K forward | 32 | 1458.08 | 1245.59 | 4243.32 | 2421.95 | 650.37 |

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
| Decode Q1 | 0.234× | 0.252× |
| Decode Q8 + commit | 0.807× | 0.850× |
| Chunk prefill | 0.937× | 0.967× |
| Prefill | 1.012× | 1.001× |
| 192K forward | 1.017× | 1.026× |

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
