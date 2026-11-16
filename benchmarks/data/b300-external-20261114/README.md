# Historical B300 external-reference comparisons, 2026-11-14

Frozen implementations measured on B300 SXM6 AC, driver 615.71.10,
Python 3.12.3, PyTorch 2.12.0+cu132, CUDA 13.2 and CuTe DSL 4.8.1.
Two-GPU NV18 topology was verified; performance measurements used one GPU serially.
These are historical snapshots, not new measurements of the current standalone package.

## Results

Geometric means of paired per-case latency ratios; lower is better.
BF16/reference is recomputed from the paired CSV values (equivalently,
FP32/reference divided by FP32/BF16 in the same experiment).

| Workload | Reference = 1× | BF16 state | FP32 state |
|---|---|---:|---:|
| Decode Q1 | FlashInfer | 0.85× | 1.35× |
| Decode Q8 | FlashInfer | 0.97× | 1.05× |
| Chunk prefill | FlashKDA | 2.71× | 2.33× |
| Prefill | FlashKDA | 3.47× | 2.84× |
| 192K forward | FlashKDA | 1.73× | 1.50× |

[Q1/forward cases](q1-forward-cases.csv) and [summary](q1-forward-summary.json)
contain 210 cases: Q1 21, chunk prefill 84, prefill 9, training forward 96.
The CSV column `candidate_us` means frozen FP32-state latency; `fp32_seed_us`
is an earlier FP32 reference, not included in the table above.
[Q8 cases](q8-cases.csv) and [summary](q8-summary.json) contain a separate 21-case
checkpoint experiment. Original summary statistics and per-case medians are preserved.

## Workloads and numerical contracts

H=12/16/32, D=128. Decode B=64, Q1/Q8, historical KV limits
8K/16K/32K/64K/128K/256K/512K. Chunk prefill uses B=8/16;
prefill uses B=16 and maximum lengths 8K/16K/32K, with explicit varlen lengths.
192K training forward uses 32 selected CP16 rank layouts, each totaling 12288 query
tokens, multiplied by three H settings. It includes neither backward nor communication.

Q/K/V/g/beta/output are BF16; gate parameters and computation are FP32 in the
recurrent implementations. Recurrent state/checkpoints follow the BF16 or FP32
contract. FlashInfer and FlashKDA use BF16 state storage. A shared FP32 entry
state is generated from the real historical prefix; BF16 paths round it at entry.
This compares the current call, not each implementation's complete historical trajectory.
Q8 includes all eight outputs and checkpoints. FlashInfer Q8 recurs internally in
FP32 and writes BF16 checkpoints; BF16 recurrent rounds state at each token.
Including FlashInfer/FlashKDA does not imply they satisfy our segmentation contract.

## Source snapshots

- BF16 recurrent: `364a0e515dd9e0247bfef168159933df5589d9e7`.
- FP32 frozen candidate SHA256:
  `ad59d7aad83e541c7cac7b6daf5a8edf3a228f5e3760056d0307c7ca55dadc7f`.
  Q8 uses its separately measured native checkpoint extension. These snapshots
  predate production cleanup; historical measurements do not certify the current API.
- FlashInfer: `c3af0c13328fd23b256c53b9b4f94d9e160d118c`.
- FlashKDA: `d7e3b3a9f448500630fa60fb5200f4f9c9b1fc72`;
  CUTLASS: `5c149f52a436782210263fb2f19b354443a61c6a`.
- Q1/forward experiment manifest SHA256:
  `731a1ffbeca53899897e14967d03c99f75c09946346823cff1191656706d37db`.
- Q8 experiment manifest SHA256:
  `a0fba1044b311447d75f06d3ab89a013bc1ee31a297425f540ed044d3f175bc7`.

## Timing and interpretation

Complete prepared GPU API, fixed-address hot-cache CUDA Graph, three replays per
sample, three warmup samples and nine retained samples, with rotating variant order.
Q1 uses 32 independent complete calls per graph; Q8 uses eight; forward uses one.
Reported latency is per complete call. Compilation, allocation, host preparation,
historical-prefix construction and acceptance selection are excluded.

The separate [B300 four-way comparison](../four-way-20261114/README.md) uses
one complete call per graph and prepares history under each implementation's own
contract. Do not multiply ratios across these experiments to infer unmeasured
CF-BI/FlashInfer or CF-BI/FlashKDA comparisons.
The [B200 comparison](../b200-20261116/README.md) uses different source snapshots
and a different PyTorch/CUDA stack; it is not a controlled hardware-only comparison.
The public smoke benchmark CLI does not reproduce these external-reference experiments.
