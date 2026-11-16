# K3/K3.1 runtime state management benchmark

- [B200](b200/README.md): paired external comparison, indexed pools, complete decode cycles and FP64 diagnostics.
- [B300](b300/README.md): the same workload and timing protocol.

Q8 now includes accepted-prefix state selection. Historical no-selection measurements
remain in their original data directories. The software stacks differ between GPUs;
these tables do not isolate hardware-only speedups.

## Existing API regression

[B300 paired results](b300-regression.csv), 231 cases × two state contracts.
The frozen pre-change source is identified by [baseline-sources.json](baseline-sources.json).
This check retains the old obligations: compact state and no Q8 commit.
Outputs, terminal states and all checkpoints match bitwise for all 462 pairs.
Ratios are new / old latency; all decode cases stay within the 10% regression limit.

| Workload | BF16 geomean | BF16 worst | FP32 geomean | FP32 worst |
|---|---:|---:|---:|---:|
| Decode Q1 | 0.996× | 1.042× | 1.000× | 1.034× |
| Decode Q8 | 0.999× | 1.013× | 1.000× | 1.006× |
| Chunk prefill | 0.974× | 1.043× | 1.012× | 1.182× |
| Prefill | 1.014× | 1.037× | 1.000× | 1.001× |
| 192K forward | 1.030× | 1.160× | 1.043× | 1.188× |

Some forward cases regress by up to 18.8%; the generic scheduling change is not a universal speedup. No overall forward regression threshold was set.
