# Historical four-way benchmark, 2026-11-14

These CSV/JSON files preserve the original experiment (CSV line endings may be normalized by Git): B300 SXM6 AC, driver 615.71.10,
Python 3.12, PyTorch 2.12.0+cu132, CUDA 13.2, CuTe DSL 4.8.1. The node's two-GPU NVLink
topology was verified; timing used one GPU serially. They are **not new measurements
of this extracted package**.

BF16 source: `364a0e515dd9e0247bfef168159933df5589d9e7`.
FP32 source: frozen r22 candidate SHA256
`ad59d7aad83e541c7cac7b6daf5a8edf3a228f5e3760056d0307c7ca55dadc7f`
plus the Q8 checkpoint module SHA256
`3c11c6c97cbec05e990762996c23c767fe0501e473d1aa8abac912bb3fc6ec8f`.
This predates the FP32 cleanup commit. CF-BI source: Inferact/zero-mismatch-rl,
`vllm/kimi-k3.1`, commit `5735778f43f7d94108d432706dc5b17765bf723a`,
C16/v6 and C64/v5. The original CUDA kernels were unchanged.

231 cases: Q1 21, Q8 21, chunk prefill 84, prefill 9, 192K training-forward 96,
H=12/16/32, D=128. Decode B=64; prefill lengths are packed/varlen.
The training cases use selected CP16 rank fragments totaling 12288 query tokens,
without backward or communication. Q/K/V/g/beta/output BF16, gate parameters FP32;
state storage follows each contract.

Timings include the prepared complete GPU API and preservation of entry state.
CF state copies, gather/scatter, open-chunk completion and output packing are included.
Q8 includes all eight checkpoints; CF uses eight public Q1 calls plus complete
E/cache/position snapshots, **not an optimized native Q8 checkpoint kernel**.
Compilation, allocation, host metadata construction, prefix preparation and acceptance
selection are excluded. Each contract prepares its own state from the same history.
Fixed-address hot-cache Graphs: one API call per graph, three replays per sample,
rotating variant order, three warmup rounds then nine measured rounds. See cases.csv
for per-case median timings and lengths, phase-cases.csv for the 384-position decode-cycle diagnostic,
and summary.json for geometric ratios and separately weighted phase aggregates.
No aggregate mixes decode and prefill.

Accuracy: 18 common inputs per variant, H=3, lengths 1/8/64/256/4096/32768,
normal/open/saturated gates, shared nonzero BF16-representable initial state.
Independent FP64 recurrence; state errors compared only at common closed chunk
boundaries. NRMSE=RMS(error)/RMS(reference), max_normalized=max(abs(error))/max(abs(reference)),
denominators clamped at 1e-30. Output remains BF16. See accuracy.csv and
accuracy-by-gate.csv; observed maxima are diagnostics, not accepted error bounds.

These results compare different implementations and floating-point programs;
differences cannot be attributed solely to state storage dtype. The simple public
benchmark CLI does not reproduce this four-way experiment. No CF-BI consistency or
performance guarantee is added by including the historical data.
