# Benchmarks

Run from the installed repository checkout:

```bash
python -m benchmarks.bench --mode prefill --lengths 0,128,1024,4096 --heads 16 --state-dtype bf16
python -m benchmarks.bench --mode decode --batch 64 --query-length 1 --state-dtype fp32
python -m benchmarks.bench --mode decode --batch 64 --query-length 8 --state-dtype fp32 --state-pool --accept-counts 0,1,2,3,4,5,6,7,8
```

CLI defaults are prefill, `--lengths 0,128,1024,4096`, `--heads 16`,
`--state-dtype fp32`, `--samples 9`, and `--workspace-mib 256`.
Decode defaults to `--batch 64 --query-length 1`; `--lengths` applies only to prefill.
The workspace limit is passed to both prepare APIs and includes internal metadata,
but excludes caller-owned inputs, outputs and checkpoints. The CLI default is
256 MiB for both modes; the public decode API's own default is 64 MiB.
Q1 and FP32 Q8 need no internal device workspace; BF16 Q8 allocates coefficients
and indices subject to the limit. Insufficient capacity raises an error.

Each Graph contains one complete API call. After three eager warmup calls, the
CLI discards three timed warmup samples, then retains `--samples` samples.
Each sample replays the Graph **10 times**; elapsed GPU time is divided by 10
to report per-call latency. JSON output includes all retained samples and their median.
The published comparison experiments use **three replays per sample** and may
have different calls per Graph, as specified on each data page.

Times include the complete prepared GPU API, including all eight Q8 checkpoints.
Prepare/compile/allocation, historical-prefix preparation, host overhead, and
acceptance decision generation are excluded. `--accept-counts` includes device checkpoint
selection and pool writes via `commit()`; omitting it measures run only.
`--state-pool` uses permuted device slots in a pool larger than the batch. Fixed input/output addresses and repeated
CUDA Graph replay give hot-address timings, not cold-L2 or framework end-to-end
performance. Inputs are synthetic; explicit varlen lengths are not replaced by
dense inputs. The smoke cases above are not the full accepted workload matrix.

[Current B200/B300 comparison](data/k3-runtime-20261116/README.md) covers the same
231-case workload on both GPUs, including Q8 accepted-state selection, indexed-pool
A/B measurements, all decode positions and independent FP64 diagnostics.
Each platform retains its software versions, source hashes and all nine timing samples.

[Direct indexed-state comparison with FlashInfer](data/k3-flashinfer-pool-20261116/README.md)
covers Q1/Q8, dynamic slots, in-place/out-of-place writes and acceptance counts
0–8 on B200/B300. Both sides index the pool directly; this is separate from the
old-versus-new API comparison that includes legacy gather/scatter costs.

Historical results remain available: [B200](data/b200-20261116/README.md),
[B300 external references](data/b300-external-20261114/README.md) and
[B300 four-way comparison](data/four-way-20261114/README.md).
They predate the state-pool API and exclude accepted-state selection; their source
and Graph protocols are recorded on each page. The comparative runner and external
CF-BI build are not dependencies of the kernel package.
