# Benchmarks

Run from the installed repository checkout:

```bash
python -m benchmarks.bench --mode prefill --lengths 0,128,1024,4096 --heads 16 --state-dtype bf16
python -m benchmarks.bench --mode decode --batch 64 --query-length 1 --state-dtype fp32
python -m benchmarks.bench --mode decode --batch 64 --query-length 8 --state-dtype fp32
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
accept/reject selection are excluded. Fixed input/output addresses and repeated
CUDA Graph replay give hot-address timings, not cold-L2 or framework end-to-end
performance. Inputs are synthetic; explicit varlen lengths are not replaced by
dense inputs. The smoke cases above are not the full accepted workload matrix.

[B200 comparison](data/b200-20261116/README.md) adds paired measurements of the
current public BF16/FP32 APIs against FlashInfer, FlashKDA, and CF-BI C16/C64,
including per-case samples, decode positions, and FP64 diagnostics.

[Historical B300 external-reference data](data/b300-external-20261114/README.md)
provide the separate Q1/forward and Q8 comparisons against FlashInfer/FlashKDA.
Their Graph call counts and entry-state preparation differ from the four-way run.

[Historical four-way data](data/four-way-20261114/README.md) compare frozen recurrent
FP32/BF16 and open-source CF-BI C16/C64. They are retained as measured evidence,
not relabeled as timings of the current package. The complete comparative runner
and external CF-BI build are not shipped as dependencies of the kernel package.
