# Numerical contracts

[简体中文](numerics.md) · [API](api.en.md)

Both contracts use BF16 inputs/outputs and FP32 Q/K normalization, gate, beta,
projections, reductions, delta, state multiply-add and accumulation. Elementary
functions include approximate exp2, rsqrt and reciprocal, not guaranteed correctly
rounded functions. There are no BF16 projection intermediates or TF32 matrix operations.
Packed f32x2 means two FP32 operations.

Each token applies state decay, key projection, delta, state update, then query
projection/output. The BF16 contract computes output from the updated FP32 state,
then rounds state to BF16 for the next token; checkpoints store that same BF16
continuation state. The FP32 contract retains FP32 between tokens and calls, including
checkpoints. Complete continuation state consists only of the state matrix, with no
open-chunk position or historical-row cache.

Q1/Q8/prefill within each contract preserve arithmetic/reduction order and rounding
boundaries; scheduling and workspace windows do not change that program. On the same
GPU model/software stack, tests require bitwise valid outputs, complete states and
all checkpoints across segmentation, PD, batching/reordering, varlen, full-head
sharding and eager/Graph execution. NaN/Inf fail. This does not require agreement with
FlashInfer, FlashKDA or the other state contract, or guarantee full-model logits.
Changing state dtype does not preserve the other trajectory; keep a contract for a
request's lifetime.

`python -m tests.accuracy --state-dtype fp32 --length 32768 --gate open` measures
output/state error against an independent FP64 recurrence. Use `--state-dtype bf16`
for BF16 state. The reference does not round its internal state to BF16 per token,
so it measures accumulated state-quantization differences. Metrics are diagnostics;
internal equality does not replace mathematical validation, and observed maxima are
not universal error bounds.

Measurements with their source versions, environments and timing scope:

- [B200 comparison](../benchmarks/data/b200-20261116/README.md): current public APIs versus FlashInfer, FlashKDA and CF-BI; [FP64 diagnostics](../benchmarks/data/b200-20261116/accuracy.csv).
- [B300 historical external comparisons](../benchmarks/data/b300-external-20261114/README.md): frozen BF16/FP32 implementations versus FlashInfer and FlashKDA.
- [B300 historical four-way comparison](../benchmarks/data/four-way-20261114/README.md): frozen BF16/FP32 versus CF-BI C16/C64; [FP64 diagnostics](../benchmarks/data/four-way-20261114/accuracy.csv).

Historical data are not a new acceptance run of the current source. Do not combine
different versions or measurement protocols to infer unmeasured results. Consult
each data page for FP64 input coverage and state-comparison boundaries; maxima
over different coverage sets are not directly comparable.
