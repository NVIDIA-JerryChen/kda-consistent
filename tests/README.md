# Tests

Run from the repository checkout after `pip install '.[test]'`:

```bash
python -m pytest -q tests
python -m tests.accuracy --state-dtype bf16 --length 4096 --gate saturated
python -m tests.accuracy --state-dtype fp32 --length 32768 --gate open
```

All consistency executions use the public `prepare_prefill` and `prepare_decode`
APIs for both BF16 and FP32 state. Tests compare valid outputs, full continuation
states and all Q8 checkpoints byte for byte; NaN/Inf fail.

- `test_api.py`: Q1/Q8/prefill, acceptance counts 0..8, dynamic Graph metadata,
  empty decode, input ownership and invalid contracts.
- `test_prefill.py`: arbitrary varlen splits, empty requests, padding, in-place
  state and batch changes/reordering.
- `test_state_pool.py`: direct pool reads/writes, dynamic slots, acceptance 0..8,
  checkpoint commit, continuation, Graph replay, untouched slots and pool-capacity cache reuse.
- `test_decode.py`: Q8 checkpoints, acceptance/continuation, batch reordering
  and bounded coefficient workspace.
- `test_consistency.py`: every PD cut in a short sequence, gate extremes,
  nonzero initial states, and 4K/32K prefill followed by Q8/Q1 continuation.
- `test_graph_cache.py`: dynamic metadata, CUDA Graph replay, runtime scalar
  changes and compilation-cache reuse.
- `test_multigpu.py`: complete-head shards and PD state handoff on two GPUs.
- `test_validation.py`: scalar, dtype, alignment, aliasing, no_grad and empty
  decode contracts.

Private cache counters are inspected only to verify compilation reuse; tests
do not execute private forward interfaces or compare against legacy kernels.
Two-GPU tests skip without a second GPU. Read pass/skip counts: skips do not
prove coverage. These are no_grad kernel checks, not backward, model-logit or
framework-communication tests.

`fp64_reference.py` is an independent Triton mathematical recurrence. It keeps
FP64 state without per-token BF16 rounding and imports no production arithmetic.
The accuracy command measures output/state error for either public contract.
Metrics are diagnostics, not universal error bounds; nonfinite results fail.
