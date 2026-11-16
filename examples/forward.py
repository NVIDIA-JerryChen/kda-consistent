"""Run packed prefill, Q1, and Q8 with either state dtype."""

import argparse
import torch
from kda_consistent import prepare_decode, prepare_prefill


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dtype", choices=["bf16", "fp32"], default="fp32")
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.state_dtype == "bf16" else torch.float32
    torch.manual_seed(7)
    h = 12
    a, bias = torch.zeros(h, device="cuda"), torch.zeros(h, 128, device="cuda")

    def inputs(shape):
        return [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)] + [
            torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
        ]

    xs = inputs((1, 19, h, 128))
    initial = torch.zeros(2, h, 128, 128, device="cuda", dtype=dtype)
    state = torch.empty_like(initial)
    out = torch.empty_like(xs[0])
    cu = torch.tensor([0, 2, 19], device="cuda", dtype=torch.int32)
    prepare_prefill(*xs, a, bias, initial, out, state, state_dtype=dtype, cu_seqlens=cu).run()

    xs = inputs((2, 1, h, 128))
    next_state, out = torch.empty_like(state), torch.empty_like(xs[0])
    prepare_decode(*xs, a, bias, state, out, next_state, state_dtype=dtype).run()

    xs = inputs((2, 8, h, 128))
    checkpoints = torch.empty(2, 8, h, 128, 128, device="cuda", dtype=dtype)
    plan = prepare_decode(*xs, a, bias, next_state, torch.empty_like(xs[0]),
                          state_dtype=dtype, checkpoint_states=checkpoints)
    out, terminal = plan.run()
    # Two requests accept zero and four tokens. Selection is outside the kernel API.
    accepted = torch.tensor([0, 4], device="cuda")
    rows = torch.arange(2, device="cuda")
    selected = torch.where((accepted > 0)[:, None, None, None],
                           checkpoints[rows, (accepted - 1).clamp_min(0)], next_state).contiguous()
    assert torch.isfinite(out).all() and torch.isfinite(selected).all()
    print(f"{args.state_dtype}: prefill/Q1/Q8 complete; terminal={tuple(terminal.shape)}")


if __name__ == "__main__":
    main()
