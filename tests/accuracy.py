"""Compare either public state contract with an independent full FP64 recurrence."""
import argparse
import json
import torch
from kda_consistent import prepare_prefill
from .fp64_reference import run as reference
from .helpers import case

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dtype", choices=["bf16", "fp32"], default="fp32")
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--gate", choices=["normal", "open", "saturated"], default="normal")
    args = parser.parse_args()
    if args.length < 1 or args.heads < 1:
        parser.error("length and heads must be positive")
    dtype = torch.bfloat16 if args.state_dtype == "bf16" else torch.float32
    x = case([args.length], h=args.heads)
    x["state_in"] = x["state_in"].to(dtype)
    if args.gate != "normal":
        x["g"].fill_(-1000)
        if args.gate == "saturated":
            x["g"][..., ::2] = 1000
            x["beta"][:, ::2] = 1000
            x["beta"][:, 1::2] = -1000
        x["A_log"].zero_()
        x["dt_bias"].zero_()
    out, state = torch.empty_like(x["q"]), torch.empty_like(x["state_in"])
    prepare_prefill(**x, out=out, state_out=state, state_dtype=dtype).run()
    ref = reference(*[x[k] for k in ("q", "k", "v", "g", "beta", "A_log", "dt_bias", "state_in", "cu_seqlens")], full_precision=True)
    result = {}
    for name, value in [("output", out), ("state_out", state)]:
        want = ref[name]
        assert torch.isfinite(value).all() and torch.isfinite(want).all()
        diff = value.double() - want
        result[name] = dict(nrmse=float(diff.square().mean().sqrt() / want.square().mean().sqrt().clamp_min(1e-300)), max_normalized=float(diff.abs().max() / want.abs().max().clamp_min(1e-300)))
    print(json.dumps(dict(state_dtype=args.state_dtype, length=args.length, heads=args.heads, gate=args.gate, metrics=result), indent=2))

if __name__ == "__main__":
    main()
