"""Validate the public buffer and scalar contracts without legacy entry points."""
import pytest
import torch
from kda_consistent import prepare_decode, prepare_prefill
from .helpers import bits, case, prefill

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32]),
]


@pytest.mark.parametrize("mode", ["prefill", "q1", "q8"])
def test_buffer_scalar_and_grad_contract(dtype, mode):
    x = case([1 if mode == "q1" else 8], dtype, h=3)
    x.pop("cu_seqlens")
    original = x["state_in"].clone()
    out = torch.empty_like(x["q"])
    state = torch.empty_like(x["state_in"])
    cp = torch.empty(1, 8, 3, 128, 128, device="cuda", dtype=dtype)
    def prepare(inputs=None, **kwargs):
        options = dict(out=out, state_dtype=dtype)
        if mode == "q8":
            options["checkpoint_states"] = cp
        else:
            options["state_out"] = state
        options.update(kwargs)
        return (prepare_prefill if mode == "prefill" else prepare_decode)(**(x if inputs is None else inputs), **options)
    for kwargs in [dict(eps=0), dict(eps=1e-100), dict(scale=float("inf")),
                   dict(lower_bound=1), dict(lower_bound=-6), dict(scale=1e100)]:
        with pytest.raises(ValueError):
            prepare(**kwargs)
    for key in ("q", "k", "v", "g", "beta", "A_log", "dt_bias", "state_in"):
        with pytest.raises(ValueError):
            prepare(dict(x, **{key:x[key].double()}))
    with pytest.raises(ValueError, match="overlapping"):
        prepare(out=x["q"])
    unaligned = torch.empty(x["q"].numel()+1, device="cuda", dtype=torch.bfloat16)[1:].view_as(x["q"])
    with pytest.raises(ValueError, match="align"):
        prepare(dict(x, q=unaligned))
    with pytest.raises(ValueError, match="contiguous"):
        prepare(dict(x, q=torch.empty(*x["q"].shape[:-1], 256, device="cuda", dtype=torch.bfloat16)[..., ::2]))
    # Grad-enabled data must be rejected at prepare and at a later run.
    grad_x = dict(x, q=x["q"].clone().requires_grad_())
    with pytest.raises(RuntimeError, match="no_grad"):
        prepare(grad_x)
    with torch.no_grad():
        plan = prepare(grad_x)
        plan.run()
    with pytest.raises(RuntimeError, match="no_grad"):
        plan.run()
    bits(x["state_in"], original)
    if mode == "q1":
        expected = state.clone()
        prepare(state_out=x["state_in"]).run()
        bits(x["state_in"], expected)


def test_api_errors_and_empty_decode(dtype):
    x = case([8], dtype, h=3)
    for patch in [
        dict(state_in=x["state_in"].double()),
        dict(cu_seqlens=x["cu_seqlens"].long()),
    ]:
        with pytest.raises(ValueError):
            prefill(dict(x, **patch))
    with pytest.raises(ValueError):
        prefill(x, workspace_limit_bytes=0)
    with pytest.raises(ValueError):
        prefill(x, eps=0)
    with pytest.raises(ValueError):
        prepare_prefill(state_dtype=dtype, **x, out=x["q"], state_out=torch.empty_like(x["state_in"]))
    for q in [1, 8]:
        d = {
            k: torch.empty(
                (0, q, 3, 128) if k != "beta" else (0, q, 3),
                device="cuda",
                dtype=torch.bfloat16,
            )
            for k in ("q", "k", "v", "g", "beta")
        }
        d.update(A_log=x["A_log"], dt_bias=x["dt_bias"], state_in=x["state_in"][:0])
        kw = (
            dict(state_out=torch.empty_like(d["state_in"]))
            if q == 1
            else dict(
                checkpoint_states=torch.empty(
                    0, 8, 3, 128, 128, device="cuda", dtype=dtype
                )
            )
        )
        plan = prepare_decode(state_dtype=dtype, **d, out=torch.empty_like(d["q"]), **kw)
        a, b = plan.run()
        assert a.numel() == b.numel() == 0

