"""Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Adapted from FlashInfer c3af0c13328fd23b256c53b9b4f94d9e160d118c.
Changes: fix the D128 dense T=1 row16 dual-accumulator configuration,
remove unused optional metadata paths, and support separate input/output state.
Arithmetic, normalization, gate, reduction order and rounding are unchanged.

Shared arithmetic for the BF16-state contract.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import math as mlir_math


@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(), b.ir_value()],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _fma_rn(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(), b.ir_value(), c.ir_value()],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _prediction_dot8(state, row, key):
    # Match pinned FI's generated prediction contraction, which differs from
    # its output contraction: terms 2/3 round first, then terms 0/1 are fused.
    even = _mul_rn(state[row, 2], key[2])
    odd = _mul_rn(state[row, 3], key[3])
    even = _fma_rn(state[row, 0], key[0], even)
    odd = _fma_rn(state[row, 1], key[1], odd)
    for pair in cutlass.range_constexpr(2, 4):
        i = 2 * pair
        even = _fma_rn(state[row, i], key[i], even)
        odd = _fma_rn(state[row, i + 1], key[i + 1], odd)
    return even + odd


@cute.jit
def _output_dot8(state, row, query):
    even = _mul_rn(state[row, 0], query[0])
    odd = _mul_rn(state[row, 1], query[1])
    for pair in cutlass.range_constexpr(1, 4):
        i = 2 * pair
        even = _fma_rn(state[row, i], query[i], even)
        odd = _fma_rn(state[row, i + 1], query[i + 1], odd)
    return even + odd


DOT_REDUCTION_DUAL_ACCUM = 1

@cute.jit
def compute_gate_value(
    g_val,
    k_idx,
    A_log_val,
    gDtBias,
    h_K_offset,
    lower_bound_val,
    USE_GATE_IN_KERNEL: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    USE_LOWER_BOUND: cutlass.Constexpr[int],
):
    """Compute one exp(gate) value in registers.

    When USE_GATE_IN_KERNEL=0: g is pre-computed log-space gate, just exp it.
    When USE_GATE_IN_KERNEL=1, USE_LOWER_BOUND=0:
        g_log = -exp(A_log) * softplus(g + dt_bias); return exp(g_log)
    When USE_GATE_IN_KERNEL=1, USE_LOWER_BOUND=1:
        g_log = lower_bound * sigmoid(exp(A_log) * (g + dt_bias)); return exp(g_log)

    Args:
        g_val: raw or log-space gate scalar
        k_idx: K-dimension index for the optional dt_bias load
        A_log_val: exp(A_log) for this head (Float32), precomputed outside
        gDtBias: dt_bias tensor [H*K] Float32 (or dummy if HAS_DT_BIAS=0)
        h_K_offset: query_head_idx * K offset into dt_bias
        lower_bound_val: lower bound float (negative, e.g. -5.0)
        USE_GATE_IN_KERNEL: 0 = pre-computed gate, 1 = in-kernel gate
        HAS_DT_BIAS: 0 = no dt_bias, 1 = add dt_bias
        USE_LOWER_BOUND: 0 = softplus formula, 1 = lower_bound * sigmoid formula

    The optional dt_bias load remains at the token-local gate computation site.
    """
    if USE_GATE_IN_KERNEL == 1:
        if HAS_DT_BIAS == 1:
            g_val = g_val + gDtBias[h_K_offset + k_idx].to(cutlass.Float32)
        if USE_LOWER_BOUND == 1:
            LOG2_E = cutlass.Float32(1.4426950408889634)
            neg_A_log2e = -A_log_val * LOG2_E
            lb_log2e = lower_bound_val * LOG2_E
            one = cutlass.Float32(1.0)
            fm = mlir_arith.FastMathFlags.fast
            one_ir = one.ir_value()
            neg_A_ir = neg_A_log2e.ir_value()
            lb_ir = lb_log2e.ir_value()
            ag = mlir_arith.mulf(neg_A_ir, g_val.ir_value(), fastmath=fm)
            exp_neg = mlir_math.exp2(ag, fastmath=fm)
            denom = mlir_arith.addf(one_ir, exp_neg, fastmath=fm)
            sig = mlir_arith.divf(one_ir, denom, fastmath=fm)
            ls = mlir_arith.mulf(lb_ir, sig, fastmath=fm)
            g_val = mlir_math.exp2(ls, fastmath=fm)
        else:
            exp_g = cute.exp(g_val, fastmath=True)
            one = cutlass.Float32(1.0)
            log2_v = cute.log2(one + exp_g, fastmath=True)
            g_val = cute.exp2(-A_log_val * log2_v, fastmath=True)
    else:
        g_val = cute.exp(g_val, fastmath=True)
    return g_val


@cute.jit
def _dot8_row(lhs, row, rhs, reduction_schedule: cutlass.Constexpr[int]):
    """Evaluate an eight-element dot product with the selected dependency graph."""
    if cutlass.const_expr(reduction_schedule == DOT_REDUCTION_DUAL_ACCUM):
        even = lhs[row, 0] * rhs[0]
        odd = lhs[row, 1] * rhs[1]
        for pair in cutlass.range_constexpr(1, 4):
            i = 2 * pair
            even = lhs[row, i] * rhs[i] + even
            odd = lhs[row, i + 1] * rhs[i + 1] + odd
        return even + odd
    return (
        lhs[row, 0] * rhs[0]
        + lhs[row, 1] * rhs[1]
        + (lhs[row, 2] * rhs[2] + lhs[row, 3] * rhs[3])
        + (
            lhs[row, 4] * rhs[4]
            + lhs[row, 5] * rhs[5]
            + (lhs[row, 6] * rhs[6] + lhs[row, 7] * rhs[7])
        )
    )


@cute.jit
def _reduce_k_group(value, HEAD_DIM: cutlass.Constexpr[int]):
    """Reduce one partial dot product across the lanes that partition K."""
    if cutlass.const_expr(HEAD_DIM == 64):
        for offset in [4, 2, 1]:
            value = value + cute.arch.shuffle_sync_bfly(
                value, offset=offset, mask=4294967295
            )
    else:
        for offset in [8, 4, 2, 1]:
            value = value + cute.arch.shuffle_sync_bfly(
                value, offset=offset, mask=4294967295
            )
    return value

