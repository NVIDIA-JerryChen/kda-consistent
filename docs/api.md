# API 与数据契约

[English](api.en.md) · [首页](../README.zh-CN.md)

公开导出：`prepare_prefill`、`prepare_decode`、`PrefillPlan`、`DecodePlan`。
两个 prepare 的前九个参数相同：`q,k,v,g,beta,A_log,dt_bias,state_in,out`。
`state_dtype` 为必填 keyword，必须与 state/checkpoint dtype 一致；不做隐式转换。
所有 tensor 在同一 CUDA device，连续，非空地址对齐 32 字节。D=128、H>0。
状态最后两维为 V,K。

| 参数 | Prefill | Decode | dtype |
|---|---|---|---|
| q/k/v/g/out | dense `[B,L,H,128]` 或 packed `[1,T,H,128]` | `[B,Q,H,128]`，Q=1/8 | BF16 |
| beta | 对应前三维 | `[B,Q,H]` | BF16 |
| A_log / dt_bias | `[H]` / `[H,128]` | 同左 | FP32 |
| state_in | `[N,H,128,128]`，dense N=B | `[B,H,128,128]` | state_dtype |
| state_out | 必填，同 state_in | Q1 必填；Q8 启用 commit 时必填，否则 None | state_dtype |
| checkpoint_states | 不支持 | Q1 不支持；Q8 必填 `[B,8,H,128,128]` | state_dtype |
| cu_seqlens | packed Int32 `[N+1]`，dense 省略 | 不使用 | CUDA Int32 |
| state_indices | 可选 `[N]` | 可选 `[B]` | CUDA Int32 |
| num_accepted_tokens | 不支持 | Q8 可选 `[B]`，值为 0～8 | CUDA Int32 |

Q/K 在内部 L2 归一化；g 和 beta 为激活前值。默认 `scale=128**-0.5`、`eps=1e-6`、
`lower_bound=-5.0`；eps 在 FP32 中须大于零，lower_bound 范围 `[-5,0]`，所有标量须为有限 FP32。
`workspace_limit_bytes` 默认为 prefill 256 MiB、decode 64 MiB，包含内部 metadata；
不足以容纳最小执行窗口时报错。Q1 和 FP32 Q8 不使用内部设备 workspace，BF16 Q8 需要系数及索引空间。

Prefill 最多 1023 请求；容量/grid 有 Int32 边界检查。空 batch、空请求有效。
`cu_seqlens` 从零开始、非递减、末项不超过 T；尾部无效输出不写入。
metadata 内容由调用方保证，API 不读取到 host 检查。历史 KV 已包含在入口状态中；
不传历史 KV 长度。非连续序列片段必须分别提供正确入口状态。

## 状态与 checkpoint

`run()` 返回 `(out,state_out)`。Q8 的第二项为 `checkpoint_states[:, -1]`，通常不连续。
再次作为入口前使用 `.contiguous()` 或复制到连续 buffer。
提供 `state_indices` 时，state_in/state_out 改为同容量 `[P,H,128,128]` 状态池，`N≤P<2³¹`（decode 的 N=B）。
请求 i 直接读写 `state_indices[i]`，无需随 batch 重排搬运状态；输出与 checkpoint 仍按请求顺序排列。
slot 必须在 `[0,P)` 内，同次调用的 slot 必须唯一。未被索引的 slot 保持不变。
Q1/prefill 允许输入和输出状态池完全相同；禁止部分重叠及其他 buffer 别名。

Q8 始终生成全部八个 checkpoint。默认 `state_out=None`；启用提交时，同时传入
`state_out` 和 `num_accepted_tokens`。`run()` 不修改状态池；接受数在设备端准备好后，
调用 `plan.commit()`：接受 0 个时保留/复制原始状态，否则复制第 a−1 个 checkpoint。
`commit()` 返回目标池，也可通过 `plan.accepted_state` 访问。

```python
plan = prepare_decode(
    q, k, v, g, beta, A_log, dt_bias, state_pool, out, state_pool,
    state_dtype=torch.float32, checkpoint_states=checkpoints,
    state_indices=slots, num_accepted_tokens=accepted,
)
plan.run()
# Fill accepted on the same CUDA stream, then commit the accepted prefix.
plan.commit()
```

checkpoint 与状态池必须独立。run 到 commit 之间不能改变 slot 映射、原始状态或 checkpoint；
允许在此期间更新接受数。原地提交后，再次 run 从已提交状态继续；
非原地提交须将目标池作为后续 plan 的入口。
写入某个 slot 时，不得与其他调用对该 slot 的读取或写入并发；
跨 stream 依赖由调用方用 event 等方式同步。

## 生命周期、Graph 与编译

prepare 在 CUDA Graph capture 外校验、编译并绑定固定容量/地址，分配有上限的 workspace。
run/commit 不分配设备内存、不编译、不 D2H，向绑定 device 的 current stream 提交。
输入、标量和所有 buffers/plan 在 GPU 完成前须存活；plan 的输入内容及合法设备 metadata
可以更新，改变地址/shape/标量需重新 prepare。跨 stream 依赖由调用方同步。
同一 plan/workspace/output 不得并发复用。Graph replay 原地写状态时会推进状态。

Prefill 不按请求长度选择编译实例；B/H/token 数和 metadata 内容保持 runtime。
Decode 按 Q 和有限配置选择实现，不按精确 batch 或历史 KV 长度特化。
无效类型/shape/dtype/device/alignment/alias/标量抛出 TypeError 或 ValueError；
capture 内 prepare、启用 autograd 且输入需要梯度时抛出 RuntimeError。
有效输入/结果必须有限，run 不额外扫描 NaN/Inf。支持范围为 no_grad forward。
