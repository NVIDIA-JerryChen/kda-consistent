# 数值契约

[English](numerics.en.md) · [API](api.md)

两种状态契约都使用 BF16 输入/输出，Q/K 归一化、gate、beta、投影、归约、delta、
状态乘加与累加使用 FP32。初等函数包含近似 exp2、rsqrt 和 reciprocal，不要求正确舍入。
不使用 BF16 投影中间结果或 TF32 矩阵运算。packed f32x2 是两个 FP32 运算。

每个 token 依次执行：状态衰减 → key 投影 → delta → 状态更新 → query 投影输出。
BF16 契约在更新后的 FP32 状态上计算输出，再将状态舍入 BF16 供下一 token 使用；
checkpoint 保存同一 BF16 续算状态。FP32 契约在 token 间及调用边界保持 FP32，
checkpoint 同样保持 FP32。完整有效续算状态只有状态矩阵，无开放块位置或历史行缓存。

同契约的 Q1、Q8、prefill 保持固定的运算/归约顺序与舍入边界；执行窗口和调度不改变
数值程序。同 GPU 型号、同软件栈下，测试要求有效输出、完整状态以及每个 checkpoint
在调用切分、PD、batch/重排、varlen、完整 head 分片和 eager/Graph 间逐位一致；NaN/Inf 失败。
这不要求与 FlashInfer、FlashKDA 或另一状态契约逐位相同，也不保证完整模型 logits。
切换状态 dtype 不能直接继承另一数值轨迹；调用方应在请求生命周期内保持契约。

`python -m tests.accuracy --state-dtype fp32 --length 32768 --gate open`
以独立 FP64 递推测量输出及状态误差，BF16 契约可用 `--state-dtype bf16`。
参考的内部状态不逐 token 舍入 BF16，因此会测出 BF16 状态量化导致的累计差异。
误差指标用于诊断，不以内部逐位相同替代数学检查，也不将有限样本最大值视为误差上限。

测量数据及对应源码、环境与计时范围：

- [B200 对比](../benchmarks/data/b200-20261116/README.md)：当前公开 API 与 FlashInfer、FlashKDA、CF-BI 的性能对比；[FP64 诊断明细](../benchmarks/data/b200-20261116/accuracy.csv)。
- [B300 历史外部对比](../benchmarks/data/b300-external-20261114/README.md)：冻结 BF16/FP32 实现相对 FlashInfer、FlashKDA 的性能。
- [B300 历史四路对比](../benchmarks/data/four-way-20261114/README.md)：冻结 BF16/FP32 与 CF-BI C16/C64 的性能；[FP64 诊断明细](../benchmarks/data/four-way-20261114/accuracy.csv)。

历史数据不代表当前源码的重新验收；不同版本或测量协议的数据不得拼接推导未测结果。
FP64 诊断的输入覆盖和状态比较边界以对应数据页为准，不跨不同覆盖范围直接比较最大误差。
