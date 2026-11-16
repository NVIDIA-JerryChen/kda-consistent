# KDA Consistent

[English](README.md)

独立的 KDA no_grad forward kernel，支持 **BF16 或 FP32 续算状态**。
同 GPU 型号、同软件栈、同数值契约内，prefill、Q1/Q8 decode、请求组合/重排、
任意切分及完整 head 分片保持有效输出与完整续算状态逐位一致。
两种状态契约不要求相互逐位一致；不包含 backward、完整模型 logits、框架接入或通信。

## 安装

目标为 SM100/SM103 Blackwell；使用 Python 3.12、CuTe DSL 4.8.1，已验证
B300（PyTorch 2.12.0+cu132、CUDA 13.2）和 B200
（PyTorch 2.14.0a0+b2c75dd062.nv26.09、CUDA 13.4）。其他环境需独立验证。
先安装与 CUDA 环境匹配的 PyTorch，再在本仓库执行：

```bash
python -m pip install '.[test]'
python -m examples.forward --state-dtype bf16
python -m examples.forward --state-dtype fp32
```

Python 包可在无 GPU 环境构建；首次 prepare 在目标 GPU 编译 CuTe DSL kernel。
Triton 仅用于独立 FP64 精度诊断。

## API

```python
import torch
from kda_consistent import prepare_prefill, prepare_decode

# Buffers are caller-owned CUDA tensors; see the runnable example above.
plan = prepare_prefill(
    q, k, v, g, beta, A_log, dt_bias, state_in, out, state_out,
    state_dtype=torch.float32, cu_seqlens=cu_seqlens,
)
with torch.no_grad():
    output, terminal = plan.run()
```

`state_dtype` 必填：`torch.bfloat16` 或 `torch.float32`；不隐式转换状态。
Q/K/V/g/beta/output 为 BF16，gate 参数为 FP32。
Q1 输出终态，Q8 必须输出全部八个 checkpoint，并返回最后一个 checkpoint 的 view。
调用方负责按接受数选择续算状态。Packed varlen prefill 使用设备 Int32 `cu_seqlens`；
decode 每请求 Q 固定。

- [完整 API 与生命周期](docs/api.md)
- [数值契约与验收](docs/numerics.md)
- [测试](tests/README.md)与 [benchmark / 历史数据](benchmarks/README.md)
- [源码归属](provenance/README.md)

## 性能

### B200

[B200 对比与原始数据](benchmarks/data/b200-20261116/README.md)：
231 个 varlen case，H=12/16/32、D=128，decode B=64。
下表为逐 case **延迟比的几何平均**，decode 相对 FlashInfer、forward 相对 FlashKDA；越小越好。

| 场景 | 对照 = 1× | BF16 state | FP32 state | CF-BI C16 | CF-BI C64 |
|---|---|---:|---:|---:|---:|
| Decode Q1 | FlashInfer | 0.90× | 1.19× | 5.37× | 8.50× |
| Decode Q8 | FlashInfer | 0.93× | 0.98× | 8.84× | 12.18× |
| Chunk prefill | FlashKDA | 2.71× | 2.35× | 8.80× | 7.15× |
| Prefill | FlashKDA | 3.45× | 2.83× | 9.09× | 5.24× |
| 192K forward | FlashKDA | 1.73× | 1.52× | 5.09× | 2.90× |

计时覆盖完整 prepared GPU API，采用固定地址热缓存 CUDA Graph；Q8 包含全部八个 checkpoint。
CF-BI Q8 使用八次公开 Q1 调用并保存完整状态快照。B200 使用
PyTorch 2.14.0a0 / CUDA 13.4 / CuTe DSL 4.8.1。完整版本、计时边界、
逐 case 数据、decode 完整周期和 FP64 误差见数据页。

### B300：历史对比

B300 SXM6 AC、PyTorch 2.12.0+cu132、CUDA 13.2、CuTe DSL 4.8.1；
H=12/16/32、D=128，decode B=64。以下使用生产清理前的冻结源码，
不是当前独立 repo 的重新测量。

[外部参考对比](benchmarks/data/b300-external-20261114/README.md)：
Q1/forward 与单独的 Q8 checkpoint 测量；数值为延迟比几何平均，
以 FlashInfer 或 FlashKDA 为 1×。

| 场景 | 对照 = 1× | BF16 state | FP32 state |
|---|---|---:|---:|
| Decode Q1 | FlashInfer | 0.85× | 1.35× |
| Decode Q8 | FlashInfer | 0.97× | 1.05× |
| Chunk prefill | FlashKDA | 2.71× | 2.33× |
| Prefill | FlashKDA | 3.47× | 2.84× |
| 192K forward | FlashKDA | 1.73× | 1.50× |

[另一轮四路对比](benchmarks/data/four-way-20261114/README.md)：
231 个 case，以下以 **BF16 state 为 1×**。

| 场景 | BF16 state = 1× | FP32 state | CF-BI C16 | CF-BI C64 |
|---|---:|---:|---:|---:|
| Decode Q1 | 1.00× | 1.39× | 5.68× | 9.42× |
| Decode Q8 | 1.00× | 1.07× | 8.79× | 12.55× |
| Chunk prefill | 1.00× | 0.86× | 3.09× | 2.44× |
| Prefill | 1.00× | 0.82× | 2.62× | 1.52× |
| 192K forward | 1.00× | 0.87× | 2.86× | 1.60× |

两批均计入完整 prepared GPU API，采用固定地址热缓存 CUDA Graph，Q8 包含全部 checkpoint。
CF-BI Q8 使用八次公开 Q1 调用并保存完整状态快照。
外部对比每图包含 32 次 Q1 或八次 Q8，四路对比每图一次调用；入口状态准备方式也不同。
不能跨表相乘推导未测对比。当前独立 repo 尚未在 B300 上按本次 B200 口径重测，
这些数据也不能直接用于比较 B200/B300 的纯硬件差异。

## 许可证

许可证与第三方声明见 [LICENSE](LICENSE) 和 [licenses/](licenses/)。
