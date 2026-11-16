# KDA Consistent

[English](README.md)

KDA forward kernel，支持 **BF16/FP32 续算状态**及 prefill/decode 逐位一致。
**精度优先推荐 FP32 state**（`state_dtype=torch.float32`），避免逐 token 的 BF16 状态量化误差。

## 安装

CuTe DSL 4.8.1+，支持 SM100/SM103。

```bash
python -m pip install '.[test]'
python -m examples.forward --state-dtype bf16
python -m examples.forward --state-dtype fp32
```

首次 `prepare` 编译 kernel。

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
支持设备端 `state_indices` 状态池索引；Q8 可用 `commit()` 按设备端接受数提交续算状态。
Packed varlen prefill 使用设备 Int32 `cu_seqlens`；
decode 每请求 Q 固定。

- [完整 API 与生命周期](docs/api.md)
- [数值契约与验收](docs/numerics.md)
- [测试](tests/README.md)与 [benchmark / 历史数据](benchmarks/README.md)
- [源码归属](provenance/README.md)

## 训推一致性与 CP 范围

同 GPU 型号、同软件栈、同状态契约内，prefill、Q1/Q8、分段/PD、batch/重排、varlen 和
eager/CUDA Graph 的输出、完整状态及 checkpoint 逐位一致。BF16/FP32 两种契约之间不要求
逐位相同。

Prefill 和 decode 使用相同的逐 token 递推、运算/归约顺序和舍入边界。
BF16 state 用更新后的 FP32 状态计算输出，再将状态舍入 BF16 供下一 token 使用；
FP32 state 在 token 间和调用边界保持 FP32。调度不改变该数值程序。

支持 Ulysses Parallel（UP）所需的完整 head 分片，不支持 Ring CP。
调用方负责通信和框架接入，输入、gate 参数和状态按相同 head 切分。

## 性能

231 个 case：H=12/16/32、D=128、decode B=64，forward 使用 varlen。
表中为延迟比几何平均，越小越好，基线见表头。计时覆盖完整 prepared GPU API，
每图一次完整调用，采用固定地址热缓存 CUDA Graph；Q8 包含全部 checkpoint、接受状态选择与写回。
CF-BI Q8 使用八次 Q1 调用和完整状态快照。编译、历史状态准备及接受数决策不计时。

### 对照实现版本

B200/B300 使用以下参考版本。

| 实现 | 仓库 | Commit | 测量路径 |
|---|---|---|---|
| FlashInfer | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | [c3af0c13328fd23b256c53b9b4f94d9e160d118c](https://github.com/flashinfer-ai/flashinfer/commit/c3af0c13328fd23b256c53b9b4f94d9e160d118c) | Q1 单 token；原生 Q8 grouped checkpoint |
| FlashKDA | [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) | [d7e3b3a9f448500630fa60fb5200f4f9c9b1fc72](https://github.com/MoonshotAI/FlashKDA/commit/d7e3b3a9f448500630fa60fb5200f4f9c9b1fc72) | 原始 packed forward |
| CF-BI | [Inferact/zero-mismatch-rl](https://github.com/Inferact/zero-mismatch-rl) | [5735778f43f7d94108d432706dc5b17765bf723a](https://github.com/Inferact/zero-mismatch-rl/commit/5735778f43f7d94108d432706dc5b17765bf723a) | C16/v6、C64/v5；分支 `vllm/kimi-k3.1` |

FlashKDA 使用的 [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass) 子模块版本为
[5c149f52a436782210263fb2f19b354443a61c6a](https://github.com/NVIDIA/cutlass/commit/5c149f52a436782210263fb2f19b354443a61c6a).

### B200

CuTe DSL 4.8.1 · CUDA 13.4 · PyTorch 2.14.0a0+b2c75dd062.nv26.09

[数据与环境](benchmarks/data/k3-runtime-20261116/b200/README.md)。

| 场景 | 对照 = 1× | BF16 state | FP32 state | CF-BI C16 | CF-BI C64 |
|---|---|---:|---:|---:|---:|
| Decode Q1 | FlashInfer | 0.90× | 1.19× | 5.39× | 8.51× |
| Decode Q8 + commit | FlashInfer | 0.88× | 1.01× | 8.43× | 12.82× |
| Chunk prefill | FlashKDA | 2.70× | 2.37× | 8.80× | 7.15× |
| Prefill | FlashKDA | 3.50× | 2.83× | 9.09× | 5.25× |
| 192K forward | FlashKDA | 1.79× | 1.58× | 5.08× | 2.90× |

### B300

CuTe DSL 4.8.1 · CUDA 13.2 · PyTorch 2.12.0+cu132

[数据与环境](benchmarks/data/k3-runtime-20261116/b300/README.md)。

| 场景 | 对照 = 1× | BF16 state | FP32 state | CF-BI C16 | CF-BI C64 |
|---|---|---:|---:|---:|---:|
| Decode Q1 | FlashInfer | 0.87× | 1.23× | 4.95× | 8.21× |
| Decode Q8 + commit | FlashInfer | 0.89× | 1.03× | 8.00× | 12.61× |
| Chunk prefill | FlashKDA | 2.73× | 2.38× | 8.40× | 6.63× |
| Prefill | FlashKDA | 3.54× | 2.86× | 9.09× | 5.27× |
| 192K forward | FlashKDA | 1.80× | 1.59× | 4.97× | 2.78× |

[状态池完整 API 与改动前回归对比](benchmarks/data/k3-runtime-20261116/README.md)。

### 状态池 Decode 与 FlashInfer

双方直接索引状态池，B=64、H=12/16/32；Q8 包含全部 checkpoint 与接受状态写回。
两台机器均使用 CuTe DSL 4.8.1 / CUDA 13.4 / PyTorch 2.14.0a0+b2c75dd062.nv26.09。
下表为相对 FlashInfer 的延迟比几何平均，越小越好。

| GPU | 状态写回 | Q1 BF16 | Q1 FP32 | Q8 BF16 | Q8 FP32 |
|---|---|---:|---:|---:|---:|
| B200 | 非原地 | 0.90× | 1.22× | 0.79× | 1.00× |
| B200 | 原地 | 0.92× | 1.20× | 0.79× | 1.00× |
| B300 | 非原地 | 0.92× | 1.25× | 0.79× | 1.03× |
| B300 | 原地 | 0.95× | 1.24× | 0.80× | 1.03× |

FlashInfer Q8 使用相同的状态复制 helper 完成目标池写回；不比较省略物理写回、
仅更新后续 checkpoint 索引的集成方式。FP32 Q1 原地 H=32 的延迟比分别为 1.41×/1.47×。
[计时口径、按 H 汇总和逐 case 数据](benchmarks/data/k3-flashinfer-pool-20261116/README.md)。

## 数学精度

相对独立 FP64 递推的逐 case 误差最大值，取自
[B200 数据](benchmarks/data/k3-runtime-20261116/b200/accuracy.csv)
（[按 gate 分类](benchmarks/data/k3-runtime-20261116/b200/accuracy-by-gate.csv)）。
H=3、D=128；长度 1/8/64/256/4096/32768；normal/open/saturated gate；
相同的非零 BF16 可表示初态。输出为 BF16，参考保持 FP64 状态、不做 BF16 舍入。

NRMSE = RMS(error) / RMS(reference)；最大归一化误差 =
max(abs(error)) / max(abs(reference))。数值为比值，不是百分比。

| 实现 | 输出 NRMSE | 输出最大归一化误差 | 状态 NRMSE | 状态最大归一化误差 |
|---|---:|---:|---:|---:|
| BF16 state | 2.197e-02 | 2.569e-02 | 2.195e-02 | 4.149e-02 |
| FP32 state | 1.739e-03 | 3.588e-03 | 3.575e-07 | 7.516e-07 |
| CF-BI C16 | 1.739e-03 | 3.588e-03 | 6.517e-07 | 1.254e-06 |
| CF-BI C64 | 1.739e-03 | 3.588e-03 | 1.626e-06 | 3.123e-06 |
| FlashInfer | 1.739e-03 | 2.636e-03 | 1.676e-03 | 3.202e-03 |
| FlashKDA | 8.029e-03 | 9.402e-03 | 7.440e-03 | 1.189e-02 |

FlashInfer 覆盖六个 Q1/Q8 case，其余实现各有 18 个输出 case。
CF-BI 状态误差仅覆盖长度 64/256/4096/32768 的 12 个闭合 chunk case；
BF16 state、FP32 state 和 FlashKDA 各有 18 个状态 case。
不同覆盖范围的最大值不作为同条件排名或误差上限。指标详情与一致性检查见[数值契约](docs/numerics.md)。

## 许可证

许可证与第三方声明见 [LICENSE](LICENSE) 和 [licenses/](licenses/)。
