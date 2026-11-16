# FlashInfer 状态池 Decode 对比

B=64、H=12/16/32、D=128，Q=1/8；每种 Q 有 21 个正式 case。两个平台均为
PyTorch 2.14.0a0+b2c75dd062.nv26.09 / CUDA 13.4 / CuTe DSL 4.8.1 / driver 615.71.10。
每个平台在双卡 NV18 节点的单张 GPU 上测量，不包含跨卡通信。

FlashInfer 使用 [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) 的
[c3af0c13328fd23b256c53b9b4f94d9e160d118c](https://github.com/flashinfer-ai/flashinfer/commit/c3af0c13328fd23b256c53b9b4f94d9e160d118c)，
Q1 为原生单 token kernel，Q8 为原生 grouped kernel；函数体不变，仅去除无关 imports/dispatchers。
双方直接读取设备端 slot 索引；FlashInfer 不附加 gather/scatter。FlashInfer state/checkpoint 为 BF16，
本项目分别报告 BF16 和 FP32 状态契约。生产 kernel 与前一批 k3-runtime 测量相同。

## 配对结果

以下为反序奇数 slot、Q8 接受数按请求索引模 9 分布的主测量。延迟和比值均为几何平均；越小越好。

| GPU | 写回 | Q | FlashInfer μs | BF16 μs | FP32 μs | BF16 / FI | FP32 / FI |
|---|---|---:|---:|---:|---:|---:|---:|
| B200 | 非原地 | 1 | 21.59 | 19.36 | 26.31 | 0.897× | 1.219× |
| B200 | 非原地 | 8 | 143.01 | 112.45 | 143.66 | 0.786× | 1.005× |
| B200 | 原地 | 1 | 19.41 | 17.95 | 23.29 | 0.925× | 1.200× |
| B200 | 原地 | 8 | 139.74 | 110.54 | 140.20 | 0.791× | 1.003× |
| B300 | 非原地 | 1 | 20.87 | 19.11 | 26.10 | 0.916× | 1.251× |
| B300 | 非原地 | 8 | 137.32 | 108.50 | 140.93 | 0.790× | 1.026× |
| B300 | 原地 | 1 | 18.72 | 17.85 | 23.17 | 0.953× | 1.238× |
| B300 | 原地 | 8 | 133.77 | 106.73 | 137.42 | 0.798× | 1.027× |

FP32 Q1 的差距随 H 变化：原地 H=32 在 B200/B300 分别为 1.409×/1.465×，
主测量最慢单 case 分别为 1.465×/1.466×。不得用总体均值代表每个 head 配置。
这些是相对 FlashInfer 的差距，不是相对本项目改动前版本的回退。

## 输入、计时与状态义务

- 沿用确认的 14 个 decode 长度 case × 三个 H；历史上限 8K～512K，精确长度见 [workloads.json](workloads.json)。历史以非零随机状态表示，不重新执行历史 KV；每个 case 的 seed、slot 和接受数均保存在 CSV。FP32 初态与 BF16 初态数值相同。
- 状态池容量 P=129；主测量使用反序奇数 slot。补充测量使用随机唯一 slot/全部接受 0、连续 slot/全部接受 8。Q1 不使用接受数。这三种设置分开汇总，未混合进入主表。
- Q1 直接读写索引 slot。Q8 均生成全部八个输出及 checkpoint，再按接受数 0～8 把状态写回目标 slot；0 保留或复制初态。checkpoint 按请求顺序存储。
- FlashInfer Q8 后使用与本项目 BF16 路径相同的 `compiled_commit` 复制 helper；它属于本测量适配层，不是 FlashInfer 原生 API。FlashInfer 也能通过后续入口索引直接使用 checkpoint；省略物理写回的集成方式不在此表范围内。
- 每图一次完整调用；Q8 包含 commit。固定地址热缓存；每样本三次 replay，三轮预热、九轮有效样本，三实现轮换顺序。每样本计时外重置初态，原地模式在三次 replay 中连续推进状态。
- 计时排除 metadata 更新、接受数决策、初态准备、分配、编译、capture、正确性比较；包含 kernel 中的索引读取和状态选择写回。
- 时钟采用默认策略，未锁频。这里是同节点配对测量，不把两台机器的绝对延迟差异解释为纯硬件收益。

## 正确性

每个平台完成 252 组配置 × 三实现。每个实现与自身的连续状态参考比较，
同一个 CUDA Graph 更换 slot 和接受数，测量前后各检查三次连续调用的全部输出、
Q8 每个 checkpoint、所选状态及未选 slot；全部 finite/bitwise 通过。
该检查验证索引和提交语义；不要求不同实现的数值契约逐位相同，
也不替代已有的独立 FP64 数学精度诊断。

## 原始数据

- [B200 逐 case 与全部样本](b200-cases.csv)、[环境](b200-environment.json)
- [B300 逐 case 与全部样本](b300-cases.csv)、[环境](b300-environment.json)
- [按 Q/H/原地/slot 汇总](summary.json)、[源码 SHA256 与参考版本](sources.json)
- [前一批完整五路比较](../k3-runtime-20261116/README.md)

此处直接比较双方的状态池索引路径，不能与“本项目旧 API 搬出/搬回状态”的收益混为一谈。
