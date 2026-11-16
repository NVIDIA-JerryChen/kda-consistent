# KDA Consistent 开发约束

- 使用简体中文沟通；代码注释用英文。agent 产物放在忽略的 `agent/` 下。
- 保持 BF16/FP32 两套独立数值契约；同契约的 prefill/Q1/Q8、分段/PD、batch/重排、varlen、完整 head 分片、eager/Graph 必须逐位一致。
- 比较全部有效输出、完整续算状态和每 token checkpoint，NaN/Inf 失败。内部一致性不能代替独立 FP64 数学诊断。
- gate、归一化、投影、归约、delta、状态乘加保持 FP32；不得引入 BF16 中间计算或投影舍入。状态存储按选定契约为 BF16 或 FP32。
- Prefill 使用统一入口，B/H/token 数/varlen metadata 保持 runtime，不按长度或 B×H 分档编译。
- Prefill 调度不得使用固定 H 数值的经验门槛（如 `H > 12`）；按 CTA 工作量、SM 容量及 workspace 推导调度，不为 benchmark shape 设置专用分支。
- 状态池索引及接受数保持设备端 runtime metadata；不得为分派或校验内容引入 D2H。slot 写入须唯一，原地更新只允许完整输入/输出池别名。
- Q8 始终保留全部 checkpoint；commit 按 0～8 接受数提交，0 保留原始状态。验收比较所选完整状态及后续续算，benchmark 计入选择与写回。
- 通用调度成本模型须有跨形状测量依据；不针对 benchmark case ID、固定 H/N/L 值设特判。模型/窗口调整仍须验证同一编译实例及数值程序不变。
- Decode 可按 Q 和确实影响代码生成的有限配置编译，不按精确 batch、历史 KV 长度、指针或设备 metadata 内容特化；构造 key 不得 D2H。
- 只负责 no_grad forward，不包含 backward、完整模型 logits 或框架通信。
- 保留许可证与 attribution；修改后的正确性与性能须由对应版本的验证证据支持。
- 性能比较匹配状态契约、有效输出和 checkpoint 义务，计入完整 prepared GPU API，注明计时边界与缓存方式。
- Benchmark 数据须记录源码版本或 hash、硬件与软件栈、workload、状态/checkpoint 精度、Graph 调用次数、入口状态准备方式及统计方法，提供可核对的逐 case 数据。
- 不同源码版本或测量协议的数据分开展示，不跨批次相乘推导未测结果；历史数据不得标为当前实现性能，不同软件栈的跨卡结果不得归因为纯硬件差异。
- 正式 benchmark 只使用经核验的 B200/B300 多卡 NVLink 节点；排除 B300A 和 galaxy-*。GPU 资源按 gpu-resources skill 管理并及时释放。
- README 面向安装、调用和集成，不放内部集群信息、个人路径、调优流水账或未经验证的支持/性能承诺；中英文语义一致。
- 每次 push 前提供最终 diff 供用户 review，取得新的明确授权和目标远端分支。不得复用旧授权。
