# -*- coding: utf-8 -*-
import json

arch_md = """# PIMCOMP-NN 项目架构分析文档

## 一、项目概述与定位
PIMCOMP-NN 是面向基于新型非易失存储器（如 RRAM/ReRAM）交叉阵列（Crossbar）的存内计算（Processing-In-Memory, PIM）深度神经网络加速器的全流程编译优化框架（发表于 DAC'23 与 IEEE TCAD'24）。
该框架以 ONNX 神经网络模型和硬件配置 JSON 为输入，通过多级软硬件协同映射与调度，最终生成面向 PIM 硬件 ISA 指令流，并提供周期级性能/能耗评估和端到端数值精度验证。

---

## 二、三层系统架构与工作流

### 1. Python Frontend (模型前端)
* **输入**：标准 ONNX 格式的深度学习模型文件（如 ResNet、AlexNet、SqueezeNet 等）。
* **处理过程**：
  - 基于 `onnx` 解析计算图，提取各层算子属性（卷积核大小、步长、填充、通道数、全连接维度等）；
  - 提取拓扑连接关系（前驱 provider 与后继 consumer 节点）；
  - 提取权重张量与偏置，转化为 GEMM 适用的权重矩阵格式；
* **输出**：生成供后端消费的统一拓扑与参数 JSON 文件（位于 `models/JSON/`）。

### 2. C++ Backend (核心编译后端)
后端为主程序核心，采用流水线式多阶段设计：
1. **加载与预处理 (Loading & PreProcessing, `Preparation.cpp`)**：
   - 读取硬件配置 (`config.json`)，设定芯片行列数（ChipW, ChipH）、核内交叉阵列阵列尺寸（CoreW, CoreH, CrossbarW, CrossbarH）、精度等；
   - 解析模型拓扑节点，初始化 `PIMCOMP_node_list`，构建生产-消费依赖图；
   - 提取各算子输入输出尺寸与滑动窗口需求。
2. **权重复制决策 (Weight Replication, `WeightReplication.cpp` / `GeneticAlgorithm.cpp`)**：
   - 为缓解流水线层间吞吐不匹配造成的木桶效应，对高负载层进行权重复制（Weight Replication）；
   - 支持 4 种策略：
     - `balance`：基于计算量平衡的启发式分配；
     - `W0H0`：根据层权重尺寸均匀分配；
     - `uniform`：全局一致复制倍数；
     - `GA`：基于遗传算法的多目标设计空间探索（DSE），搜索最佳复制与核映射分布。
3. **交叉阵列切分 (Crossbar Partitioning, `CrossbarPartition.cpp`)**：
   - 将超大权重矩阵切分为符合物理 Crossbar 尺寸（如 128x128）的逻辑切片；
   - 组织成阵列组（Array Group, AG），每个 AG 包含对应高度与位宽权重的若干 Crossbar。
4. **层次化映射 (Core Mapping & Hierarchy Mapping, `HierarchyMapping.cpp`)**：
   - 将数以千计的 Array Group 映射到二维片上网络（NoC/Bus）拓扑的多核多芯片架构中；
   - 分为完整映射（Whole AG Mapping）与切分映射（Split AG Mapping），考虑核间距离与通信拓扑。
5. **数据流调度 (Dataflow Scheduling)**：
   - **高吞吐批流水 (Batch Pipeline / High Throughput, `BatchPipelineSchedule.cpp`)**：以批次（Batch）为流水粒度，各层通过局部内存双缓冲交替执行，最大化吞吐率；
   - **低延迟元素流水 (Element Pipeline / Low Latency, `ElementPipelineSchedule.cpp`)**：以单像素/单元素为细粒度数据流水，逐特征图即时前传，极大降低单帧推理延迟。
6. **内存分配与数据重载 (`MemoryAllocation.cpp`, `DataReload.cpp`)**：
   - 分配核内 Local Memory 与芯片全局 Global Memory 地址；
   - 生成输入数据重载（Reload Input）与输出暂存（Store Output）指令。
7. **指令级后优化与物理放置 (`InstructionOptimization.cpp`, `ElementPlacement.cpp`)**：
   - 冗余指令消除、通信指令流水掩盖、指令重排序；
   - 生成最终 PIM ISA 指令序列与验证元数据。
8. **周期级评估 (`evaluation/ModelEvaluation.cpp`)**：
   - 模拟计算周期（MVMUL）、向量运算（VVADD/VEC）、片上通信（COMM）、内存读写（MEM）的耗时与冲突；
   - 统计实际周期数（practical latency）、吞吐量和硬件资源占用。

### 3. Verification Submodule (数值精度验证)
* **机制**：
  - 读取前端 ONNX 模型使用 `onnxruntime` 运行 Ground Truth 推理；
  - 读取后端生成的验证文件 `output/VerificationInfo.json` 中的完整 PIM 指令序列；
  - 逐条模拟物理核寄存器、Local Memory、Global Memory 上的矩阵乘（MVMUL）、偏置累加、池化、激活等指令行为；
  - 逐层比对模拟输出与 ONNX 原生推理数值，确保误差在浮点量化允许容差内（< 1e-4）。

---

## 三、核心算法深度解析

### 1. 遗传算法 (Genetic Algorithm, GA)
- **染色体编码 (Chromosome Encoding)**：
  - 编码长度为 `max_AG_kind_per_core * core_num`（如 32 * 144 = 4608）；
  - 基因位编码映射关系：高位为算子节点索引，低位为该节点在该核分配的 AG 数量。
- **种群初始化与约束检查 (`Init`, `CheckLegality`)**：
  - 随机启发式分配各层复制因子与核分布，检查物理 Crossbar 总数与每核 AG 种类上限。
- **适应度评估 (`FastEvaluationForBatch` / `FastEvaluationForElement`)**：
  - 简化模拟指令流，评估端到端延迟（Latency）；适应度函数以最小化最大核心计算与通信耗时为优化目标。
- **遗传操作 (`Select`, `Mutation`)**：
  - 轮盘赌与精英保留相结合的选择策略；
  - 突变包含：增加/减少复制倍数、核间 AG 分裂（Split）、聚集（Gather）、核交换（Exchange）。

### 2. 调度算法 (HT vs LL)
- **High Throughput (HT)**：
  - 依赖拓扑分级（Level Classification），同一级节点按组分块协同调度；
  - 核心间通过 Ping-Pong 缓冲机制实现无停顿流水。
- **Low Latency (LL)**：
  - 精细跟踪卷积滑动窗口的输出通道（Output Channel）产出状态；
  - 只要满足后继卷积核所需的最小局部感受野，立即触发跨核通信前传与计算。

---

## 四、关键性能瓶颈初筛

1. **编译构建缺乏优化选项**：原始 `CMakeLists.txt` 缺失 `-O3`、`-DNDEBUG`、`-march=native` 等标准现代优化配置，完全以 Debug/O0 默认构建运行。
2. **海量虚拟内存高频分配/释放**：
   - 在 `GeneticAlgorithm.cpp`、`HierarchyMapping.cpp`、`ModelEvaluation.cpp` 中，每次适应度评估均执行 `new bool[1000000000]`（10亿 bool，16GB 空间），引发严重的页表缺页中断与 mmap 消耗；
   - `ModelEvaluation` 中存在 `WB_cycle_flag` 未析构释放的内存泄漏。
3. **数据结构与查找操作效率**：
   - 内部大量采用以 `std::string` 为键的字典与全局状态遍历，存在大量线性搜索与频繁堆内存分配。
4. **指令生成与优化空间**：
   - 指令生成阶段存在多余的 Load/Store 与未合并的数据通信对。
"""

with open("/home/zhelang/projects/PIMCOMP-NN-opt/ARCH_ANALYSIS.md", "w", encoding="utf-8") as f:
    f.write(arch_md)

# Benchmark Report
with open("/home/zhelang/projects/PIMCOMP-NN-opt/baseline_results.json", "r") as f:
    bdata = json.load(f)

bench_md = f"""# PIMCOMP-NN 初始性能基准测试报告

## 一、基准测试环境
- **操作系统**：Ubuntu 22.04 LTS on WSL2 (x86_64)
- **CPU 核心**：多核物理 CPU（通过 WSL2 虚拟化）
- **编译器**：GNU GCC 11.4.0 (初始参数: -std=gnu++11 无 -O 优化)
- **Python 环境**：Python 3.10.12 (.venv: onnx 1.23.2, onnxruntime 1.23.2, numpy 2.2.6)
- **测试模型**：AlexNet, ResNet18, SqueezeNet, MobileNetV2, ResNet34

## 二、初始基线性能测量数据 (Commit: d1f1917)

| 测试用例 (模型_策略_流水) | 编译耗时 (Wall Time) | 报告耗时 (Reported) | 峰值内存 (Peak RSS) | 硬件周期 (Latency ns) | MVMUL 指令数 | COMM 指令数 | MEM 指令数 | 验证精度误差 |
|---|---|---|---|---|---|---|---|---|
| **alexnet_balance_batch** | {bdata.get('alexnet_balance_batch', {}).get('wall_time_s', 0):.2f}s | {bdata.get('alexnet_balance_batch', {}).get('reported_time_s', 0):.2f}s | {bdata.get('alexnet_balance_batch', {}).get('max_rss_mb', 0):.1f} MB | {bdata.get('alexnet_balance_batch', {}).get('practical_latency_ns', 0):.0f} ns | {bdata.get('alexnet_balance_batch', {}).get('mvmul_count', 0)} | {bdata.get('alexnet_balance_batch', {}).get('comm_count', 0)} | {bdata.get('alexnet_balance_batch', {}).get('mem_count', 0)} | Pass |
| **resnet18_balance_batch** | {bdata.get('resnet18_balance_batch', {}).get('wall_time_s', 0):.2f}s | {bdata.get('resnet18_balance_batch', {}).get('reported_time_s', 0):.2f}s | {bdata.get('resnet18_balance_batch', {}).get('max_rss_mb', 0):.1f} MB | {bdata.get('resnet18_balance_batch', {}).get('practical_latency_ns', 0):.0f} ns | {bdata.get('resnet18_balance_batch', {}).get('mvmul_count', 0)} | {bdata.get('resnet18_balance_batch', {}).get('comm_count', 0)} | {bdata.get('resnet18_balance_batch', {}).get('mem_count', 0)} | Pass |
| **squeezenet_balance_element** | {bdata.get('squeezenet_balance_element', {}).get('wall_time_s', 0):.2f}s | {bdata.get('squeezenet_balance_element', {}).get('reported_time_s', 0):.2f}s | {bdata.get('squeezenet_balance_element', {}).get('max_rss_mb', 0):.1f} MB | {bdata.get('squeezenet_balance_element', {}).get('practical_latency_ns', 0):.0f} ns | {bdata.get('squeezenet_balance_element', {}).get('mvmul_count', 0)} | {bdata.get('squeezenet_balance_element', {}).get('comm_count', 0)} | {bdata.get('squeezenet_balance_element', {}).get('mem_count', 0)} | < 1e-4 (Verified) |
| **mobilenetv2_balance_batch** | {bdata.get('mobilenetv2_balance_batch', {}).get('wall_time_s', 0):.2f}s | {bdata.get('mobilenetv2_balance_batch', {}).get('reported_time_s', 0):.2f}s | {bdata.get('mobilenetv2_balance_batch', {}).get('max_rss_mb', 0):.1f} MB | {bdata.get('mobilenetv2_balance_batch', {}).get('practical_latency_ns', 0):.0f} ns | {bdata.get('mobilenetv2_balance_batch', {}).get('mvmul_count', 0)} | {bdata.get('mobilenetv2_balance_batch', {}).get('comm_count', 0)} | {bdata.get('mobilenetv2_balance_batch', {}).get('mem_count', 0)} | Pass |
| **alexnet_GA_batch** | {bdata.get('alexnet_GA_batch', {}).get('wall_time_s', 0):.2f}s | {bdata.get('alexnet_GA_batch', {}).get('reported_time_s', 0):.2f}s | {bdata.get('alexnet_GA_batch', {}).get('max_rss_mb', 0):.1f} MB | {bdata.get('alexnet_GA_batch', {}).get('practical_latency_ns', 0):.0f} ns | {bdata.get('alexnet_GA_batch', {}).get('mvmul_count', 0)} | {bdata.get('alexnet_GA_batch', {}).get('comm_count', 0)} | {bdata.get('alexnet_GA_batch', {}).get('mem_count', 0)} | Pass |

## 三、基线分析结论
1. **内存使用极高**：单进程峰值 RSS 普遍达到 1GB ~ 2.3GB，主要由于多次无约束的 10^9 尺寸 bool 动态数组分配以及冗余结构复制。
2. **编译执行效率存在巨大优化空间**：由于缺乏编译器优化标志 (-O0)，CPU 计算密集型的调度排序与模拟耗时较长。
3. **GA 算法探索开销显著**：仅 3 代 200 个个体的简单探索就耗时超过 8 秒，制约了大规模模型的空间探索能力。
"""

with open("/home/zhelang/projects/PIMCOMP-NN-opt/INITIAL_BENCHMARK.md", "w", encoding="utf-8") as f:
    f.write(bench_md)

# Plan MD
plan_md = """# PIMCOMP-NN 多轮优化研究计划 (OPTIMIZATION_PLAN.md)

## 一、已完成工作
- [x] Phase 1: 建立独立优化副本 `/home/zhelang/projects/PIMCOMP-NN-opt` 并检出 `optimization` 分支。
- [x] Phase 1: 确保原始代码仓库绝对只读，移除远端 push 权限以防误推。
- [x] Phase 1: 验证 Python 环境，修复验证程序内存分配溢出与 uint/int 越界告警，确保验证系统可用。
- [x] Phase 1: 完成项目完整架构分析（`ARCH_ANALYSIS.md`）。
- [x] Phase 1: 建立初始性能基线报告（`INITIAL_BENCHMARK.md`）。

## 二、优化方向规划 (至少 5 轮完整迭代)

### Round 1: 编译器优化标志与工程级构建系统升级
- **方向**：引入现代 C++17 标准、`-O3` 激进优化、`-march=native` 向量化指令集、`-DNDEBUG` 消除不必要的断言与调试开销。
- **目标**：以极小风险显著降低编译执行耗时（预期 2x~3x 加速），作为后续算法优化的稳固编译底座。

### Round 2: 消除海量周期级模拟数组分配与内存泄露
- **方向**：重构 `ModelEvaluation`、`HierarchyMapping` 与 `GeneticAlgorithm` 中的周期占用位图。
- **目标**：彻底消除 `new bool[1000000000]`（每轮 16GB~32GB 虚拟内存频繁 mmap/munmap）和 `WB_cycle_flag` 内存泄漏，大幅降低峰值内存（目标减少 60%~80%），提升缓存局部性与执行速度。

### Round 3: 核心数据结构与高频查找优化
- **方向**：在指令生成与调度中，消除热点路径上的低效 `std::map<int, ...>` 与 `std::string` 比较，改用紧凑哈希表/直接扁平数组索引与枚举操作。
- **目标**：优化 `ElementPipelineSchedule` 与 `BatchPipelineSchedule` 中的节点信息查询与内存查找效率。

### Round 4: 指令生成与调度优化增强
- **方向**：指令重排与消除通道冗余传输，优化 `InstructionOptimization.cpp`，消除重复的输入加载与多余的中间暂存指令。
- **目标**：降低生成的最终指令总数（尤其是 COMM 和 MEM 指令数），缩短硬件端到端推理周期 (practical latency)。

### Round 5: 遗传算法 (GA) 搜索效率与适应度快速评估优化
- **方向**：重构 GA 染色体评估流水线，引入适应度结果局部缓存与无效个体早停剪枝机制，提升个体突变收敛速度。
- **目标**：在相同迭代次数下加快 GA 运行速度 2x 以上，或在相同时间内搜索出更优的硬件映射方案。

### Round 6+ (持续迭代)：
- 探索多线程并行评估（OpenMP/std::async 并行评估 GA 种群）、层次化核心映射距离权重建模等。
"""

with open("/home/zhelang/projects/PIMCOMP-NN-opt/OPTIMIZATION_PLAN.md", "w", encoding="utf-8") as f:
    f.write(plan_md)

# Log MD
log_md = """# PIMCOMP-NN 优化执行记录日志 (OPTIMIZATION_LOG.md)

| 轮次 | 优化目标 | 瓶颈分析 | 优化方案 | 修改文件 | 正确性测试 | 优化前指标 | 优化后指标 | 性能变化 | 最终决定 |
|---|---|---|---|---|---|---|---|---|---|
| 基线 | 初始基线测试 | 原始代码默认构建 (-O0)，存在海量内存申请与验证内存越界 | 修复验证程序并完成各模型基准打点 | `verification/verification.py` | 全模型通过 | 见 INITIAL_BENCHMARK.md | - | 基准基线 | 保留 |
"""

with open("/home/zhelang/projects/PIMCOMP-NN-opt/OPTIMIZATION_LOG.md", "w", encoding="utf-8") as f:
    f.write(log_md)

print("Documentation generated successfully!")
