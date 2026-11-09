# PIMCOMP-NN 多轮优化研究计划 (OPTIMIZATION_PLAN.md)

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
