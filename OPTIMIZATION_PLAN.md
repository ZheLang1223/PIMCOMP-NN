# PIMCOMP-NN 多轮优化研究计划 (OPTIMIZATION_PLAN.md)

## 一、已完成工作
- [x] Phase 1: 建立独立优化副本 `/home/zhelang/projects/PIMCOMP-NN-opt` 并检出 `optimization` 分支。
- [x] Phase 1: 确保原始代码仓库绝对只读，移除远端 push 权限以防误推。
- [x] Phase 1: 验证 Python 环境，修复验证程序内存分配溢出与 uint/int 越界告警，确保验证系统可用。
- [x] Phase 1: 完成项目完整架构分析（`ARCH_ANALYSIS.md`）。
- [x] Phase 1: 建立初始性能基线报告（`INITIAL_BENCHMARK.md`）。
- [x] Round 1: 编译器与构建系统升级（C++17, -O3, -march=native, -DNDEBUG），实现全模型 2.7x 平均加速，已验证提交（Commit: 007d28f）。
- [x] Round 2: 消除海量 10^9 周期模拟数组与内存泄漏，引入动态位图与 GA Epoch ID 代际标记复位，已验证提交（Commit: 2510c1b）。
- [x] Round 3: 消除指令优化与放置阶段深拷贝，引入移动语义、引用传递、容量预留与 unordered_map，全模型耗时进一步减半（较基准加速 2.36x ~ 3.78x），常驻内存减少 150MB~367MB，已验证提交（Commit: 47b1e06）。
- [x] Round 4: 遗传算法 (GA) 适应度缓存与种群演化优化，引入代际缓存、哈希查重、操作码枚举、栈数组合法性检查，GA 耗时缩短至 1.47s (较基线加速 5.56x)。

## 二、后续优化方向规划

### Round 5: 指令生成与调度流水冗余指令优化 (Peephole / Redundant Load & Sync Elimination)
- **方向**：在指令合并与优化阶段 (`backend/InstructionOptimization.cpp` 与 `backend/BatchPipelineSchedule.cpp` / `ElementPipelineSchedule.cpp`)，深入分析相邻计算核心间的数据传输，识别并消除相邻周期间冗余的数据 Reload 与多余同步指令。
- **目标**：缩减生成的总指令数（尤其是 COMM 和 MEM 指令），直接缩短硬件仿真延迟 (Practical Latency)。

### Round 6: OpenMP 多核多线程并行化加速 (GA Population & Hierarchy Mapping)
- **方向**：在 GA 种群适应度评估循环及独立 Array Group 拓扑切分计算中引入 OpenMP 多核多线程并行化 (`#pragma omp parallel for`)。
- **目标**：充分利用现代多核 CPU，将空间探索速度提升数倍。
