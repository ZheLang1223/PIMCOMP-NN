# PIMCOMP-NN 初始性能基准测试报告

## 一、基准测试环境
- **操作系统**：Ubuntu 22.04 LTS on WSL2 (x86_64)
- **CPU 核心**：多核物理 CPU（通过 WSL2 虚拟化）
- **编译器**：GNU GCC 11.4.0 (初始参数: -std=gnu++11 无 -O 优化)
- **Python 环境**：Python 3.10.12 (.venv: onnx 1.23.2, onnxruntime 1.23.2, numpy 2.2.6)
- **测试模型**：AlexNet, ResNet18, SqueezeNet, MobileNetV2, ResNet34

## 二、初始基线性能测量数据 (Commit: d1f1917)

| 测试用例 (模型_策略_流水) | 编译耗时 (Wall Time) | 报告耗时 (Reported) | 峰值内存 (Peak RSS) | 硬件周期 (Latency ns) | MVMUL 指令数 | COMM 指令数 | MEM 指令数 | 验证精度误差 |
|---|---|---|---|---|---|---|---|---|
| **alexnet_balance_batch** | 3.08s | 2.83s | 903.6 MB | 293802 ns | 32096 | 1832 | 35110 | Pass |
| **resnet18_balance_batch** | 7.08s | 6.47s | 1944.1 MB | 470023 ns | 132500 | 3604 | 111633 | Pass |
| **squeezenet_balance_element** | 5.82s | 5.04s | 1685.6 MB | 2971658 ns | 50962 | 144198 | 37778 | < 1e-4 (Verified) |
| **mobilenetv2_balance_batch** | 5.87s | 6.61s | 2322.8 MB | 1252413 ns | 244961 | 16870 | 289762 | Pass |
| **alexnet_GA_batch** | 8.18s | 7.70s | 907.4 MB | 172702 ns | 32096 | 1850 | 33436 | Pass |

## 三、基线分析结论
1. **内存使用极高**：单进程峰值 RSS 普遍达到 1GB ~ 2.3GB，主要由于多次无约束的 10^9 尺寸 bool 动态数组分配以及冗余结构复制。
2. **编译执行效率存在巨大优化空间**：由于缺乏编译器优化标志 (-O0)，CPU 计算密集型的调度排序与模拟耗时较长。
3. **GA 算法探索开销显著**：仅 3 代 200 个个体的简单探索就耗时超过 8 秒，制约了大规模模型的空间探索能力。
