# v0.3.0 微调与泛化验证

## 固定实验方案

配置位于 `configs/v0.3.0.json`，实际数据 ID 和训练构型序号位于
`configs/v0.3.0_partitions.json`。源数据延续 v0.2.0 固定 revision 的
Transition1x JMP 格式导出，坐标与力为 float32，能量为 float64。

- 原十条已用于模型比较的 test 反应纳入开发集。
- 使用固定种子扩展开发集到 77 条，剩余 210 条作为本轮最终评价集。
- 官方 val 中固定抽取 20 条反应，使用 200 个最终路径构型做验证。
- 官方 train 的第一个公开 Parquet shard 中选择 256 个互异反应，各随机取一个构型。
  64 构型实验使用该列表的前 64 个，构成嵌套的数据规模比较。
- 按官方生成代码的化学式列表检查 train/val/test 归属，反应 ID 也检查互斥。

第一个 train shard 是本轮小规模试验的数据池，完整训练集仍需后续覆盖。
开发集与最终集共享官方 test 的化学式范围，其反应 ID 分离；统计同时报告按反应
与化学式聚类的 bootstrap。预训练数据重叠核查仍待更完整的来源结构数据。

最终数据在准备阶段进行格式和数值完整性检查，模型选择只使用 validation 指标。
所有既定实验完成后，按验证集的采样路径能垒 MAE 排序，同分时比较非反应物相对
能量 MAE。结果写入 selection.json 后，再计算预留最终集的三个基线和选定模型。
最终结果用于报告本轮泛化表现。
本轮最终评价已完成。该集合的使用状态已记录，后续依据本轮结果作方案选择时，
需要设计后续独立评价。

## 官方训练入口

使用已安装 MACE 0.3.17 的 `mace_run_train`。先复用官方 `remove_pt_head`
提取 MH-1/omol 或 MH-0/rgd1_b3lyp 单头权重，核对单头，再使用
`E0s="estimated"` 从 train 估计参考能。

提取 head 前必须像官方 CLI 一样设置源权重 dtype，提取后转为本轮 float64。
本轮初次尝试漏掉前一步，导致 estimated 阶段出现 TorchScript 张量 dtype
不一致。修正后重新开始训练。该 dtype 设置仅作用于当前 Python 进程。

训练使用明确的 REF_energy / REF_forces 键，分别为 eV 和 eV/Å。固定 8 epochs、
batch size 4、学习率 1e-4，能量损失权重 10、力权重 1，使用官方 weighted loss。
原生训练器依据 validation loss 选择每个实验的最佳 checkpoint，实验之间再使用
上述路径能垒指标选择候选。

四个实验覆盖 MH-1 全模型 64 构型、MH-1 readout-only 256 构型、MH-1 全模型
256 构型和 MH-0 全模型 256 构型。官方 `freeze=6` 冻结 embedding、interactions
和 products，保留 readouts。当前两种基础模型的 radial embedding 没有独立参数。

这是一轮单种子、小数据、短程微调试验。数据规模、冻结策略和模型初始化的比较
使用固定方案，充分收敛与多种子统计属于后续实验。

## 诊断与不确定性

使用现有 CSV 即可计算非反应物相对能量 MAE、DFT TS 构型相对能量误差和峰值偏移。
精确分解为采样能垒误差 = DFT TS 处相对能量误差 + 模型峰值选择增益
− DFT 峰值相对 TS 的增益。程序逐反应验证该等式。

跨 DFT 方法的绝对能量差包含参考基准差异，单独的反应物和 TS 绝对准确度需要
统一参考后解释。本轮可识别的是二者差分误差。

每个反应的十图像整体作为统计单元，进行 2,000 次反应级和化学式级聚类 bootstrap。
配对模型比较使用相同反应顺序与聚类重采样。能量指标报告反应均值，整体力 RMSE
按原子 Cartesian 分量加权，同时保留反应平均力 RMSE，字段名称明确区分。

按化学式、原子数和端点连接变化分层。端点连接变化使用 ASE covalent radii × 1.2
识别成键、断键或二者皆有，属于几何启发式标签。精细反应机理分类仍待验证。

## v0.3.0 实测与恢复

验证集选定 mh1_full_256，最佳 validation loss 权重来自 epoch 6。
其最终 210 条反应能垒 MAE 为 0.131641 eV，未微调 MH-1 为 0.245429 eV。
对应整体力分量 RMSE 为 0.111649 和 0.251791 eV/Å。

采样能垒平均带符号误差从 -0.231239 变为 -0.041327 eV，低估数从 192 变为
117。模型减基线的配对能垒 MAE 差为 -0.113789 eV，化学式聚类 95% 区间
[-0.124152, -0.081254] eV，八个化学式簇。

MH-1 的标准 `.model` 保存、重新加载和推理均通过，可选 TorchScript 编译产物
因上游 gate.py 的 o3.Irreps 类型注解失败。该问题保留在官方训练日志中。
MH-0 编译成功，本项目推理统一使用标准 `.model`。

发布时保留每个实验一个标准模型以及日志、配置、学习曲线，删除同一实验的重复
checkpoint 和可选编译副本。原官方基础权重和单头提取权重承担不同复现用途。

## 实际执行

```bash
mamba run -n ai4chem python prepare_research_data.py
mamba run -n ai4chem python run_finetuning.py
mamba run -n ai4chem python run_research_evaluation.py
mamba run -n ai4chem python analyze_benchmark.py results/v0.2.0 --output results/v0.3.0/legacy_diagnosis
```

数据准备程序自动获取并缓存固定 revision 的 test、val 和 train 第零个 Parquet
文件。源文件、原始划分和缓存路径保留在数据准备程序及分区清单中。
