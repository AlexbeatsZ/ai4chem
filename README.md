# ai4chem

在 OMEN 的 WSL Ubuntu 24.04 中，通过 Mamba 独立环境运行分子势推理与反应路径基准。

- v0.1.0 为本地推理环境验收，计算水分子势能和受力。
- v0.2.0 为 Transition1x 十条反应的数据读取及三个预训练模型的零微调基准。
- v0.3.0 为扩大基准、短程微调和独立最终集评价。
- v0.4.0 为端点初始化 CI-NEB、鞍点频率和 IRC 检查，以及 Gaussian DFT 精修。

## 创建环境

以下命令在 WSL 中运行。

```bash
cd /mnt/c/Users/Meta/Project/Workspaces/ai4chem
mamba env create -f environment.yml
mamba run -n ai4chem python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu134
mamba run -n ai4chem python -m pip install mace-torch==0.3.17 ase
mamba run -n ai4chem python -m pip install sella==2.6.0
mamba run -n ai4chem python -m pip check
```

`requirements.lock.txt` 记录通过测试的完整 pip 依赖版本，适用于
Linux x86_64 / WSL / Python 3.12 / CUDA 13.4。Mac 的依赖应在其独立环境中
单独解析，Mac 安装验收待做。需要重建相同 Linux CUDA 环境时，创建 Mamba
环境后执行以下命令。

```bash
mamba run -n ai4chem python -m pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu134
```

## 执行测试

```bash
cd /mnt/c/Users/Meta/Project/Workspaces/ai4chem
mamba run -n ai4chem python smoke_test.py
```

也可以在交互式终端使用 `mamba activate ai4chem`，再运行
`python smoke_test.py`。首次运行自动下载官方 MACE-OFF23-small 权重。

程序以 float32 在 CUDA 上执行。通过条件包括 GPU 矩阵运算成功、模型参数位于
CUDA、能量为有限值、三个原子的受力均为有限值，以及保存后重新读取的数据一致。
通过后输出 `PASS - CUDA inference and saved energy/forces verified`。

## 端点驱动的过渡态搜索

v0.4.0 在 v0.3.0 已评价过的反应中固定选择十条，覆盖八个化学式，比较原始
MH-1/omol 与微调 mh1_full_256。搜索程序从两个端点坐标开始，执行端点优化、
IDPP、CI-NEB、Sella 鞍点精修、投影振动分析和双向 MLIP IRC。
先完成 v0.3.0 数据准备与训练，或从该版本发布资产恢复微调权重。
本轮复用 `models/mh1-omol.model` 和 `models/v0.3.0/mh1_full_256.model`。
从发布资产恢复微调权重时，可先用官方命令补齐原始 MH-1 单头模型。

```bash
mamba run -n ai4chem python inspect_models.py
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 mamba run -n ai4chem mace_select_head \
  --head_name omol --target_device cpu \
  --output_file models/mh1-omol.model models/mace-mh-1.model
```

v0.4.0 结果包中的 `inputs/` 与 `selection.json` 可直接恢复搜索输入，搜索恢复阶段
只需端点文件。生成包含原始 DFT 路径比较的报告还需 v0.3.0 的参考 HDF5。

```bash
export OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu
mamba run -n ai4chem python ts_search.py --prepare-only
mamba run -n ai4chem python ts_search.py --models mh1-finetuned mh1-original
mamba run -n ai4chem python validate_ts_dft.py
mamba run -n ai4chem python summarize_ts_search.py
mamba run -n ai4chem python -m unittest discover -s tests -v
```

Gaussian 验证使用 OMEN 已安装的 Gaussian 16W，经 `run_gaussian_job.ps1` 调用，
选择五个微调候选，执行 ωB97X/6-31G(d) TS 优化、频率、双向 IRC 和末端最小值检查。
每个任务的工作目录和 scratch 位于对应结果子目录。

`results/v0.4.0/selection.json` 固定反应及协议，`inputs/` 保存端点输入。
每个模型、反应的 `result.json` 保留完整阶段判定，轨迹和 extxyz 供几何复核。
`search_metrics.csv`、`dft_metrics.csv`、`summary.json` 和 `report.md` 为汇总，
`optimized_paths.png/pdf` 展示模型优化路径与原始 DFT 离散路径。
正常结束和已记录结果可复用，具体失败原因修复后使用 `--retry-failed` 恢复对应阶段。

十条反应属于开发试验。详细选择规则、数值容差及 DFT 验收条件见
[过渡态搜索设计文档](docs/design/ts-search.md)。

### v0.4.0 路径搜索结果

十条反应覆盖八个化学式，按端点几何成断键规则涵盖成键、断键和重排。
每个模型独立优化端点和路径，各运行一次 IDPP 初始化。

| 模型 | CI-NEB 收敛 | 模型势能面全阶段通过 |
| --- | --- | --- |
| 原始 MH-1 / omol | 9/10 | 7/10 |
| MH-1 全模型，256 构型微调 | 8/10 | 6/10 |

全阶段检查包含端点收敛及连接保持、NEB 收敛、鞍点最大受力不超过 0.01 eV/Å、
一个显著虚频、双向 IRC 收敛并连接预期端点，以及逐连通片段的几何匹配。
本轮通过的十三个模型鞍点在投影振动空间中均恰有一个负频率。
解离产物采用逐片段配准，完整体系 RMSD 同时保留。全部二十个结果已统一复核。

五个固定微调候选完成了 Gaussian DFT TS 精修、唯一虚频、双向 IRC 与稳定端点
验证。它们的 MLIP→DFT TS 几何 RMSD 为 0.0113–0.0690 Å。在这五个所选且已
验证的反应上，模型势能面电子能垒相对 Gaussian 的 MAE 为原始 MH-1 的
0.197624 eV、微调 MH-1 的 0.103424 eV。

rxn1152 的 IRC 先达到 160 点预算，checkpoint 续算在第 161 点找到最小值，
随后 Gaussian 路径汇总报 `PESDtA: Invalid IPoint`。末端另行通过正常结束的
DFT 优化、频率和几何匹配，原始错误日志及恢复标记保留。其余四条完成正常流程。
实际数据来自已使用过的开发池，五条 DFT 子集按微调候选的验收结果选择。

## 本地推理结果

- `results/smoke_h2o.extxyz` 保存元素、坐标、势能和原子受力。
- `results/smoke_summary.json` 保存版本、设备、数值结果和运行时间。
- `results/smoke_test.log` 和 `results/smoke_test_cached.log` 保存首次运行和缓存重跑日志。

结果文件可以使用文本编辑器打开，也可以使用 ASE 读取。

```python
from ase.io import read

atoms = read("results/smoke_h2o.extxyz")
print(atoms.get_potential_energy())
print(atoms.get_forces())
```

v0.1.0 的验收范围是本地 CUDA 推理与结果文件读写。Transition1x 基准、
微调和路径搜索的执行方式与结果分别保存在对应版本章节和设计文档中。

## v0.3.0 微调与泛化验证

使用现有 Linux CUDA 依赖锁环境，按以下顺序执行。

```bash
mamba run -n ai4chem python inspect_models.py
mamba run -n ai4chem python prepare_research_data.py
mamba run -n ai4chem python run_finetuning.py
mamba run -n ai4chem python run_research_evaluation.py
mamba run -n ai4chem python -m unittest discover -s tests -v
```

数据准备程序自动下载固定 revision 的 test、val 和 train 第零个 shard，
缓存大小约 647 MB。沿用 WSL Mamba `ai4chem` 环境和 OMEN CUDA。

本轮使用 77 条开发反应、20 条官方 val 反应与 210 条预留最终反应。
原十条位于开发集，训练样本来自官方 train。训练规模为嵌套的 64/256 构型，
每个构型对应一个不同反应，数据源限定为本轮选定的一个 train shard。
分区与构型序号固定于 `configs/v0.3.0_partitions.json`。

四个八-epoch 试验覆盖 MH-1 全模型 64 构型、MH-1 readout-only 256 构型、
MH-1 全模型 256 构型和 MH-0 全模型 256 构型。复用官方训练入口，提取单头后
采用 `E0s="estimated"`。训练配置和日志保存在 `results/v0.3.0/training/`，
模型保存在 `models/v0.3.0/`。训练脚本可复用配置相同且已完成的实验。

验证集选定候选之后，再评估预留最终集。三个原始基线在开发和最终集合计覆盖
全部 287 条 test 反应。当前图像均采用固定 DFT 路径几何。

结果集中保存非反应物和 TS 图像能量误差、带符号能垒误差、峰值偏移、TS 区域
力误差及按化学式、原子数和端点连接变化分层的指标。置信区间按反应与化学式
分别进行 2,000 次聚类 bootstrap，整体力 RMSE 保留 Cartesian 分量权重。

- `results/v0.3.0/selection.json` 记录 validation 排名和最终候选。
- `results/v0.3.0/experiment_comparison.csv` 汇总三个角色的指标和置信区间。
- `results/v0.3.0/finetuning_generalization.png` 和 `.pdf` 展示学习曲线与泛化表现。
- `results/v0.3.0/report.md` 为实测结果报告。
- 各角色的 `analysis/` 保存分层 CSV、误差分解和配对置信区间。

### v0.3.0 实测结果

四组实验各运行八个 epochs。验证集按固定规则选择 MH-1 全模型、256 构型方案，
所选权重来自该实验最佳 validation loss 的 epoch 6（从 0 开始编号）。

| 模型 | 最终能垒 MAE / eV | 非反应物相对能量 MAE / eV | 力分量 RMSE / (eV/Å) | TS 区域力分量 RMSE / (eV/Å) |
| --- | --- | --- | --- | --- |
| OFF23-small | 0.606971 | 0.372728 | 0.587378 | 0.845369 |
| MH-1 / omol | 0.245429 | 0.126206 | 0.251791 | 0.276627 |
| MH-0 / RGD1 | 0.565433 | 0.258092 | 0.233540 | 0.296187 |
| MH-1 全模型，256 构型 | 0.131641 | 0.063770 | 0.111649 | 0.169773 |

选定模型的最终能垒 MAE 反应级 bootstrap 95% 区间为 [0.114183, 0.149081] eV，
化学式级区间为 [0.061627, 0.177177] eV。相对未微调 MH-1 的配对能垒 MAE
差值为 -0.113789 eV，化学式级 95% 区间为 [-0.124152, -0.081254] eV，
共八个化学式簇。平均带符号能垒误差从 -0.231239 变为 -0.041327 eV，低估比例
从 192/210 变为 117/210。

readout-only 方案在 validation 的力误差改善，而能垒 MAE 从原模型的 0.253158
升至 0.346186 eV。MH-1 的可选 TorchScript 编译导出遇到上游类型错误，标准
`.model` 权重均已重新加载并完成基准。所有实测配置、日志和错误提示保留在结果包。

[已发布的学习与泛化图](https://github.com/AlexbeatsZ/ai4chem/releases/download/v0.3.0/finetuning_generalization.png)

完整约定位于 [微调与泛化设计文档](docs/design/finetuning.md)。
本轮采用公开导出副本、一个 train shard、单种子和短程训练，官方原始 HDF5
对照、预训练重叠、更充分收敛和多种子仍待验证。端点驱动的模型路径优化见 v0.4.0。
本轮最终集已用于报告。后续依据这些结果改变方案时，需要记录其已使用状态，
并重新设计后续独立评价。

## v0.2.0 反应路径基准

在现有环境中安装新增的数据读取依赖，随后执行完整流程。

```bash
cd /mnt/c/Users/Meta/Project/Workspaces/ai4chem
mamba run -n ai4chem python -m pip install pyarrow==26.0.1
mamba run -n ai4chem python inspect_models.py
mamba run -n ai4chem python prepare_transition1x.py --count 10 --seed 42
mamba run -n ai4chem python benchmark_transition1x.py
mamba run -n ai4chem python -m unittest discover -s tests -v
```

实测数据来自固定 revision 的 JMP 作者公开 Transition1x test Parquet 导出。
本轮 Figshare 官方原始 HDF5 下载返回 HTTP 403，程序从该副本生成兼容原始 schema
的 HDF5。选取其 287 条 test 反应中的十条，保留全部 13,348 个源构型，并恢复
每条反应最后一轮的十图像 NEB band，最终评估 100 个固定几何。

该副本的坐标和力采用 float32，能量采用 float64，尚未与官方完整 HDF5
逐数值对照。来源、revision、随机种子、反应 ID 和重建方式记录在
`data/transition1x_test_10.manifest.json`。读取器会核对原始 test 化学式划分、
构型顺序、维度、端点和 TS，并检查 ASE 写出读回。

取得官方完整 HDF5 后，可直接通过同一个读取入口进行基准。

```bash
mamba run -n ai4chem python benchmark_transition1x.py --input /path/to/transition1x.h5 --split test --count 10 --seed 42 --output results/original_hdf5
```

所有模型使用 CUDA float64。相对能量逐反应减去各自反应物能量，能垒取固定离散
路径的最大相对能量。力误差按 Cartesian 分量汇总。相对能量统计包括十个误差为零
的反应物图像。理论水平差异和预训练数据重叠仍需在解释结果时保留。

| 模型 | 相对能量 MAE / eV | 能垒 MAE / eV | 力分量 RMSE / (eV/Å) | TS 区域力分量 RMSE / (eV/Å) |
| --- | --- | --- | --- | --- |
| OFF23-small | 0.266570 | 0.614896 | 0.617353 | 0.967881 |
| MH-1 / omol | 0.125486 | 0.311887 | 0.236213 | 0.262136 |
| MH-0 / rgd1_b3lyp | 0.225021 | 0.606168 | 0.230551 | 0.315144 |

在本次十反应样本上，MH-1/omol 的相对能量与能垒误差最低，MH-0/RGD1 的整体
力分量 RMSE 略低。扩大样本后的统计、模型重新优化路径和微调训练仍待验证。

主要输出位于 `results/v0.2.0/`。

- `reaction_energy_profiles.png` 和 `.pdf` 为十条反应的能量曲线。
- `reaction_metrics.csv` 为逐反应能垒、相对能量、反应能和力误差。
- `image_predictions.csv` 为逐图像相对能量、原始构型序号和力误差。
- `benchmark_summary.json` 为完整配置、数据来源、模型 heads 和指标。
- `reference_paths.extxyz` 与各模型的 extxyz 为可重新读取的坐标、能量和力。
- `model_heads.json` 为官方权重实际提供的 heads。

[已发布的十条反应曲线 PNG](https://github.com/AlexbeatsZ/ai4chem/releases/download/v0.2.0/reaction_energy_profiles.png)

完整的方法和数据约定位于 [Transition1x 设计文档](docs/design/transition1x.md)。
E0 文件位于 `configs/transition1x_e0s.json`，内容取自官方 Transition1x 读取器。
当前 MACE 0.3.17 支持训练入口的 `E0s="estimated"`；多头模型需要核对该入口默认
取首个 head 的行为。E0 配置和实际微调训练将在后续阶段验收。

### 更新 Linux 依赖锁

使用 `pip list --format=freeze` 导出版本，可避免 conda 构建路径进入锁文件。
在 WSL 项目目录执行，输出目标为本项目的明确绝对路径。

```bash
mamba run -n ai4chem python -m pip list --format=freeze > /mnt/c/Users/Meta/Project/Workspaces/ai4chem/requirements.lock.txt
```

发布资产包含本次数据子集和结果。Git 仓库中忽略 `data/`、`models/` 与 `results/`。

## 2026-11-16 实测结果

初次下载权重和缓存重跑均通过，依赖检查输出 `No broken requirements found.`。

| 项目 | 实测值 |
| --- | --- |
| Mamba 环境 | `ai4chem`，Python 3.12.16 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB |
| PyTorch / CUDA runtime | 2.15.0+cu134 / 13.4 |
| MACE / ASE | 0.3.17 / 3.29.0 |
| 模型设备 | `cuda:0` |
| 精度 | float32 |
| 水分子势能 | -2081.1162109375 eV |
| 缓存重跑的单次推理 | 约 0.741 秒，含该进程首次推理开销 |
| PyTorch 峰值分配显存 | 60 MiB |

受力单位为 eV/Å，顺序与 ASE 水分子的 O、H、H 一致。

```text
O   0.00000000   0.00000000  -0.65114212
H   0.00000000  -0.34527943   0.32557106
H   0.00000000   0.34527943   0.32557106
```

运行日志包含 TorchScript 弃用、float64 权重转换为 float32、Triton 浮点融合
默认值和可选 cuequivariance 加速库缺失等提示。上述测试均正常完成，CUDA
矩阵运算和模型参数设备检查通过。

## 上游资料

- [MACE v0.3.17 源码](https://github.com/ACEsuit/mace/tree/v0.3.17)
- [MACE-OFF 官方模型仓库](https://github.com/ACEsuit/mace-off)
- [PyTorch CUDA 13.4 安装索引](https://download.pytorch.org/whl/cu134)
- [Transition1x 官方代码](https://gitlab.com/matschreiner/Transition1x)
- [JMP 作者公开数据副本](https://huggingface.co/datasets/nimashoghi/transition1x)
