# ai4chem

在 OMEN 的 WSL Ubuntu 24.04 中，通过 Mamba 独立环境运行分子势推理与反应路径基准。

- v0.1.0 为本地推理环境验收，计算水分子势能和受力。
- v0.2.0 为 Transition1x 十条反应的数据读取及三个预训练模型的零微调基准。

## 创建环境

以下命令在 WSL 中运行。

```bash
cd /mnt/c/Users/Meta/Project/Workspaces/ai4chem
mamba env create -f environment.yml
mamba run -n ai4chem python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu134
mamba run -n ai4chem python -m pip install mace-torch==0.3.17 ase
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

## 输出

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

v0.1.0 的验收范围是本地 CUDA 推理与结果文件读写。Transition1x 数据基准的
进展见下一节，微调训练和反应路径搜索待后续验收。

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
