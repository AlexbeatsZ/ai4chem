# ai4chem

在 OMEN 的 WSL Ubuntu 24.04 中，通过 Mamba 独立环境运行
MACE-OFF23-small，输入 ASE 构建的水分子三维坐标，计算势能和三个原子的受力。

## 创建环境

以下命令在 WSL 中运行。

```bash
cd /mnt/c/Users/Meta/Project/Workspaces/ai4chem
mamba env create -f environment.yml
mamba run -n ai4chem python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu134
mamba run -n ai4chem python -m pip install mace-torch==0.3.17 ase
mamba run -n ai4chem python -m pip check
```

`requirements.lock.txt` 记录本次通过测试的完整 pip 依赖版本。需要重建相同
Linux CUDA 环境时，创建 Mamba 环境后执行以下命令。

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

本阶段的验收范围是本地 CUDA 推理与结果文件读写。Transition1x、微调训练和
反应路径搜索仍待后续验证。

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
