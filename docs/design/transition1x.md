# Transition1x 数据与基准约定

## 数据来源和划分

目标数据为 Transition1x，DOI `10.6084/m9.figshare.19614657.v4`。
其官方读取器和生成器确定能量单位为 eV、坐标为 Å、力为 eV/Å。
原始 HDF5 以 `data/<formula>/<reaction>` 保存数据，train、val、test
按化学式通过软链接划分。

2026-11-16 的 Figshare 页面、API 和下载端点均返回 HTTP 403。v0.2.0 实测采用
JMP 作者 nimashoghi 公开的 Hugging Face 导出，固定 revision
`fac02fc7019ae95a6957b0723b467f6e56745fd2` 的 test Parquet 文件。
JMP 预处理源码将 `rxn` 的数字部分保存为 `sid`，将反应组内序号保存为 `fid`，
并保存原始 eV 能量。其坐标和力被转为 float32。本次实测通过该副本生成兼容
原始结构的十反应 HDF5，manifest 明确记录重建来源。反应物、产物和 TS 子组
依据原始 writer 的构型顺序重建。

按 seed 42 对该 test 文件中的反应 ID 无放回随机选择十条，选择过程仅使用 ID。
每条反应必须具有从 0 开始连续且无重复的 fid、恒定的元素顺序和有限的能量、力、
坐标。由元素推导的化学式必须属于官方 test 化学式列表。该检查保留原始划分，
test 样本用于预训练模型诊断，后续训练与拟合 E0 使用 train。

本次尚未取得官方完整 HDF5，尚未将导出副本逐数值对照原始文件。
`benchmark_transition1x.py --input <original.h5>` 可读取原始 schema，默认 split
为 test。原始大文件入口已通过同 schema 合成夹具测试，官方完整文件仍待实测。

## 最后一轮 NEB 路径

官方 `combine_dbs.py` 先保存十个构型，包括固定反应物和产物。后续每个被保留的
NEB band 追加八个内部构型，最后一轮强制保留。反应组里的全部历史构型沿优化
迭代拼接。

设反应组构型数为 n。必须满足 `n >= 10` 且 `(n - 10) % 8 == 0`。

- 单个 band 的最终路径序号为 `0, 1, ..., 9`。
- 多个 band 的最终路径序号为 `0, n-8, ..., n-1, 9`。
- 多 band 情况的 TS 参考点为最后八个内部构型中 DFT 能量最大的点。
- 单 band 情况遵循原始 writer，从全部十个图像取最大值，包括端点。

读取器核对路径端点与反应物、产物子组的元素、坐标、能量、力，并核对 TS
子组的元素、坐标、能量和力。每条路径转成十个 ASE Atoms，写出 extxyz 后重新读取
并比较参考坐标、能量和力。测试覆盖单 band、多 band、非法帧数和 TS 不一致。

## 能量和力指标

所有模型在同一组 DFT 路径几何上单点推理，使用 CUDA float64。能量逐反应、逐模型
减去各自反应物的能量。CSV 保留逐图像数据、原始构型序号及逐反应指标。

- 相对能量 MAE/RMSE 在十条路径的全部 100 个图像上计算，包括误差为零的反应物。
- 采样路径正向能垒为该路径十个点的最大能量减去反应物能量。
- 同时记录模型在 DFT TS 图像处的相对能量误差、模型路径最大点序号和反应能误差。
- 力 MAE/RMSE 按全部原子 Cartesian 分量统计。不同原子数的反应贡献不同分量数。
- TS 区域定义为上述 writer 定义的 TS 图像及相邻图像，在端点截断，单独统计力分量 RMSE。

图的横轴为最终 NEB 图像序号。能垒来自固定离散路径上的最大值，模型自行搜索的
鞍点与优化路径由后续 v0.4.0 另行验收，见 `ts-search.md`。十条样本的聚合指标用于可行性诊断。

OFF23、MH-1/omol、MH-0/rgd1_b3lyp 与 Transition1x 的参考 DFT 方法存在差异，
因此将本次结果记作跨理论水平的零微调对照。Transition1x test 表示该数据集的
内部划分，预训练数据与该样本是否重叠尚待核查。

## 模型 heads

下载 MACE foundations `mace_mh_1` release 的实际权重，通过 torch 和官方
`mace_select_head --list_heads` 同时检查。模型 URL、文件大小、全部 heads
和选定 head 记入 `results/v0.2.0/model_heads.json`。

实测 MH-1 提供 `omol`，MH-0 提供 `rgd1_b3lyp`。加载模型时显式指定 head，
并检查模型参数所在设备。

## 训练前的 E0 约定

`configs/transition1x_e0s.json` 保存官方 dataloader 发布的 H/C/N/O 孤立原子参考能，
单位 eV，目标理论水平为 Transition1x 的 ωB97X/6-31G(d)。v0.2.0 的相对能量
基准直接使用总能量。v0.3.0 训练采用下述单头 `estimated`，详见 `finetuning.md`。

已安装的 MACE 0.3.17 在 `tools/scripts_utils.py` 中将 `E0s="average"`
解释为按元素计数做最小二乘拟合。`cli/run_train.py` 单独处理
`E0s="estimated"`，要求 foundation 模型和 XYZ 训练数据，并从 foundation
预测与训练数据估计 E0。只查看底层 `get_atomic_energies` 会漏掉该训练入口。

该版本的 `estimated` 分支对多 head 的 foundation E0 数组默认取第一个 head。
采用该方式时，应先通过官方 `mace_select_head` 导出目标单 head checkpoint，
再验证估计时实际使用的 head。后续训练统一选择显式的 Transition1x JSON 或
经过 head 核验的 `estimated` 配置，并核对目标能量设置和原子参考能覆盖。
估计或拟合参考能统一仅使用 train 构型。v0.3.0 四组微调已完成该入口的实测。

## 上游来源

- [Transition1x 论文](https://www.nature.com/articles/s41597-022-01870-w)
- [官方数据读取器与参考能](https://gitlab.com/matschreiner/Transition1x/-/blob/08f728925d76e22ce04ec1ab3b75770d32e617ca/transition1x/dataloader.py)
- [官方 HDF5 生成器](https://gitlab.com/matschreiner/Transition1x/-/blob/08f728925d76e22ce04ec1ab3b75770d32e617ca/scripts/combine_dbs.py)
- [JMP 数据导出](https://huggingface.co/datasets/nimashoghi/transition1x/tree/fac02fc7019ae95a6957b0723b467f6e56745fd2)
- [JMP 原始数据预处理](https://github.com/facebookresearch/JMP/tree/main/src/jmp/datasets/scripts/transition1x_preprocess)
- [MACE 0.3.17 E0 处理](https://github.com/ACEsuit/mace/blob/v0.3.17/mace/tools/scripts_utils.py)
- [MACE 0.3.17 训练入口](https://github.com/ACEsuit/mace/blob/v0.3.17/mace/cli/run_train.py)
