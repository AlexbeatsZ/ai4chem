# v0.7：模型改进与后续独立评价

## 目标与当前边界

用户在 2026-11-23 明确：报告稍后自行撰写，约两周用于提升模型，并研究
本地决策模型是否能选择互补的势能专家。v0.3 的 256 帧 / 8 epoch / 单种子
结果是先导实验。v0.5 搜索和 v0.6 定向 DFT 诊断保持原记录。

本轮先提高 MACE，再以独立的决策层研究专家互补。不能靠增加统计表、把
已有测试集改叫训练集或只延长失败任务，替代模型改进。

## 数据与角色

`configs/v0.7.0.json` 固定第一阶段方案，`v0.7.0_partitions.json` 在新训练前
冻结具体 ID、端点、时间和排除关系。

- MACE train：保持 v0.3 的 256 条反应及原先随机帧作为 anchor。
- MACE validation：官方 val 60 条，包括原先 20 条和新选择的 40 条。
  这 60 条用于 checkpoint/方案选择，不是独立测试。
- Meta development：128 条此前未用于项目的官方 train 反应，用于专家互补、
  决策层拟合和按反应/化学式分组的交叉验证。与 256 条 MACE train 互斥。
- Prospective holdout：64 条新官方 train + 64 条新官方 val，全部排除以往
  项目反应。前者检验同化学式范围的新反应，后者的化学式与 MACE train 不同。
  第一阶段只冻结端点，不生成参考能量/力文件，也不评价这些标签。

旧 287 条官方 test 反应已经用于本项目，不能重新宣称未见，也不用于拟合
新的决策模型。基础模型预训练重叠仍未证实；新集合只能称项目后续 holdout。

公开 Parquet 按帧跨 13 个 shard 分散；第零 shard 的 9,561 条反应均不能
直接当作完整路径。路径训练必须获取全部 13 文件，并按 sid 合并后检查 fid
从零连续、无重复，再应用已有 final-band 规则。固定 HF revision 不变。
只读取所需 sid，避免同时把全部原子坐标、力解码到 WSL 内存。

## 第一阶段：数据与训练长度对照

四个实验均从相同 MH-1/omol 单头基础模型开始，全参数训练，CUDA float64。
使用已安装的 MACE 0.3.17 官方 `mace_run_train`，不修改训练库。
其余损失/学习率设置保持 v0.3：energy 10、forces 1、lr 1e-4、batch 4、
E0s estimated、no_scaling。最多 40 epoch，validation loss patience 12。

| 实验 | 帧数 | 每条反应 | 主要目的 |
| --- | ---: | ---: | --- |
| anchor256_long | 256 | 1 | 延长原先短训练的对照 |
| path2048 | 2,048 | 8 | 增加端点、TS 邻域及几何多样性覆盖 |
| random4096 | 4,096 | 16 | 等反应数、等标签数的随机取样对照 |
| path4096 | 4,096 | 16 | 在相同预算下完整最终 band 与路径历史覆盖 |

所有组保留原 anchor；random 在其余帧无放回随机选择。path8 强制覆盖两个
端点、DFT TS 及相邻图像，再用原子对距离描述符做 farthest-point sampling。
path16 包含 path8、完整十图像最终 band，再补几何多样性帧。原子对距离对
整体平移/旋转不变；每条反应内相同原子映射。TS 参考能只用于 train 采样。

random4096 与 path4096 的反应数、标签数、epoch 上限相同。不同数据规模
的梯度更新数不同，不能称等计算比较。实际 epoch、估计更新数、wall time
均保存。官方训练器选 validation-loss 最优权重；方案之间按 validation
采样能垒 MAE，平局再按 TS 区域力 RMSE 排序。与原模型和 v0.3 模型同时比较。

第一阶段只是筛选。下一阶段在 validation 上固定合理的优化器/力权重与
replay 对照，并对有效方案做三个种子重复。需要统计所有尝试，不能只报告
表现最好的种子。不会预先保证增加数据必然改善结果。

## 后续决策模型与独立测试

见 `decision-routing.md`。Meta development 的参考标签只用于决策层；
第一阶段的 MACE 参数不使用这些标签。只在冻结模型、路由策略和预算后，
打开后续 holdout。先做静态路径基准，再进行端点搜索和预先选定的 DFT 验证。
静态 DFT 路径上的优势不能直接替代实际 NEB/TS 成功率。

若搜索失败暴露出新的构型区域，可在独立 active-learning 池上按固定预算
比较随机、几何多样性、模型分歧、学习到的风险排序，再用同一 DFT 方法产生
能量和力，微调新版本。用于评价的反应不能回流其训练。

## 运行与恢复

OMEN：`scripts/launch_model_study.ps1 -Stage download` 补齐固定源文件；
`-Stage study` 启动一次性四实验、验证和 meta-development 基准。
PowerShell owner 独立于 SSH 会话，WSL 项目路径由当前脚本位置转换。
启动前检查 `results/coordination/model-v07/*-launch.json` 和实际 owner。

`run_model_study.py --freeze-only` 仅冻结角色和准备验证集。
运行状态 `results/v0.7.0/state.json`；逐实验官方日志在 training 子目录。
已完成实验按配置与产物复用；有日志但无完成记录的实验停止，先检查实际
进程与失败证据，不自动覆盖。第一阶段完成后停在 screening_complete，
不会自动打开 holdout 或声称研究已经完成。

Mac 决策模型隔离环境：`environments/decision-models/pixi.toml`。
底层机器学习库、模型权重和真实设备推理分别验收。

## 参考实现

- MACE 官方微调接口：https://mace-docs.readthedocs.io/en/latest/guide/finetuning.html
- 官方建议与研究：https://mace-docs.readthedocs.io/en/latest/guide/finetuning_guidance.html
- 多头 replay：https://mace-docs.readthedocs.io/en/latest/guide/multihead_finetuning.html

本轮先保留旧优化设置做数据对照；文档中较新的默认值不是改动已安装训练器
或跳过受控比较的理由。后续实际参数支持须以安装版本源码为准。
