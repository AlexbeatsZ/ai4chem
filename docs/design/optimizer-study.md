# 学习率、力损失和三种子续接

用户要求把剩余两周用于更好的模型，并希望夜间继续等待和接续计算。
`configs/v0.9.0.json` 在新训练前固定有限 CUDA 研究。它是已知开发结果
启发的后续改进，不回改 v0.7/v0.8，也不打开正式 holdout。

## 前置和相同标签

等待 v0.7 四臂全部结束、写入 screening_summary 并释放原 runner.lock。
新 runner 必须获得该锁和自己的 v0.9 锁，才会占用 GPU；不靠进程名猜测
前一计算是否结束。原 256 条训练反应、两个已有的 4096 帧数据集中，按
validation 能垒 MAE/TS 力 RMSE 选择较好的 random/path 设置，先写不可变
results/v0.9.0/frozen_inputs.json。训练帧、validation60、meta128 均复用。
没有新反应标签、prospective 或旧 test 回流。

## 三个新设置和两次固定重复

官方 MACE 0.3.17 训练器、同一 MH-1/omol、float64、batch4、energy10、
E0 estimated、no_scaling、种子20261116，四十实际 epoch。比较学习率
1e-4/1e-3 与 force 权重1/10 的两因子网格，复用已经完成的1e-4/1基线，
只新增另外三个设置。固定patience100确保不早于预算停止；旧基线虽然
patience12，已安装非 distributed 实际没有早停，必须核对四十日志 epoch。

学习率和 force 权重由已读的官方 finetuning 指南启发；不修改库或换版本。
第一种子结束后，只在 validation 力 RMSE 和 TS 力 RMSE 均不差于基线的
设置中按能垒选择；基线始终可选。其余能量指标、失败都保留，不保证全指标
自然变好。然后冻结设置，补20261117和20261118两种子；报告全部三种子
和均值/样本标准差，不依据重复结果改超参数或只展示最佳种子。

每新臂4096×40/batch4，估计40960次更新；合计最多五个新训练，旧基线成本
明确保留。网格整体多花算力，不能称为对旧模型等总计算预算的优越证明。

## 运行和交接

OMEN `scripts/launch_model_study.ps1 -Stage optimizer-study` 创建原生 WMI
owner，前台 WSL执行 run_optimizer_study.py。等待前置最多十小时，此后完成
有限五臂；不加计划任务。状态 results/v0.9.0/state.json；具体冻结输入、选择、
attempts、逐臂日志及 screening_summary 都在同版本目录，协调记录在
results/coordination/model-v09。活动 v0.7 使用原已加载代码，不重启。

失败训练保留并继续其他已固定设置，不自动重算或覆盖不完整日志。任一重复
失败时 summary 标记 incomplete_seed_repeats，不能声称三种子完成。
Meta 只用于开发分析，不改变 validation 选择。最终仍需冻结模型/路由后
独立 static、端点搜索和预选 DFT；数据宽度及其他架构目标 DFT 适配继续待做。

参考：MACE 官方 finetuning / finetuning_guidance 文档，所读版本与实际
安装接口以项目日志和环境锁为准。
