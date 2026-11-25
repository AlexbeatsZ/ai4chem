# 训练更强的通用模型：反应覆盖扩展

2026-11-25 用户再次明确，当前优先级是训练更好的模型，供小组成果说明。
v0.8 的 512 反应通用模型在同一 validation60 上能垒 MAE 为
0.111033270 eV，优于 v0.7 random4096 的 0.125968657 eV；力 RMSE
0.090200620 对 0.090547554 eV/A。v0.9 的学习率/力权重网格没有取代
原设置。因此下一步沿扩大反应覆盖训练，暂不扩展路由器或超参数网格。
这是开发证据引出的训练决策，不能预先宣称新规模一定更好。

## 数据和隔离

`configs/v0.10.0.json` 固定两档嵌套数据：1024/2048 条官方 train 反应，
每反应八帧，总计 8192/16384 个标签。原 v0.8 的 512 条反应和全部
4096 个具体 `(sid,fid)` 标签原样保留；不重新抽样既有部分。
新增 1536 条通过既有 endpoint-only `diverse_subset` 选择，前 512 条
构成 1024 档，其余构成 2048 档。选择只读端点元素/坐标和帧数量，
不按 validation/meta 误差挑训练反应。新增部分必须排除 v0.7 的
validation/meta/两组 holdout、v0.3 所有旧评价 ID 和 v0.5 搜索 ID。

新增八帧复用已验证的 path8：确定一个固定随机 anchor，覆盖两端点、
训练侧参考 TS 邻域及成对距离多样性。先跨全部十三个固定源 shard
检查 fid 连续和原子顺序，再选样。每次只读有限批反应完整历史，避免
把所有反应转成 Python 对象占满 WSL 内存。所有训练仍使用原目标
DFT 的公开标签；室友新 force 标签继续隔离，不混入本轮。

在任何新增标签读取/训练前，冻结具体 ID、端点记录和前置 partition
时间到 `configs/v0.10.0_partitions.json`。准备后保存每帧来源，读回核对
能量、力、坐标及所有旧帧；缓存身份改变或不完整证据均停止，不覆盖。
validation60 和 meta128 原样复用，prospective holdout 继续封存。

## 训练和晋级

复用已安装 MACE 0.3.17 官方训练器和 `run_model_study.train`，从同一
MH-1/omol foundation 全参数微调，float64、batch4、lr1e-4、energy10/
force1、E0 estimated、no_scaling。两档先各训练 seed20261116、20 个
实际 epoch、patience100，仍按 validation loss 保存 checkpoint。
1024 档估计40960次更新，与旧512×8/40epoch相同；2048档81920次，
增加标签也增加计算，不能称等总成本优势。全部尝试/失败/实际成本保留。

只根据相同 validation60，在能垒 MAE 最低且全路径力及 TS 区域力
RMSE 均不差于 512 基线的候选中晋级；基线始终可选。若新增档都未
超过基线，终止本轮晋级，不制造胜者。若有改进，先冻结具体方案，
再从同一 foundation 训练两个固定种子20261117/18，报告全部三种子，
不根据重复结果改训练参数或挑最好种子。meta 仅作开发诊断，不改选择。
候选/重复失败保留，禁止自动重试或覆盖不完整模型。

## 执行

OMEN 的 CPU 数据准备可与现有 CUDA 任务并行；新训练必须等待真实
`overnight-20261124/omen/owner.lock`、v0.9/v0.7 runner.lock 释放，
并确认原队列全部终态。保持当前训练和冻结的配对任务，不重启模型。
新 owner 使用已验收的原生 Windows WMI → 前台 WSL 链，不增加计划任务。
前置等待上限72h，随后有限2个初始模型及最多2个胜出方案重复。

入口：`scripts/launch_model_study.ps1 -Stage growth-prepare`；准备通过后
`-Stage growth`。状态 `results/v0.10.0/state.json`，独立准备状态为
`data/v0.10.0/preparation_state.json`。已有 owner/证据必须先核对。
训练完成后冻结最终模型和评价预算，再进入 prospective static/search/DFT。
当前 validation 改进和 meta 开发结果不替代这一独立验收。
