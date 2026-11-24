# 多设备计算与交接

最后核实：2026-11-24T00:34Z（洛杉矶 November 23）。先读根 AGENTS.md。
用户要求持续推进长计算，包括夜间；等待期间检查完成并接续已授权研究。
用户稍后自行写报告，当前主线是改进模型和验证条件专长。

## 当前分工

| 节点 | 当前任务 | 权威状态与恢复入口 |
| --- | --- | --- |
| OMEN | v0.7 四臂 CUDA 对照，前三臂完成、path4096 运行 | results/v0.7.0/state.json；native owner 39092，启动命令在 results/coordination/model-v07 |
| Mac | 八反应三专家端点搜索，AIMNet2-rxn 八项完成，其余仍运行 | results/v0.7.0/diversity_search/<expert>/state.json；results/coordination/diversity-search |
| Mac | v0.8 七项等更新专门化训练，数据准备/传输/读回完成，尚等待 CPU 调度 | results/v0.8.0/state.json；results/coordination/specialization-v08 |
| 室友 Gaussian | AIMNet2-rxn/rxn8694，四线程/4 GB TS→频率→双向 IRC→稳定末端 | worker results/rxn8694-aimnet2-rxn-gaussian16w-r1；native owner 5232，00:33Z 启动 |
| ROG | 本轮无已验收计算 | 需要时重新检查，不把历史不可达当作当前事实 |

OMEN 已完成 random4096：60 条 validation 能垒 MAE 0.125969 eV、力 RMSE
0.090548 eV/Å；这比旧微调同集 0.137937/0.095218 好，但仍是开发结果。
path4096 尚未完成。所有 prospective holdout 参考标签继续封存。

## 目录和访问

Mac 项目 /Users/meta/Developer/ai4chem；OMEN 项目
%USERPROFILE%\source\repos\ai4chem，WSL /mnt/c/Users/Meta/source/repos/ai4chem。
室友为原授权最小 worker %USERPROFILE%\Project\Workspaces\ai4chem-worker，
不安装完整科研环境；现有 Gaussian C:\G16W。OMEN 的已有 SSH 身份可中转，
不用复制私钥。地址、用户和具体命令在同步任务日志
~/.agents/ai4chem--adaptive-ts-search/AGENTS.md，不进入公开仓库。

代码/协议经 AlexbeatsZ/ai4chem main 同步；数据、权重、大运行证据不进 Git。
室友不自动同步 dot-agents，任务包更新其 worker AGENTS.md 和 task.json。
启动前检查实际 owner、原生 Gaussian 和资源，不能只看旧日志中的 PID。
Windows 活动 Gaussian log 使用 bounded shared byte snapshot，不用 Get-Content -Tail。

## 科学协议和接力

- docs/design/model-improvement.md：v0.7 数据/优化比较及封存。
- docs/design/expert-diversity.md：不同架构、24 个搜索单元、六单元 DFT 开发诊断。
- docs/design/specialization.md：512 反应、4096 共享标签、四系统各 40960 更新。
- scripts/continue_gaussian_relay.py：有限一候选接力，经 OMEN 查询室友，终态
  回传后调用独立 assessor；不自动重算、更改判据或创建计划任务。
- scripts/status_gaussian_chain.ps1：原生 owner、Gaussian 进程和有界日志。
  raw_chain_completed 仍须独立映射端点/频率检查，才可标记 DFT verified。

## 已完成和历史

v0.5 已发布，v0.6 r1/r2 全部终态。Gaussian rxn9456 r2 完整通过；Mac 同候选
因真实负曲率末端和 IRC 预算失败，保留负结果。Mac rxn0666 正对照四项容差通过。
这些任务不得因旧节点文档仍显示“进行中”而重启；见
 docs/reports/v0.6.0-pilot.md、docs/log/2026-11.md 和同步任务日志历史。
旧四专家路由先导已发布 v0.7.0-routing-pilot；新八专家记录在
 docs/reports/v0.7.0-diversity，尚未证明可部署的综合改进。

原 OMEN chat：01a1e76a-d5bf-7881-8023-349081a2e012（测试 MACE 水分子 GPU 推理）。
Mac 接续 chat：01a206de-3464-7071-ad42-580c51d4e3e5。
组员可分担独立 Gaussian 全链，见 group-compute.md；目前没有发送人际消息。
