# 多设备计算与交接

最后核实：2026-11-24T00:56Z（洛杉矶 November 23）。先读根 AGENTS.md。
用户要求持续推进长计算，包括夜间；等待期间检查完成并接续已授权研究。
用户稍后自行写报告，当前主线是改进模型和验证条件专长。

## 当前分工

| 节点 | 当前任务 | 权威状态与恢复入口 |
| --- | --- | --- |
| OMEN | 原2048s2训练→TS两臂先导/条件完整应用→原预留GPU验收 | growth67371；TS35116/75451；验收25432/75777等待真实锁，验收不与Mac重复 |
| Mac | 两份真实训练：8480帧20轮公共回放完整对照、544帧4轮公共/新5D适配 | owner69899；worker69904/69905；results/until-0800-20261126；释放slot可接16个固定端点验收 |
| 室友 Gaussian | 原32反应5D批次继续；1152项新储备等待首批/实际适配gate | 原16128；接续3248；serial4threads/2GB/4GiBgate，00:00Z停接新作业；collector50081/73508 |
| ROG | 本轮无新计算 | 现有分析环境，不把连接当作训练/DFT可用 |

November26用户将三台任务窗口明确到November27 08:00UTC+8（00:00Z）。
协议`configs/three-host-until-0800-20261126.json`，设计
`docs/design/overnight-until-0800.md`。Mac到点后不启新阶段、当前树nice至少10
自然排空；室友新增reserve停止接新作业并收尾；OMEN原冻结训练自然完成。
数据扩展门槛与模型晋级不同，gate拒绝/科学失败不以空转填时长。prospective封存。

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

## November24 本地长队列

用户已授权三台现有节点至少18h任务。OMEN新native22604/WSL61077在等待
v0.9 runner.lock，之后三组成对训练约21h。原准备的五组训练和三DFT全链
已认领，不能再向其他人派发；新force标签暂不合并训练。
Mac回收器状态位于results/coordination/local-queue-20261124；每链独立assessor，
总队列终态ZIP有限72h回收。readiness/提交/科学结果分别记账。
详见docs/design/local-compute-queue.md。

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
