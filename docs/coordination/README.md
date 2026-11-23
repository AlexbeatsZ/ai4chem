# 多设备计算与交接

最后核实：2026-11-23 01:29（用户时区 America/Los_Angeles），
即 2026-11-23T09:29:56Z。时间戳文件统一使用 UTC。

## 从这里继续

先读项目根目录 `AGENTS.md`、本文件和 `docs/design/v0.5.0-protocol.md`。
本轮目标是完成冻结的 v0.5.0：4 条开发与 12 条保留反应、两模型、三策略，
共 96 个评价单元；随后完成预选三反应的 9 个 Gaussian 策略评价单元、
独立核验、报告与 release。开发阶段与冻结已完成。

原始对话：OMEN「测试 MACE 水分子 GPU 推理」，
`01a1e76a-d5bf-7881-8023-349081a2e012`。
跨设备接续对话：Mac `01a206de-3464-7071-ad42-580c51d4e3e5`。

## 工作目录与职责

| 设备 | 项目目录 | 当前职责 | 已核实环境 |
| --- | --- | --- | --- |
| OMEN | `C:\Users\Meta\Project\Workspaces\ai4chem` | 唯一 v0.5 CUDA 搜索和 Gaussian DFT 写入节点 | WSL Ubuntu-24.04 / Mamba ai4chem，RTX 4060 8 GB，现有 Gaussian 16W A.03 |
| Mac | `/Users/meta/Project/Workspaces/ai4chem` | 协调、代码与 CPU 报告测试、图表检查、发布整合 | 项目隔离 uv `.venv`，分析依赖单独记录 |
| ROG | `C:\Users\Meta\Project\Workspaces\ai4chem` | 已完成结果快照的独立文件与数值核验 | Ryzen Z1 Extreme；无已安装 Gaussian/ORCA；分析环境准备中 |

代码和人工交接记录通过 `AlexbeatsZ/ai4chem` 的当前分支同步。
模型、原始数据和计算结果沿用被 Git 忽略的 `models/`、`data/`、`results/`，
正式结果由 release 提供；跨节点核验传输已结束任务的只读快照。
每台设备的安装状态只适用于该设备。

## 正在运行的任务与状态

接管时 OMEN 原始搜索 Python PID 为 39916，来自临时
`ai4chem-v05-holdout.sh`。PID 只是观察记录，恢复时必须重新检查实际进程。
搜索矩阵现已完成 96/96 单元、83 次唯一尝试。保留集原始 S0/S1/S2 的鞍点与
IRC 通过为 3/3/3，微调为 5/6/6；全阶段通过为 3/3/3 与 5/5/5，分母均为十二反应。
Gaussian 已完成 rxn0666 和 rxn2737，两个均通过完整 DFT 验证。
rxn8204 仍在正向 IRC，第 61 点附近，原生 g16/link 进程存在且日志持续更新。
其反向 IRC 和端点验证仍待执行，最终 DFT 验收和 release 未完成。

`scripts/continue_v05.py` 会先等待现有 OMEN 搜索，随后按现有入口运行
剩余矩阵、回归、Gaussian、报告、依赖检查、水分子 CUDA 验证和最终产物核验。
科学失败和无候选保留；不会自动重试失败的 Gaussian，不自动修改协议或发布。

预选反应的三种策略均完成后，可通过 `bash scripts/run_ready_dft_omen.sh`
提前并行复核已就绪的 DFT 候选（一个 Gaussian 作业，四核/4 GB）。
`--ready-only` 只改变调度顺序；候选、精确去重身份、输出目录和验证函数不变。
部分就绪时仅写 `results/coordination/dft-ready-selection.json`，
正式 `results/v0.5.0/dft/selection.json` 仍需完整九单元清单。
coordinator 在完整 DFT 阶段前等待这一早期进程，复用其既有终态结果。

运行时进程、阶段、日志、完成计数保存在 OMEN：

- `results/coordination/omen-state.json`：当前阶段、PID、命令、退出码和矩阵计数。
- `results/coordination/omen-events.jsonl`：阶段开始、结束和失败记录。
- `results/coordination/<stage>.log`：每个阶段的标准输出与错误。
- `results/coordination/*-launch.json`：Windows WSL 启动器 PID 和外层日志。
- `results/v0.5.0/`：策略、attempt、原始轨迹和 Gaussian 权威结果。

运行时文件不进 Git；任何设备均可经 SSH 读取 OMEN 状态。换模型不需要重启进程。

## 查看和恢复

任意设备有 Python 即可读取已有结果目录的计数，无需加载模型：

```bash
python scripts/experiment_status.py --root /path/to/ai4chem
```

OMEN 后台启动使用项目内 PowerShell 启动器，它会保留整个任务的 Windows
`wsl.exe` 父进程，维持 Gaussian 所需的互操作会话。外层 PowerShell 通过
WMI 创建独立进程，避免 SSH 会话结束时连带终止；未添加计划任务或开机配置：

```powershell
powershell -NoProfile -File C:\Users\Meta\Project\Workspaces\ai4chem\scripts\launch_v05_omen.ps1 -Mode coordinator
powershell -NoProfile -File C:\Users\Meta\Project\Workspaces\ai4chem\scripts\launch_v05_omen.ps1 -Mode ready-dft
# 搜索全部结束后，可由协调节点显式并行补齐预选 DFT；不重复已运行候选：
powershell -NoProfile -File C:\Users\Meta\Project\Workspaces\ai4chem\scripts\launch_v05_omen.ps1 -Mode dft
```

前台 OMEN WSL 的等价命令（应先读取状态并确认没有同一 coordinator）：

```bash
cd /mnt/c/Users/Meta/Project/Workspaces/ai4chem
/home/meta/.local/opt/miniforge3/bin/mamba run -n ai4chem \
  python -u scripts/continue_v05.py --dft-jobs 2
```

不要从短命 WSL 会话中 detach Python 后关闭其互操作父进程：本轮实际遇到
`UtilAcceptVsock` 超时，Gaussian 尚未启动。原失败 JSON/TXT 留在对应 DFT 目录，
确认无遗留原生进程后改用 Windows 持续启动器恢复。文件锁防止第二个 coordinator
同时进入，但人为重启前仍须检查 Python、PowerShell、g16.exe 和 Gaussian links。
若状态为 stopped，先读对应日志并确认旧计算已退出，再决定恢复；
既有 `--retry-failed` 仅在明确排除原生进程还在运行并诊断失败后使用。

冻结的是五个核心实现、软件、GPU、模型与输入。Mac/ROG 分析环境不满足
这份 OMEN CUDA 身份，不能把其模型计算悄悄合并进本轮 holdout。

## 后续工作与组员算力

- 当前：跑完 holdout、Gaussian 与最终验收，提交报告与 release。
- 独立核验：ROG 核对已完成产物；Mac 核对 CPU 报告与图表。
- 用户确认：若有大量明确、对时效要求不高的计算，可找组员并行，可能一天后回传。
- 独立 Gaussian 反应可以分给组员。冻结 OMEN CUDA 身份约束 MACE 对照计算，
  不限制 DFT 必须在同一主机；固定输入/科学参数与验收、记录运行来源并统一回收。
  优先每人整条反应验证链；依赖关系和回传清单见 `group-compute.md`。
- 组员任务应先形成可复现任务包：固定输入和版本、独立任务 ID、资源/时长估计、
  单条执行命令、成功/失败输出约定与回传清单。当前未向任何组员发消息。
- 后续多随机种子、更多独立反应与更大训练数据规模尚未启动，须在新实验协议下分配。

节点详情见同目录下的 `mac-reporting.md`、`rog-readiness.md`、`rog-audit.md`
及 `omen.md`（各文件随实测创建）。
同步任务入口：`~/.agents/ai4chem--adaptive-ts-search/AGENTS.md`。
