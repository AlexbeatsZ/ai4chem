# 多设备计算与交接

最后核实：2026-11-23 03:40 America/Los_Angeles（11:40Z）。
先读根 `AGENTS.md` 和 `docs/design/v0.6.0-pilot.md`。v0.5.0 已完成发布，
本轮继续单独的 v0.6.0-pilot。原始 OMEN 对话「测试 MACE 水分子 GPU 推理」
ID 为 `01a1e76a-d5bf-7881-8023-349081a2e012`；Mac 接续对话为
`01a206de-3464-7071-ad42-580c51d4e3e5`。

## 当前两个任务

| 任务 | 节点 | 当前证据 | 尚待完成 |
| --- | --- | --- | --- |
| rxn9456 唯一额外恢复的 DFT 诊断 | 室友 Windows / Gaussian 16W A.03 | TS 优化和 24 模频率通过，唯一虚频 -721.2605 cm⁻¹ | 双向 LQA IRC、稳定末端和映射端点验收 |
| rxn0666 十原子 DFT 引擎校准 | Mac / PySCF 2.15.1 + geomeTRIC 1.1.1 | 相同原始 MLIP 初值；TS 五次评价收敛，24 模且唯一虚频 -1568.0823 cm⁻¹ | 双向 IRC、稳定末端和 Gaussian 最终数值对照 |

协议和输入在 11:19:00Z 固定并提交，先于任何新计算。它们属于看过旧结果后的
定向诊断和已知正对照，不构成新 holdout。正式输出使用 `results/v0.6.0-pilot/`，
不覆盖 v0.5.0。各计算完成后仍需 `scripts/assess_dft_chain.py` 统一核验。

## 设备和权威目录

- Mac：`/Users/meta/Project/Workspaces/ai4chem`，协调、分析、原生 CPU 量化。
  原生环境 `environments/mac-qc/` 的 Pixi lock；分析环境 `.venv`。M3 Max / 128 GiB。
  当前全链在 `results/v0.6.0-pilot/mac/rxn0666-r1/`，主进程 5437。
- 室友：`%USERPROFILE%\Project\Workspaces\ai4chem-worker`，最小原生 PowerShell
  worker；i9-14900HX / 16 GB，既有 `C:\G16W`。当前任务 `results\rxn9456-v06-r1`。
  独立 native owner 35464，11:29Z 显式恢复，只复用了已正常结束的 TS。
- OMEN：`C:\Users\Meta\Project\Workspaces\ai4chem`，v0.5 CUDA/Gaussian 原始结果。
  本轮没有启动新的 OMEN 计算，不重跑已完成的冻结实验。
- ROG：同标准 Windows 项目目录，最近不可达。本轮不依赖其计算结果。

节点连接信息只存在同步任务日志 `~/.agents/ai4chem--adaptive-ts-search/AGENTS.md`。
代码/设计通过 AlexbeatsZ/ai4chem main 同步；运行时证据不进 Git。室友不自动同步
任务日志，worker AGENTS 和 task.json 必须随分配更新。

## 查看与恢复

一次性 Mac 接力器 PID 15336 已在独立会话中运行（新连接实测 PPID=1），
`results/coordination/v06/state.json` 是两节点的实时任务板；每 60 秒观察，最多 8 小时。
完成后自动回收原始 Gaussian 日志/checkpoint、独立几何验收，输出同目录 `report.md`，
并更新用户 outputs 中 `ai4chem-next-investigation.md/.json`。结束于 ready_for_review；
异常保留，不重算、不改协议、不自动发布。终态导出/传输已用真实 TS 日志往返验证。
不要重复启动接力器或 Gaussian；程序和科学任务的 PID 分别记录。

Mac 查看 `summary.json` 及各分支 `result.json`；现有进程运行时禁止重启。
原生 DFT 每支四线程，双向并行，避免外层再开三反应造成 24 线程嵌套。
完整分析命令：

```bash
.venv/bin/python scripts/assess_dft_chain.py --protocol configs/v0.6.0-pilot/protocol.json \
  --reaction rxn0666 --engine mac --folder results/v0.6.0-pilot/mac/rxn0666-r1 \
  --output results/v0.6.0-pilot/mac/rxn0666-r1/assessment.json
```

室友 worker 在 SSH 断开后的存活已通过新连接观察 native owner、Gaussian 和增长日志：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\status_gaussian_chain.ps1 -JobDirectory C:\path\job
# 只有状态 failed、确认旧 owner 和 native Gaussian 均已退出、诊断原因后：
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_gaussian_chain.ps1 -JobDirectory C:\path\job -Resume
```

显式 Resume 会归档旧状态，复用已正常结束的阶段，拒绝覆盖不完整日志。
首次 TS 后的读取错误来自 PowerShell 对 .NET MatchCollection 的负索引，已改用
Count-1；真实 TS 结果由 PowerShell 与 Mac ASE 解析相互核对，未重新计算。
活动 Gaussian log 上 `Get-Content -Tail` 曾忙循环，已停止两条只读 helper；
新 status 工具以 8192 字节单次快照读取活动日志，不追逐变化的 EOF。
无用户/系统环境、代理、软件安装或计划任务变更。

## 已完成 v0.5.0 与后续边界

96/96 策略单元、83 次唯一尝试；原始 S0/S1/S2 saddle+IRC 为 3/3/3，微调为
5/6/6，分母均 12。严格全阶段为 3/3/3 和 5/5/5。预选的 rxn2737/rxn8204/rxn0666
三个 Gaussian 候选全部通过，九策略单元复用它们，每策略 3/3；条件能垒 MAE
0.206068 eV。49 回归、Mac 14,084 项独立搜索产物核对及最终 verification 通过。
v0.5.0 2026-11-23T11:10:37Z 发布，8 资产已验收；旧 coordinator ready_for_review
只是发布前保留快照，不代表尚有计算。不要再次启动旧 coordinator。

若扩大到多反应/多随机种子，先写新独立协议和任务包。组员可分担完整独立反应链，
约一天回传适合大批量，见 `group-compute.md`。当前没有必要组织额外大批量。
节点历史见 `omen.md`、`mac-reporting.md`、`roommate-gaussian.md`、`rog-audit.md`。
