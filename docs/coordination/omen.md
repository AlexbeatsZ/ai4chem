# OMEN 节点

Last Verified: 2026-11-23T09:29:56Z。权威科学结果节点，Windows + WSL Ubuntu-24.04。

- 项目：`C:\Users\Meta\Project\Workspaces\ai4chem`。
- Mamba：`/home/meta/.local/opt/miniforge3/bin/mamba`，环境 `ai4chem`。
- 冻结提交：`761e580`；当前全部 96 个搜索单元已完成，端点阈值诊断改动已整合。
- Gaussian rxn0666/rxn2737 完成且验证通过；rxn8204 正向 IRC 第 61 点附近。
- coordinator PID 41374，running / wait_ready_dft，等待 validator 42193。
  本次看到原生 g16.exe 48760 及计算 link，日志持续更新；PID 需重新核实后使用。
- 本轮继续运行冻结实验，核心实现不变；CPU分析与独立核验分给 Mac/ROG。
- coordinator 与每阶段日志：`results/coordination/`；以实际 `omen-state.json` 为准。
- 恢复命令和任务阶段见 `README.md`。

本文件不以旧 shell session ID 作为恢复入口；shell session 属于旧对话，跨设备
应查询真实 PID 和保存的结果。运行状态与证据都写到项目目录。

## 接续实测

2026-11-23T05:53Z：48 项回归全部通过。等待中的旧 coordinator 已有界停止，
新 coordinator PID 41115，仍附着原始 GPU 搜索 PID 39916；原始科学进程未动。
提前 DFT PID 41129 首次启动因 WSL `UtilAcceptVsock` 超时失败，未产生
Gaussian 日志或原生 g16 进程。原失败 JSON/TXT 保存在候选目录。

2026-11-23T05:56:59Z：改为 Windows 持续 wsl.exe 启动器，coordinator
Windows PID 32152、ready-dft Windows PID 50908；Linux PID 与当前阶段见状态文件。
外层日志 `results/coordination/coordinator-20261123T055659Z.log/.err` 和
`ready-dft-20261123T055659Z.log/.err`。正式九单元选择尚未创建。
运行状态仍以 JSON 和真实进程为准。

2026-11-23T05:59Z：SSH 直接 Start-Process 的 wsl.exe 未能在会话结束后存活，
现通过 WMI 创建独立 PowerShell owner。coordinator owner PID 6588，
ready-dft owner PID 19496；Linux coordinator PID 41238。在启动 SSH 退出后的
独立调用中确认 g16.exe PID 15252 和 l502.exe 运行，ts.log 已写出 17 KB，
证明 Gaussian 已实际启动。前一次基础设施失败仍保留，未计为化学失败。

## Durable handoff checkpoint

At 2026-11-23T06:51:20.960180+00:00, all 96 search units and 49 regression tests passed; coordinator Linux PID 41374 remains running in wait_ready_dft. Native Gaussian processes were separately observed after launcher SSH exit. rxn0666/rxn2737 are in IRC, rxn8204 queued. Formal nine-unit selection exists, deduplicated into three jobs. The two current validator PIDs were [41255, 42193]. These are observations; recheck before recovery.

Task-owned transfer archives and temporary launch/inspection scripts were cleaned after integration. Authoritative scripts are in project scripts/, logs/results remain in results/. Scientific processes and Windows owners were left running. ROG cleanup remains unverified while its connection is unavailable.
