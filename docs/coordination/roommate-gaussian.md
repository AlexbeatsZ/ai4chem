# 室友 Gaussian 节点

最后核实：2026-11-23T10:00:06Z（洛杉矶 2026-11-23 02:00）。
用户明确授权通过 SSH 使用室友电脑的现有 Gaussian 一起计算。
连接账户与地址保存在同步任务日志，不写入公开仓库。

## 已验证能力

- Windows 11，Intel i9-14900HX，24 核 / 32 逻辑处理器，约 16 GB 内存。
- Gaussian `C:\G16W\g16.exe`，实测日志为 Gaussian 16W A.03，与 OMEN 版本一致。
- 探测时未运行 Gaussian；可用物理内存约 5.7 GB，仅代表当时状态。
- 工作区：`%USERPROFILE%\Project\Workspaces\ai4chem-worker`。
  这是最小 PowerShell 任务包，不是完整科研环境或 Git checkout。
  SSH PATH 中未找到 Git；使用现有工具即可执行，无需安装程序。
- 首次水分子 Opt/Freq：ωB97X/6-31G(d)、中性单重态、4 线程 / 2 GB、
  UltraFine、XQC，约 20.60 s；优化完成、整个作业正常结束、三个正振动频率。
- 能量 -76.3882482121 Hartree；频率 1723.6309、3828.8085、3953.2144 cm⁻¹。
  这是节点可用性测试，不是 v0.5.0 反应评价或新增验证样本。

原始输入、日志、checkpoint 和摘要保留在节点工作区的
`results/node-readiness/<host>/20261123T095946Z/`。
Mac 另保存输入、日志与摘要；运行时证据位于 Git 忽略的 results/，不发布个人路径。

## 执行入口

任务包包含项目 `run_gaussian_job.ps1` 与 `scripts/smoke_gaussian_node.ps1`：

```powershell
powershell -NoProfile -File scripts\smoke_gaussian_node.ps1 -GaussianDirectory C:\G16W
powershell -NoProfile -File run_gaussian_job.ps1 -InputFile C:\path\job.gjf -GaussianDirectory C:\G16W
```

readiness 每次创建新 UTC 目录，保留完整输出并检查能量、优化结束和三条正频率。
summary.json 记录主机、实际 Gaussian banner、输入方法、资源、用时和成功/失败。
runner 的 Gaussian 环境变量只作用于进程，不更改用户或系统配置。

## 当前分配与下一步

- 本节点已完成节点测试，尚未分配正式反应。OMEN 的 rxn8204 已进入反向 IRC，
  继续由其现有 validator 执行，避免重复计算和交接运行中的 checkpoint。
- 后续按 `group-compute.md` 的唯一候选任务包分配，优先整条反应验证链。
- 先每次一个作业。正式 DFT 目前配置 4 核 / 4 GB，执行前重新检查可用内存和
  其他任务；readiness 的 2 GB 设置不能作为正式反应参数变更。
- 尚未验证本节点 SSH 断开后的长期作业存活。长任务派发前先使用独立 Windows
  owner 启动方式并核实其存活；不能仅凭这次前台 21 秒测试宣称后台调度已完成。
- 当前未部署自动任务队列、全局环境或第三方程序，Gaussian 安装保持原样。
