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

2026-11-23 11:21Z 已分配 v0.6.0-pilot 的 rxn9456（C3H4O3，十原子），
初始可用内存 7.88 GiB、无其他 Gaussian。固定 4 线程/4 GB、ωB97X/6-31G(d)。
TS 正常结束且优化收敛，完整 24 模，唯一虚频 -721.2605 cm⁻¹；Mac ASE 与
PowerShell 独立解析一致，MLIP→DFT TS 对齐 RMSD 为 0.01599 Å。
自动读取曾因 .NET MatchCollection 负索引出错，Count-1 修复后显式 Resume，
保存旧状态和全部日志，复用原 TS checkpoint，未重算。11:29Z owner 为 35464，
在 `results\rxn9456-v06-r1` 继续双向 LQA IRC 和末端 Opt/Freq。

- `scripts/run_gaussian_chain.ps1`：原生 Windows 独立 WMI owner + 单作业文件锁。
  已用新 SSH 连接观察 Gaussian/owner 存活及增长日志，证实后台会话存活。
- `scripts/status_gaussian_chain.ps1`：状态、PID 和最多 8192 字节日志快照。
  活动 Gaussian 文件上 Get-Content -Tail 曾忙循环，已结束两条只读 helper。
- `scripts/export_gaussian_chain.ps1`：只导出终态科学输入、日志、checkpoint 和状态，
  不收集 scratch。实际 TS 日志导出回传通过；失败证据同样返回。
- Mac 独立接力器每 60 秒观察，完成后统一判定映射端点，才可标记 DFT verified。
  目前仅 TS 通过，完整验证仍未完成。接续见协调 README 和 v0.6 pilot 协议。
- 首先每次一个作业；无全局环境、软件安装、代理或计划任务变更。
