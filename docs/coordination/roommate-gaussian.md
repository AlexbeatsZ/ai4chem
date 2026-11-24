# 室友 Gaussian 节点

最后核实：2026-11-24T00:34Z（洛杉矶 November 23）。
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

v0.6 rxn9456 r2 已完成且经 Mac 独立验收通过；r1 checkpoint 续算失败保留。
旧 owner 35464/39032 是历史，不能作为活动任务恢复。

2026-11-24T00:31Z 经 OMEN 现有 SSH 身份再次连通：无原生 Gaussian、
可用内存 8.418 GiB。00:33:18Z 新分配 AIMNet2-rxn/rxn8694，12 原子，
四线程/4 GB，ωB97X/6-31G(d)，中性单重态，UltraFine/XQC/NoSymm。
Native owner 5232；新 SSH 会话观察 g16/l502 和增长 TS 日志，后台存活。
工作目录 results\rxn8694-aimnet2-rxn-gaussian16w-r1。

协议 configs/v0.7.0_diversity_dft.json；本候选单独协议为
configs/v0.7.0-diversity-dft/aimnet2-rxn-rxn8694.json。复用 TS 80 步、LQA
每向初始 320 点/步长 5，无额外续算。完整 30 模、唯一低于 -50 的负模、
双向 IRC、全非负末端和映射端点仍须独立验收，尚不能标记 DFT verified。

run_gaussian_chain.ps1 创建 WMI owner；status_gaussian_chain.ps1 读取有界日志；
export_gaussian_chain.ps1 只导出终态证据。OMEN 中转 relay_gaussian_chain.ps1，
Mac 有限接力 continue_gaussian_relay.py。这是事后开发候选，不改变 v0.5
结果，也不充当 prospective holdout。

每次一个四线程/4 GB 全链，先检查资源和进程。不改变全局环境、程序安装、
代理或计划任务，不复制私钥，节点不加入 dot-agents 自动同步。
