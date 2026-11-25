# 室友 Gaussian 节点

最后核实：2026-11-25T12:13:40Z（洛杉矶 November 25 04:13）。
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

## 当前分配

November25 11:16Z用户因朋友需要用电脑要求暂停；无Gaussian在算，精确挂起
原pool owner29700及其console21336，本机四个回收器同步挂起。12:13Z用户
重新允许使用后，核对原命令、创建时间、全部挂起线程和未过期截止，原样
恢复这两个进程及四个回收器，没有另起pool或重算旧项。12:13:40Z实际观察
到训练侧force项rxn0559-r10-minus的g16运行，可用内存6.83GiB；新增标签仍
隔离等待对齐。原November25 17:24:51Z（洛杉矶09:24）接单截止不延长，仍然
max2、每项四线程/2GB、6GiB接单门槛，到点排空。两次操作回执保留在worker
queue的user-pause-20261125.json与user-resume-20261125.json；Mac核对证据
位于results/coordination/model-v10/*user-resume*。

用户在November24明确要求本节点、OMEN、本机都有至少18小时工作储备。
16:13Z原serial owner13616启动队列；November24 17:24:51Z按用户进一步授权，
退休的只有串行调度父进程，新pool29700接管原0255 child33396，IRC进程保持。
新增3342 child26464/g16 29852并行TS；原0255 g16 26692仍在正向IRC。
17:25:33Z fresh SSH实际看见两组g16/l502，free7.566GiB。4757随后优先接空槽，
再接原force储备；原候选不可重复分配。

启动前实测free4.886GiB、无Gaussian、磁盘余量430GiB。资源修订四线程/2GB，
启动需4GiB空闲（2GiB余量）；方法/输入/频率/IRC预算和判据保持。运行后free
约2.99GiB是已分配内存后的状态，不用据此增加并发或更改系统配置。

原单任务资源不变；双槽位新增前需6GiB可用，最多两项并行，各四线程/2GB。
三链后接训练池512反应的4608项force储备（512参考、4096扰动）。新pool从
17:24:51Z起24h墙钟（November25 17:24:51Z）停止接新项并排空，旧累计串行
24h门槛被本次用户授权取代。所有新force隔离，待完整力/坐标/DFT参考对齐后才能训练。
运行状态为worker `results/overnight-20261124/roommate/queue-state.json`；
协议、任务输入、每项receipt保留于同目录。Mac每链回收器独立调用assessor，
总队列回收器经OMEN传输终态ZIP，有限72h，不创建计划任务或重算任务。

先前rxn9456 r2、rxn8694均已完成独立验收。8694于02:35Z完成，Mac于03:43Z
完整验收，电子能垒2.625419123eV；owner5232为历史。不得因旧文档重启。
详细科学定义见`docs/design/local-compute-queue.md`和`expert-diversity.md`。

节点继续使用现有Gaussian，不加入dot-agents自动同步，不改安装/全局环境/
代理/计划任务；账号与地址仅保留在同步私有任务日志。
