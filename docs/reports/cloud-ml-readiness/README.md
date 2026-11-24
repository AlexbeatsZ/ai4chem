# 云训练准备与本地无人值守验收

2026-11-23 晚（UTC 2026-11-24），OMEN 新隔离环境通过实际 CUDA 推理、
力差分、两 epoch 训练和保存重载；后台完整流程在 SSH 提交返回后继续执行，
自动完成并归档，Mac 收回归档后加载最终模型并确认全部参数有限。
这是基础设施控制，使用 MLIP 生成的水分子标签；化学精度、真实云吞吐、
原生 Linux Gaussian 分别等待对应验收。机器可读记录见 [summary.json](summary.json)。

## 实际通过

| 检查 | 结果 |
| --- | --- |
| 新环境 | OMEN WSL Ubuntu 24.04，独立 ai4chem-cloud-worker 前缀，Python 3.12.16 |
| 固定软件 | torch 2.15.0+cu134、MACE 0.3.17、ASE 3.29.0；完整 lock 版本检查和 pip check 通过 |
| CUDA | RTX 4060 Laptop GPU，约 8 GiB；真实能量/力推理通过 |
| 数值力 | 有限差分与解析力的绝对差约 1.59e-7 eV/Å |
| 官方训练 | 两个实际 epoch，标准模型保存及重载推理通过 |
| 后台完整链 | Windows 独立 owner 49228 托管前台 WSL；SSH 提交返回后继续运行，最终退出码 0 |
| 队列与返包 | 固定任务 completed，自动重试 0；返回包所需模型、配置、状态、日志存在 |
| Mac 回收 | 返回包读取成功，最终模型在 Mac CPU 加载且参数全部有限 |

首轮隔离安装的 cuDNN SSL 失败证据保留；第二轮使用连接/续传重试、保持
证书校验并完成安装。研究用 ai4chem 环境、既有训练和代理均保持。

首次通用 nohup 在 WSL 中未能存活到 SSH 退出后；第二次 Windows 启动器把
`-d` 等选项全加引号，导致 WSL 交给默认 zsh，训练没有开始。两次基础设施
启动失败均保留。修复后的 WMI 原生 Windows owner 托管前台 WSL 通过完整链；
WSL 的通用 detach 入口现在明确拒绝假提交。普通 Linux 云节点继续用 Linux
后台入口，并保留实际云节点验收门槛。

完整原始控制归档为 567,837,409 字节。保留最终模型、配置、状态与日志，
排除优化器 checkpoints 后回传包为 95,488,341 字节、37 条目；Mac 实际收回
并检查了该包。节点原始 checkpoints 仍在原输出目录，未通过重新训练生成。

## 五人交付

已准备五份自包含 ZIP，每份 72,566,277 字节，包含部署代码、依赖锁、
foundation、两套训练帧、validation60 和固定任务。每人使用独立训练种子
20261124–20261128，顺序跑 random4096/path4096；每模型 40 epoch、
估计 40,960 更新、12 小时上限。合计十个新训练均未启动。

成员在解压目录只执行对应命令，例如：

```bash
bash member-01/start.sh
```

后台自动安装/检查、做小控制、顺序训练、保存模型并打包。任务失败保留错误，
继续下一个独立固定任务，不自动重跑。最后回传 `member-01/result.tar.gz`，
成功和失败用同一种交付方式。负责人统一评价全部种子的配对差异、离散度和
成本；旧 v0.7 单种子的开发选型身份保留，不挑最好种子。

现有 OMEN 单组 4096 帧/40 epoch 约 3.5 小时，故每份约一晚是本机估算，
未承诺云 GPU 耗时。八项部署/搬运/失败续接/WSL 拒绝假后台检查通过；
包内两组各 4096 帧、验证 600 帧及能量/力标签实际读回通过。

## 证据与边界

- Mac：`results/node-readiness/cloud-worker/omen-ml-compute-r2/` 存 runtime 和首轮实际计算收据。
- Mac：`results/node-readiness/cloud-worker/omen-unattended-water-r3-return.tar.gz` 与 `unattended-return-verification.json` 为实际返回及本机加载证据。
- OMEN：同项目 `results/node-readiness/cloud-worker/omen-unattended-water-r{1,2,3}/` 保留失败/成功启动与完整原始输出；原生 owner 最终状态在 r3/native-owner.json。
- Mac/OMEN：`results/group-training/paired-sampling-r1/` 存五份准备包和清单。
- 没有创建云服务器、运行十个新研究任务或向组员发送消息。原生 Linux Gaussian 与用户提供云节点的原有验收/分发顺序保持。
