# 云节点与组员任务交接

用户规定：先 OMEN 本地实际安装/计算验收，再用其提供的单台服务器验证，
成功后才分发给五名组员。当前未创建服务器、未分发，也不发送组员消息。

## 当前可执行入口

Linux x86_64/Ubuntu，从 `~/Projects/ai4chem` 仓库运行：

```bash
bash scripts/cloud_worker.sh install
bash scripts/cloud_worker.sh doctor -- --device cuda
bash scripts/cloud_worker.sh smoke -- --foundation "$PWD/models/mh1-omol.model" \
  --output "$PWD/results/node-readiness/cloud-worker/server-r1"
bash scripts/cloud_worker.sh train -- --task /absolute/bundle/task.json \
  --output "$PWD/results/group-tasks/member1-r1" --detach
```

训练清单包含相对自身目录的 train_file、valid_file、foundation_model、name、
seed、settings、expected_epochs、max_wall_seconds。上传包移动后仍可运行；
不能覆盖已有输出。detach 记录 PID、精确命令和日志，但服务器停机会中断计算。
安装器只使用项目引导器/独立环境，不改 base、系统驱动、PATH、代理。
普通云镜像需已有 bash/curl/tar/python3/flock 与适配 NVIDIA 驱动；缺失就报告。

Gaussian 用已有安装，原生 Linux 冒烟入口为：

```bash
# 参数指向包含 g16 可执行文件和 bsd/ 的实际 g16 目录。
bash scripts/cloud_worker.sh doctor -- --device cpu
"$PWD/environments/cloud-worker/.runtime/env/bin/python" scripts/cloud_gaussian_smoke.py \
  --g16-root /actual/g16 --output "$PWD/results/node-readiness/cloud-worker/native-g16-r1"
```

没有在公开包里携带或安装 Gaussian；云节点的原生二进制仍须实际验证。
用户明确要求 OMEN WSL 先安装并运行原生 Linux Gaussian。此前的 Windows
桥接验收方向错误，不满足该要求；`validate_cloud_omen.sh` 中的旧桥接步骤待
提供 Linux 安装包后替换，当前禁止将其结果当作本地部署通过。用户会稍后
提供 g16，本轮先检查已有本地研究的进度。尚未编写/验证 Gaussian 安装器。

## OMEN 本地验收

- 启动 2026-11-24T02:39:30Z（November 23 本地），原生 owner19712，WSL39116。
- `scripts/launch_cloud_validation.ps1 -ValidationId omen-bootstrap-r1`。
- 隔离 Python 前缀 `/home/meta/.local/opt/miniforge3/envs/ai4chem-cloud-worker`。
- 项目范围 micromamba2.6.0 引导器；从零创建 Python3.12.16 并按现有 lock 安装。
- 日志 `results/coordination/cloud-validation/omen-bootstrap-r1/{launch.json,stdout.log,stderr.log}`。
- 实际验收 `results/node-readiness/cloud-worker/omen-bootstrap-r1/validation_summary.json`。
- 首次安装已失败：2026-11-24T03:00Z 退出码2，固定依赖下载中的
  nvidia-cudnn-cu13 出现 SSL record layer failure；只创建了独立 Python，
  没有通过 MACE 推理/短训练/pip check 或 Gaussian。原 GPU 研究保持。
- 已检查 Mac 下载/相关资料、OMEN 下载/文档/桌面/资料盘和 WSL 常用软件
  目录；仅发现 Windows G16W ZIP/现有安装，未找到 Linux 安装包。

## 2026-11-23 配置比较（页面实时观察）

| 资源 | 可选单卡/CPU配置 | 计算资源报价 CNY/h | 页面状态 |
| --- | --- | ---: | --- |
| RTX4090 | 10vCPU/32GB，24GB显存 | 3.60 | 中卫二区/山东一区可选 |
| RTX3090 | 10vCPU/32GB，24GB显存 | 3.30 | 中卫一区可选 |
| RTX5090 | 14vCPU/120GB，32GB显存 | 5.50 | 山东二区可选 |
| V100-32 | 8vCPU/32GB，32GB显存 | 5.60 | 北京一区可选 |
| V100-PCIe-32 | 6vCPU/26GB，32GB显存 | 3.14 | 缺货 |
| L4/T4/A10/L20/A100 | 各自单卡 | 4.10/3.50起/6.12/11.25/16.16 | 缺货 |
| 纯CPU | 4vCPU/8GB | 0.60 | 中卫一区/二区可选 |
| 纯CPU | 8vCPU/16GB | 1.20 | 中卫一区/二区可选 |
| 纯CPU | 16vCPU/32GB | 2.40 | 中卫一区/二区可选 |

这只是报价和库存，不是 MACE 吞吐实测。4090比3090贵9.09%，只有速度超过
相同比例才有更低计算费用；5090相对4090贵52.78%，额外显存/内存也需任务证明。
没有在租赁卡上执行 benchmark，不能断言已证明最优每元训练吞吐。

页面预选4090单卡、10核/32GB、Ubuntu22.04的Jittor镜像、100GBSSD，按量总价
3.70元/h；出口流量另计1元/GB。2/4/8/24h静态预算约7.4/14.8/29.6/88.8元，
不包含额外流量。镜像只是宿主，使用本项目独立 MACE 环境，不使用计图训练。
未点“立即创建”。GPU研究先单卡验证；Gaussian 批量优先另看纯CPU，避免占卡空转。

原始报价/选择截图保留在 Mac 的
`results/coordination/cloud-selection/{quotes-2026-11-23.json,selected-4090.jpg}`。
包天模式切换后提供的是其他大型资源，没有核实到4090单卡包天优惠，不计入比较。

## 已准备、等待云验收的五人任务包

前三份是已有严格 MLIP 候选的独立 DFT 完整链（AIMNet2-rxn rxn0255、rxn3342、
rxn4757）；rxn8694 已在室友节点，禁止重复派发。后两份各选训练区域内最小的
一个成键、一个裂解、两个重排，分别算原始训练几何和固定0.03Å RMS扰动：
每人8个能量/力单点，4个已有参考用于对齐，4个新几何用于后续定向训练。
新数据隔离待参考方法/网格/基组/力坐标系验证，不能自动混入训练。

具体IDs/事后开发身份固定在 `configs/group-dft-development.json`；五份ZIP位于
`results/group-jobs/2026-11-23-r1/member-0{1,2,3,4,5}.zip`。只是准备，未运行或
派发。每份README写了输入、预算、返回内容及Linux/Windows入口，不改现有队列。
Linux完整链/单点队列使用 `scripts/run_gaussian_portable.py`，沿用原Windows
科学判据和独立assessor。新增NH3/HF/3-21G小控制用于本地编排验证，不计科学结果；
WSL桥接成功也不代表云端Linux二进制验收。
