# 本地三节点持续计算

2026-11-24 用户明确要求给室友电脑、OMEN 和本机发布至少够运行18小时的任务。
本轮直接使用已有的已验收环境；云服务器采购、原生Linux Gaussian验收和其他
组员的分发仍保留各自的门槛。这次授权直接覆盖三个本地节点的有限任务下发。

## 科学任务和去重

`configs/overnight-20261124.json` 在新增训练前冻结七组成对取样重复。
原先准备给五人的种子20261124–28按固定归属落地：前三组给OMEN，后两组
给Mac；再在训练前增加20261129/30两组给Mac。原ZIP保留作准备证据，其成员
01–05已被本轮认领，不得再交给其他机器运行。与v0.8/v0.9的种子16–18不重叠。

每组按random4096→path4096顺序，从同一MH1/omol foundation重新微调：
256条训练反应、各4096帧、同一validation60、float64、batch4、lr1e-4、
energy/force10/1、E0estimated、40实际epoch、patience100。每模型40960
估计更新，共14个新模型573440次更新。共享标签不增加DFT标签预算。
验证集用于checkpoint；meta128只作开发评价；prospective holdout仍封存。
最终报告全部七个配对差值、离散度、失败和成本，旧开发seed16另列。

OMEN单worker等待v0.9和v0.7的实际文件锁释放，并检查前置终态才占CUDA。
六个新模型按现有每模型约3.5小时，预计新增约21小时，另有现有任务剩余时间。
Mac新增两个四线程worker，各自顺序完成分配的配对；八模型按最近约14–16
分钟/epoch的CPU观测，预计约36–44小时。并发变化会影响耗时；这是任务储备
估计，无法承诺失败或硬件异常下仍能完成同样时长。原有v0.8worker保持运行。

## 室友Gaussian任务

先认领原组员开发包中的AIMNet2-rxn候选rxn0255、rxn3342、rxn4757，原member
01–03不得重复派发。rxn8694已经独立验证，不能重复计算。三条都是事后开发
诊断，不能计作新的独立holdout或改写v0.5结果。

节点实测空闲约4.89GiB，将新任务资源修订为四线程/2GB，每次启动要求4GiB
空闲（Gaussian2GB+2GiB余量）。原方法、几何、80步TS、LQA每向320点/步长5、
完整模式/唯一虚频、稳定端点与映射连接判据不变。新job_id和protocol记录资源
修订及原准备包ID，不覆盖旧准备输入。每次只有一个原生Gaussian作业。

三条链后，依次计算v0.8冻结的512条训练反应的能量/力储备：先按区域和端点
反应类别轮转、各桶按原子数和ID排序。每反应选已有训练帧中最高能帧，保留
一个reference，再用固定seed2026112401生成0.01/0.03/0.06/0.10Å各一对正负
扰动，共512个复算参考+4096个新离路径几何。此规则不是不确定性主动学习。
每个任务是同方法固定几何force计算，不做几何优化；不生成无用checkpoint。

有限queue累计原生Gaussian执行时间达到24小时后不再启动新作业，允许当前
作业正常结束；资源等待不算计算时长。4608项是备用池，不要求全算完，更不
将准备项当作成功标签。三个连续force失败则停止保留错误，禁止用失败空转
填时长。原始日志仅通过最低输出完整性检查，必须独立核对全3N力、坐标旋转、
6D基组/网格/方法一致性和参考差异后才能训练；禁止自动合并新标签。

## 执行与恢复

- Mac：现有mac-mace解释器运行`scripts/run_paired_queue.py --protocol
  configs/overnight-20261124.json --node mac`，任务级caffeinate，不改电源配置。
- OMEN：原生`scripts/launch_paired_queue.ps1 -Protocol configs\overnight-20261124.json`
  创建WMI owner，持有前台WSL；现有ai4chem环境，不再安装依赖。
- 室友：`scripts/run_gaussian_queue.ps1 -QueueDirectory <已解压roommate目录>`
  创建原生WMI owner，持有各完整链和force子进程。每链独立子进程的终态PID
  可供既有exporter/独立assessor在整个queue结束前回收。

源代码和配置随仓库提交；输入ZIP、完整queue.json、状态、日志和返回证据在
git忽略的`results/overnight-20261124/`。启动前检查已有输出和活动owner；所有
任务仅一次尝试，训练单臂失败后继续其配对及其他固定任务，禁止自动重试或
覆盖中断证据。恢复时先读三台真实状态，不把提交PID当作计算已启动。

`scripts/collect_local_queue.py`在Mac有限观察72小时，经OMEN中转终态队列ZIP，
核对字节数/ZIP CRC/清单数量；不启动化学计算、不重试作业。队列ZIP保留全部
输入/状态/原始日志和force receipt，省略checkpoint；三个完整链另用既有
`continue_gaussian_relay.py`各自回收checkpoint并调用独立assessor。断线重连
只重试传输。终态ZIP只在清单/长度/JSON身份一致时复用，不覆盖差异证据。
