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

November25恢复节点时发现：Windows PowerShell直接用`$result = & ssh ...`
捕获终态导出回执会挂起，中转进程在Mac240秒超时后残留；Status流式输出
正常不代表Export也正常。两个relay的Export改用`worker_transport.ps1`原生
Process，先异步排空stdout/stderr，再有限180秒WaitForExit；非零退出传出
错误，超时只终止传输进程。不能用Start-Process再Refresh后空ExitCode当作
成功。Windows真进程测试覆盖双流各1MiB、退出7和500ms超时；继续保留终态
ZIP身份校验，科学进程、数据预算与固定有限collector不因传输修复而重启。

## November24 清理后台后的双槽位修订

用户在关闭Chrome/Codex/微信/Steam等闲置应用、可用内存4.49→8.01GiB后，
明确授权继续加入更多计算。17:18Z实测free7.99GiB、CPU47%，当前0255正向IRC。
`configs/roommate-pool-20261124.json`先冻结两个worker上限；每worker仍四线程/
2GB，新增任务启动前要求至少6GiB可用，不杀已有计算释放内存。低于3GiB记下
内存压力，等待自然完成释放；内存恢复至6GiB才可继续接新任务，不允许第三个
并发。此时原来的“每次一个原生Gaussian”由本修订取代，科学输入/接受判据
保持原值。

`scripts/run_gaussian_pool.ps1`仅停止确认精确命令和owner.lock的串行调度父进程，
保留其正在运行的链与Gaussian子进程；新WMI pool owner取得原queue.lock，
核对原worker命令/PID后接管，不能重启IRC。pool-start.json保存原状态、认领
的科学child和新owner；pool-worker-start.json禁止隐式再次执行。所有链/force
在claims下用CreateNew一次认领，已有证据或未匹配的活动Gaussian拒绝新分配。
每链仍是独立原生子进程，可以继续用既有三个collector和assessor。

池先把未开始的3342/4757接到空槽，之后才启动原4608 force储备。不是新增
模型训练/新holdout，不复制同一几何任务。原输入queue.json保持不改；资源
调度另记execution。此前24h累计串行预算改为从pool接管起24h墙钟接新任务，
最多两项并行；到点只排空当前项，不中断科学计算，理论不超过48worker小时
加当前项收尾。实际compute_seconds为各终态作业耗时之和，不能当作墙钟。
此增加的计算预算来自用户本次授权，旧串行成本和并发差异均保留。

观察器仍查询同一个queue-state.json；status额外给active/pending/deadline。
串行owner退出到新state写入间用pool-start.new_owner_pid判断handoff，不能误报
丢失owner。导出器除调度owner外还检查active科学child，防止调度器异常时
导出仍增长的日志。无科学自动重试、无标签自动训练合并、无全局环境变更。

## 北京时间 November25 15:00 前的 Mac 增载

November24 18:37Z用户说明至UTC+8的15:00都不需要使用电脑，允许更高负载。
当前当地为November25 02:37，截止明确为2026-11-25T07:00:00Z。
Mac实测16核/128GiB，约50%CPU空闲和50GiB未使用内存；v0.8只剩一个收尾
任务，因此将已冻结的27–30四组全部并发，每组仍四线程，原两组的科学进程
原样保留。OMEN同轮GPU86%/3709MiB，继续单CUDA任务及原锁序；室友保持
已授权的双槽位池。只改变执行并发，七对种子、数据、40epoch与评价预算不变。

`configs/mac-idle-window-20261124.json`单独记录临时执行窗口。运行中的旧
Mac调度器不支持动态扩容，新入口先核对精确命令、state和全部直接子进程，
暂停旧调度父进程，保存handoff快照，再仅终止该调度PID；新调度器取得原
owner.lock并接管原科学child。任一核对失败恢复旧调度器，禁止隐式重试。
新入口：原`--protocol ... --node mac`命令加`--execution
configs/mac-idle-window-20261124.json --adopt-owner <已核对旧调度PID>`。
这不是一般重启入口；已经接管的队列必须先读真实state和handoff证据。

到07:00Z后最多30秒内接新任务容量恢复2，对所有仍活跃的成对worker及其
训练/评价子进程设nice至少10，后续子进程继承低优先级。已运行的科学计算
自然排空，届时活跃任务数可能仍大于2；这是让交互程序优先获得CPU，不承诺
立即将CPU占用降至某个百分比。没有强杀训练或丢弃进度。此边界已用真实
合成进程跨过短截止时间验证，实际15:00回落尚待届时观察。
