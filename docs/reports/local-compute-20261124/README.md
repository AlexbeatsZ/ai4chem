# 室友原队列恢复与三条开发候选的独立核对

2026-11-25用户重新允许使用后，恢复原调度器与结果回收器；当前继续计算
冻结训练池的离路径受力储备。原双槽位、四线程/2GB、6GiB内存门槛和
November25 17:24:51Z接单截止保持不变。新力标签仍隔离等待独立对齐。

回传故障修复后，三条原有Gaussian候选的全部归档成功回到Mac，ZIP CRC和
每项身份/字节校验通过。以下是恢复已有结果的核对，没有重新执行化学任务。

| 候选 | 独立核对结果 | DFT电子能垒 | 说明 |
| --- | --- | ---: | --- |
| rxn0255 | TS/唯一虚频、双向IRC、稳定端点及映射连接通过 | 3.085938552 eV | dft_verified=true |
| rxn3342 | TS/唯一虚频、双向IRC、稳定端点及映射连接通过 | 4.821321403 eV | dft_verified=true |
| rxn4757 | 失败证据完整保留 | 不报告 | 正向IRC达到冻结的320点预算；无重试 |

这些候选来自事后开发搜索，既不是独立prospective holdout，也不能用来
改写v0.5结果、外推整体成功率或声称新512模型已经完成搜索评价。

摘要为本目录summary.json；完整原始输入、日志、checkpoint、assessor结果
在Mac的results/coordination/local-queue-20261124/chains/<job>/evidence。
回收器终态分别为assessed/assessed/worker_failed，原科学queue仍继续force。

传输修复v0.7.9-roommate-transfer已发布源码和验收资料；它不包含新模型权重。
