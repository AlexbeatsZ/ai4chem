# Mac 原生量化计算与并行验收

## 范围

Mac 作为额外 CPU 量化计算节点，优先使用原生 arm64 工具和项目隔离环境。
本模块不改变 v0.5.0 冻结的 CUDA 搜索和 Gaussian TS/Freq/IRC 验收。
对已完成 Gaussian TS 几何的 PySCF 单点/解析梯度仅用于节点测试及独立数值检查，
不能计为新增 TS、频率、IRC 或正式 DFT 验证。
另有独立 HCN/HNC 小算例，实际验证 Mac 上的 TS 优化、频率和双向 IRC 全流程。
这些能力无需 Gaussian；Gaussian 安装包只在需要使用 Gaussian 引擎时才需要。

## 环境和资源

2026-11-23 实测：M3 Max，128 GiB 内存，12 性能核 + 4 能效核，
macOS 26.6.2。初始检查内存压力低、无 swap I/O。
PATH 与常见安装目录未找到 Gaussian/ORCA/NWChem/Psi4。
既有 `.venv` 继续保持分析环境；新环境位于 `environments/mac-qc/.pixi/`。
`pixi.lock` 固定原生 osx-arm64 Python 3.12.16 / PySCF 2.15.1 等依赖。
geomeTRIC 1.1.1 同样安装在这个隔离环境中。

初始并发为 3 进程、每进程 4 线程、每进程 PySCF 内存预算 4000 MB。
设置进程内 OMP/OpenBLAS 等线程数并调用 `pyscf.lib.num_threads(4)`，避免
三个任务各自占满全部核心。PySCF 内存预算不等于操作系统的硬内存限制。
该起点已通过三个十原子单点/梯度算例，但尚未证明是所有任务的最优并发，
也不是持续长时 IRC、训练或 128 GiB 满负载测试。

## 执行与证据

```bash
pixi install --manifest-path environments/mac-qc/pixi.toml --locked
pixi run --manifest-path environments/mac-qc/pixi.toml --locked python scripts/check_mac_qc.py \
  --manifest /absolute/path/inputs.json --output /absolute/path/fresh-output \
  --jobs 3 --threads 4 --memory-mb 4000
```

输出必须是新目录，输入 manifest 的 reaction ID 唯一；每项包含 symbols、
positions_A、candidate_key_prefix、Gaussian SCF 参考能量及基函数数目。
本次输入来自 OMEN 已完成的三个 ts_dft.extxyz，顺序和坐标保持。
父进程使用 spawn 启动独立 worker；每个 worker 保存完整 PySCF 日志及结果 JSON。
验收包括 SCF 收敛、有限能量和 N×3 解析梯度、基函数数目一致。
summary 另记录真实 PID、开始/结束区间、同时运行数量、进程峰值 RSS 和墙钟时间。
结果属于节点 readiness，JSON 的 passed 只表示这些检查通过。

2026-11-23T10:23:17Z 的最终验收：三进程确实重叠，整批 18.47 s，
每任务 15.29–18.36 s，进程峰值 RSS 1.49–1.68 GiB。
权威证据在 Mac `results/node-readiness/mac-qc/parallel-3x4-cartesian/`。

## 跨引擎的基组约定

不要只按方法/基组名称认定 Gaussian 与 PySCF 等价。
本次 Gaussian ts.log 明确为 `6-31G(d) (6D, 7F)`，三个反应基函数数目为
98、111、98。PySCF 默认球谐 d 函数；首次未指定 cart 的检查虽收敛，
但与 Gaussian 能量相差 0.003920–0.006067 Hartree，属于不同基组表示。
保留首次结果 `parallel-3x4/`，随后明确 `cart=True` 并检查基函数数目再复核。
本轮 CHNO / 6-31G(d) 没有 f 壳层；以后含 f 及更高壳层时不能把全 Cartesian
开关直接等同于 Gaussian 的混合 6D/7F 约定。

对齐后 PySCF 与 Gaussian 参考能量差仍为 -1.91e-5 至 -2.47e-5 Hartree，
保留在 JSON，不据此声明全流程等价。PySCF 使用未剪枝 99×590 积分网格，
Gaussian 原作业使用 UltraFine，两引擎的积分细节未完全对齐。
不得用这些单点覆盖 Gaussian 能量或其正式 TS/IRC 结果。

## TS、频率与双向 IRC 的开源路线

`scripts/check_mac_ts_irc.py` 采用 geomeTRIC 官方 HCN/HNC HF/3-21G 算例。
PySCF 提供能量、解析梯度、解析 Hessian 及 harmonic_analysis；geomeTRIC 提供
transition=True 鞍点优化和 irc=True 路径积分。初始结构来源见 examples/mac-qc/README.md。

```bash
pixi run --manifest-path environments/mac-qc/pixi.toml --locked python scripts/check_mac_ts_irc.py \
  --output /absolute/path/fresh-output --threads 4
```

2026-11-23T10:55:30Z 实际通过：TS 优化收敛，恰有一个 -1215.845 cm⁻¹ 虚频；
双向 IRC 同时运行、各四线程，分别经过 48/36 次能量/梯度评价后收敛，连接 HNC/HCN。
两端另行优化和频率分析，均无虚频，分别有四个正振动频率；键距确认连接不同端点。
全流程 2.53 s，仅代表这个三原子 HF 小算例。
结果和全部原始日志位于 `results/node-readiness/mac-qc/hcn-hnc-ts-irc-topology/`。

### PySCF/geomeTRIC 的拓扑衔接

PySCF 2.15.1 的 generic kernel 创建 PySCFEngine 后未建立 molecule topology。
geomeTRIC 1.1.1 在 IRC 坐标系设置阶段先访问 M.molecules，直接调用会出现
AttributeError。首次失败保存在 `hcn-hnc-ts-irc/`，不能因已有 TS/频率成功而标记整条通过。
项目在 IRC 阶段使用现有公开 customengine 接口，显式调用 engine.M.build_topology()，
传入 PySCF 解析 Hessian，保留收敛检查和每次能量/梯度评价。不修改已安装库。

当前通过的是独立 HF/3-21G 完整小算例，以及三个 wB97X/6-31G(d) 单点/梯度。
不能把这两项组合说成已完成十原子 DFT 全流程；后续生产 DFT 链需要单独验收。

## 若要使用 Gaussian 引擎

优先准备 Gaussian 16 C.02 Apple M-Series 原生二进制及适用许可，需能运行 g16。
Gaussian 官方平台表确认该版本支持 M 系列和共享内存并行，但表格更新日期为
2025-03-18，列出的 macOS 到 15.x，未列本机的 26.6.2。
收到软件后先完成实际 Opt/Freq、线程并发、后台存活和版本差异检查再接正式任务。
当前 Mac 没有安装 Gaussian；不需要用户重配既有 Python/Pixi 或提供 Windows g16w。
这不是 Mac 执行 TS、频率或 IRC 的前置条件，PySCF/geomeTRIC 路线已经实际跑通。

官方参考：

- https://gaussian.com/g16/g16_plat.pdf
- https://pyscf.org/user/install.html
- https://github.com/pyscf/pyscf/blob/master/examples/gto/00-input_mole.py
- https://pyscf.org/user/geomopt.html
- https://pyscf.org/user/hessian.html
- https://geometric.readthedocs.io/en/latest/irc.html

## v0.6.0-pilot 的十原子 DFT 进展

2026-11-23 后续单独固定 rxn0666 已知正对照，使用同一 MLIP 初值。
`run_mac_dft_chain.py` 实际 TS 优化五次评价收敛，解析 Hessian 得到完整 24 模，
唯一负频率 -1568.0823 cm⁻¹；与已有 Gaussian -1568.9618 cm⁻¹ 相差 0.8795。
TS 对齐 RMSD 4.5423e-5 Å、能量差 -2.2704e-5 Ha。当前并行双向 IRC 正在运行，
尚未完成稳定末端或全链 DFT 验收；这不是 v0.5 的新增样本。
接续读取 `docs/design/v0.6.0-pilot.md` 和 `results/coordination/v06/state.json`。
