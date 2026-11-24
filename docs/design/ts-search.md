# v0.4.0 端点驱动的过渡态搜索

## 范围与数据使用状态

使用 v0.3.0 已评价的 210 条反应作为开发池，固定十条反应，比较原始 MH-1/omol
与 mh1_full_256。全部分子为原子映射保持一致的中性闭壳层 CHNO 小分子。
本轮的成功率用于开发验证，外部反应和独立最终测试仍需后续建立。

`configs/v0.4.0.json` 固定优化预算和验收阈值。先按化学式各取一条反应，按端点
成断键数之和、reaction ID 排序。再补充尚未覆盖的几何连接变化类别，最后补充
不同成断键数的组合，优先较小分子。几何键采用 ASE 共价半径乘以 1.2 的规则。
该规则提供可复现的结构分类，详细化学机理需要人工核验。

`ts_search.endpoint_pool` 只访问 atomic_numbers 及 reactant/product 的 positions。
准备后的 `inputs/` 仅保存端点 XYZ，搜索端点携带坐标和元素。DFT 中间图像、
能量、受力及 TS 子组在评价程序中于搜索完成后加载。回归测试使用只有端点的
HDF5 夹具检查此边界。

## MLIP 搜索

1. 在各模型自己的势能面用 ASE FIRE 优化两个端点，最大原子受力阈值 0.02 eV/Å，
   最多 250 步。记录优化后是否保持输入端点的原子映射键集合。
2. 保持原子顺序，用仅包含正旋转和平移的刚体配准对齐产物。
3. 构建含两个端点的十图像路径，用 ASE IDPP 初始化八个内部图像。
4. 使用 improved-tangent NEB，弹簧常数 0.1 eV/Å²，逐次去除图像间刚体运动。
   FIRE 先运行最多 150 步至 0.15 eV/Å，再开启 climbing image，最多 500 步至
   NEB 有效最大受力 0.05 eV/Å。保留初始化路径、最终路径、轨迹及优化日志。
   `mace_neb.py` 复用 MACE 的 AtomicData 和 Batch 接口，将各图像作为独立图批量
   推理。每条新路径先与标准 ASE MACECalculator 逐图像结果核对，能量最大差
   允许 1e-7 eV、受力最大差允许 1e-8 eV/Å。早期三个已完成搜索使用逐图像接口，
   其后搜索使用已核对的批量接口，优化算法和预算相同。
5. 以内部最高能量图像作为 Sella 一阶鞍点优化初值，使用内部坐标，最多 150 步，
   最大原子受力阈值 0.01 eV/Å。即使 NEB 未收敛，也保留其候选精修结果供诊断，
   综合验收仍要求 NEB 收敛。

不同架构的专家比较可显式设置 `neb_backend=sequential_ase`，全体候选统一
使用标准 ASE 逐图像计算，不进入 MACE 专用批量器；NEB、IDPP 和收敛预算
保持一致。省略该设置仍为原来的 `batched_mace`。新专家先导另见
`expert-diversity.md`，不能将其结果并入旧版本冻结分母。
6. 用中心差分受力计算完整 Cartesian Hessian，位移为 0.005 Å。先质量加权，
   再投影平移和旋转子空间。非线性分子去除六个刚体模式，线性分子去除五个。
   保存完整 Hessian、本征模式和以负数表示虚频的 cm⁻¹ 频率。
7. 显著虚频阈值设为 50 cm⁻¹，同时报告所有负频率数。只有精修收敛、真实受力
   满足阈值且恰有一个显著虚频时，启动 Sella 双向质量加权 IRC。步长为
   0.1 Å√amu，每方向最多 400 步，受力阈值 0.05 eV/Å。保持其默认内部步失败
   即停止的行为，失败原样记录。每支末端再用 FIRE 优化至 0.02 eV/Å。
8. 两支末端必须分别匹配预期两个端点，允许正反交换。要求原子映射键集合相同，
   与该模型优化端点逐连通片段配准后的最大 RMSD 不超过 0.5 Å，并要求 IRC 和
   末端优化都收敛。完整体系 RMSD 保留作诊断。解离产物中不同片段的相对平移、
   转动不应被解释为化学连接变化。

综合 `mlip_verified` 还要求两个端点均收敛且保持输入键集合。结果逐层报告端点、
NEB、鞍点受力、频率和 IRC，软负频率保留供复核。上述频率阈值用于数值容差下
的开发验收。Gaussian 的精确负频率数另行报告。
另行报告 `saddle_irc_verified`，以保留达到 NEB 步数预算后，Sella 继续精修并完成
驻点、频率和 IRC 验证的有效鞍点。该字段与全阶段收敛率共同描述搜索结果。

## Gaussian DFT 精修

OMEN 已有 Gaussian 16W A.03。`run_gaussian_job.ps1` 直接调用现有 g16.exe，
将工作目录和 scratch 限定在对应结果目录，环境变量仅作用于该子进程。
实测 Gaussian 16W 的 Link 0 明确传入 `/scrdir=".\\"`，scratch 实际位于
该作业的工作目录，正常结束后自动清理，中止产生的残余在作业结束后清理。
Windows 可执行文件通过 WSL 的 powershell.exe 桥接调用。首次实测发现该 WSL
没有将 Windows PowerShell 目录加入 PATH，调用使用其明确的系统路径作为后备。

初次按 MLIP 综合通过、单显著虚频、驻点受力、原子数和 reaction ID 排序固定
五个微调候选。随后发现完整体系 RMSD 会误判含分离片段的产物，修正为逐片段
配准并重判全部已保存结果。五个 DFT 候选仍全部通过，保留已经固定的 DFT
子集。固定清单同时写入 `configs/v0.4.0_dft_selection.json` 和结果包中的
`dft/selection.json`，保证后续复现继续使用同五条。首条五原子反应提前验证。

使用与 Transition1x 名义一致的 ωB97X/6-31G(d)，中性单重态，四 CPU 线程、
4 GB 内存、UltraFine 积分网格及 XQC SCF。Gaussian 与源数据计算程序的实现
细节仍存在差异，原始 Transition1x 离散峰值只作为路径参考。

- TS 使用 `Opt=(TS,CalcFC,NoEigenTest,MaxCycles=80) Freq`。NoEigenTest 允许
  初始 Hessian 不满足鞍点阶数，最后仍须优化收敛且恰有一个负频率，并超过
  50 cm⁻¹，才能进入 IRC。
- 从该频率 checkpoint 分别运行 forward/reverse IRC，使用 RCFC，步长 10，
  每方向最多 80 个点。分别保存 checkpoint，避免两方向覆盖。
  首次 rxn4439 的 HPC 积分达到默认 20 次 corrector 上限。对此类已识别错误，
  按统一顺序尝试从同一 TS 重算，步长降至 5、最多 160 点、corrector 上限 80。
  仍发生同类错误时尝试 LQA、相同步长和点数。保留各次输入输出及选用算法。
  其他错误保留供诊断。
  本轮继续观察到小步长 HPC 在同一路径点超过二十次校正，四个未完成反应的
  HPC 分支因此中止，保留 `.interrupted.json` 及原日志，并统一从已验证 TS
  使用 LQA、步长 5、最多 160 点、MaxCycle 80 重算双向 IRC。
  rxn4439 已完成的 HPC 结果保留。具体反应清单和调整原因固定在
  `configs/v0.4.0_dft_selection.json`。这是开发过程中的自适应调整，结果报告
  保留每条反应实际使用的算法。
- 两支末端进一步进行最小值优化和频率分析，要求优化收敛、无虚频。
- 达到 160 点预算的 IRC 从其自身 checkpoint 以 `IRC=(Restart,MaxPoints=320)`
  续算，另存日志和 checkpoint。末端优化也使用独立的 extended 文件名，避免
  复用原先 160 点末端的优化结果。
- DFT 综合通过要求 TS 频率、双向 IRC 完成、两个稳定端点的输入键集合连接关系
  同时通过。每个连通片段与相应输入端点的配准 RMSD 还需不超过 0.5 Å，完整
  体系 RMSD 同时保存。键连接关系与包含几何条件的判定分别记录。
  IRC 完成还要求 Gaussian 明确检测到 PES minimum。组合 Opt/Freq 作业可能在
  优化子任务末尾输出一次 Normal termination，必须检查整个日志末尾的结束状态，
  并核对完整振动数目。频率判定采用最后一组 Harmonic frequencies。
- rxn1152 续算在第 161 点明确检测到最小值并完成路径，随后本机 Gaussian
  16W A.03 在路径汇总中报告 `PESDtA: Invalid IPoint`。恢复分支只接受这个
  特定错误，要求最小值检测和路径完成出现在错误之前，仅有一次 Error termination，
  并有保存的 checkpoint。其末端必须另行完成正常结束的 DFT 最小值优化、完整
  频率和映射几何验证。保留 normal_termination=false、恢复标记、错误输出和
  checkpoint。该条以显式恢复状态纳入五条 DFT 验证，汇总逐条披露。
- 电子能垒使用 DFT 鞍点减去该 IRC 连接并优化后的 DFT 反应物能量。零点能和
  热力学校正需单独处理，当前电子能垒字段不包含这些项。

失败结果保留完整输入、输出和异常，汇总分母保留全部预定反应。重复执行复用
已记录结果和正常结束的 Gaussian 阶段，未正常结束的日志要求先检查具体原因。
实测终止 WSL Python 后，PowerShell 包装进程会退出，原生 g16.exe 及计算 link
仍可能运行。此次积分器切换按已记录的父子 PID 和命令行确认归属后，结束对应
Gaussian 进程树。恢复同一作业时，先核对原生进程状态。

## 执行与恢复

在 WSL 项目目录运行，继续使用 ai4chem Mamba 环境。

```bash
export OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu
mamba run -n ai4chem python ts_search.py --prepare-only
mamba run -n ai4chem python ts_search.py --models mh1-finetuned mh1-original
mamba run -n ai4chem python ts_search.py --recheck
mamba run -n ai4chem python validate_ts_dft.py
mamba run -n ai4chem python summarize_ts_search.py
```

JAX 在 CPU 上执行 Sella 内部坐标计算，MACE 始终使用 CUDA float64。
`--models` 和 `--reactions` 可限定恢复范围。新配置使用新输出目录，防止混合协议。
DFT 默认最多并行两个反应，每个 Gaussian 作业使用独立目录和 scratch。`--jobs`
可调整反应并发数。每个反应使用文件锁串行化重复调用，正常阶段可重新解析，
`validate_ts_dft.py --recheck` 复核已保存输出。

## 参考实现

- [ASE NEB、CI-NEB 与 IDPP](https://docs.ase-lib.org/ase/neb.html)
- [Sella 源码](https://github.com/zadorlab/sella)
- [Sella IRC](https://github.com/zadorlab/sella/wiki/IRC)
- [Gaussian TS 优化](https://gaussian.com/opt/)
- [Gaussian IRC](https://gaussian.com/irc/)
- [2025 年 MLIP TS 搜索和 IRC 研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC12442704/)
- [2026 年混合 MLIP/DFT TS 搜索研究](https://arxiv.org/abs/2604.00405)
