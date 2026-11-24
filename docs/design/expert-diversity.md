# 不同架构的分子势专家扩展

2026-11-23 用户指出，原先四个模型全部属于 MACE，不能代表足够广的专家池。
保留已完成的四模型负结果，同时优先加入真正不同架构，而不是继续只增加
同一个基础模型的随机种子。原有 OMEN 训练对照继续完成，不改变已冻结方案。

## 冻结的先导比较

`configs/molecular-experts.json` 在新增专家的化学结果产生前登记候选、权重和
适用范围。新增 AIMNet2、AIMNet2-rxn、ANI-2x、OrbMol conservative：四个
候选、三个新增架构家族；ANI 使用官方八成员集合，其成本不能当单成员。

仍在原先 128 条 meta-development 上诊断互补性，也可在 60 条 validation
上选择候选。这是看到第一轮结果后的探索性扩展，不是全新的无偏检验。
后续 64+64 条 prospective holdout 保持封存，不能用新名字重新打开旧 test。

统一 ASE 输出 eV / eV per angstrom，固定相同几何。每个专家减去自己的反应物
能量；不混合不同理论水平的绝对能量。AIMNet2 官方实现使用原生混合精度，
其他新候选先用 float64；记录真实参数 dtype，不能只看 CLI 的 dtype 声明。
同名 DFT 标准仍需留意色散、基组和数据导出精度；原始跨方法误差和模型
本身误差尚不能完全分离。AIMNet2-rxn 卡片和当前计算器对 DFT 方法的表述
有差异，保留所用版本、外加 D3 和原始来源，不能声称全部专家已对齐 DFT。

## 接入和验收

Mac 项目隔离环境：`environments/aimnet`（AIMNet2 / TorchANI）和
`environments/orb`（OrbMol）。分开是由于当前发行版对 nvalchemi-toolkit-ops
的依赖范围冲突。无需改变 MACE 训练环境或全局 Python。

先真实加载权重，在同一 meta 路径的端点和 TS 检查有限能量/力、原子映射、
能量差分与力的一致性，然后完成 128 条路径。OrbMol 只用 conservative
版本；direct-force 版本不进入需要能量负梯度的 NEB/TS 链。

`external_experts.py` 使用官方 ASE 接口，`benchmark_transition1x.py` 沿用
旧指标定义。模型来源单独写入每个结果，MACE 默认候选不变。

## 判断标准

1. 同时比较能垒、力、TS 区域力、反应能、尾部和实际耗时。
2. 固定最优专家与 barrier-only oracle / force-guarded oracle 一起报告；
   oracle 用了真实标签，只能说明潜力。
3. 对比旧四个专家和加入新家族的专家池。如果更多模型只改善能垒却明显
   损伤力，不能声称得到了综合更好的势能面。
4. 有可观多指标潜力后再更新端点路由，并按化学式分组；新增模型由已知
   结果启发，最终结论需要冻结后在 prospective 数据和实际搜索上检验。
5. 基础训练的反应重叠未逐条证明，不能以项目未见替代基础模型未见。

## 端点搜索先导

`configs/v0.7.0_diversity_search.json` 在搜索前冻结八条 meta-development
反应：仅按端点的类别、化学式、大小和成断键特征选取，不按已知模型误差
筛选。三个候选为旧微调 MH-1、AIMNet2-rxn、OrbMol conservative，合计
24 个配对的模型/反应单元。此集合有六条重排和两条裂解，仍无成键类。

所有候选都用 Mac CPU 三线程、标准 sequential ASE 十图像 NEB，同样的
v0.4 迭代预算和精度阈值。端点先统一检查收敛及映射键保留，失败保留在
分母。通过鞍点/IRC 后另外检查两侧最小值的投影 Hessian，无低于 -50 cm^-1
的显著负模；软负模也原样保存。分别报告严格全链和允许 NEB 达预算后的
有效鞍点，不放宽驻点、频率、IRC 或映射连接阈值。

`scripts/run_diversity_search.py --expert NAME` 每个专家一个进程，三个
进程可以并行。逐反应 `pipeline.json` 含完整冻结协议、失败和实际墙钟时间。
异常单元保留证据并继续其他反应；存在中断目录而无完成记录时先停止检查，
不自动覆盖。有意义的鞍点还需要独立 DFT，MLIP 通过不等于 DFT 通过。

## 原始来源

- https://github.com/isayevlab/aimnetcentral
- https://isayevlab.github.io/aimnetcentral/models/aimnet2_rxn/
- https://huggingface.co/isayevlab/aimnet2-rxn
- https://github.com/aiqm/torchani
- https://huggingface.co/Orbital-Materials/OrbMol
- https://github.com/orbital-materials/orb-models
