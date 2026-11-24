# AIMNet2-rxn 候选的独立 Gaussian 验证

rxn8694（C4H5NO2，12 原子）完成了从 AIMNet2-rxn 候选出发的独立
Gaussian TS、频率、双向 IRC 和稳定端点验证，并通过映射几何核验。
这是事后开发诊断中的一个候选，不能作为独立总体成功率或更优势能模型的证明。

| 检查 | 实际结果 |
| --- | --- |
| TS 优化和完整振动模式 | 正常收敛，30 个模式 |
| TS 虚频 | 仅一个，−567.5748 cm⁻¹ |
| 正向/反向 IRC | 分别 90 / 95 点，均正常终止并找到最小值 |
| 两个端点优化和频率 | 均正常收敛，每个 30 个模式、全部非负 |
| 映射键和对应端点的最大片段 RMSD | 正确，0.011072 / 0.068461 Å |
| 候选与 DFT TS 对齐 RMSD | 0.185763 Å |
| Gaussian 电子能垒 | 2.625419123 eV |
| AIMNet2-rxn 电子能垒 | 2.044812811 eV |
| MLIP − DFT 电子能垒 | −0.580606313 eV |
| Gaussian 实际完整链墙钟 | 7321.279 秒（约 122 分钟） |

方法固定为 wB97X/6-31G(d)，中性单重态，UltraFine/XQC/NoSymm，
四线程/4 GB。LQA IRC 每向上限 320 点，没有延伸或放宽判据。原生计算
运行于室友 Windows 节点；这项成功不能代替 Linux Gaussian 的安装验收。
能垒是电子能差，不含零点能和热力学修正。不同专家参考方法差异仍未完全分离。

本候选的 MLIP 全链约 64.760 秒，Gaussian 后处理验证约 122 分钟。
两者运行设备、算法和承担的任务不同，这两个时间不能作为公平吞吐比。
该例支持“异构专家能提供可验证候选”，同时暴露约 0.581 eV 的能垒偏差。
搜索/连接成功与定量能垒准确性需要分别评价。

原始计算在 2026-11-24T02:35:20Z 完成；Mac 在 03:43:32Z 完成独立评估，
对应用户本地 November 23。导出中断留下完整 ZIP，恢复时核对源文件清单和
JSON 身份后复用，随后收到 8,861,220 字节/32 文件并通过 ZIP CRC。
计算没有重跑。原始输入、日志和检查点留在 Mac 的
`results/coordination/diversity-dft/rxn8694-aimnet2-rxn-gaussian16w-r1/evidence`。

`summary.json` 保留完整独立 assessment。复核命令：

```bash
.venv/bin/python scripts/assess_dft_chain.py \
  --protocol configs/v0.7.0-diversity-dft/aimnet2-rxn-rxn8694.json \
  --reaction rxn8694 --engine gaussian \
  --folder results/coordination/diversity-dft/rxn8694-aimnet2-rxn-gaussian16w-r1/evidence \
  --output /tmp/.agents/ai4chem-rxn8694-assessment.json
```

其余专家的搜索仍按冻结预算继续。完整 24 单元比较和前瞻 holdout 尚未完成；
此记录不修改已发布 v0.5/v0.6，也不开放 holdout 标签。
