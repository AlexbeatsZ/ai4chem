# 本地决策模型与势能专家选择

## 研究问题

用户希望在 Mac 本地运行 Jev 类决策模型，不依赖付费 API，并尝试根据分子
和反应特征选择势能专家。目标是检验组合是否比固定使用最佳专家更准确，
以及选择开销是否合理。不是把一个概率输出当作 DFT 精度保证。

用户提供的 `jevmodel/jev-ai` 当前是网站导出与说明，没有模型权重。
TypeSafe Jev 是另一官方产品。`bnsd55/jevmlx` 是独立 MIT 项目，可在
Apple Silicon 上对 MLX 模型的有限答案做约束打分；不是 Jev 原权重或 RLCD。
TabPFN/TabICL 则直接从标注表格拟合数值误差或专家选择。

第一轮本地候选：

- `jevmlx` 固定 Git revision，使用其公开 MLX 权重做决策接口验证。
- TabPFN v2 开放权重；新版本可能需要许可证登录，不自动接受账户条款。
- TabICLv2 官方开放实现。与常数规则、Ridge、ExtraTrees/随机森林比较。

## 先验证互补性

固定 OFF23-small、MH-1/omol、MH-0/rgd1、v0.3 微调 MH-1 四个专家。
仅在 `v0.7` 的 128 条 meta-development 反应上计算已有参考路径指标。
先比较最佳固定专家与逐反应 reference oracle，后者仅是不可部署的收益上限。
若上限收益也很小，停止增加路由复杂度，优先加强/增加互补专家。

主要路由目标是反应能垒误差，同时报告力误差、尾部误差、成本与选择错误
造成的损失。能垒最佳专家不自动等于受力、优化或动力学最佳专家。

## 可部署特征与信息边界

第一版端点特征只使用元素数、原子数、映射成断键、端点原子对距离与局部
几何摘要。反应 ID、DFT 能量/力/参考 TS、参考误差、真实最佳专家不能作为
预测特征。读取 extxyz 时采用白名单，不能把文件里的 reference_* 字段传入。

可另设一个付费 probe 版本，使用明确预算的多专家单点/粗路径分歧；必须
计入所有被调用专家的成本。若输入来自 DFT 优化的最终路径，要明确这属于
固定几何诊断，不能声称端点输入即可达到该收益。

按反应分组；多构型或多个专家产生的行不能跨折。优先以化学式分组作更强
泛化检查。所有预处理/选择阈值在每折训练部分拟合。最佳固定专家也只能在
该折训练部分选择；另显示全 meta 集固定最佳值供描述，不能作为无偏基线。
Oracle 可使用评价标签作上限，名字和图例必须显式标记不可部署。

先做 grouped out-of-fold 估计；冻结模型与规则后，才评价新 holdout。
现有 287 条已打开的 test 不纳入决策模型训练。

## 势能面的物理边界

第一版是 reaction-level model routing。输入端点、任务和电子态，选定一个
专家后，整条 NEB/TS/IRC 链固定该专家。登记元素覆盖、电子态和目标 DFT
理论水平；不支持的输入拒绝进入候选集合。零电荷/单重态验证不外推到离子
或任意自旋。此方案没有实现对所有构型的一张统一平滑混合势能面。

构型相关软混合若为 E(R)=sum w_k(R)E_k(R)，力包含
sum w_k F_k - sum E_k grad(w_k)。省略门控导数一般不保守。
不同理论水平/原子参考能必须先对齐；路由也须满足所需对称性。
原子环境专家、协同训练或蒸馏属于后续有收益才进入的架构研究。

## 本地验收与产物

环境位于 `environments/decision-models`；不修改全局 Python/PATH/代理。
权重保存在项目 `models/decision/`，推理日志和版本写入 results，模型许可
与版本需保留。测试合成数据只证明运行能力，不是化学精度证据。
需要分别记录 package 安装、权重取得、MPS/MLX 推理、真实化学数据结果。

## 原始来源

- https://github.com/jevmodel/jev-ai
- https://typesafe.ai/blog/introducing-system-one-models-and-jev
- https://github.com/bnsd55/jevmlx
- https://github.com/PriorLabs/TabPFN
- https://huggingface.co/Prior-Labs/TabPFN-v2-reg
- https://github.com/soda-inria/tabicl
