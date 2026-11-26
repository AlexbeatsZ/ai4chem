# v0.11：固定组成门控的保守软混合

2026-11-25用户要求研究论文的成功机制并实际推进。此先导补上现有端点
硬选择没有检验的软混合；不声称复现材料论文的数值收益，也不称为内部
联合训练的稀疏MoE。冻结配置为`configs/v0.11.0.json`，执行前另保存实际
反应身份、化学式分组折与配置快照到`results/v0.11.0/frozen_protocol.json`。

## 与论文和已有实验的对应

- Liu等DPA3在网络内部替换不变特征模块，使用元素门控、非线性专家和
  常开共享专家。没有共享专家的配置也会变差；MACE等变张量不能直接替换
  为不受约束的普通MLP。[论文](https://arxiv.org/html/2603.07977v3)
- UMA将线性专家权重按组成/电子态/DFT任务混合，并有意排除随坐标变化的
  门控信息。[论文](https://arxiv.org/html/2506.23971v2)
- Zhang等Cu工作实际包含分别微调MACE专家、保留通用专家、冻结专家后
  训练尺寸软门控的方案。其能量与力共同参与门控训练。原始MoE的保守势与
  后加energy-only残差必须分开，后者不直接照搬进TS搜索。
  [作者机构全文](https://orca.cardiff.ac.uk/id/eprint/189300/1/s41524-026-02280-x_reference.pdf)
- Nascimento等Allegro在固定原子区域上联合训练并施加公共体相一致性约束；
  主要结果是接近大模型精度并加速，不能解读成精度胜过完整大模型。
  [论文](https://arxiv.org/html/2604.26143v1)

此前v0.8两分区专家/跨区回放以及事后风险路由均未稳定胜过通用模型。
该观察不覆盖软混合、通用成员参与混合或内部共享专家。本先导先利用已
保存的完整预测，不额外训练MACE，不新增DFT，不占OMEN的既定GPU队列。

## 模型和物理定义

固定三个已完成、同目标ωB97X/6-31G(d)的MACE权重：global512、replay0、
replay1，分别56,945,307/56,944,983/56,944,983字节。总170,835,273字节；
三个成员的历史训练和全部推理调用都计入成本，通用模型不是免费回退。

每个体系使用组成特征`log(N), x_H, x_C, x_N, x_O`。不读当前坐标、反应ID、
参考标签、TS身份或误差。门控给出凸权重，整条固定组成路径保持权重不变：
`E(R)=sum_k w_k(Z) E_k(R)`，`F(R)=sum_k w_k(Z) F_k(R)`。
因权重不依赖坐标，后式等于前式的负梯度。推理复用已安装ASE3.29的
`LinearCombinationCalculator`，不自行修改MACE或混合不一致的力场。

只接受非周期、中性单重态CHNO；其他电子态拒绝。对ASE缓存命中也重新
检查charge/spin，避免只改变info字段时返回先前结果。所有专家均参与推理，
不宣称稀疏加速。组成全局门控还不能保证不同分离子体系的独立组合一致性，
此先导仅在固定分子/反应组成范围内验收，不作为通用材料基础势。

## 数据角色和可解释对照

复用v0.7的128条meta-development作为门控拟合池，符合其原来的路由开发
角色；不更新底层MACE。五折按化学式隔离，seed20261125。预处理只在各折
训练部分拟合；输出折外预测。60条既有validation只检查全meta拟合后的门控；
它已参与MACE checkpoint选择，不能称独立测试。prospective64+64不打开。

逐构型读取v0.8缓存，与原角色HDF5逐项配对：反应/化学式、原子顺序、坐标、
完整能量与3N力；每反应必须有十图像。单模型重建值必须复现旧报告。不得
对已汇总能垒直接平均：先组合完整能量曲线，再取采样最大值。受力误差
保留专家误差的交叉项，不能只平均各自RMSE。

四个固定对照：单通用模型、均匀混合、可学习常数凸权重、组成softmax。
softmax是线性logit，不额外搜索MLP结构；最后一项logit固定为0，共12个
可训练系数。常数权重用SLSQP，条件门控用L-BFGS-B及解析梯度；每个最多
300次迭代，不自动换初始化或扩展超参数。斜率L2为1e-5，其余见冻结配置。

目标为每反应平均的`10*mean((E_error/N)^2)+mean(F_component_error^2)`。
用误差Gram矩阵与凸权重计算，精确等价于完整缓存预测的损失；无额外能量
偏置或energy-only校正。门控标定虽然便宜，也必须交付可加载参数与推理接口。

## 预算、晋级和验收

最多12次拟合：两种学习门控各5折加全meta一次。Mac单线程/nice10，墙钟
上限900秒，0新增GPU训练小时/0新增DFT。完成或失败保留证据；已有state
则拒绝覆盖，科学失败不自动重试。此预算不授权新的大型专家训练。

晋级要求折外与validation同时降低能垒MAE、全局/TS力RMSE不劣化，且按
化学式聚类的折外能垒差95%区间上界小于0。通过只进入实际成本/搜索复核；
不自动替换通用模型或打开最终保留集。全部对照与失败均报告，不仅展示最优。

检查顺序：门控解析梯度、组成/排列不变、混合势中心差分、平移/旋转、保存
重载、非法电子态与数据角色拒绝；再跑真实缓存拟合；最后真实三MACE重载、
解析力差分/对称性/缓存数值复现与实际推理耗时。接口通过与化学收益分开。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  environments/decision-models/.pixi/envs/default/bin/python scripts/run_soft_mixture.py
# Inspect existing results/v0.11.0/state.json first; never rerun completed evidence.
```

## 共享主干残差专家：接口与有限训练

外部软混合完成后，两个训练构型/三次更新的真实接口检查已通过：保留
MACE主干坐标梯度，提取1024个0e标量通道（完整描述符2560维），共享
残差分支加两个元素门控专家联合更新。主干6,178,198个参数冻结；适配层
49,256个参数，普通宽单头对照49,248个参数。此处共同训练的是适配层，
没有共同更新主干，且两个专家均计算，不称稀疏加速或完整DPA3复现。

`residual_moe.py`按每层真实irreps选择0e通道，避免对矢量通道应用普通
非线性；原子残差能量求和，再对总能量求导。禁止在冻结主干时使用
`no_grad()`，否则会丢失残差力和力损失对专家的梯度。初始化末层为零，
初始输出精确等于global512。0e特征用无可训练仿射项的LayerNorm归一化。

`configs/residual-moe-pilot-20261125.json`另冻结小训练：从旧两个训练区域
各取32条完整反应，共512个已有标签；所有分支都看相同数据，不再分割专家
数据。两模型各2实际epoch、batch4、Adam1e-4、energy/force10/1，256次更新。
先普通单头、再共享专家，均只取最终权重并保存；不搜索学习率或种子。
使用Mac单线程/nice10，上限1800秒、0新GPU/DFT，原Mac队列须已完成且
owner锁可用。v0.10 GPU队列保持。选择名单在读取训练标签前另存快照。

只做既有validation60全600图像验收，比较global512与相近参数单头。门控和
专家适配不使用meta标签；外部软混合使用meta的角色独立保留。共同指标
通过只允许进入更完整的匹配训练/重复；短先导不能排除所有训练长度，也不能
声称新泛化率。所有原始状态、完整训练损失和失败留在
`results/v0.11.0/residual_pilot`，拒绝自动重试或覆盖。

```bash
environments/mac-mace/.pixi/envs/default/bin/python scripts/check_residual_moe.py
# Interface acceptance is a one-time finite check, not chemical training evidence.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  environments/mac-mace/.pixi/envs/default/bin/python scripts/run_residual_moe_pilot.py
```
