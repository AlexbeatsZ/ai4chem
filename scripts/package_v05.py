import json
from pathlib import Path
import sys
import zipfile
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from adaptive_ts_search import digest, file_digest, implementation_identity
output = root / 'results/v0.5.0'
verification = json.loads((output / 'verification.json').read_text())
assert verification['passed'] and verification['strategy_units'] == 96
manifest = json.loads((root / 'configs/v0.5.0_reactions.json').read_text())
frozen = json.loads((root / 'configs/v0.5.0_frozen.json').read_text())
assert frozen['manifest_sha256'] == digest(manifest)
assert frozen['implementation'] == implementation_identity()
for specification in frozen['models'].values():
    assert file_digest(root / specification['path']) == specification['sha256']
summary = json.loads((output / 'summary.json').read_text())
assert summary['completed_strategy_units'] == 96 and summary['dft_evaluation_complete']
staging = Path('/mnt/c/Users/Meta/AppData/Local/Temp/.agents/ai4chem-v05-release')
staging.mkdir(parents=True, exist_ok=True)
results_zip = staging / 'ai4chem-v0.5.0-results.zip'
paths = sorted(path for path in output.rglob('*') if path.is_file()
               and not path.name.endswith(('.lock','.tmp')) and not path.name.startswith('Gau-'))
paths += sorted((root / 'configs').glob('v0.5.0*.json'))
with zipfile.ZipFile(results_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in paths:
        archive.write(path, path.relative_to(root).as_posix())
models_zip = staging / 'ai4chem-v0.5.0-models.zip'
with zipfile.ZipFile(models_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for relative in ('models/mh1-omol.model','models/v0.3.0/mh1_full_256.model'):
        archive.write(root / relative, relative)
metadata = []
for path in (results_zip,models_zip):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        metadata.append({'name':path.name,'size_bytes':path.stat().st_size,'entries':len(archive.infolist())})
notes = ['# v0.5.0 失败感知自适应过渡态搜索','',
         '完成 4 条开发反应、12 条项目级保留反应，两种固定 MH-1 模型和三种有界搜索策略，共 96 个评价单元。','',
         '## 十二条保留反应结果','',
         '| 模型 | 策略 | 验证鞍点与 IRC | 全阶段通过 | 平均尝试 |',
         '| --- | --- | --- | --- | --- |']
for model in ('mh1-original','mh1-finetuned'):
    for strategy in ('S0','S1','S2'):
        group = summary['groups'][f'holdout/{model}/{strategy}']
        notes.append(f"| {model} | {strategy} | {group['verified_saddles']}/12 | {group['strict_pipeline']}/12 | {group['average_attempts']:.2f} |")
notes += ['', 'S0 为单起点，S1 为固定多起点，S2 为失败感知恢复。全部失败反应保留在分母中。', '',
          '## Gaussian 验证','',
          '三条反应在搜索前固定，分别评价微调模型三种策略的最终候选，保留九个评价单元。相同坐标候选复用同一作业。','']
for strategy,values in summary['dft'].items():
    error = values['conditional_barrier_mae_eV']
    error_text = f'{error:.6f} eV' if error is not None else '无通过候选'
    notes.append(f"- {strategy} 的 DFT 综合通过为 {values['verified']}/3，已通过候选的电子能垒 MAE 为 {error_text}。")
notes += ['', '## 复现与结果包','',
          '- results ZIP 包含全部路径、轨迹、候选、Hessian、频率、IRC、Gaussian 原始记录、策略轨迹、CSV、图和验证结果。',
          '- models ZIP 包含两种实验的精确权重，供内容 SHA256 身份复现。两个 ZIP 均可解压到仓库根目录。',
          '- 冻结配置保存模型、输入、策略、软件和实现标识，运行命令见 README 与 docs/design/v0.5.0-protocol.md。','',
          '## 解释范围','',
          '样本来自固定 revision 的 Transition1x train 导出 shards 0 和 1，排除全部既有项目反应。预训练重叠尚待核对。',
          '几何成断键判据存在距离阈值敏感性，报告逐对列出变化距离。DFT 使用数据源的电荷零、单重态参考约定。',
          '本轮采用有限初始化与小样本探索性配对评价。全部失败、无候选及特殊 Gaussian 恢复均在完整报告中披露。','']
(staging / 'release-notes.md').write_text('\n'.join(notes),encoding='utf-8')
(staging / 'package-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata,indent=2),flush=True)
