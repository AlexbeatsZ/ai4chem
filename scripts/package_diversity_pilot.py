"""Package completed eight-expert static results, excluding live computations."""
from collections import Counter
import csv
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'results/v0.7.0/expert_diversity'
    report = ROOT / 'docs/reports/v0.7.0-diversity'
    summary = json.loads((base/'summary.json').read_text())
    router = json.loads((base/'router/summary.json').read_text())
    if summary['status'] != 'completed' or summary['holdout_labels_evaluated']:
        raise ValueError('Only completed development results may be packaged')
    if router['status'] != 'meta_development_oof_complete' or router['prospective_holdout_opened']:
        raise ValueError('Grouped routing must be complete with prospective labels sealed')
    experts = router['experts']
    with (base/'reaction_metrics.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter(row['model'] for row in rows)
    if len(experts) != 8 or counts != Counter({name:128 for name in experts}):
        raise ValueError('Expected exactly 128 paired reactions for each of eight experts')
    identity = {row['reaction'] for row in rows if row['model'] == experts[0]}
    if any({row['reaction'] for row in rows if row['model'] == name} != identity for name in experts):
        raise ValueError('Expert reaction identities differ')
    files = [ROOT/name for name in ('configs/molecular-experts.json','configs/decision-models.json',
        'configs/v0.7.0.json','configs/v0.7.0_partitions.json','docs/design/expert-diversity.md',
        'docs/design/decision-routing.md','docs/design/transition1x.md')]
    files += [base/'summary.json',base/'reaction_metrics.csv']
    files += list(report.glob('*'))
    for environment in ('mac-mace','decision-models','aimnet','orb'):
        files += [ROOT/'environments'/environment/name for name in ('pixi.toml','pixi.lock')]
    for name in ('summary.json','headroom.json','selected_errors.csv','ridge.json',
                 'extratrees.json','tabpfn.json','tabicl.json'):
        files.append(base/'router'/name)
    for expert in ('aimnet2-wb97m','aimnet2-rxn','ani2x','orbmol-conservative'):
        state = json.loads((base/expert/'state.json').read_text())
        acceptance = json.loads((base/expert/'acceptance.json').read_text())
        if state['stage'] != 'completed' or acceptance['status'] != 'passed':
            raise ValueError(f'Incomplete real-inference acceptance: {expert}')
        files.append(base/expert/'acceptance.json')
        for role,count in (('meta_development',128),('validation',60)):
            folder = base/expert/role
            with (folder/'reaction_metrics.csv').open() as handle:
                metrics = list(csv.DictReader(handle))
            if len(metrics) != count or {row['model'] for row in metrics} != {expert}:
                raise ValueError(f'Wrong paired benchmark: {expert}/{role}')
            files += [folder/name for name in ('reaction_metrics.csv','image_predictions.csv','benchmark_summary.json')]
    output = ROOT/'results/releases/v0.7.1-diversity-pilot/ai4chem-v0.7.1-diversity-pilot.zip'
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            if not path.is_file() or path.suffix in {'.model','.pt','.pth','.ckpt'}:
                raise ValueError(f'Missing or excluded artifact: {path}')
            archive.write(path,str(path.relative_to(ROOT)))
        archive.writestr('SCOPE.txt',
            'Completed static eight-expert/four-family and formula-grouped routing DEVELOPMENT pilot.\n'
            'No prospective holdout labels opened. No deployable joint accuracy improvement demonstrated.\n'
            'Live v0.7 training, endpoint searches, DFT and v0.8 specialization are not completed by this release.\n'
            'Different foundation DFT targets and unverified pretraining reaction overlap limit interpretation.\n'
            'Weights are not redistributed; use the pinned official sources and original licenses.\n'
            'Transition1x-derived metrics retain the pinned source provenance in benchmark summaries.\n'
            'Source code is available in the corresponding Git release archive.\n')
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Evidence ZIP failed CRC')
        entries = len(archive.namelist())
    print(json.dumps({'artifact':str(output),'bytes':output.stat().st_size,
        'entries':entries,'models':len(experts),'paired_meta_rows':len(rows),
        'scope':'completed static development pilot only'}))


if __name__ == '__main__':
    main()
