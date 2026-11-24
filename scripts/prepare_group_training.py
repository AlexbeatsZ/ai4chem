"""Prepare five independent paired sampling repeats; do not dispatch or run them."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import zipfile
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [20261124, 20261125, 20261126, 20261127, 20261128]


def prepare(output):
    configs = {arm: yaml.safe_load((ROOT/f'results/v0.7.0/training/mh1_{arm}/config.yaml').read_text())
               for arm in ('random4096', 'path4096')}
    excluded = {'train_file','valid_file','foundation_model','name','seed','model_dir',
                'checkpoints_dir','log_dir','results_dir','restart_latest'}
    settings = {arm: {k:v for k,v in config.items() if k not in excluded} for arm,config in configs.items()}
    if settings['random4096'] != settings['path4096']:
        raise ValueError('Paired controls have different source settings')
    for config in configs.values():
        if config['max_num_epochs'] != 40 or config['batch_size'] != 4 or config['default_dtype'] != 'float64':
            raise ValueError('Unexpected original frozen budget/precision')
    if any(configs['random4096'][key] != configs['path4096'][key]
           for key in ('valid_file','foundation_model')):
        raise ValueError('Paired controls must share validation and foundation')
    inputs = {'foundation.model': Path(configs['random4096']['foundation_model']),
              'valid.extxyz': Path(configs['random4096']['valid_file'])}
    inputs.update({f'{arm}.extxyz':Path(config['train_file']) for arm,config in configs.items()})
    inputs = {name:path if path.is_absolute() else ROOT/path for name,path in inputs.items()}
    for path in inputs.values():
        if not path.is_file(): raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)
    shared = output/'shared'; shared.mkdir()
    for name,path in inputs.items(): shutil.copyfile(path,shared/name)
    members = []
    for index,seed in enumerate(SEEDS,1):
        member = output/f'member-{index:02d}'; member.mkdir()
        tasks = []
        for arm in configs:
            filename = f'{arm}.json'; tasks.append(filename)
            task = {'name':f'member{index:02d}_{arm}_s{seed}', 'seed':seed,
                    'train_file':f'../shared/{arm}.extxyz', 'valid_file':'../shared/valid.extxyz',
                    'foundation_model':'../shared/foundation.model', 'expected_epochs':40,
                    'max_wall_seconds':43200,
                    'settings':{**settings[arm],'patience':100},
                    'purpose':'paired development replication of v0.7 sampling; no holdout evaluation'}
            (member/filename).write_text(json.dumps(task,indent=2)+'\n')
        (member/'task.json').write_text(json.dumps({'tasks':tasks},indent=2)+'\n')
        (member/'README.md').write_text(f'''# Member {index:02d}: paired sampling, seed {seed}

Two fixed, sequential 40-epoch tasks: random4096 and path4096. No parameter choice
or intermediate handoff. Each task has a 12-hour maximum, no training retry.
Use only the cloud node/environment accepted by the coordinator. No Gaussian.

Extract the whole ZIP so shared/ is next to member-{index:02d}/. In the matching
ai4chem deployment checkout, submit once:

```bash
bash scripts/run_cloud_training_bundle.sh /absolute/member-{index:02d} /fresh/absolute/member-{index:02d}-result
```

The command returns immediately with an owner PID. On completion return
member-{index:02d}-result.tar.gz, including on failure. Keep the node available
until the archive exists. Do not alter settings, rerun a failed model, interpret
accuracy, or send just screenshots. The coordinator collects all outcomes and
runs common chemistry evaluation. Contact the coordinator only if the node or
launcher itself is unavailable. This package is prepared, not yet dispatched.
''')
        archive = output/f'member-{index:02d}.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
            for folder in (shared,member):
                for path in sorted(folder.iterdir()): z.write(path,path.relative_to(output))
        members.append({'member':index,'seed':seed,'tasks':2,'epochs_each':40,
                        'estimated_updates_each':40960,'archive':archive.name,'bytes':archive.stat().st_size})
    manifest = {'prepared_at_utc':datetime.now(timezone.utc).isoformat(),
                'status':'prepared_not_dispatched', 'members':members,
                'scientific_question':'Does the small random-vs-path validation difference persist across paired independent seeds?',
                'reference':'Existing v0.7 seed20261116 was used for development; retain separately, not one of the five new pairs.',
                'data':'Existing256 train reactions x16 frames per arm; same validation60 and MH1/omol. No new labels.',
                'analysis':'Report all five pairs, paired differences, seed variability and cost; never pick a best seed alone.',
                'budget':'Ten new40-epoch runs,409600 estimated gradient updates; roughly7h/member on observed OMEN timing, not a cloud guarantee.',
                'failure_policy':'No training retries; attempt the next independent fixed task, archive successes and failures.',
                'gates':'Prepared only. Original native Linux Gaussian / user-provided cloud acceptance gates remain before group dispatch unless the user separately authorizes an ML-only dispatch. No server creation or human messages.',
                'holdout_labels_evaluated':False}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.output.resolve()),indent=2))
