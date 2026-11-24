"""Export a completed development run as a portable, bounded cloud timing task.

Only explicit train/validation inputs are copied. No holdout or active queue changes.
The two-epoch result is infrastructure evidence, never a new research model.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import yaml

ROOT = Path(__file__).resolve().parents[1]


def prepare(source, output):
    config = yaml.safe_load(source.read_text())
    inputs = {}
    for key in ('train_file', 'valid_file', 'foundation_model'):
        path = Path(config[key])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        inputs[key] = path
    output.mkdir(parents=True, exist_ok=False)
    names = {'train_file': 'train.extxyz', 'valid_file': 'valid.extxyz',
             'foundation_model': 'foundation.model'}
    for key, path in inputs.items():
        shutil.copyfile(path, output / names[key])
    excluded = set(inputs) | {'name', 'seed', 'model_dir', 'checkpoints_dir', 'log_dir', 'results_dir', 'restart_latest'}
    settings = {key: value for key, value in config.items() if key not in excluded}
    settings.update(max_num_epochs=2, patience=100, device='cuda')
    task = {'name': 'cloud_development_timing', 'seed': config['seed'], **names,
            'expected_epochs': 2, 'max_wall_seconds': 1800, 'settings': settings,
            'purpose': 'infrastructure timing on existing development labels; no scientific model selection',
            'source_name': config['name'], 'source_config': str(source),
            'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
            'holdout_labels_evaluated': False}
    (output / 'task.json').write_text(json.dumps(task, indent=2) + '\n')
    (output / 'README.md').write_text('''# Existing-label CUDA timing task

This bundle repeats only two epochs on existing development train/validation data.
It does not alter any active queue or open prospective holdout labels. Keep all logs.
It is not a new scientific training arm or a trained-model accuracy result.

Clone the ai4chem deployment release on Linux x86_64, then from that checkout run:

```bash
bash scripts/run_cloud_training_bundle.sh /absolute/bundle "$PWD/results/node-readiness/timing-r1"
```

The command returns immediately; return timing-r1.tar.gz on success or failure.
Inspect timing-r1/training/state.json, per-task console.log, logs and saved model. Retain actual
GPU/driver/runtime, epoch count and elapsed time. Compare logged epoch times after
initialization; two epochs give a rough estimate, not a promised full-run speedup.
The timeout is 30 minutes. Output paths must be fresh. No Gaussian is required.
Do not run this timing task beside the occupied OMEN research GPU. New cloud
instances, scientific job reassignment and group dispatch are separate actions.
''')
    return task


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_config.resolve(), args.output.resolve()), indent=2))
