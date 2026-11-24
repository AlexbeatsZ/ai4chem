"""Run frozen paired sampling repeats on an already accepted local environment."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def worker(protocol, node, seed):
    import torch
    from run_model_study import train, benchmark
    torch.set_num_threads(4)
    output = ROOT/protocol['output']/node
    directory = output/f'pair-{seed}'
    directory.mkdir(exist_ok=False)
    state = dict(owner_pid=os.getpid(), seed=seed, node=node, status='running',
                 started_at_utc=now(), outcomes=[], holdout_labels_evaluated=False)
    def update(stage, **extra):
        state.update(stage=stage, updated_at_utc=now(), **extra)
        save(directory/'state.json', state)
    for arm in ('random4096', 'path4096'):
        name = f'mh1_{arm}_s{seed}'
        manifest = []
        experiment = dict(name=name, dataset=arm, train_size=4096, seed=seed,
            model_file=f'models/paired-sampling-r1/{name}.model')
        settings = {**protocol['training'], 'device':protocol['nodes'][node]['device']}
        try:
            train(experiment, {'training':settings, 'training_seed':seed}, manifest, output_root=output,
                  manifest_path=directory/f'{arm}-manifest.json', state_writer=update)
            if manifest[0]['trained_epochs'] != 40:
                raise RuntimeError('Expected forty actual logged epochs')
            model = str(ROOT/experiment['model_file'])
            metrics = {}
            for role, count in [('validation',60),('meta_development',128)]:
                update(role, experiment=name)
                metrics[role] = benchmark(f'{role}/{name}', [name], {name:model},
                    role=role, count=count, output_root=output,
                    device=settings['device'])['models'][name]['aggregate']
            state['outcomes'].append(dict(name=name, status='completed', run=manifest[0], metrics=metrics))
        except Exception as error:
            state['outcomes'].append(dict(name=name, status='failed', error=f'{type(error).__name__}: {error}'))
        update('arm_terminal')
    state.update(status='completed' if all(x['status']=='completed' for x in state['outcomes'])
                 else 'completed_with_failures', finished_at_utc=now())
    update('pair_terminal')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--node', choices=['mac','omen'], required=True)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    config = protocol['nodes'][args.node]
    if args.seed is not None:
        if args.seed not in config['seeds']: raise ValueError('Seed is not assigned to this host')
        worker(protocol, args.node, args.seed)
        return
    output = ROOT/protocol['output']/args.node
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'state.json').exists(): raise FileExistsError('Retain existing queue evidence; no implicit restart')
    state = dict(owner_pid=os.getpid(), node=args.node, started_at_utc=now(),
                 status='starting', pending=config['seeds'][:], active={}, outcomes=[],
                 holdout_labels_evaluated=False, automatic_training_retries=0)
    save(output/'state.json', state)
    predecessor_locks=[]
    if args.node=='omen':
        # Keep the established CUDA owner and its exact locks authoritative.
        for version in ('v0.9.0','v0.7.0'):
            predecessor = (ROOT/'results'/version/'runner.lock').open('a')
            while True:
                try:
                    fcntl.flock(predecessor, fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    state.update(status='waiting_for_existing_cuda_owner', waiting_for=version, updated_at_utc=now())
                    save(output/'state.json', state)
                    time.sleep(30)
            predecessor_locks.append(predecessor)
        prior = json.loads((ROOT/'results/v0.9.0/state.json').read_text())
        if prior['stage'] not in ('optimizer_screening_and_seed_repeats_complete','incomplete_seed_repeats'):
            state.update(status='blocked_predecessor', predecessor=prior, finished_at_utc=now())
            save(output/'state.json',state)
            raise RuntimeError('CUDA predecessor did not reach a documented terminal state')
    active = {}
    state['status']='running'
    while state['pending'] or active:
        while state['pending'] and len(active)<config['workers']:
            seed = state['pending'].pop(0)
            if (output/f'pair-{seed}').exists(): raise FileExistsError('Existing pair directory; inspect before retry')
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--protocol',str(args.protocol.resolve()),
                     '--node',args.node,'--seed',str(seed)]
            with (output/f'pair-{seed}-owner.log').open('xb') as log:
                process = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT)
            active[seed]=process
        for seed,process in list(active.items()):
            code=process.poll()
            if code is not None:
                path=output/f'pair-{seed}'/'state.json'
                record=json.loads(path.read_text()) if path.exists() else {}
                state['outcomes'].append(dict(seed=seed,exit_code=code,
                    status=record.get('status','owner_failed'),state=str(path.relative_to(ROOT))))
                del active[seed]
        state.update(active={str(seed):p.pid for seed,p in active.items()},updated_at_utc=now())
        save(output/'state.json',state)
        if active: time.sleep(30)
    state.update(status='completed' if all(x['status']=='completed' and x['exit_code']==0 for x in state['outcomes'])
        else 'completed_with_failures',finished_at_utc=now())
    save(output/'state.json',state)


if __name__=='__main__': main()
