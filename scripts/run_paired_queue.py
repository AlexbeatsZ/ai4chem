"""Run frozen paired sampling repeats on an already accepted local environment."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
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


def identity(pid):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'lstart=,command='],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


class AdoptedWorker:
    def __init__(self, pid):
        self.pid = pid
        self.identity = identity(pid)

    def poll(self):
        current = identity(self.pid)
        if current is None:
            return 0  # Terminal scientific state is independently required below.
        if current != self.identity:
            raise RuntimeError('Adopted worker PID changed identity')
        return None


def command(protocol_path, node, seed=None):
    value = [sys.executable, '-u', str(Path(__file__).resolve()),
             '--protocol', str(protocol_path.resolve()), '--node', node]
    return value + (['--seed', str(seed)] if seed is not None else [])


def adopt_owner(pid, output, protocol_path, node):
    """Replace only an exact Mac scheduler, retaining every scientific child."""
    if node != 'mac' or pid == os.getpid():
        raise ValueError('Only a distinct Mac scheduler may be adopted')
    expected = command(protocol_path, node)
    observed = subprocess.check_output(['ps', '-p', str(pid), '-o', 'command='], text=True).strip()
    if shlex.split(observed) != expected:
        raise RuntimeError('Old scheduler command does not match')
    os.kill(pid, signal.SIGSTOP)
    killed = False
    try:
        state = json.loads((output/'state.json').read_text())
        if state['owner_pid'] != pid or state['status'] != 'running':
            raise RuntimeError('Old scheduler state does not match')
        children = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,command='], text=True)
        direct = {}
        for line in children.splitlines():
            parts = line.strip().split(None, 2)
            if len(parts) == 3 and int(parts[1]) == pid:
                direct[int(parts[0])] = shlex.split(parts[2])
        recorded = {int(seed): int(child) for seed, child in state['active'].items()}
        if set(direct) != set(recorded.values()):
            raise RuntimeError('Scheduler is between dispatch/state writes; resume and inspect again')
        for seed, child in recorded.items():
            if direct[child] != command(protocol_path, node, seed):
                raise RuntimeError('Scientific child command does not match')
        # Exclusive receipt prevents a second takeover from discarding history.
        with (output/f'handoff-from-{pid}.json').open('x') as receipt:
            json.dump(dict(old_state=state, old_command=observed, new_owner_pid=os.getpid(),
                           adopted_at_utc=now()), receipt, indent=2)
        active = {seed: AdoptedWorker(child) for seed, child in recorded.items()}
        # SIGKILL targets the stopped, verified scheduler PID, never its process group.
        os.kill(pid, signal.SIGKILL)
        killed = True
        return state, active
    finally:
        if not killed:
            os.kill(pid, signal.SIGCONT)


def execution_policy(execution, default_workers, at=None):
    if execution is None:
        return default_workers, 0
    at = at or datetime.now(timezone.utc)
    deadline = datetime.fromisoformat(execution['boost_until_utc'])
    if deadline.utcoffset() is None or not 1 <= execution['boost_workers'] <= 4:
        raise ValueError('Invalid bounded execution window')
    if execution['baseline_workers'] != default_workers or execution['after_window_nice'] != 10:
        raise ValueError('Baseline or priority policy changed unexpectedly')
    return ((execution['boost_workers'], 0) if at < deadline
            else (default_workers, execution['after_window_nice']))


def lower_priority(active, niceness):
    """Apply to exact live worker trees, including official training subprocesses."""
    rows = subprocess.check_output(['ps', '-axo', 'pid=,ppid='], text=True)
    parents = {int(pid): int(parent) for pid, parent in (line.split() for line in rows.splitlines())}
    owned = {process.pid for process in active.values() if process.poll() is None}
    while True:
        descendants = {pid for pid, parent in parents.items() if parent in owned}
        if descendants <= owned:
            break
        owned |= descendants
    for pid in owned:
        try:
            os.setpriority(os.PRIO_PROCESS, pid, max(niceness, os.getpriority(os.PRIO_PROCESS, pid)))
        except ProcessLookupError:
            pass


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
    parser.add_argument('--execution', type=Path)
    parser.add_argument('--adopt-owner', type=int)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    config = protocol['nodes'][args.node]
    execution = json.loads(args.execution.read_text()) if args.execution else None
    execution_policy(execution, config['workers'])
    if execution is not None and args.node != 'mac':
        raise ValueError('Temporary CPU boost is accepted only on Mac')
    if args.seed is not None:
        if args.seed not in config['seeds']: raise ValueError('Seed is not assigned to this host')
        worker(protocol, args.node, args.seed)
        return
    output = ROOT/protocol['output']/args.node
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'owner.lock').open('a')
    adopted = None
    if args.adopt_owner is not None:
        if execution is None:
            raise ValueError('Takeover requires a bounded execution policy')
        adopted = adopt_owner(args.adopt_owner, output, args.protocol, args.node)
        # The dead owner's kernel releases its lock; retain its exact children.
        for attempt in range(50):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if attempt == 49:
                    raise
                time.sleep(.1)
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if adopted is None and (output/'state.json').exists(): raise FileExistsError('Retain existing queue evidence; no implicit restart')
    state = adopted[0] if adopted else dict(owner_pid=os.getpid(), node=args.node, started_at_utc=now(),
                 status='starting', pending=config['seeds'][:], active={}, outcomes=[],
                 holdout_labels_evaluated=False, automatic_training_retries=0)
    state.update(owner_pid=os.getpid(), execution=execution)
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
    active = adopted[1] if adopted else {}
    state['status']='running'
    while state['pending'] or active:
        capacity, niceness = execution_policy(execution, config['workers'])
        state.update(admission_capacity=capacity, worker_nice=niceness)
        while state['pending'] and len(active)<capacity:
            seed = state['pending'].pop(0)
            if (output/f'pair-{seed}').exists(): raise FileExistsError('Existing pair directory; inspect before retry')
            invocation=command(args.protocol, args.node, seed)
            with (output/f'pair-{seed}-owner.log').open('xb') as log:
                process = subprocess.Popen(invocation,cwd=ROOT,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT)
            active[seed]=process
        if niceness:
            lower_priority(active, niceness)
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
