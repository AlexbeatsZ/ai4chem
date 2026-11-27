"""Finite November27 continuation: one full EMA model and pending meta searches."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from model_study_data import ROOT, save_json

PROTOCOL = ROOT / 'configs/four-hours-20261127.json'
OUTPUT = ROOT / 'results/four-hours-20261127'


def now():
    return datetime.now(timezone.utc).isoformat()


def open_window(protocol):
    return datetime.now(timezone.utc) < datetime.fromisoformat(protocol['deadline_utc'])


def predecessors_complete(ts, growth):
    return (ts.get('status') == 'completed' and growth.get('stage') == 'growth_training_and_repeats_complete'
        and bool(growth.get('finished_at_utc')) and len(growth.get('attempts',{})) == 4
        and all(row.get('status') == 'completed' for row in growth['attempts'].values()))


def omen(protocol, update):
    import numpy as np
    import torch
    from ase.io import read, write
    from mace.calculators import MACECalculator
    from run_model_growth import wait_for_cuda, valid_metrics
    from run_model_study import train, benchmark
    torch.set_num_threads(4)
    locks = []
    for relative in ('results/until-0800-20261126-omen/owner.lock',
                     'results/ts-weight-20261126/runner.lock', 'results/v0.10.0/runner.lock'):
        handle = (ROOT/relative).open('a'); locks.append(handle)
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                if not open_window(protocol):
                    return
                update('waiting_for_previous_owner', waiting_for=relative); time.sleep(30)
    ts = json.loads((ROOT/'results/ts-weight-20261126/state.json').read_text())
    growth = json.loads((ROOT/'results/v0.10.0/state.json').read_text())
    if not predecessors_complete(ts, growth):
        raise RuntimeError('Frozen training predecessor is not complete')
    remaining = (datetime.fromisoformat(protocol['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()/3600
    if remaining <= 0:
        return
    locks.extend(wait_for_cuda({'predecessor_wait_hours':remaining}, update))
    if not open_window(protocol):
        return
    settings = {'training_seed':protocol['omen']['seed'], 'training':protocol['omen']['training']}
    frames = read(ROOT/protocol['omen']['train_file'], index=':')
    if len(frames) != 8480:
        raise ValueError('Matched full control must have8480 frames')
    # Accept the real official EMA save/reload/force interface before full training.
    interface = OUTPUT/'omen/interface'; interface.mkdir(parents=True, exist_ok=False)
    write(interface/'train.extxyz', frames[:8]); write(interface/'valid.extxyz', frames[:4])
    experiment = dict(name='ema_interface8', dataset='interface', train_size=8,
        train_file=str((interface/'train.extxyz').relative_to(ROOT)),
        valid_file=str((interface/'valid.extxyz').relative_to(ROOT)),
        foundation_model=protocol['omen']['foundation'],
        model_file='models/four-hours-20261127/ema_interface8.model',
        training_overrides={'epochs':1})
    runs = []
    train(experiment, settings, runs, output_root=OUTPUT/'omen/interface',
        manifest_path=interface/'manifest.json', state_writer=update)
    model = ROOT/experiment['model_file']
    console = (interface/'training/ema_interface8/console.log').read_text()
    if 'Using Exponential Moving Average with decay: 0.99' not in console:
        raise RuntimeError('Official EMA was not enabled')
    calc = MACECalculator(model_paths=str(model), device='cuda', default_dtype='float64')
    atoms = frames[0].copy(); atoms.calc = calc
    energy = atoms.get_potential_energy(); force = atoms.get_forces()
    step = 1e-4; original = atoms.positions[0,0]
    atoms.positions[0,0] = original+step; plus = atoms.get_potential_energy()
    atoms.positions[0,0] = original-step; minus = atoms.get_potential_energy()
    error = abs(-(plus-minus)/(2*step)-force[0,0])
    if not np.isfinite(energy) or not np.isfinite(force).all() or error > 2e-5:
        raise RuntimeError('EMA standard-model reload/force-gradient check failed')
    save_json(interface/'acceptance.json', dict(status='passed', real_updates=2,
        force_fd_error_eV_A=float(error), scientific_model=False), immutable=True)
    del calc
    model.unlink()
    if not open_window(protocol):
        return
    experiment = dict(name='ema_replay1024_e20', dataset='public-replay', train_size=8480,
        train_file=protocol['omen']['train_file'], foundation_model=protocol['omen']['foundation'],
        seed=protocol['omen']['seed'], model_file='models/four-hours-20261127/ema_replay1024_e20.model')
    runs = []
    train(experiment, settings, runs, output_root=OUTPUT/'omen',
        manifest_path=OUTPUT/'omen/training/manifest.json', state_writer=update)
    if runs[0]['trained_epochs'] != 20:
        raise RuntimeError('Full EMA actual epoch budget differs')
    metrics = benchmark('validation/ema_replay1024_e20', [experiment['name']],
        {experiment['name']:str(ROOT/experiment['model_file'])}, output_root=OUTPUT/'omen', device='cuda')
    metrics = metrics['models'][experiment['name']]['aggregate']; valid_metrics(metrics)
    update('model_complete', model=experiment['model_file'], validation=metrics,
        promotion='Pending matched Mac full-control metrics; no automatic promotion or repeat')


def mac(protocol, update):
    active = {}; outcomes = []; handles = []
    for expert in protocol['mac']['experts']:
        if not open_window(protocol):
            break
        command = [sys.executable,'-u','scripts/run_diversity_search.py','--expert',expert,
            '--protocol',protocol['mac']['search_protocol'],'--output',str(OUTPUT/'search'),
            '--device','cpu','--deadline-utc',protocol['deadline_utc'],'--max-hours','3']
        log = (OUTPUT/'mac'/(expert+'.log')).open('x'); handles.append(log)
        env = {**os.environ, **{key:'3' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}}
        child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        active[expert] = child
    while active:
        for expert, child in list(active.items()):
            if child.poll() is not None:
                outcomes.append(dict(expert=expert,exit_code=child.returncode)); del active[expert]
        update('searching', active={key:child.pid for key,child in active.items()}, outcomes=outcomes)
        if active:
            time.sleep(30)
    for handle in handles:
        handle.close()
    if any(row['exit_code'] for row in outcomes):
        raise RuntimeError('Search worker failed; preserve evidence, no automatic retry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', choices=['mac','omen'], required=True)
    args = parser.parse_args(); protocol = json.loads(PROTOCOL.read_text())
    directory = OUTPUT/args.node; directory.mkdir(parents=True, exist_ok=True)
    lock = (directory/'owner.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (directory/'state.json').exists():
        raise FileExistsError('Existing continuation evidence; no implicit restart')
    os.nice(10)
    state = dict(owner_pid=os.getpid(),node=args.node,status='running',started_at_utc=now(),
        deadline_utc=protocol['deadline_utc'],holdout_labels_evaluated=False,automatic_retries=0)
    def update(stage, **extra):
        state.update(stage=stage,updated_at_utc=now(),**extra); save_json(directory/'state.json',state)
    update('starting')
    try:
        if open_window(protocol):
            (omen if args.node=='omen' else mac)(protocol, update)
        update('completed',status='completed',finished_at_utc=now())
    except Exception as error:
        update('failed',status='failed',error=f'{type(error).__name__}: {error}',finished_at_utc=now())
        raise


if __name__ == '__main__':
    main()
