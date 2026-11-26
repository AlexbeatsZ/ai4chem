"""Mac full-weight replay control and bounded new-label adaptation until08:00UTC+8."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from ase.io import read, write

from model_study_data import ROOT, save_json
from run_model_growth import valid_metrics
from run_ts_weight_study import promotes

PROTOCOL = ROOT / 'configs/three-host-until-0800-20261126.json'
OUTPUT = ROOT / 'results/until-0800-20261126'
DATA = ROOT / 'data/until-0800-20261126'
RETURN = ROOT / 'results/coordination/compute-20261126/offpath-return'


def now():
    return datetime.now(timezone.utc).isoformat()


def before_deadline(protocol):
    return datetime.now(timezone.utc) < datetime.fromisoformat(protocol['deadline_utc'])


def extension_allowed(control, adapted, baseline, control_new_force, adapted_new_force):
    for metrics in (control, adapted, baseline):
        valid_metrics(metrics)
    return (np.isfinite([control_new_force, adapted_new_force]).all()
            and 0 <= adapted_new_force < control_new_force
            and adapted['barrier_mae_eV'] <= baseline['barrier_mae_eV'] * 1.10
            and all(adapted[key] <= baseline[key] * 1.05 for key in
                    ('force_component_rmse_eV_per_A', 'ts_region_force_component_rmse_eV_per_A')))


def write_checked(path, atoms):
    if path.exists():
        raise FileExistsError('Prepared task data exists; inspect without overwriting')
    write(path, atoms)
    restored = read(path, index=':')
    for original, saved in zip(atoms, restored, strict=True):
        np.testing.assert_allclose(original.positions, saved.positions, atol=1e-8, rtol=0)
        np.testing.assert_allclose(original.arrays['REF_forces'], saved.arrays['REF_forces'], atol=1e-8, rtol=0)
        np.testing.assert_allclose(original.info['REF_energy'], saved.info['REF_energy'], atol=1e-8, rtol=0)


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    small = read(ROOT/'data/v0.8.0/general4096.extxyz', index=':')
    coverage = read(ROOT/'data/v0.10.0/coverage1024.extxyz', index=':')
    if len(small) != 4096 or len(coverage) != 8192:
        raise ValueError('Public replay data count differs')
    selected = json.loads((ROOT/'configs/offpath32-20261126.json').read_text())['training_reactions']
    roles = json.loads((ROOT/'configs/v0.7.0_partitions.json').read_text())
    blocked = {f'rxn{sid:04d}' for key in ('validation','meta_development','holdout_train','holdout_val') for sid in roles[key]}
    groups = defaultdict(list)
    for atoms in small:
        if atoms.info['reaction'] in blocked:
            raise ValueError('Evaluation reaction leaked into public replay')
        groups[atoms.info['reaction']].append(atoms)
    if len(groups) != 512 or any(len(group) != 8 for group in groups.values()):
        raise ValueError('Public512 reaction membership changed')
    if any(atoms.info['reaction'] in blocked for atoms in coverage):
        raise ValueError('Evaluation reaction leaked into full replay')
    replay_ids = sorted(set(groups)-set(selected))[:32]
    replay = [atoms.copy() for name in replay_ids for atoms in groups[name]]
    matched = [groups[name][index % 8].copy() for name in selected for index in range(9)]
    write_checked(DATA/'pilot-control.extxyz', replay+matched)
    write_checked(DATA/'full-control.extxyz', coverage+matched)
    write_checked(DATA/'pilot-replay.extxyz', replay)
    write_checked(DATA/'full-replay.extxyz', coverage)
    manifest = dict(prepared_at_utc=now(), pilot_replay_reactions=replay_ids,
        comparison_reactions=selected, pilot_frames=544, full_frames=8480,
        matched_control='Same32reactions,9public replay frames each(8unique+one repeat); treatment replaces288slots with reference+8Gaussian perturbations.',
        foundation='models/v0.10.0/mh1_coverage2048_s0.model', holdout_labels_evaluated=False)
    save_json(DATA/'manifest.json', manifest, immutable=True)
    print('PASS - fixed544/8480-frame public controls, labels read back', flush=True)


def deliver_gate(gate):
    path = OUTPUT/'adaptation-gate.json'
    save_json(path, gate, immutable=True)
    remote = 'C:/Users/Meta/source/repos/ai4chem/results/coordination/compute-20261126/mac-adaptation-gate.json'
    subprocess.run(['scp','-q',str(path),'meta@100.113.70.121:'+remote], check=True, timeout=90)
    command = ['ssh','-n','-T','meta@100.113.70.121','powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',
        'C:/Users/Meta/source/repos/ai4chem/scripts/deliver_gaussian_gate.ps1','-SourceFile',remote,
        '-QueueDirectory','C:/Users/Boywe/Project/Workspaces/ai4chem-worker/results/offpath128-until-0800-20261126']
    subprocess.run(command, check=True, timeout=240)


def training_force_rmse(model, frames):
    from mace.calculators import MACECalculator
    calculator = MACECalculator(model_paths=str(model), device='cpu', default_dtype='float64')
    squared, components = 0., 0
    for original in frames:
        atoms = original.copy(); atoms.calc = calculator
        delta = atoms.get_forces()-original.arrays['REF_forces']
        squared += float(np.sum(delta**2)); components += delta.size
    return (squared/components)**.5


def worker(role, protocol):
    import torch
    from run_model_study import train, benchmark
    torch.set_num_threads(4)
    state = dict(owner_pid=os.getpid(), role=role, status='running', started_at_utc=now(), attempts={}, holdout_labels_evaluated=False)
    def update(stage, **extra):
        state.update(stage=stage, updated_at_utc=now(), **extra)
        save_json(OUTPUT/f'{role}-state.json', state)

    def attempt(name, dataset, epochs):
        if not before_deadline(protocol):
            raise TimeoutError('08:00 window closed; no new training stage')
        count = sum(1 for _ in read(dataset, index=':'))
        experiment = dict(name=name, dataset='offpath-application', train_file=str(dataset.relative_to(ROOT)),
            train_size=count, seed=protocol['mac']['seed'], foundation_model=protocol['mac']['foundation'],
            model_file=f'models/until-0800-20261126/{name}.model')
        settings = {'training_seed':protocol['mac']['seed'], 'training':{**protocol['mac']['training'],'epochs':epochs}}
        record = dict(status='running', started_at_utc=now()); state['attempts'][name] = record
        runs = []
        train(experiment, settings, runs, output_root=OUTPUT, manifest_path=OUTPUT/f'training/{name}/manifest.json', state_writer=update)
        if runs[0]['trained_epochs'] != epochs:
            raise RuntimeError('Actual training epochs differ from frozen budget')
        metrics = benchmark(f'validation/{name}', [name], {name:str(ROOT/experiment['model_file'])},
            output_root=OUTPUT, device='cpu')['models'][name]['aggregate']
        valid_metrics(metrics)
        record.update(status='completed', run=runs[0], validation=metrics, finished_at_utc=now())
        update('model_terminal')
        return record, ROOT/experiment['model_file']

    gate_sent = False
    try:
        update('starting')
        if role == 'full-control':
            attempt('public_replay1024_e20', DATA/'full-control.extxyz', 20)
        else:
            control, control_model = attempt('public_pilot32_e4', DATA/'pilot-control.extxyz', 4)
            update('waiting_for_aligned_offpath_labels')
            while True:
                if not before_deadline(protocol):
                    raise TimeoutError('08:00 arrived before aligned labels')
                returned = json.loads((RETURN/'state.json').read_text())
                if returned['status'] == 'offpath_returned_and_aligned_quarantined':
                    break
                if returned['status'] in ('owner_missing','observation_deadline'):
                    raise RuntimeError('Finite label return did not complete')
                time.sleep(30)
            fresh = read(RETURN/'alignment/quarantined.extxyz', index=':')
            manifest = json.loads((DATA/'manifest.json').read_text())
            if len(fresh) != 288 or {x.info['reaction'] for x in fresh} != set(manifest['comparison_reactions']):
                raise ValueError('Returned new-label identities differ')
            for atoms in fresh:
                if atoms.info['label_method'] != 'gaussian5d_ultrafine':
                    raise ValueError('Unaligned label method')
            write_checked(DATA/'pilot-adapted.extxyz', read(DATA/'pilot-replay.extxyz',index=':')+fresh)
            adapted, adapted_model = attempt('gaussian5d_pilot32_e4', DATA/'pilot-adapted.extxyz', 4)
            update('checking_training_label_force_fit')
            force_control = training_force_rmse(control_model, fresh)
            force_adapted = training_force_rmse(adapted_model, fresh)
            baseline = json.loads((ROOT/'results/coordination/compute-20261126/growth-state.json').read_text())['attempts']['mh1_coverage2048_s0']['validation']
            gate = dict(protocol=protocol['version'], decided_at_utc=now(),
                allow_label_extension=bool(extension_allowed(control['validation'], adapted['validation'], baseline, force_control, force_adapted)),
                actual_pilot_models_complete=True, frames=288, training_force_control=force_control,
                training_force_adapted=force_adapted, control_validation=control['validation'], adapted_validation=adapted['validation'],
                baseline_validation=baseline, training_merge_authorized=False,
                criterion=protocol['mac']['next_label_gate'])
            deliver_gate(gate); gate_sent = True
            promotion = promotes(adapted['validation'], control['validation'], baseline)
            state['label_gate'] = gate; state['full_promotion'] = promotion
            if promotion and before_deadline(protocol):
                write_checked(DATA/'full-adapted.extxyz',read(DATA/'full-replay.extxyz',index=':')+fresh)
                attempt('gaussian5d_replay1024_e20', DATA/'full-adapted.extxyz',20)
        update('completed', status='completed', finished_at_utc=now())
    except Exception as error:
        update('failed', status='failed', error=f'{type(error).__name__}: {error}', finished_at_utc=now())
        if role == 'adaptation' and not gate_sent:
            try:
                deliver_gate(dict(protocol=protocol['version'],allow_label_extension=False,decided_at_utc=now(),reason=state['error']))
            except Exception as transfer_error:
                update('failed',gate_transfer_error=str(transfer_error))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--worker',choices=['full-control','adaptation'])
    args=parser.parse_args()
    protocol=json.loads(PROTOCOL.read_text())
    if args.prepare_only:
        prepare();return
    if args.worker:
        worker(args.worker,protocol);return
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=(OUTPUT/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUTPUT/'state.json').exists():
        raise FileExistsError('Existing Mac owner/evidence; no implicit retry')
    state=dict(owner_pid=os.getpid(),status='running',started_at_utc=now(),deadline_utc=protocol['deadline_utc'],active={},outcomes=[])
    active={}
    pending_search=['global512','coverage2048s0']
    from scripts.run_paired_queue import lower_priority
    for role in ('full-control','adaptation'):
        log=(OUTPUT/f'{role}-owner.log').open('x')
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',role],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        log.close();active[role]=child;state['active'][role]=child.pid
    save_json(OUTPUT/'state.json',state)
    while active:
        if not before_deadline(protocol):
            lower_priority(active,10);state['after_deadline_nice']=10
        for role,child in list(active.items()):
            code=child.poll()
            if code is not None:
                state['outcomes'].append(dict(role=role,exit_code=code));del active[role];state['active'].pop(role)
        while pending_search and len(active)<2 and before_deadline(protocol):
            expert=pending_search.pop(0);role='search-'+expert
            log=(OUTPUT/f'{role}-owner.log').open('x')
            command=[sys.executable,'-u',str(ROOT/'scripts/run_diversity_search.py'),'--expert',expert,
                '--protocol',str(ROOT/'configs/growth-search-until-0800-20261126.json'),
                '--output',str(OUTPUT/'development-search'),'--deadline-utc',protocol['deadline_utc']]
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            log.close();active[role]=child;state['active'][role]=child.pid
        state['updated_at_utc']=now();save_json(OUTPUT/'state.json',state)
        if active:time.sleep(15)
    state.update(status='completed' if all(x['exit_code']==0 for x in state['outcomes']) else 'completed_with_failures',finished_at_utc=now())
    save_json(OUTPUT/'state.json',state)


if __name__=='__main__':
    main()
