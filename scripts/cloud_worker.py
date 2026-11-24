"""Linux installation receipts, actual MACE acceptance and portable training jobs."""

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def installed():
    versions = {}
    for line in (ROOT/'requirements.lock.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, expected = line.strip().split('==', 1)
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f'Package differs from the frozen lock: {name} {actual} != {expected}')
        versions[name] = actual
    return {'host':socket.gethostname(), 'platform':platform.platform(),
            'python':sys.version, 'executable':sys.executable, 'packages':versions,
            'observed_at_utc':now()}


def doctor(device='cuda'):
    import torch
    record = installed()
    record.update(device_requested=device, torch_cuda_runtime=torch.version.cuda,
                  cuda_available=torch.cuda.is_available(), gaussian_executable=shutil.which('g16'))
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no automatic CPU fallback or driver installation')
        free, total = torch.cuda.mem_get_info()
        record.update(gpu=torch.cuda.get_device_name(0), gpu_capability=torch.cuda.get_device_capability(0),
                      gpu_free_MiB=free/1024**2, gpu_total_MiB=total/1024**2)
    record.update(status='PASS_RUNTIME_CHECK', model_inference_verified=False,
                  official_training_verified=False, native_linux_gaussian_verified=False)
    return record


def resolve_inputs(task, source):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', task['name']):
        raise ValueError('Task name must be a simple artifact name')
    if task['expected_epochs'] != task['settings']['max_num_epochs']:
        raise ValueError('Expected epochs must equal the submitted training budget')
    paths = {}
    for key in ('train_file','valid_file','foundation_model'):
        path = Path(task[key])
        path = path if path.is_absolute() else source.parent/path
        if not path.is_file():
            raise FileNotFoundError(f'Missing uploaded input: {key}={path}')
        paths[key] = str(path.resolve())
    forbidden = {'train_file','valid_file','foundation_model','name','seed','model_dir',
                 'checkpoints_dir','log_dir','results_dir'} & task.get('settings',{}).keys()
    if forbidden:
        raise ValueError(f'Settings override protected task identity/paths: {sorted(forbidden)}')
    return paths


def train_task(task, source, output):
    import torch
    settings = task['settings']
    device = settings['device']
    runtime = doctor(device)
    paths = resolve_inputs(task, source)
    config = {**settings, **paths, 'name':task['name'], 'seed':task['seed'],
              'model_dir':str(output/'models'), 'checkpoints_dir':str(output/'checkpoints'),
              'log_dir':str(output/'logs'), 'results_dir':str(output/'metrics')}
    # JSON is a valid YAML mapping and preserves the submitted scalar values.
    config_path = output/'config.yaml'
    save(config_path, config)
    state = {'host':socket.gethostname(),'owner_pid':os.getpid(),'started_at_utc':now(),
             'task':task,'runtime':runtime,'status':'training','holdout_labels_evaluated':False}
    save(output/'state.json',state)
    started=time.perf_counter()
    try:
        with (output/'console.log').open('xb') as log:
            process = subprocess.run([str(Path(sys.executable).with_name('mace_run_train')),
                '--config',str(config_path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                timeout=task.get('max_wall_seconds',86400))
        if process.returncode:
            raise RuntimeError(f'Official trainer exited {process.returncode}; retain console.log')
        text=(output/'console.log').read_text(errors='replace')
        epochs=[int(x) for x in re.findall(r'INFO: Epoch (\d+):',text)]
        actual=max(epochs)+1 if epochs else 0
        if actual != task['expected_epochs']:
            raise RuntimeError(f'Actual logged epochs {actual} != expected {task["expected_epochs"]}')
        model_path=output/'models'/f'{task["name"]}.model'
        model=torch.load(model_path,map_location='cpu',weights_only=False)
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise RuntimeError('Saved model has nonfinite parameters')
        state.update(status='completed',actual_epochs=actual,model=str(model_path),
                     elapsed_seconds=time.perf_counter()-started,finished_at_utc=now(),
                     model_finite=True)
        save(output/'state.json',state)
        return state
    except Exception as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}',finished_at_utc=now())
        save(output/'state.json',state)
        raise


def batch(source, output):
    """Finish all independent frozen tasks once; preserve failures and keep going."""
    manifest = json.loads(source.read_text())
    sources = [source.parent / name for name in manifest.get('tasks', [source.name])]
    if not sources:
        raise ValueError('Empty training queue')
    tasks = [json.loads(path.read_text()) for path in sources]
    names = [task['name'] for task in tasks]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate task names in queue')
    for task, path in zip(tasks, sources):
        resolve_inputs(task, path)
    output.mkdir(parents=True, exist_ok=False)
    state = {'status':'running', 'started_at_utc':now(), 'outcomes':[],
             'automatic_training_retries':0, 'holdout_labels_evaluated':False}
    save(output/'state.json', state)
    for task, path in zip(tasks, sources):
        directory = output/task['name']
        directory.mkdir()
        try:
            result = train_task(task, path, directory)
            state['outcomes'].append({'name':task['name'], 'status':result['status']})
        except Exception as error:
            state['outcomes'].append({'name':task['name'], 'status':'failed',
                                      'error':f'{type(error).__name__}: {error}'})
        save(output/'state.json', state)
    failed = any(row['status'] != 'completed' for row in state['outcomes'])
    state.update(status='completed_with_failures' if failed else 'completed', finished_at_utc=now())
    save(output/'state.json', state)
    if failed:
        raise RuntimeError('One or more fixed tasks failed; return all evidence without retry')
    return state


def smoke(foundation, output):
    import gc
    import numpy as np
    import torch
    from ase.build import molecule
    from ase.io import read, write
    from mace.calculators import MACECalculator
    runtime=doctor('cuda')
    minimum=json.loads((ROOT/'configs/cloud-worker.json').read_text())['minimum_smoke_free_vram_MiB']
    if runtime['gpu_free_MiB'] < minimum:
        raise RuntimeError(f'GPU busy: require {minimum} MiB free before the bounded probe')
    if not foundation.is_file():
        raise FileNotFoundError('Upload the explicit frozen foundation with the task inputs')
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    calculator=MACECalculator(model_paths=str(foundation),device='cuda',default_dtype='float64')
    water=molecule('H2O'); water.calc=calculator
    energy=float(water.get_potential_energy()); force=water.get_forces().copy()
    if force.shape != (3,3) or not np.isfinite(energy) or not np.isfinite(force).all():
        raise RuntimeError('GPU energy/forces are not finite')
    epsilon=1e-4
    atom,axis=np.unravel_index(np.abs(force).argmax(),force.shape)
    plus=water.copy(); plus.calc=calculator; plus.positions[atom,axis]+=epsilon
    minus=water.copy(); minus.calc=calculator; minus.positions[atom,axis]-=epsilon
    difference=-(plus.get_potential_energy()-minus.get_potential_energy())/(2*epsilon)
    np.testing.assert_allclose(difference,force[atom,axis],atol=1e-4,rtol=1e-3)
    frames=[]
    for displacement in [-.03,-.02,-.01,.01,.02,.03,.015,-.015]:
        atoms=water.copy(); atoms.positions[1,0]+=displacement; atoms.calc=calculator
        reference_energy=float(atoms.get_potential_energy()); reference_force=atoms.get_forces().copy()
        atoms.calc=None; atoms.info['REF_energy']=reference_energy; atoms.arrays['REF_forces']=reference_force
        frames.append(atoms)
    write(output/'train.extxyz',frames[:6]); write(output/'valid.extxyz',frames[6:])
    water.calc=None; plus.calc=None; minus.calc=None
    del calculator,water,plus,minus
    gc.collect(); torch.cuda.empty_cache()
    settings={'device':'cuda','default_dtype':'float64','foundation_model_readout':True,
        'multiheads_finetuning':False,'energy_key':'REF_energy','forces_key':'REF_forces',
        'E0s':'estimated','model':'ScaleShiftMACE','loss':'weighted','scaling':'no_scaling',
        'energy_weight':10.,'forces_weight':1.,'batch_size':2,'valid_batch_size':2,
        'max_num_epochs':2,'patience':100,'lr':1e-4,'freeze':0,'num_workers':0,
        'eval_interval':1,'save_cpu':True}
    task={'name':'cloud_readiness_water','seed':20261123,'purpose':'infrastructure_readiness_only',
        'labels':'generated by the supplied MLIP, not DFT or scientific training data',
        'train_file':'train.extxyz','valid_file':'valid.extxyz','foundation_model':str(foundation.resolve()),
        'settings':settings,'expected_epochs':2,'max_wall_seconds':600}
    save(output/'task.json',task)
    trained=train_task(task,output/'task.json',output)
    calculator=MACECalculator(model_paths=trained['model'],device='cuda',default_dtype='float64')
    restored=read(output/'valid.extxyz'); restored.calc=calculator
    if not np.isfinite(restored.get_potential_energy()) or not np.isfinite(restored.get_forces()).all():
        raise RuntimeError('Reloaded trained model inference failed')
    result={'status':'PASS_MACE_INFERENCE_AND_TWO_EPOCH_TRAINING','host':socket.gethostname(),
        'finished_at_utc':now(),'runtime':runtime,'energy_eV':energy,'forces_eV_A':force.tolist(),
        'finite_difference_force_eV_A':float(difference),'checked_atom_axis':[int(atom),int(axis)],
        'actual_epochs':trained['actual_epochs'],
        'trained_model_reloaded':True,'scientific_model_improvement_claimed':False,
        'native_linux_gaussian_verified':False,'real_cloud_compute_verified':False}
    save(output/'acceptance.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['installation','doctor','smoke','train','batch'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--foundation',type=Path)
    parser.add_argument('--task',type=Path)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.action=='installation':
        result={**installed(),'status':'PASS_INSTALLED','model_inference_verified':False,
                'official_training_verified':False}
        save(args.output,result)
    elif args.action=='doctor':
        result=doctor(args.device)
        if args.output: save(args.output,result)
    elif args.action=='smoke':
        if not args.foundation or not args.output: parser.error('smoke requires --foundation and fresh --output')
        result=smoke(args.foundation,args.output)
    elif args.action=='batch':
        if not args.task or not args.output: parser.error('batch requires --task and fresh --output')
        result=batch(args.task.resolve(),args.output.resolve())
    else:
        if not args.task or not args.output: parser.error('train requires --task and fresh --output')
        source=args.task.resolve(); output=args.output.resolve(); task=json.loads(source.read_text())
        resolve_inputs(task,source)
        if not args.worker: output.mkdir(parents=True,exist_ok=False)
        if args.detach:
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'train','--task',str(source),
                     '--output',str(output),'--worker']
            with (output/'owner-console.log').open('xb') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
            result={'status':'submitted','host':socket.gethostname(),'owner_pid':process.pid,
                    'command':command,'submitted_at_utc':now(),'output':str(output)}
            save(output/'launch.json',result)
        else:
            lock=(output/'owner.lock').open('a')
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=train_task(task,source,output)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
