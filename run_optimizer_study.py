"""Finite CUDA learning-rate/force-weight controls and two extra seed repeats.

Wait for the existing data study and acquire its lock before spending GPU time.
This runner never opens prospective holdout labels or retries failed training.
"""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import yaml

from model_study_data import save_json
from run_model_study import benchmark, train

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'results/v0.9.0'


def now():
    return datetime.now(timezone.utc).isoformat()


def choose_guarded(metrics, baseline):
    """Validation barrier rank, subject to baseline force and TS-force guards."""
    reference = metrics[baseline]
    eligible = [name for name,item in metrics.items() if
        item['force_component_rmse_eV_per_A'] <= reference['force_component_rmse_eV_per_A']+1e-12 and
        item['ts_region_force_component_rmse_eV_per_A'] <= reference['ts_region_force_component_rmse_eV_per_A']+1e-12]
    if not all(np.isfinite(value) for item in metrics.values() for value in item.values()):
        raise ValueError('Nonfinite selection metrics')
    return min(eligible,key=lambda name:(metrics[name]['barrier_mae_eV'],
        metrics[name]['ts_region_force_component_rmse_eV_per_A'],name))


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock = (OUTPUT/'runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads((ROOT/'configs/v0.9.0.json').read_text())
    state = {'host':'OMEN','owner_started_at_utc':now(),'holdout_labels_evaluated':False}
    import os
    state['owner_pid'] = os.getpid()

    def update(stage,**extra):
        state.update(stage=stage,updated_at_utc=now(),**extra)
        save_json(OUTPUT/'state.json',state)

    predecessor_lock = None
    try:
        deadline = time.monotonic()+protocol['predecessor_wait_hours']*3600
        predecessor = ROOT/'results/v0.7.0'
        while predecessor_lock is None:
            source = json.loads((predecessor/'state.json').read_text())
            if source['stage']=='first_stage_screening_complete':
                candidate_lock = (predecessor/'runner.lock').open('a')
                try:
                    fcntl.flock(candidate_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
                    predecessor_lock = candidate_lock
                    break
                except BlockingIOError:
                    candidate_lock.close()
            if source['stage']=='failed':
                raise RuntimeError('Predecessor failed; inspect without starting new training')
            if time.monotonic()>deadline:
                raise TimeoutError('Predecessor has not completed/released its GPU study lock')
            update('waiting_for_predecessor',predecessor_stage=source['stage'],predecessor_experiment=source.get('experiment'))
            time.sleep(60)
        if not torch.cuda.is_available():
            raise RuntimeError('This optimization study requires the accepted CUDA environment')
        screen = json.loads((predecessor/'screening_summary.json').read_text())
        names = protocol['candidate_data_arms']
        selected_data = min(names,key=lambda name:(screen['validation_metrics'][name]['barrier_mae_eV'],
            screen['validation_metrics'][name]['ts_region_force_component_rmse_eV_per_A'],name))
        original = next(row for row in screen['runs'] if row['name']==selected_data)
        old_config = yaml.safe_load((ROOT/original['config']).read_text())
        expected = {'seed':protocol['training_seed'],'batch_size':4,'lr':.0001,'energy_weight':10.,
            'forces_weight':1.,'max_num_epochs':40,'default_dtype':'float64','scaling':'no_scaling'}
        if original['train_size']!=4096 or original['trained_epochs']!=40 or any(old_config[key]!=value for key,value in expected.items()):
            raise ValueError('Predecessor baseline is not the fixed matched-budget configuration')
        frozen = {'protocol':protocol,'selected_data_arm':selected_data,'source_run':original,
                  'source_validation_metrics':screen['validation_metrics'][selected_data]}
        frozen_path = OUTPUT/'frozen_inputs.json'
        if frozen_path.exists():
            prior = json.loads(frozen_path.read_text())
            if {key:value for key,value in prior.items() if key!='frozen_at_utc'}!=frozen:
                raise ValueError('Concrete frozen data selection changed')
            frozen = prior
        else:
            frozen['frozen_at_utc'] = now()
            save_json(frozen_path,frozen,immutable=True)
        metrics = {selected_data:screen['validation_metrics'][selected_data]}
        parameters = {selected_data:{'learning_rate':.0001,'forces_weight':1.}}
        benchmark(f'meta_development/{selected_data}',[selected_data],
                  {selected_data:str(ROOT/original['model'])},role='meta_development',count=128,output_root=OUTPUT)
        path = OUTPUT/'attempts.json'
        attempts = json.loads(path.read_text()) if path.exists() else {}

        def run_arm(name,settings,seed):
            if name in attempts and attempts[name]['status']=='failed':
                return None
            experiment = {'name':name,'dataset':original['dataset'],'train_size':4096,
                'seed':seed,'training_overrides':settings,'model_file':f'models/v0.9.0/{name}.model'}
            manifest_path = OUTPUT/'training'/name/'manifest.json'
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
            try:
                train(experiment,protocol,manifest,output_root=OUTPUT,manifest_path=manifest_path,state_writer=update)
                if manifest[0]['trained_epochs']!=40:
                    raise ValueError('Actual epoch count differs from the frozen budget')
                model = str(ROOT/experiment['model_file'])
                validation = benchmark(f'validation/{name}',[name],{name:model},output_root=OUTPUT)['models'][name]['aggregate']
                benchmark(f'meta_development/{name}',[name],{name:model},role='meta_development',count=128,output_root=OUTPUT)
                attempts[name] = {'status':'completed','experiment':experiment,'run':manifest[0],
                                  'validation':validation,'finished_at_utc':now()}
                save_json(path,attempts)
                return validation
            except Exception:
                attempts[name] = {'status':'failed','experiment':experiment,'error':traceback.format_exc(),'finished_at_utc':now()}
                save_json(path,attempts)
                update('arm_failed_preserved',experiment=name)
                return None

        for experiment in protocol['grid_new_arms']:
            settings = experiment['training_overrides']
            validation = run_arm(experiment['name'],settings,protocol['training_seed'])
            if validation is not None:
                metrics[experiment['name']] = validation
                parameters[experiment['name']] = settings
        selected = choose_guarded(metrics,selected_data)
        selection = {'selected_hyperparameters_arm':selected,'selected_data_arm':selected_data,
            'parameters':parameters[selected],'validation_metrics_all_completed':metrics,
            'frozen_at_utc':now(),'criterion':protocol['hyperparameter_selection']}
        selection_path = OUTPUT/'selected_hyperparameters.json'
        if selection_path.exists():
            prior = json.loads(selection_path.read_text())
            if {k:v for k,v in prior.items() if k!='frozen_at_utc'}!={k:v for k,v in selection.items() if k!='frozen_at_utc'}:
                raise ValueError('Selected hyperparameters changed on resume')
            selection = prior
        else:
            save_json(selection_path,selection,immutable=True)
        repeat_names = []
        for index,seed in enumerate(protocol['additional_seeds'],1):
            name = f'mh1_opt_selected_s{index}'
            repeat_names.append(name)
            run_arm(name,parameters[selected],seed)
        repetitions_complete = all(attempts[name]['status']=='completed' for name in repeat_names)
        seed_metrics = [{'seed':protocol['training_seed'],'arm':selected,'validation':metrics[selected]}]
        seed_metrics += [{'seed':attempts[name]['experiment']['seed'],'arm':name,'validation':attempts[name]['validation']}
                         for name in repeat_names if attempts[name]['status']=='completed']
        aggregate = {key:{'mean':float(np.mean([row['validation'][key] for row in seed_metrics])),
                          'sample_std':float(np.std([row['validation'][key] for row in seed_metrics],ddof=1)) if len(seed_metrics)>1 else None}
                     for key in metrics[selected]}
        summary = {'status':'optimizer_screening_and_seed_repeats_complete' if repetitions_complete else 'incomplete_seed_repeats',
            'finished_at_utc':now(),'protocol':protocol,'frozen_inputs':frozen,'selection':selection,
            'attempts':attempts,'first_seed_reference':selected,'additional_seed_names':repeat_names,
            'seed_validation_results':seed_metrics,'seed_validation_aggregate':aggregate,
            'failed_attempts':[name for name,row in attempts.items() if row['status']=='failed'],
            'holdout_labels_evaluated':False,'remaining':['assess every seed and metric jointly','compare data breadth and adapted architectures',
                'freeze final models/router','prospective static/search and preset DFT evaluation']}
        save_json(OUTPUT/'screening_summary.json',summary)
        update(summary['status'],finished_at_utc=now(),selected_hyperparameters_arm=selected)
    except Exception as error:
        update('failed',error=f'{type(error).__name__}: {error}',finished_at_utc=now())
        raise


if __name__=='__main__':
    main()
