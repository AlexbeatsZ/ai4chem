"""Compare fixed regional experts and grouped development routing after training.

The finite --wait-hours mode observes existing workers and does no training or
prospective evaluation. All deployable systems use their own trained members
and the same pre-existing fallback; the new global model is not a free expert.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from expert_routing import endpoint_features, grouped_delta_interval
from specialization_gate import predict_gate

OUTPUT = ROOT/'results/v0.8.0/development_analysis'
FALLBACK = 'legacy_mh1_256'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def metric_summary(table, reactions, choices):
    selected = table.set_index(['reaction','model']).loc[list(zip(reactions,choices))]
    errors = selected.barrier_error_eV.abs().to_numpy()
    return {'barrier_MAE_eV':float(errors.mean()),
        'barrier_absolute_error_p90_eV':float(np.quantile(errors,.9)),
        'force_global_component_RMSE_eV_per_A':float(np.sqrt(np.average(
            selected.force_component_rmse_eV_per_A.to_numpy()**2,weights=selected.atoms))),
        'mean_reaction_TS_force_RMSE_eV_per_A':float(selected.ts_region_force_component_rmse_eV_per_A.mean()),
        'reaction_energy_MAE_eV':float(selected.reaction_energy_error_eV.abs().mean()),
        'choices':dict(Counter(choices))}


def fixed_choices(members, labels, accepted):
    if set(members) not in ({'0','1'},{'global'}):
        raise ValueError('System must have a global model or exactly the two frozen regions')
    selected = ([members['global']]*len(labels) if 'global' in members else
                [members[str(region)] for region in labels])
    return [model if accept else FALLBACK for model,accept in zip(selected,accepted)]


def analyze(protocol, partition):
    previous = json.loads((ROOT/'configs/v0.7.0_partitions.json').read_text())
    records = sorted(previous['endpoint_records']['meta_development'],key=lambda row:row['sid'])
    reactions = [row['reaction'] for row in records]
    groups = np.array([row['formula'] for row in records])
    labels,accepted = predict_gate(records,partition['endpoint_gate'])
    base = pd.read_csv(ROOT/'results/v0.7.0/meta_development_mac/reaction_metrics.csv')
    tables = [base[base.model==FALLBACK]]
    systems = {}
    manifests = []
    for experiment in protocol['experiments']:
        name = experiment['name']
        manifest = json.loads((ROOT/f'results/v0.8.0/training/{name}/manifest.json').read_text())
        if len(manifest)!=1 or manifest[0]['name']!=name:
            raise ValueError(f'Wrong training manifest identity: {name}')
        expected_epochs = experiment.get('training_overrides',{}).get('epochs',protocol['training']['epochs'])
        if manifest[0]['trained_epochs'] != expected_epochs:
            raise ValueError(f'Logged epochs differ from the frozen budget: {name}')
        manifests += manifest
        table = pd.read_csv(ROOT/f'results/v0.8.0/meta_development/{name}/reaction_metrics.csv')
        if set(table.model)!={name}:
            raise ValueError(f'Wrong benchmark identity: {name}')
        tables.append(table)
        members = systems.setdefault(experiment['system'],{})
        members[str(experiment['region']) if 'region' in experiment else 'global'] = name
    table = pd.concat(tables,ignore_index=True)
    models = [FALLBACK]+[row['name'] for row in protocol['experiments']]
    if len(table)!=len(reactions)*len(models) or table.duplicated(['reaction','model']).any():
        raise ValueError('Expected exactly one paired row per model/reaction')
    for model in models:
        subset = table[table.model==model].set_index('reaction')
        if set(subset.index)!=set(reactions):
            raise ValueError(f'Development reaction identity differs: {model}')
        for key in ('reference_barrier_eV','atoms','formula'):
            expected = base[base.model==FALLBACK].set_index('reaction').loc[reactions,key]
            actual = subset.loc[reactions,key]
            if key=='reference_barrier_eV':
                same = np.allclose(actual.astype(float),expected.astype(float),atol=1e-10,rtol=0)
            else:
                same = actual.tolist()==expected.tolist()
            if not same:
                raise ValueError(f'Paired reference differs for {model}/{key}')
    numeric = table.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError('Nonfinite benchmark values')
    losses = table.pivot(index='reaction',columns='model',values='barrier_error_eV').loc[reactions,models].abs()
    baseline_choices = fixed_choices(systems['single_general'],labels,accepted)
    baseline_errors = np.array([losses.iloc[i][model] for i,model in enumerate(baseline_choices)])
    features = pd.DataFrame([endpoint_features(row) for row in records]).to_numpy(float)
    config = json.loads((ROOT/'configs/v0.8.0_analysis.json').read_text())
    if config['partition_frozen_at_utc']!=partition['frozen_at_utc'] or config['methods']!=['ridge','extratrees']:
        raise ValueError('Analysis protocol identity differs')
    folds = list(GroupKFold(n_splits=config['folds'],shuffle=True,random_state=config['seed']).split(features,groups=groups))
    report = {'status':'completed','scope':'Exploratory meta-development only; endpoint gate fixed before training; grouped OOF routing uses only fold-training labels',
        'finished_at_utc':datetime.now(timezone.utc).isoformat(),'holdout_labels_evaluated':False,
        'partition_frozen_at_utc':partition['frozen_at_utc'],'analysis_protocol':config,'reactions':len(records),
        'formulas':len(set(groups)),'region_counts':np.bincount(labels,minlength=2).tolist(),
        'geometry_fallback_count':int((~accepted).sum()),'training_manifests':manifests,
        'fallback_policy':'Same frozen legacy_mh1_256 for every system; global model belongs only to the single_general system',
        'fixed_models':{model:metric_summary(table,reactions,[model]*len(records)) for model in models},
        'systems':{}}
    choices_records = []
    for system,members in systems.items():
        pool = list(members.values())+[FALLBACK]
        risks = losses[pool].to_numpy()
        candidates = {'fixed_endpoint_gate':fixed_choices(members,labels,accepted)}
        oracle = np.asarray(pool,dtype=object)[risks.argmin(axis=1)]
        oracle[~accepted] = FALLBACK
        candidates['barrier_oracle_unattainable'] = oracle.tolist()
        fold_fixed = np.empty(len(records),dtype=object)
        for train,test in folds:
            fold_fixed[test] = pool[int(risks[train].mean(axis=0).argmin())]
        fold_fixed[~accepted] = FALLBACK
        candidates['fold_selected_fixed_with_geometric_fallback'] = fold_fixed.tolist()
        for method in config['methods']:
            chosen = np.empty(len(records),dtype=object)
            for fold,(train,test) in enumerate(folds):
                if set(groups[train]) & set(groups[test]):
                    raise ValueError('Formula leakage across router folds')
                estimator = make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                    StandardScaler(),Ridge(alpha=config['ridge_alpha'])) if method=='ridge' else make_pipeline(
                    SimpleImputer(strategy='median',keep_empty_features=True),ExtraTreesRegressor(
                        random_state=config['seed']+fold,**config['extratrees']))
                estimator.fit(features[train],risks[train])
                predicted = np.maximum(estimator.predict(features[test]),0.)
                if not np.isfinite(predicted).all():
                    raise ValueError('Nonfinite OOF predicted errors')
                chosen[test] = np.asarray(pool,dtype=object)[predicted.argmin(axis=1)]
            chosen[~accepted] = FALLBACK
            candidates[method+'_oof_with_geometric_fallback'] = chosen.tolist()
        entry = {'members':members,'deployable_pool':pool,'policies':{},'conditional_fixed_models':{}}
        for model in members.values():
            entry['conditional_fixed_models'][model] = {str(region):metric_summary(table,
                list(np.asarray(reactions)[labels==region]),[model]*int((labels==region).sum()))
                for region in (0,1) if (labels==region).any()}
        for policy,choice in candidates.items():
            if not set(choice).issubset(pool):
                raise ValueError('A system received an unbudgeted expert')
            errors = np.array([losses.iloc[i][model] for i,model in enumerate(choice)])
            entry['policies'][policy] = {**metric_summary(table,reactions,choice),
                'paired_barrier_delta_vs_single_general_fixed_gate':grouped_delta_interval(errors-baseline_errors,groups),
                'uses_evaluation_labels_for_choice':policy=='barrier_oracle_unattainable'}
            choices_records += [{'system':system,'policy':policy,'reaction':reaction,'region':int(labels[i]),
                'geometry_accepted':bool(accepted[i]),'selected_model':choice[i]} for i,reaction in enumerate(reactions)]
        report['systems'][system] = entry
    OUTPUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(choices_records).to_csv(OUTPUT/'choices.csv',index=False)
    table.to_csv(OUTPUT/'reaction_metrics.csv',index=False)
    save(OUTPUT/'summary.json',report)
    print(json.dumps({'status':report['status'],'reactions':len(records),'systems':list(systems),'holdout_labels_evaluated':False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-hours',type=float,default=0)
    args = parser.parse_args()
    if not 0 <= args.wait_hours <= 96:
        parser.error('Use a finite wait from zero to 96 hours')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock = (OUTPUT/'owner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUTPUT/'summary.json').exists():
        raise FileExistsError('Completed analysis exists; inspect without overwriting')
    protocol = json.loads((ROOT/'configs/v0.8.0.json').read_text())
    partition = json.loads((ROOT/'configs/v0.8.0_partitions.json').read_text())
    if partition['protocol']!=protocol:
        raise ValueError('Frozen partition and protocol differ')
    deadline = time.monotonic()+args.wait_hours*3600
    while True:
        stages = {}
        for experiment in protocol['experiments']:
            path = ROOT/f"results/v0.8.0/training/{experiment['name']}/state.json"
            stages[experiment['name']] = json.loads(path.read_text())['stage'] if path.exists() else 'pending'
        if all(stage=='completed' for stage in stages.values()):
            state = {'host':'Mac','owner_pid':os.getpid(),'stage':'analyzing',
                     'holdout_labels_evaluated':False}
            save(OUTPUT/'state.json',state)
            try:
                analyze(protocol,partition)
                state.update(stage='completed',finished_at_utc=datetime.now(timezone.utc).isoformat())
            except Exception as error:
                state.update(stage='failed',error=f'{type(error).__name__}: {error}')
                save(OUTPUT/'state.json',state)
                raise
            save(OUTPUT/'state.json',state)
            return
        state = {'host':'Mac','owner_pid':os.getpid(),'stage':'waiting_for_training',
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'worker_stages':stages,
            'holdout_labels_evaluated':False}
        parent_path = ROOT/'results/v0.8.0/state.json'
        if 'failed' in stages.values() or (parent_path.exists() and json.loads(parent_path.read_text())['stage']=='failed'):
            state['stage']='stopped_training_failure';save(OUTPUT/'state.json',state)
            raise RuntimeError('Training failure retained; no analysis or automatic retries')
        if time.monotonic()>=deadline:
            state['stage']='stopped_deadline';save(OUTPUT/'state.json',state)
            raise TimeoutError('Seven development benchmarks are not complete')
        save(OUTPUT/'state.json',state)
        time.sleep(60)


if __name__=='__main__':
    main()
