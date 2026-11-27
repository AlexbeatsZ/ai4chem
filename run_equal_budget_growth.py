"""4096x4 versus existing 2048x8, equal labels and optimizer steps."""
import json, os, fcntl, gc
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from ase.io import read, write
from model_study_data import ROOT, COLUMNS, endpoint_pool, diverse_subset, save_json, final_band_indices
from model_growth_data import excluded_roles, now
from prepare_research_data import formula_splits, training_atoms
from run_model_growth import wait_for_cuda
from run_model_study import train, benchmark
OUT=ROOT/'results/equal-budget-20261127'
DATA=ROOT/'data/equal-budget-20261127'
def load(path): return json.loads((ROOT/path).read_text())
def main():
    OUT.mkdir(parents=True,exist_ok=False); DATA.mkdir(parents=True,exist_ok=False)
    state=dict(owner_pid=os.getpid(),holdout_labels_evaluated=False,started_at_utc=now())
    def update(stage,**kw):
        state.update(stage=stage,updated_at_utc=now(),**kw); save_json(OUT/'state.json',state)
    try:
        protocol=load('configs/v0.10.0.json')
        protocol['training_seed']=20261116
        locks=wait_for_cuda(protocol,update)
        for p in ('results/v0.10.0/runner.lock','results/ts-weight-20261126/runner.lock'):
            path=ROOT/p
            if path.exists():
                lock=path.open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); locks.append(lock)
        sources=[ROOT/f'data/raw/transition1x_train_{i:02d}.parquet' for i in range(13)]
        assert all(p.exists() for p in sources)
        prior=load('configs/v0.10.0_partitions.json')
        base=set(prior['datasets']['2048'])
        roles=excluded_roles(load('configs/v0.7.0_partitions.json'),load('configs/v0.3.0_partitions.json'),load('configs/v0.5.0_reactions.json'))
        blocked=set().union(*roles.values())
        assert not base & blocked
        allowed=formula_splits()['train']
        update('freezing_endpoint_selection')
        table=pq.read_table(sources,columns=['sid','fid']).group_by('sid').aggregate([('fid','count')])
        counts=dict(zip(table['sid'].to_pylist(),table['fid_count'].to_pylist())); del table
        pool=endpoint_pool(sources,blocked|base,allowed)
        added=diverse_subset([x for x in pool if counts[x['sid']]>=16],2048,20261127)
        ids=sorted(base|{x['sid'] for x in added})
        assert len(ids)==4096 and not set(ids)&blocked
        save_json(OUT/'partition.json',dict(training_ids=ids,prior_ids=sorted(base),new_ids=[x['sid'] for x in added],frozen_at_utc=now(),selection='endpoint diversity, no evaluation labels',holdout_labels_evaluated=False),immutable=True)
        dataset=DATA/'coverage4096x4.extxyz'; records=[]
        for first in range(0,len(ids),128):
            batch=ids[first:first+128]
            table=pq.read_table(sources,columns=COLUMNS,filters=[('sid','in',batch)],use_threads=False).sort_by([('sid','ascending'),('fid','ascending')])
            c=Counter(table['sid'].to_pylist()); offset=0; frames=[]
            for sid in batch:
                rows=table.slice(offset,c[sid]).to_pylist(); offset+=c[sid]
                assert len(rows)==counts[sid] and [r['fid'] for r in rows]==list(range(len(rows)))
                assert all(r['atomic_numbers']==rows[0]['atomic_numbers'] for r in rows)
                final=final_band_indices(len(rows)).tolist()
                peak=max(final[1:-1],key=lambda i:rows[i]['y'])
                other=[i for i in final[1:-1] if i!=peak]
                fourth=max(other,key=lambda i:min(abs(final.index(i)-final.index(j)) for j in (0,9,peak)))
                for fid in (0,9,peak,fourth):
                    frames.append(training_atoms(rows[fid],'train',allowed)); records.append(dict(sid=sid,fid=fid))
            write(dataset,frames,append=first>0); del table,frames;gc.collect()
            update('preparing',reactions=first+len(batch))
        restored=read(dataset,index=':')
        assert len(restored)==16384 and len({(r['sid'],r['fid']) for r in records})==16384
        for a,r in zip(restored,records,strict=True):
            assert a.info['reaction']==f"rxn{r['sid']:04d}" and a.info['source_frame']==r['fid']
            assert np.isfinite(a.info['REF_energy']) and np.isfinite(a.arrays['REF_forces']).all()
        del restored
        save_json(DATA/'manifest.json',dict(records=records,holdout_labels_evaluated=False),immutable=True)
        experiment=dict(name='mh1_coverage4096x4_s0',dataset='coverage4096x4',train_size=16384,train_file=str(dataset.relative_to(ROOT)),seed=20261116,model_file='models/equal-budget-20261127/mh1_coverage4096x4_s0.model')
        import torch
        assert torch.cuda.is_available();torch.set_num_threads(4)
        runs=[]
        train(experiment,protocol,runs,output_root=OUT,manifest_path=OUT/'training-manifest.json',state_writer=update)
        assert runs[0]['trained_epochs']==20
        update('validation',run=runs[0])
        name=experiment['name']
        metrics=benchmark('validation',[name],{name:str(ROOT/experiment['model_file'])},output_root=OUT)['models'][name]['aggregate']
        base_metrics={'barrier_mae_eV':.10628014013879389,'force_component_rmse_eV_per_A':.08269444139643234,'ts_region_force_component_rmse_eV_per_A':.12357257294325198}
        update('completed',validation=metrics,baseline=base_metrics,eligible=all(metrics[k]<=v for k,v in base_metrics.items()),finished_at_utc=now())
    except BaseException as e:
        update('failed',error=repr(e));raise
if __name__=='__main__':main()
