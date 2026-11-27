"""Generate train-only on-policy NEB candidates; freeze equal-size random and disagreement arms."""
import json, os, hashlib, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from ase import Atoms
from ase.mep import NEB
from ase.optimize import FIRE
from ase.io import write
from mace.calculators import MACECalculator
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/active-learning-pilot-20261127'
def now(): return datetime.now(timezone.utc).isoformat()
def save(name,obj):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    state=dict(pid=os.getpid(),status='running',holdout_labels_evaluated=False,started_at_utc=now())
    save('state.json',state)
    try:
        torch.set_num_threads(4)
        p=json.loads((ROOT/'configs/v0.10.0_partitions.json').read_text())
        roles=json.loads((ROOT/'configs/v0.7.0_partitions.json').read_text())
        excluded=set().union(*(set(roles[k]) for k in ['validation','meta_development','holdout_train','holdout_val']))
        previous=set(json.loads((ROOT/'configs/offpath32-20261126.json').read_text())['training_reactions'])
        pool=[r for r in p['endpoint_records'] if r['sid'] in p['datasets']['2048'] and r['sid'] not in excluded and r['reaction'] not in previous and r['atoms']<=16]
        rng=np.random.default_rng(2026112702)
        selected=[]
        for kind in sorted({r['kind'] for r in pool}):
            options=[r for r in pool if r['kind']==kind]
            selected.extend([options[i] for i in rng.permutation(len(options))[:min(4,len(options))]])
        remaining=[r for r in pool if r not in selected]
        selected=(selected+[remaining[i] for i in rng.permutation(len(remaining))[:max(0,16-len(selected))]])[:16]
        assert len(selected)==16
        save('protocol.json',dict(frozen_at_utc=now(),reactions=selected,seed=2026112702,images=7,neb_steps=24,snapshot_steps=[0,4,8,12,16,20,24],arms='2 high disagreement and 2 independent random per reaction; shared labels reused',geometry_gate='minimum distance > 0.55 A; max seed0 atom force <= 20 eV/A',holdout_labels_evaluated=False))
        calculators=[MACECalculator(model_paths=str(ROOT/f'models/v0.10.0/mh1_coverage2048_s{i}.model'),device='cpu',default_dtype='float64',head='Default') for i in range(3)]
        chosen={};records=[];errors=[]
        for row in selected:
            reaction=row['reaction'];folder=OUT/reaction;folder.mkdir()
            try:
                a=Atoms(row['atomic_numbers'],positions=row['reactant_positions']);b=Atoms(row['atomic_numbers'],positions=row['product_positions'])
                images=[a.copy() for _ in range(6)]+[b];neb=NEB(images,allow_shared_calculator=True,k=0.1)
                neb.interpolate(method='idpp')
                for im in images: im.calc=calculators[0]
                opt=FIRE(neb,logfile=str(folder/'neb.log'))
                candidates=[];seen=set()
                for step in range(0,25,4):
                    if step:opt.run(fmax=.05,steps=4)
                    for j,im in enumerate(images[1:-1],1):
                        xyz=im.positions.copy();dist=im.get_all_distances();dist[dist==0]=np.inf
                        if not np.isfinite(xyz).all() or np.min(dist)<=.55:continue
                        key=hashlib.sha256(np.round(xyz,6).tobytes()).hexdigest()
                        if key in seen:continue
                        seen.add(key)
                        forces=[]
                        for c in calculators:
                            x=im.copy();x.calc=c;forces.append(x.get_forces().copy())
                        f=np.asarray(forces)
                        if not np.isfinite(f).all() or np.linalg.norm(f[0],axis=1).max()>20:continue
                        score=float(np.sqrt(np.mean((f-f.mean(axis=0))**2)))
                        name=f'{reaction}-step{step:02d}-im{j:02d}'
                        candidates.append(dict(name=name,step=step,image=j,score=score,positions=xyz.tolist(),symbols=im.get_chemical_symbols(),reaction=reaction))
                if len(candidates)<2:raise ValueError('insufficient eligible candidates')
                ranked=sorted(candidates,key=lambda x:(-x['score'],x['name']))[:2]
                random=[candidates[i] for i in rng.choice(len(candidates),2,replace=False)]
                for arm,items in [('disagreement',ranked),('random',random)]:
                    for item in items:
                        rec=chosen.setdefault(item['name'],dict(item,arms=[]))
                        rec['arms'].append(arm)
                records.extend(candidates)
                save('candidates.json',records)
            except Exception as e:
                errors.append(dict(reaction=reaction,error=repr(e)))
            save('state.json',dict(state,processed=len(list(OUT.glob('rxn*'))),eligible_reactions=len(set(x['reaction'] for x in chosen.values())),errors=errors))
        if len({x['reaction'] for x in chosen.values()})!=16:raise RuntimeError('incomplete paired reaction pilot; no DFT admission')
        (OUT/'inputs').mkdir()
        items=[]
        for name,x in sorted(chosen.items()):
            rel=f'inputs/{name}.gjf'
            lines=[f'%chk={name}.chk','%mem=2GB','%nprocshared=4','#p wb97x/6-31g(d) 5D 7F force scf=(xqc,maxcycle=128) int=ultrafine nosymm','',f'AI4Chem on-policy {name}','','0 1']
            lines.extend(f'{s} {a:.12f} {b:.12f} {c:.12f}' for s,(a,b,c) in zip(x['symbols'],x['positions'],strict=True))
            (OUT/rel).write_text('\n'.join(lines)+'\n\n')
            items.append(dict(x,input=rel))
        assert all(sum(arm in x['arms'] for x in items)==32 for arm in ['disagreement','random'])
        save('queue.json',dict(version='active-learning-pilot-20261127',chains=[],force_items=items,target_compute_hours=4,training_merge_authorized=False,holdout_labels_evaluated=False,automatic_retries=0,policy='64 arm slots, overlapping geometries reused; no automatic training merge; finite serial 4 threads 2GB'))
        save('state.json',dict(state,status='prepared_not_dispatched',candidate_count=len(records),unique_labels=len(items),finished_at_utc=now(),errors=errors))
    except BaseException as e:
        save('state.json',dict(state,status='failed',error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
