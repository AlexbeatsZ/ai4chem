"""Freeze local training assignments and a finite, training-side Gaussian reserve."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from ase.io import read
from prepare_gaussian_chain import prepare

ROOT=Path(__file__).resolve().parents[1]


def write_new(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle: handle.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def build():
    stamp=datetime.now(timezone.utc).isoformat()
    output=ROOT/'results/overnight-20261124'
    output.mkdir(exist_ok=False)
    base=json.loads((ROOT/'configs/v0.7.0.json').read_text())
    training={**base['training'],'patience':100}
    nodes={'omen':dict(device='cuda',workers=1,seeds=[20261124,20261125,20261126]),
           'mac':dict(device='cpu',workers=2,seeds=[20261127,20261128,20261129,20261130])}
    protocol=dict(version='paired-sampling-r1-local-dispatch',frozen_at_utc=stamp,
        output='results/overnight-20261124',training=training,nodes=nodes,
        user_authorization='November24: deploy >=18h useful work to roommate, OMEN and Mac. No cloud purchase or other human dispatch.',
        original_five_seeds=[20261124,20261125,20261126,20261127,20261128],
        extension='Two additional paired repeats20261129/30 fixed before their training; seven pairs reported, no best-seed-only selection.',
        original_packages_reserved='member01-03 assigned OMEN, member04-05 assigned Mac; do not redistribute original packages.',
        analysis='Report all pairs, paired random-path differences, every seed, failures and actual compute. Original development seed16 separate.',
        admission='OMEN waits for actual v0.9/v0.7 locks; Mac two extra four-thread workers preserve existing specialization workers.',
        estimated_omen_hours=21,estimated_mac_hours='~36-44 at two workers using recent ~14-16min/epoch; actual contention varies',
        holdout_labels_evaluated=False,automatic_training_retries=0)
    write_new(ROOT/'configs/overnight-20261124.json',protocol)
    # Mac retained the final self-contained bundles, not loose training inputs.
    with zipfile.ZipFile(ROOT/'results/group-training/paired-sampling-r1/member-01.zip') as archive:
        for arm in ('random4096','path4096'):
            target=ROOT/f'data/v0.7.0/{arm}.extxyz'
            if not target.exists():
                with target.open('xb') as handle: handle.write(archive.read(f'shared/{arm}.extxyz'))
            if len(read(target,index=':'))!=4096: raise ValueError('Training frame budget differs')
    queue=output/'roommate';queue.mkdir()
    chains=[]
    for number in (1,2,3):
        source=ROOT/f'results/group-jobs/2026-11-23-r1/member-{number:02d}/protocol.json'
        candidate=json.loads(source.read_text())
        candidate.update(dft_mem='2GB',frozen_at_utc=stamp,
            resource_amendment='Roommate has4.89GiB free.2GB Gaussian+2GiB admission headroom,4threads; all scientific gates unchanged.')
        job=candidate['jobs'][0]
        job['source_prepared_job_id']=job['job_id']
        job['job_id']='roommate-20261124-'+job['reaction']+'-'+job['expert']
        relative='chains/'+job['job_id']
        prepare(candidate,job['reaction'],queue/relative)
        write_new(queue/relative/'protocol.json',candidate)
        chains.append(dict(job_id=job['job_id'],reaction=job['reaction'],directory=relative))
    frames=read(ROOT/'data/v0.8.0/general4096.extxyz',index=':')
    groups={}
    for atoms in frames: groups.setdefault(atoms.info['reaction'],[]).append(atoms)
    partition=json.loads((ROOT/'configs/v0.8.0_partitions.json').read_text())
    endpoints=json.loads((ROOT/'data/v0.8.0/training_endpoint_records.json').read_text())
    # Round robin by region and endpoint-only reaction kind; no error/holdout selection.
    buckets={}
    for record in endpoints:
        region=next(k for k,v in partition['regions'].items() if record['sid'] in v)
        buckets.setdefault((region,record['kind']),[]).append(record)
    for rows in buckets.values(): rows.sort(key=lambda x:(x['atoms'],x['sid']))
    ordered=[]
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]: ordered.append(buckets[key].pop(0))
    items=[]
    inputs=queue/'inputs';inputs.mkdir()
    rng=np.random.default_rng(2026112401)
    for record in ordered:
        reaction=record['reaction']
        atoms=max(groups[reaction],key=lambda a:float(a.info['REF_energy']))
        geometries=[('reference',atoms.positions.copy(),0.)]
        for amplitude in (.01,.03,.06,.10):
            noise=rng.normal(size=atoms.positions.shape);noise-=noise.mean(axis=0)
            noise*=amplitude/np.sqrt(np.mean(np.sum(noise**2,axis=1)))
            for sign,label in [(1,'plus'),(-1,'minus')]:
                geometries.append((f'r{round(amplitude*100):02d}-{label}',atoms.positions+sign*noise,amplitude))
        for mode,positions,amplitude in geometries:
            name=reaction+'-'+mode
            relative='inputs/'+name+'.gjf'
            lines=['%mem=2GB','%nprocshared=4',
                '#p wb97x/6-31g(d) force scf=(xqc,maxcycle=128) int=ultrafine nosymm','',
                'AI4Chem training-side quarantined force label '+name,'','0 1']
            lines += [f'{s} {x:.12f} {y:.12f} {z:.12f}' for s,(x,y,z) in zip(atoms.get_chemical_symbols(),positions,strict=True)]
            (queue/relative).write_text('\n'.join(lines)+'\n\n',encoding='ascii')
            items.append(dict(name=name,input=relative,reaction=reaction,region=int(next(k for k,v in partition['regions'].items() if record['sid'] in v)),
                kind=record['kind'],atoms=len(atoms),mode=mode,rms_displacement_A=amplitude,
                source_frame=int(atoms.info['source_frame']),symbols=atoms.get_chemical_symbols(),
                source_energy_eV=float(atoms.info['REF_energy']) if mode=='reference' else None,
                source_forces_eV_A=atoms.arrays['REF_forces'].tolist() if mode=='reference' else None))
    queue_task=dict(version='roommate-overnight-r1',frozen_at_utc=stamp,chains=chains,force_items=items,
        target_compute_hours=24,training_reactions=len(ordered),force_reserve=len(items),
        force_selection='All512 frozen training reactions, region/kind round-robin then size/id. Highest-energy existing train frame; one reference plus antithetic0.01/0.03/0.06/0.10A perturbations. Not uncertainty acquisition.',
        policy='Three chains then force labels until24h actual serial execution time. No retries. Three consecutive force failures block. No idle time counted.',
        force_alignment='Quarantine all labels until full3N coordinate/force and6D basis/grid/reference alignment independently passes. No automatic training merge.',
        prospective_holdout_labels_evaluated=False,training_merge_authorized=False,
        group_reservations='Original group fullchain members01-03 now assigned roommate. Other humans/cloud remain undispatched.')
    write_new(queue/'queue.json',queue_task)
    with zipfile.ZipFile(output/'roommate-inputs.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in queue.rglob('*'):
            if path.is_file(): archive.write(path,path.relative_to(output))
    print(json.dumps(dict(protocol='configs/overnight-20261124.json',training_nodes=nodes,chains=chains,
                         force_reserve=len(items),archive_bytes=(output/'roommate-inputs.zip').stat().st_size),indent=2))


if __name__=='__main__': build()
