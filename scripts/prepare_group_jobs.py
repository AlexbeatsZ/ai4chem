"""Prepare five independent development DFT bundles; never submit or send them."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

import numpy as np
from ase.io import read

from prepare_gaussian_chain import prepare

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'configs/group-dft-development.json'


def build(output):
    output.mkdir(parents=True,exist_ok=False)
    timestamp=datetime.now(timezone.utc).isoformat()
    search=json.loads((ROOT/'configs/v0.7.0_diversity_search.json').read_text())
    previous=json.loads((ROOT/'configs/v0.7.0_diversity_dft.json').read_text())
    partition=json.loads((ROOT/'configs/v0.8.0_partitions.json').read_text())
    endpoints=json.loads((ROOT/'data/v0.8.0/training_endpoint_records.json').read_text())
    records={row['sid']:row for row in endpoints}
    frames=read(ROOT/'data/v0.8.0/general4096.extxyz',index=':')
    group={}
    for atoms in frames: group.setdefault(atoms.info['reaction'],[]).append(atoms)
    jobs=[]
    base={key:previous[key] for key in ['dft_method','dft_nproc','dft_mem','charge',
        'multiplicity','dft_policy','irc_endpoint_rmsd_A','endpoint_tolerance_A']}
    for number,reaction in enumerate(['rxn0255','rxn3342','rxn4757'],1):
        expert='aimnet2-rxn'; source=ROOT/'results/v0.7.0/diversity_search'/expert/reaction
        pipeline=json.loads((source/'pipeline.json').read_text())
        if not (pipeline['strict_mlip_success'] and pipeline['saddle_irc_and_stable_minima']):
            raise ValueError(f'No eligible completed MLIP candidate: {reaction}')
        record=next(x for x in search['endpoint_records'] if x['reaction']==reaction)
        atoms=read(source/'ts_candidate.extxyz')
        if atoms.numbers.tolist()!=record['atomic_numbers'] or not np.isfinite(atoms.positions).all():
            raise ValueError('Candidate mapping/coordinates differ')
        job={'job_id':f'group-{reaction}-{expert}-r1','reaction':reaction,'expert':expert,
             'purpose':'post_hoc_expert_full_DFT_development','engine':'Gaussian16W-A.03',
             'symbols':atoms.get_chemical_symbols(),'numbers':atoms.numbers.tolist(),
             'positions_A':atoms.positions.tolist(),'expected_vibrational_modes':3*len(atoms)-6,
             'mlip_barrier_eV':pipeline['result']['saddle']['barrier_eV'],
             'source_candidate':str((source/'ts_candidate.extxyz').relative_to(ROOT)),
             'endpoints':{label:{'symbols':atoms.get_chemical_symbols(),'positions_A':record[key]}
                for label,key in [('reactant','reactant_positions'),('product','product_positions')]}}
        protocol={**base,'version':'group-dft-development-r1','frozen_at_utc':timestamp,
                  'study_type':'post_hoc_development_not_independent_holdout',
                  'changes_v05_results':False,'jobs':[job]}
        directory=output/f'member-{number:02d}'
        prepare(protocol,reaction,directory/'job')
        (directory/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        (directory/'README.md').write_text(f'''# Member {number}: {reaction} DFT full chain

Purpose: verify a completed AIMNet2-rxn candidate with TS/Freq, both IRC branches,
optimized stable endpoints and independent mapped-geometry assessment.
This is development evidence. A negative result is a valid deliverable.

Budget: four threads/4 GB, wB97X/6-31G(d), TS80, LQA320 points/direction,
step5, minimum100. Keep input and atom mapping unchanged.
Windows runner: scripts/run_gaussian_chain.ps1 -JobDirectory job -GaussianDirectory <existing-g16>
Linux runner: python scripts/run_gaussian_portable.py --job job --g16-dir <existing-g16-directory> --detach
Return the entire job folder (inputs, logs, checkpoint and state), no scratch.
Independent assessor: scripts/assess_dft_chain.py --protocol protocol.json
--reaction {reaction} --folder job --engine gaussian --output assessment.json

Deployment state: prepared only. Do not dispatch before the user-provided cloud
server passes real computation. Linux Gaussian full-chain execution requires its
own acceptance; do not treat WSL-to-Windows bridging as Linux binary verification.
''')
        jobs.append({'member':number,'kind':'full_DFT_chain','reaction':reaction,
                     'expert':expert,'path':directory.name,'prepared_not_dispatched':True})
    selected=set()
    for number,region in [(4,'0'),(5,'1')]:
        directory=output/f'member-{number:02d}'; directory.mkdir()
        available=[records[sid] for sid in partition['regions'][region] if sid not in selected]
        chosen=[]
        for kind in ['formation','cleavage','rearrangement','rearrangement']:
            pool=[r for r in available if r['kind']==kind and r not in chosen]
            if not pool: raise ValueError(f'No train-side {kind} in region {region}')
            chosen.append(min(pool,key=lambda r:(r['atoms'],r['sid'])))
        items=[]
        rng=np.random.default_rng(2026112300+number)
        for record in chosen:
            selected.add(record['sid'])
            candidates=group[record['reaction']]
            # Training-side labels are permitted for label preparation; no meta/holdout labels.
            atoms=max(candidates,key=lambda a:float(a.info['REF_energy']))
            for mode in ['reference','perturbed']:
                positions=atoms.positions.copy()
                if mode=='perturbed':
                    noise=rng.normal(size=positions.shape); noise-=noise.mean(axis=0)
                    noise*=.03/np.sqrt(np.mean(np.sum(noise**2,axis=1)))
                    positions+=noise
                name=f'{record["reaction"]}-{mode}'
                lines=[f'%chk={name}.chk','%mem=4GB','%nprocshared=4',
                       '#p wb97x/6-31g(d) force scf=(xqc,maxcycle=128) int=ultrafine nosymm','',
                       f'AI4Chem training-pool label diagnostic {name}','','0 1']
                lines += [f'{s} {x:.12f} {y:.12f} {z:.12f}' for s,(x,y,z) in zip(atoms.get_chemical_symbols(),positions,strict=True)]
                (directory/(name+'.gjf')).write_text('\n'.join(lines)+'\n\n',encoding='ascii')
                items.append({'name':name,'reaction':record['reaction'],'kind':record['kind'],
                    'region':int(region),'atoms':len(atoms),'source_frame':int(atoms.info['source_frame']),
                    'mode':mode,'symbols':atoms.get_chemical_symbols(),'positions_A':positions.tolist(),
                    'source_energy_eV':float(atoms.info['REF_energy']) if mode=='reference' else None,
                    'source_forces_eV_A':atoms.arrays['REF_forces'].tolist() if mode=='reference' else None})
        manifest={'member':number,'purpose':'training_pool_reference_alignment_and_new_off_path_gradients',
            'region':int(region),'items':items,'dft_method':base['dft_method'],
            'created_at_utc':timestamp,'not_active_learning_comparison':True,
            'training_use':'Quarantine until basis/grid/method alignment is numerically checked; no automatic training merge.',
            'holdout_labels_evaluated':False,'prepared_not_dispatched':True}
        (directory/'task.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (directory/'README.md').write_text(f'''# Member {number}: train-region {region} energy/force labels

Run eight independent Gaussian force jobs, sequentially or within actual machine
resources: four known training geometries for reference alignment, four paired
off-path geometries with fixed 0.03 A RMS atom displacement for new force labels.
Each job requests four threads/4 GB. Do not optimize geometry or change atom order.
Return every .gjf/.log and a hardware/runtime note, including failed jobs.

This is training-pool development. Recomputed references are not new unique
labels; only four perturbed geometries are new labels per member. All new results
stay quarantined until DFT basis/grid/method alignment is accepted. This is not
yet an uncertainty-based active-learning comparison and is not final evaluation.

Linux batch: python scripts/run_gaussian_portable.py --job <this-folder> --g16-dir <existing-g16-directory> --detach
Raw output forces require independent coordinate alignment before training.

Prepared only: no dispatch or cloud execution before the user-provided server
passes real acceptance. Gaussian executable is supplied by the group's existing
installation and is never included here.
''')
        jobs.append({'member':number,'kind':'DFT_energy_and_force','region':int(region),
            'reactions':[r['reaction'] for r in chosen],'reference_jobs':4,'new_geometry_jobs':4,
            'path':directory.name,'prepared_not_dispatched':True})
    protocol={'version':'group-dft-development-r1','frozen_at_utc':timestamp,
        'selection':'Post-hoc development: all three currently strict-passing AIMNet candidates not assigned to roommate; two training-region batches each smallest formation/cleavage/two rearrangements. No selection on Gaussian outcomes.',
        'user_provided_cloud_acceptance_required':True,'server_created':False,'dispatched':False,
        'roommate_rxn8694_excluded':True,'jobs':jobs,'holdout_labels_evaluated':False}
    if PROTOCOL.exists(): raise FileExistsError('Group protocol already frozen; inspect before a revision')
    PROTOCOL.write_text(json.dumps(protocol,indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for item in jobs:
        with zipfile.ZipFile(output/(item['path']+'.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
            for path in sorted((output/item['path']).rglob('*')):
                if path.is_file(): archive.write(path,path.relative_to(output))
    return protocol


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); print(json.dumps(build(args.output),indent=2))
