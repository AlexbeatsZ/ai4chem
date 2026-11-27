"""Package a finite, post-hoc development DFT cohort from completed searches."""
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

import numpy as np
from ase.io import read
from prepare_gaussian_chain import prepare

ROOT = Path(__file__).resolve().parents[1]


def main():
    window = json.loads((ROOT/'configs/four-hours-20261127.json').read_text())
    source = ROOT/'results/until-0800-20261126/development-search'
    search = json.loads((ROOT/'configs/growth-search-until-0800-20261126.json').read_text())
    records = sorted(search['endpoint_records'], key=lambda row:(row['atoms'],row['reaction']))[:3]
    base = json.loads((ROOT/'configs/v0.7.0_diversity_dft.json').read_text())
    output = ROOT/'results/four-hours-20261127/roommate'
    output.mkdir(parents=True, exist_ok=False)
    units, chains = [], []
    frozen = datetime.now(timezone.utc).isoformat()
    for record in records:
        for expert in ('coverage2048s0','global512'):
            folder = source/expert/record['reaction']
            pipeline = json.loads((folder/'pipeline.json').read_text())
            eligible = bool(pipeline['strict_mlip_success'] and pipeline['saddle_irc_and_stable_minima'])
            units.append(dict(expert=expert,reaction=record['reaction'],eligible=eligible,
                source_pipeline=str((folder/'pipeline.json').relative_to(ROOT))))
            if not eligible:
                continue
            atoms = read(folder/'ts_candidate.extxyz')
            if atoms.numbers.tolist() != record['atomic_numbers'] or not np.isfinite(atoms.positions).all():
                raise ValueError('Candidate coordinates/mapping differ')
            job_id = 'growth-20261127-'+record['reaction']+'-'+expert
            job = dict(job_id=job_id,reaction=record['reaction'],expert=expert,
                purpose='stronger_model_independent_DFT_development_acceptance',engine='Gaussian16W-A.03',
                formula=record['formula'],source_candidate=str((folder/'ts_candidate.extxyz').relative_to(ROOT)),
                symbols=atoms.get_chemical_symbols(),numbers=atoms.numbers.tolist(),positions_A=atoms.positions.tolist(),
                expected_vibrational_modes=3*len(atoms)-6,
                mlip_barrier_eV=pipeline['result']['saddle']['barrier_eV'],
                endpoints={label:dict(symbols=atoms.get_chemical_symbols(),positions_A=record[key]) for label,key in
                    (('reactant','reactant_positions'),('product','product_positions'))})
            protocol = {key:base[key] for key in ('dft_method','dft_nproc','charge','multiplicity',
                'dft_policy','irc_endpoint_rmsd_A','endpoint_tolerance_A')}
            protocol.update(version='four-hours-20261127-dft',frozen_at_utc=frozen,dft_mem='2GB',
                study_type='post_hoc_meta_development',changes_v05_results=False,jobs=[job],
                selection='Three smallest atom counts from previous frozen8metaendpoints; both models retained in6unit accounting; only strictMLIPeligible TS submitted. Selection after MLIP observations, not prospective.')
            directory = output/'chains'/job_id
            prepare(protocol,record['reaction'],directory)
            (directory/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
            chains.append(dict(job_id=job_id,reaction=record['reaction'],directory='chains/'+job_id))
    cohort = dict(version='four-hours-20261127-dft',frozen_at_utc=frozen,units=units,
        endpoint_records=records,admission_deadline_utc=window['deadline_utc'],
        scope='Posthoc development, no prospective labels, no automatic training merge; all6units retained including rejected candidates.')
    path = ROOT/'configs/four-hours-dft-20261127.json'
    with path.open('x') as handle:
        handle.write(json.dumps(cohort,indent=2)+'\n')
    task = dict(version='four-hours-20261127-dft',frozen_at_utc=frozen,chains=chains,force_items=[],
        target_compute_hours=4,admission_deadline_utc=window['deadline_utc'],
        training_merge_authorized=False,holdout_labels_evaluated=False,automatic_retries=0,
        policy='One chain at a time,4threads/2GB/4GiB free gate. Newchain admission ends at12:20UTC+8; admitted chain drains unchanged bounded TS80/IRC320/minimum100 settings. No retries or extra offpathlabels.')
    (output/'queue.json').write_text(json.dumps(task,indent=2)+'\n')
    with zipfile.ZipFile(output.parent/'roommate-inputs.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in output.rglob('*'):
            if path.is_file():
                archive.write(path,path.relative_to(output.parent))
    print(json.dumps(dict(cohort_units=len(units),eligible_chains=chains),indent=2))


if __name__ == '__main__':
    main()
