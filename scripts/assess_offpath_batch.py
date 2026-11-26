"""Build a quarantined32-reaction5D dataset after full fixed-geometry validation."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from ase import Atoms
from ase.io import read, write

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaussian_label_alignment import parse_label


def assess(archive, reference_archive, output):
    records, structures = [], []
    with zipfile.ZipFile(archive) as bundle, zipfile.ZipFile(reference_archive) as references:
        if bundle.testzip() or references.testzip():
            raise ValueError('Off-path evidence CRC failure')
        task = json.loads(bundle.read('queue.json'))
        state = json.loads(bundle.read('queue-state.json'))
        if state['status'] != 'prepared_pool_exhausted' or len(state['outcomes']) != len(task['force_items']):
            raise ValueError('Finite off-path batch incomplete; no implicit retry or partial training')
        reference_task = {x['name']: x for x in json.loads(references.read('queue.json'))['force_items']}

        def append(item, source, mode):
            name = item['name']
            receipt = json.loads(source.read('forces/'+name+'/receipt.json'))
            if receipt['status'] != 'raw_label_completed_pending_alignment':
                raise ValueError('Off-path label failed; retain without training')
            label = parse_label(source.read(item['input']).decode(), source.read('forces/'+name+'/force.log').decode(), item['symbols'])
            atoms = Atoms(symbols=item['symbols'], positions=label['positions_A'])
            atoms.info.update(REF_energy=label['energy_eV'], reaction=item['reaction'],
                              source_frame=item['source_frame'], label_method='gaussian5d_ultrafine', mode=mode)
            atoms.arrays['REF_forces'] = label['forces_eV_A']
            structures.append(atoms)
            records.append(dict(name=name, reaction=item['reaction'], mode=mode,
                                alignment_max_A=label['alignment_max_A']))

        for item in task['reused_reference_labels']:
            append(reference_task[item['name']], references, 'reference')
        for item in task['force_items']:
            append(item, bundle, 'reference' if item['mode'] == 'reference-5d' else item['mode'])
        if len(structures) != 288 or {x.info['reaction'] for x in structures} != set(task['training_reactions']):
            raise ValueError('Off-path dataset has the wrong frozen reaction/frame identities')
        for reaction in task['training_reactions']:
            group = [x for x in structures if x.info['reaction'] == reaction]
            if len(group) != 9 or sum(x.info['mode'] == 'reference' for x in group) != 1:
                raise ValueError('Off-path reaction lacks its single reference/eight perturbations')
    output.mkdir(parents=True, exist_ok=True)
    path = output/'quarantined.extxyz'
    if path.exists():
        raise FileExistsError('Validated off-path dataset already exists; inspect instead of overwriting')
    write(path, structures)
    restored = read(path, index=':')
    for original, saved in zip(structures, restored, strict=True):
        np.testing.assert_allclose(original.positions, saved.positions, atol=1e-8, rtol=0)
        np.testing.assert_allclose(original.arrays['REF_forces'], saved.arrays['REF_forces'], atol=1e-8, rtol=0)
        np.testing.assert_allclose(original.info['REF_energy'], saved.info['REF_energy'], atol=1e-8, rtol=0)
    summary = dict(status='offpath_labels_aligned_quarantined', reactions=32, frames=288, records=records,
        label_method='wB97X/6-31G(d),5D/7F,UltraFine,XQC128,NoSymm',
        training_merge_authorized=False, holdout_labels_evaluated=False,
        next='Bounded adaptation pilot against unchanged public-label replay control; no automatic expansion or merge.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assess(args.archive, args.references, args.output)
