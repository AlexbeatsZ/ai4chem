"""Compare completed5D/grid calibration with original6D and public train labels."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaussian_label_alignment import parse_label


def assess(archive, baseline, output):
    rows = []
    with zipfile.ZipFile(archive) as new, zipfile.ZipFile(baseline) as old:
        if new.testzip() or old.testzip():
            raise ValueError('Returned label archive CRC failure')
        task = json.loads(new.read('queue.json'))
        state = json.loads(new.read('queue-state.json'))
        if state['status'] == 'running' or len(state['outcomes']) != len(task['force_items']):
            raise ValueError('Calibration is incomplete; retain evidence without accepting it')
        original = {item['name']: item for item in json.loads(old.read('queue.json'))['force_items']}
        for item in task['force_items']:
            record = json.loads(new.read('forces/'+item['name']+'/receipt.json'))
            if record['status'] != 'raw_label_completed_pending_alignment':
                raise ValueError('Calibration has a failed label')
            label = parse_label(new.read(item['input']).decode(), new.read('forces/'+item['name']+'/force.log').decode(), item['symbols'])
            source = original[item['reference_name']]
            prior = parse_label(old.read(source['input']).decode(), old.read('forces/'+source['name']+'/force.log').decode(), source['symbols'], basis_d=6)
            if not np.allclose(label['positions_A'], prior['positions_A'], atol=1e-10, rtol=0):
                raise ValueError('Calibration changed the original fixed geometry')
            reference_force = np.asarray(source['source_forces_eV_A'])
            rows.append(dict(name=item['name'], reaction=item['reaction'], mode=item['mode'],
                energy_delta_public_eV=label['energy_eV']-source['source_energy_eV'],
                force_rmse_public_eV_A=float(np.sqrt(np.mean((label['forces_eV_A']-reference_force)**2))),
                energy_delta_6d_eV=label['energy_eV']-prior['energy_eV'],
                force_rmse_6d_eV_A=float(np.sqrt(np.mean((label['forces_eV_A']-prior['forces_eV_A'])**2))),
                geometry_alignment_max_A=label['alignment_max_A']))
    summary = dict(status='calibration_complete_pending_scientific_decision', results=rows,
                   training_merge_authorized=False, prospective_holdout_labels_evaluated=False,
                   interpretation='Basis/grid controls are diagnostic. Preserve residual public-label differences; no automatic relabel or training merge.')
    output.mkdir(parents=True, exist_ok=True)
    path = output/'summary.json'
    if path.exists() and json.loads(path.read_text()) != summary:
        raise ValueError('Existing calibration assessment differs')
    path.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assess(args.archive, args.baseline, args.output)
