import json
from pathlib import Path
import numpy as np
from ase.io import read
from adaptive_ts_search import digest, implementation_identity, next_action
from prepare_v05_reactions import excluded_ids
from ts_search import projected_frequencies, save_json
from validate_ts_dft import parse_output
root = Path('results/v0.5.0')
manifest = json.loads(Path('configs/v0.5.0_reactions.json').read_text())
frozen = json.loads(Path('configs/v0.5.0_frozen.json').read_text())
assert frozen['implementation'] == implementation_identity()
assert frozen['manifest_sha256'] == digest(manifest)
items = {item['reaction']: item for item in manifest['reactions']}
prior = excluded_ids(json.loads(Path('configs/v0.3.0_partitions.json').read_text()))
assert not {item['sid'] for item in manifest['reactions']} & prior
summ = json.loads((root / 'summary.json').read_text())
assert summ['planned_strategy_units'] == summ['completed_strategy_units'] == 96
traces = json.loads((root / 'policy_trace.json').read_text())
assert len(traces) == 96
unique = {}
for record in traces:
    assert record['status'] == 'completed'
    assert 1 <= len(record['attempts']) <= 3
    assert len({attempt['cache_key'] for attempt in record['attempts']}) == len(record['attempts'])
    prefix = []
    for attempt in record['attempts']:
        assert next_action(record['strategy'], prefix, frozen['policy']) == attempt['plan']['type']
        prefix.append(attempt)
        unique[attempt['cache_key']] = {**attempt, 'model': record['model'], 'reaction': record['reaction']}
    assert next_action(record['strategy'], prefix, frozen['policy']) is None
    if record['strategy'] == 'S0':
        assert len(record['attempts']) == 1
    assert abs(record['theoretical_seconds'] - record['endpoint_seconds'] - sum(attempt['elapsed_seconds'] for attempt in record['attempts'])) < 1e-6
    assert record['image_step_cost'] == sum(attempt['image_step_cost'] for attempt in record['attempts'])
    starts = {attempt['plan']['type']: attempt for attempt in record['attempts']}
    if 'perturbed_idpp' in starts:
        standard = read(Path(starts['idpp']['folder']) / 'initial_band.extxyz', index=':')
        perturbed = read(Path(starts['perturbed_idpp']['folder']) / 'initial_band.extxyz', index=':')
        for left, right in zip((standard[0], standard[-1]), (perturbed[0], perturbed[-1]), strict=True):
            np.testing.assert_array_equal(left.positions, right.positions)
        for left, right in zip(standard[1:-1], perturbed[1:-1], strict=True):
            delta = right.positions - left.positions
            assert abs(np.sqrt(np.mean(np.sum(delta ** 2, axis=1))) - 0.03) < 1e-7
            np.testing.assert_allclose(delta.mean(axis=0), 0, atol=1e-8, rtol=0)
checks = []
for key, attempt in unique.items():
    folder = Path(attempt['folder'])
    identity = json.loads((folder / 'identity.json').read_text())
    assert identity['implementation'] == frozen['implementation']
    assert identity['software'] == frozen['software']
    assert identity['model'] == frozen['models'][attempt['model']]
    assert digest(identity) == key
    result = json.loads((folder / 'result.json').read_text())
    assert result['cache_key'] == key
    for name in ('initial_band.extxyz', 'final_band.extxyz'):
        path = folder / name
        if path.exists():
            images = read(path,index=':')
            assert len(images) == attempt['plan']['images']
            for atoms in images:
                np.testing.assert_array_equal(atoms.numbers, items[attempt['reaction']]['atomic_numbers'])
                assert np.isfinite(atoms.positions).all()
                if name.startswith('final'):
                    assert np.isfinite(atoms.get_potential_energy())
                    assert np.isfinite(atoms.get_forces()).all()
            endpoint_folder = root / 'endpoints' / attempt['model'] / attempt['reaction']
            for side, endpoint_image in zip(('reactant', 'product'), (images[0], images[-1]), strict=True):
                shared_endpoint = read(endpoint_folder / f'{side}_relaxed.extxyz')
                np.testing.assert_allclose(shared_endpoint.get_all_distances(), endpoint_image.get_all_distances(), atol=1e-7, rtol=0)
    if attempt['plan']['type'] == 'continuation':
        assert result['neb']['warmup_steps'] == 0
        assert result['neb']['climb_steps'] <= frozen['policy']['continuation_steps']
        initial = read(folder / 'initial_band.extxyz', index=':')
        parent = read(attempt['plan']['band'], index=':')
        for left, right in zip(initial, parent, strict=True):
            np.testing.assert_allclose(left.positions, right.positions, atol=1e-8, rtol=0)
    hessian_path = folder / 'ts_hessian.npz'
    if hessian_path.exists():
        ts = read(folder / 'ts_candidate.extxyz')
        data = np.load(hessian_path)
        frequencies, modes = projected_frequencies(ts, data['hessian_eV_A2'])
        np.testing.assert_allclose(frequencies, data['frequencies_cm1'], atol=1e-3, rtol=0)
        assert np.isfinite(data['hessian_eV_A2']).all()
        assert int(np.count_nonzero(frequencies < -50)) == result['vibrations']['imaginary_modes_above_threshold']
    checks.append({'cache_key':key,'reaction':result['reaction'],'model':attempt['model'],'verified':result.get('saddle_irc_verified',False),'status':result['status']})
dft_selection = json.loads((root / 'dft/selection.json').read_text())
assert len(dft_selection['strategy_units']) == 9
assert {entry['reaction'] for entry in dft_selection['strategy_units']} == set(manifest['dft_reactions'])
dft_checks = []
for key,item in dft_selection['unique_jobs'].items():
    folder = Path(item['output_folder'])
    result = json.loads((folder / 'result.json').read_text())
    assert result['status'] != 'running'
    if result['dft_verified']:
        ts = parse_output(folder / 'ts.log')
        assert ts['normal_termination'] and ts['imaginary_modes'] == ts['imaginary_modes_above_50_cm1'] == 1
        for branch in result['branches']:
            minimum = parse_output(folder / (branch['minimum_job'] + '.log'))
            assert minimum['normal_termination'] and minimum['imaginary_modes'] == 0
            irc = parse_output(folder / (branch['irc']['job'] + '.log'))
            assert irc['irc_path_complete']
            assert irc['normal_termination'] or branch['irc'].get('recovered_postprocessing_error',False)
        assert result['stable_minima'] and result['expected_connectivity'] and result['irc_minima_detected']
    dft_checks.append({'job':key,'reaction':item['reaction'],'status':result['status'],'verified':result['dft_verified']})
assert json.loads((root / 'baseline_regression/comparison.json').read_text())['passed']
assert json.loads((root / 'baseline_regression/shared_endpoints/comparison.json').read_text())['passed']
save_json(root / 'verification.json',{'strategy_units':96,'unique_attempts':len(unique),'search_checks':checks,'dft_strategy_units':9,'dft_jobs':dft_checks,'baseline_regressions_passed':True,'frozen_identity_verified':True,'physical_perturbations_and_continuations_checked':True,'passed':True})
print('PASS v0.5.0 verification',len(unique),'searches',len(dft_checks),'Gaussian candidates')
