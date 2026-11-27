"""Finite read-only model return and matched-control assessment; no training retry."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model_study_data import save_json
from run_ts_weight_study import promotes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--relay-ssh',required=True)
    parser.add_argument('--relay-root',required=True)
    args=parser.parse_args()
    output=ROOT/'results/four-hours-20261127/model-return';output.mkdir(exist_ok=False)
    lock=(output/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(owner_pid=os.getpid(),status='observing',started_at_utc=datetime.now(timezone.utc).isoformat(),
        holdout_labels_evaluated=False,science_retries=0)
    remote=args.relay_root
    deadline=time.monotonic()+8*3600
    def update(status,**extra):
        state.update(status=status,updated_at_utc=datetime.now(timezone.utc).isoformat(),**extra)
        save_json(output/'state.json',state)
    update('observing')
    while time.monotonic()<deadline:
        try:
            result=subprocess.run(['ssh','-n','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',
                args.relay_ssh,'powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',
                remote+'/scripts/status_four_hour_model.ps1'],capture_output=True,text=True,timeout=45,check=True)
            observed=json.loads(result.stdout.lstrip('\ufeff'));update('observing',observed=observed)
            if observed['status']=='failed':
                update('model_failed',error=observed['error']);return
            if observed['status']=='completed':
                model=ROOT/'models/four-hours-20261127/ema_replay1024_e20.model'
                model.parent.mkdir(exist_ok=True)
                if not model.exists():
                    partial=model.with_suffix('.partial')
                    subprocess.run(['scp','-q',args.relay_ssh+':'+remote+'/'+observed['model'],str(partial)],check=True,timeout=240)
                    partial.replace(model)
                for relative in ('results/four-hours-20261127/omen/state.json',
                    'results/four-hours-20261127/omen/training/manifest.json',
                    'results/four-hours-20261127/omen/validation/ema_replay1024_e20/benchmark_summary.json',
                    'results/four-hours-20261127/omen/training/ema_replay1024_e20/config.yaml',
                    'results/four-hours-20261127/omen/interface/acceptance.json'):
                    target=output/('omen-state.json' if Path(relative).name=='state.json' else Path(relative).name)
                    if not target.exists():
                        subprocess.run(['scp','-q',args.relay_ssh+':'+remote+'/'+relative,str(target)],check=True,timeout=90)
                import numpy as np
                import torch
                from ase.io import read
                from mace.calculators import MACECalculator
                torch.set_num_threads(3)
                calculator=MACECalculator(model_paths=str(model),device='cpu',default_dtype='float64')
                atoms=read(ROOT/'data/until-0800-20261126/full-control.extxyz',index=0);atoms.calc=calculator
                energy=atoms.get_potential_energy();forces=atoms.get_forces();delta=1e-4
                x=atoms.positions[0,0];atoms.positions[0,0]=x+delta;plus=atoms.get_potential_energy()
                atoms.positions[0,0]=x-delta;minus=atoms.get_potential_energy()
                error=abs(-(plus-minus)/(2*delta)-forces[0,0])
                if not np.isfinite(energy) or not np.isfinite(forces).all() or error>2e-5:
                    raise ValueError('Returned standard EMA model failed energy/force/FD acceptance')
                if not (output/'reload-acceptance.json').exists():
                    save_json(output/'reload-acceptance.json',dict(status='passed',force_fd_error_eV_A=float(error)),immutable=True)
                del calculator
                break
        except json.JSONDecodeError as error:
            update('observing',connection_error=f'{type(error).__name__}: {error}')
        except (ValueError, RuntimeError) as error:
            update('model_acceptance_failed',error=f'{type(error).__name__}: {error}');return
        except Exception as error:
            update('observing',connection_error=f'{type(error).__name__}: {error}')
        time.sleep(300)
    else:
        update('observation_deadline');return
    while time.monotonic()<deadline:
        control=json.loads((ROOT/'results/until-0800-20261126/full-control-state.json').read_text())
        if control['status']=='failed':
            update('matched_control_failed',control=control);return
        if control['status']=='completed':
            matched=control['attempts']['public_replay1024_e20']['validation']
            baseline=json.loads((ROOT/'results/coordination/compute-20261126/growth-final-state.json').read_text())['attempts']['mh1_coverage2048_s0']['validation']
            summary=dict(ema_validation=observed['validation'],matched_control_validation=matched,baseline_validation=baseline,
                promoted=promotes(observed['validation'],matched,baseline),model=str(model.relative_to(ROOT)),
                device_difference='EMA trained CUDA; matched public control CPU. Same sourceweight/seed/orderedslots/settings/20epochs. No bitwise equivalence claim.',holdout_labels_evaluated=False)
            save_json(output/'assessment.json',summary,immutable=True)
            update('candidate_assessed',assessment=summary,finished_at_utc=datetime.now(timezone.utc).isoformat());return
        update('waiting_for_matched_full_control');time.sleep(300)
    update('observation_deadline')


if __name__=='__main__':main()
