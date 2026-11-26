"""Use the reserved OMEN acceptance budget after frozen training owners drain."""
from datetime import datetime, timezone
import fcntl
import json
import os
import subprocess
import sys
import time

from model_study_data import ROOT, save_json
from run_model_growth import wait_for_cuda


def main():
    output=ROOT/'results/until-0800-20261126-omen'
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'state.json').exists():
        raise FileExistsError('Acceptance already has an owner/evidence')
    deadline=datetime.fromisoformat('2026-11-27T00:00:00+00:00')
    state=dict(owner_pid=os.getpid(),status='waiting_for_training',started_at_utc=datetime.now(timezone.utc).isoformat(),outcomes=[],holdout_labels_evaluated=False)
    def update(stage,**extra):
        state.update(stage=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),**extra);save_json(output/'state.json',state)
    locks=[]
    update('waiting_for_training')
    try:
        for relative in ('results/ts-weight-20261126/runner.lock','results/v0.10.0/runner.lock'):
            handle=(ROOT/relative).open('a');locks.append(handle)
            while True:
                try:
                    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if datetime.now(timezone.utc)>=deadline:
                        update('deadline_before_admission',status='deadline_before_admission');return
                    time.sleep(30)
        previous=json.loads((ROOT/'results/ts-weight-20261126/state.json').read_text())
        if previous['status']!='completed':
            raise RuntimeError('TS predecessor did not finish normally; no implicit recovery')
        cuda_locks=wait_for_cuda({'predecessor_wait_hours':1},update)
        started=time.monotonic()
        for expert in ('global512','coverage2048s0'):
            remaining=3-(time.monotonic()-started)/3600
            if datetime.now(timezone.utc)>=deadline or remaining<=0:
                break
            update('acceptance',status='running',expert=expert)
            command=[sys.executable,'-u','scripts/run_diversity_search.py','--expert',expert,
                '--protocol','configs/omen-growth-search-until-0800-20261126.json','--output',str(output/'search'),
                '--device','cuda','--deadline-utc',deadline.isoformat(),'--max-hours',str(min(1.5,remaining))]
            with (output/(expert+'.log')).open('x') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state['outcomes'].append(dict(expert=expert,exit_code=result.returncode))
            if result.returncode:
                raise RuntimeError('Acceptance worker failed; evidence retained')
        update('completed',status='completed',finished_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        update('failed',status='failed',error=f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':
    main()
