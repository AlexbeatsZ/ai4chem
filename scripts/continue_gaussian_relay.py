"""Observe one existing Gaussian chain through an authorized SSH relay.

Finite task coordinator: no calculation retries, strategy changes or schedules.
Uses existing SSH keys on the relay; no private key is copied to the Mac.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time
import zipfile

from assess_dft_chain import assess

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('relay-ssh','relay-root','worker-root','worker-job','protocol','output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--worker-ssh', default='ai4chem-worker')
    parser.add_argument('--worker-host-name')
    parser.add_argument('--worker-host-key-alias')
    parser.add_argument('--max-hours', type=float, default=12)
    args = parser.parse_args()
    for value in (args.relay_ssh,args.relay_root,args.worker_root,args.worker_job,args.worker_ssh):
        if not re.fullmatch(r'[A-Za-z0-9_./:\\@-]+',value):
            parser.error('Use simple space-free SSH identities and worker paths')
    if not 0 < args.max_hours <= 72:
        parser.error('Use a positive deadline of at most 72 hours')
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    lock = (output/'owner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads(Path(args.protocol).read_text())
    if len(protocol['jobs']) != 1:
        raise ValueError('Exactly one immutable candidate protocol is required')
    job = protocol['jobs'][0]
    state = {'host':'Mac','owner_pid':os.getpid(),'status':'observing','started_at_utc':now(),
             'job_id':job['job_id'],'protocol_frozen_at_utc':protocol['frozen_at_utc'],
             'holdout_labels_evaluated':False}
    deadline = time.monotonic()+args.max_hours*3600

    def remote(action):
        command = ['ssh','-n','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',args.relay_ssh,
            'powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',
            args.relay_root+'/scripts/relay_gaussian_chain.ps1','-Action',action,
            '-WorkerRoot',args.worker_root,'-JobDirectory',args.worker_job,'-WorkerSsh',args.worker_ssh]
        for flag, value in (('-WorkerHostName', args.worker_host_name), ('-WorkerHostKeyAlias', args.worker_host_key_alias)):
            if value:
                command += [flag, value]
        result = subprocess.run(command,capture_output=True,text=True,timeout=240 if action=='Export' else 45)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        return result.stdout.lstrip('\ufeff')

    save(output/'state.json',state)
    while time.monotonic() < deadline:
        try:
            observed = json.loads(remote('Status'))
            state.update(last_observation=observed,updated_at_utc=now())
            state.pop('connection_error',None)
            source = observed['state']
            if source['job_id'] != job['job_id']:
                raise ValueError('Observed worker job differs from frozen candidate')
            if source['status'] not in ('running', 'queued') and not observed['owner_alive']:
                exported = json.loads(remote('Export'))
                archive = output/'return.zip'
                if not archive.exists():
                    partial = output/'return.partial.zip'
                    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',
                        args.relay_ssh+':'+exported['archive'].replace('\\','/'),str(partial)],
                        check=True,capture_output=True,text=True,timeout=240)
                    if partial.stat().st_size != exported['bytes']:
                        raise ValueError('Received archive size differs from relay receipt')
                    partial.replace(archive)
                received = output/'evidence'
                received.mkdir(exist_ok=True)
                with zipfile.ZipFile(archive) as bundle:
                    if bundle.testzip() is not None:
                        raise ValueError('Received evidence ZIP failed CRC')
                    for entry in bundle.infolist():
                        path = received/entry.filename
                        if path.resolve().parent != received.resolve():
                            raise ValueError('Worker evidence must contain only flat files')
                        if path.exists():
                            if path.read_bytes() != bundle.read(entry):
                                raise ValueError('Existing evidence differs; preserve and inspect')
                        else:
                            bundle.extract(entry,received)
                if source['raw_chain_completed']:
                    try:
                        assessment = assess(protocol,job['reaction'],received,'gaussian')
                        save(received/'assessment.json',assessment)
                        state.update(status='assessed',assessment=assessment)
                    except Exception as error:
                        state.update(status='assessment_failed',error=f'{type(error).__name__}: {error}')
                else:
                    state.update(status='worker_failed',error=source.get('error','Raw chain incomplete'))
                state.update(finished_at_utc=now(),evidence=str(received))
                save(output/'state.json',state)
                return
            if source['status'] == 'running' and not observed['owner_alive']:
                state.update(status='owner_missing',error='Recorded Gaussian owner is absent; inspect without retrying',finished_at_utc=now())
                save(output/'state.json',state)
                return
        except Exception as error:
            state.update(connection_error=f'{type(error).__name__}: {error}',updated_at_utc=now())
        save(output/'state.json',state)
        time.sleep(60)
    state.update(status='stopped_deadline',finished_at_utc=now())
    save(output/'state.json',state)


if __name__ == '__main__':
    main()
