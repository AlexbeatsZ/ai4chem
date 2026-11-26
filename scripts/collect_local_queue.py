"""Finite queue observation and terminal raw-evidence return; never runs chemistry."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('relay-ssh','relay-root','worker-root','queue','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--worker-host-name')
    parser.add_argument('--worker-host-key-alias')
    parser.add_argument('--max-hours',type=float,default=72)
    parser.add_argument('--calibration-baseline')
    parser.add_argument('--offpath-references')
    args=parser.parse_args()
    if not 0 < args.max_hours <= 72:
        parser.error('Finite observation must be within72h')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    lock=(output/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'owner_pid':os.getpid(),'status':'observing','started_at_utc':datetime.now(timezone.utc).isoformat()}
    def save():
        temporary=output/'state.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(output/'state.json')
    def remote(action):
        command=['ssh','-n','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',args.relay_ssh,
            'powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',args.relay_root+'/scripts/relay_local_queue.ps1',
            '-Action',action,'-WorkerRoot',args.worker_root,'-QueueDirectory',args.queue]
        for flag, value in (('-WorkerHostName', args.worker_host_name), ('-WorkerHostKeyAlias', args.worker_host_key_alias)):
            if value:
                command += [flag, value]
        result=subprocess.run(command,capture_output=True,text=True,timeout=900 if action=='Export' else 60)
        if result.returncode:raise RuntimeError(result.stderr or result.stdout)
        return json.loads(result.stdout.lstrip('\ufeff'))
    deadline=time.monotonic()+args.max_hours*3600
    save()
    while time.monotonic()<deadline:
        try:
            observed=remote('Status');state.update(observed=observed,updated_at_utc=datetime.now(timezone.utc).isoformat())
            state.pop('connection_error',None)
            if observed['status']!='running' and not observed['owner_alive']:
                receipt=remote('Export');archive=output/'return.zip'
                if not archive.exists():
                    partial=output/'return.partial.zip'
                    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',
                        args.relay_ssh+':'+receipt['archive'].replace('\\','/'),str(partial)],check=True,timeout=900)
                    if partial.stat().st_size!=receipt['bytes']:raise ValueError('Returned queue archive size differs')
                    partial.replace(archive)
                with zipfile.ZipFile(archive) as bundle:
                    if bundle.testzip() is not None:raise ValueError('Terminal queue archive CRC failure')
                    if len(bundle.infolist())!=receipt['files']:raise ValueError('Terminal queue inventory differs')
                state.update(status='raw_evidence_returned_pending_alignment',receipt=receipt,
                    finished_at_utc=datetime.now(timezone.utc).isoformat(),training_merge_authorized=False)
                if args.calibration_baseline:
                    subprocess.run([os.sys.executable, str(Path(__file__).with_name('assess_label_calibration.py')),
                        '--archive', str(archive), '--baseline', args.calibration_baseline,
                        '--output', str(output/'alignment')], check=True)
                    state['status']='calibration_returned_and_assessed'
                if args.offpath_references:
                    subprocess.run([os.sys.executable, str(Path(__file__).with_name('assess_offpath_batch.py')),
                        '--archive', str(archive), '--references', args.offpath_references,
                        '--output', str(output/'alignment')], check=True)
                    state['status']='offpath_returned_and_aligned_quarantined'
                save();return
            if not observed['owner_alive']:
                state.update(status='owner_missing',finished_at_utc=datetime.now(timezone.utc).isoformat());save();return
        except Exception as error:
            state.update(connection_error=f'{type(error).__name__}: {error}')
        save();time.sleep(300)
    state.update(status='observation_deadline',finished_at_utc=datetime.now(timezone.utc).isoformat());save()


if __name__=='__main__':main()
