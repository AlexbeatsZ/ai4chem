"""Portable bounded Gaussian chain, matching existing Windows scientific gates.

Only supplied native Gaussian installations are used. The explicit WSL bridge
tests orchestration on OMEN; it never counts as native Linux Gaussian acceptance.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temporary.replace(path)


def parse_stage(text):
    frequencies=[float(v) for line in re.findall(r'Frequencies\s+--([^\r\n]+)',text.rsplit('Harmonic frequencies',1)[-1]) for v in line.split()]
    if not all(math.isfinite(v) for v in frequencies): raise ValueError('Nonfinite frequencies')
    energies=re.findall(r'SCF Done:.*?=\s*([-\d.]+)',text)
    minimum='PES minimum detected' in text
    complete=bool(re.search(r'Calculation of (FORWARD|REVERSE) path complete',text))
    posterror=(minimum and complete and 'PESDtA: Invalid IPoint.' in text and
        text.rfind('PES minimum detected')<text.rfind('Reaction path calculation complete.')<text.rfind('PESDtA: Invalid IPoint.') and text.count('Error termination')==1)
    return {'normal_termination':bool(re.search(r'Normal termination of Gaussian[^\r\n]*\s*\Z',text)),
        'optimization_completed':'Optimization completed.' in text,'frequencies_cm1':frequencies,
        'imaginary_modes':sum(v<0 for v in frequencies),'significant_imaginary_modes':sum(v<-50 for v in frequencies),
        'minimum_detected':minimum,'direction_complete':complete,'postprocessing_error':bool(posterror),
        'maximum_steps_reached':'Maximum number of steps reached.' in text,
        'energy_hartree':float(energies[-1]) if energies else None,
        'irc_points':text.count('NET REACTION COORDINATE UP TO THIS POINT')}


def execute(directory,name,g16_dir,bridge,timeout):
    input_path=directory/(name+'.gjf'); log_path=directory/(name+'.log')
    if log_path.exists(): raise FileExistsError(f'Existing stage evidence must not be overwritten: {log_path}')
    if not input_path.is_file(): raise FileNotFoundError(input_path)
    if bridge:
        convert=lambda p:subprocess.check_output(['wslpath','-w',str(p)],text=True).strip()
        command=['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
                 convert(ROOT/'run_gaussian_job.ps1'),'-InputFile',convert(input_path)]
        with (directory/(name+'-bridge.log')).open('xb') as bridge_log:
            process=subprocess.run(command,stdout=bridge_log,stderr=subprocess.STDOUT,timeout=timeout)
    else:
        binary=g16_dir.resolve()/'g16'
        if not binary.is_file() or not os.access(binary,os.X_OK): raise FileNotFoundError(binary)
        scratch=directory/('scratch-'+name); scratch.mkdir()
        script='set -e; export g16root="$1"; . "$1/g16/bsd/g16.profile"; export GAUSS_SCRDIR="$2"; exec "$1/g16/g16"'
        with input_path.open('rb') as source,log_path.open('xb') as target:
            process=subprocess.run(['bash','-c',script,'ai4chem-gaussian',str(binary.parent.parent),str(scratch)],
                cwd=directory,stdin=source,stdout=target,stderr=subprocess.STDOUT,timeout=timeout)
    if not log_path.is_file(): raise RuntimeError('Gaussian produced no stage output')
    return process.returncode,parse_stage(log_path.read_text(errors='replace'))


def force_batch(args,directory,task):
    lock=(directory/'owner.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=directory/'worker-state.json'
    if state_path.exists(): raise FileExistsError('Existing force-batch evidence requires diagnosis')
    state={'host':socket.gethostname(),'owner_pid':os.getpid(),'started_at_utc':now(),
           'status':'running','items':[],'mode':'WSL_Windows_bridge' if args.windows_bridge else 'native_Linux',
           'holdout_labels_evaluated':False,'training_merge_authorized':False}
    started=time.perf_counter()
    try:
        for item in task['items']:
            state['stage']=item['name']; save(state_path,state)
            code,record=execute(directory,item['name'],args.g16_dir,args.windows_bridge,args.stage_timeout)
            text=(directory/(item['name']+'.log')).read_text(errors='replace')
            force=[]; section=text.rsplit('Forces (Hartrees/Bohr)',1)
            if len(section)>1:
                for line in section[-1].splitlines():
                    row=re.match(r'^\s*(\d+)\s+(\d+)\s+([-\d.DE+]+)\s+([-\d.DE+]+)\s+([-\d.DE+]+)\s*$',line)
                    if row: force.append([int(row[2]),*[float(row[i].replace('D','E')) for i in (3,4,5)]])
                    elif force: break
            if (code or not record['normal_termination'] or record['energy_hartree'] is None
                    or len(force)!=item['atoms'] or not all(math.isfinite(v) for row in force for v in row)):
                raise RuntimeError(f'Incomplete Gaussian force label: {item["name"]}')
            state['items'].append({'name':item['name'],'energy_hartree':record['energy_hartree'],
                'forces_hartree_bohr_output_orientation':force,
                'basis':re.findall(r'Standard basis:([^\r\n]+)',text)[-1:],
                'not_aligned_for_training':True})
            state['updated_at_utc']=now(); save(state_path,state)
        state['status']='raw_force_batch_completed_pending_alignment'
    except Exception as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}',
            gaussian_process_diagnosis_required=isinstance(error,subprocess.TimeoutExpired))
        raise
    finally:
        state.update(finished_at_utc=now(),elapsed_seconds=time.perf_counter()-started); save(state_path,state)
    return state


def run(args):
    directory=args.job.resolve(); task=json.loads((directory/'task.json').read_text())
    if 'items' in task: return force_batch(args,directory,task)
    lock=(directory/'owner.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=directory/'worker-state.json'
    if state_path.exists(): raise FileExistsError('Existing chain state; diagnose before any explicit revision')
    state={'job_id':task['job_id'],'reaction':task['reaction'],'host':socket.gethostname(),'owner_pid':os.getpid(),
        'started_at_utc':now(),'status':'running','stage':'starting','stages':[],'branches':[],
        'scientific_configuration':task['configuration'],'raw_chain_completed':False,'dft_verified':False,
        'engine_mode':'WSL_Windows_bridge' if args.windows_bridge else 'native_Linux',
        'native_linux_gaussian_verified':False}
    started=time.perf_counter()
    def update():
        state.update(updated_at_utc=now(),elapsed_seconds=time.perf_counter()-started); save(state_path,state)
    def stage(name,allow_posterror=False):
        state['stage']=name; update(); begin=time.perf_counter()
        code,record=execute(directory,name,args.g16_dir,args.windows_bridge,args.stage_timeout)
        record.update(job=name,returncode=code,elapsed_seconds=time.perf_counter()-begin)
        state['stages'].append(record); update()
        if not (code==0 and record['normal_termination']):
            if not (allow_posterror and record['postprocessing_error'] and (directory/(name+'.chk')).is_file()):
                raise RuntimeError(f'Gaussian stage {name} failed; retain all raw evidence')
            record['recovered_postprocessing_error']=True
        return record
    try:
        update(); ts=stage('ts'); expected=task['expected_vibrational_modes']
        if not(ts['optimization_completed'] and len(ts['frequencies_cm1'])==expected and
               ts['imaginary_modes']==1 and ts['significant_imaginary_modes']==1):
            raise RuntimeError('TS failed fixed optimization/frequency acceptance')
        for direction in ['forward','reverse']:
            name='irc_'+direction; irc=stage(name,True)
            if not(irc['minimum_detected'] and irc['direction_complete']):
                policy=task['configuration']['dft_policy']
                if not irc['maximum_steps_reached'] or policy.get('irc_initial_max_points',160)>=policy['irc_restart_max_points']:
                    raise RuntimeError(f'IRC did not reach a minimum within its fixed budget: {name}')
                name+='_extended'; irc=stage(name,True)
            if not(irc['minimum_detected'] and irc['direction_complete']): raise RuntimeError('IRC exhausted its fixed budget')
            minimum_name='minimum_'+direction+('_extended' if name.endswith('_extended') else '')
            minimum=stage(minimum_name)
            if not(minimum['optimization_completed'] and len(minimum['frequencies_cm1'])==expected and minimum['imaginary_modes']==0):
                raise RuntimeError(f'Unstable/incomplete minimum: {minimum_name}')
            state['branches'].append({'direction':direction,'irc_job':name,'minimum_job':minimum_name}); update()
        state.update(status='raw_chain_completed',stage='awaiting_geometry_validation',raw_chain_completed=True,
            native_linux_gaussian_verified=not args.windows_bridge)
    except Exception as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}',
            gaussian_process_diagnosis_required=isinstance(error,subprocess.TimeoutExpired))
        raise
    finally:
        state['finished_at_utc']=now(); update()
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--g16-dir',type=Path)
    parser.add_argument('--windows-bridge',action='store_true')
    parser.add_argument('--stage-timeout',type=int,default=21600)
    parser.add_argument('--detach',action='store_true')
    args=parser.parse_args()
    if bool(args.g16_dir)==args.windows_bridge: parser.error('Choose --g16-dir or --windows-bridge')
    if args.detach:
        directory=args.job.resolve()
        if (directory/'launch.json').exists() or (directory/'worker-state.json').exists():
            raise FileExistsError('Existing chain launch/state must not be overwritten')
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(directory),
                 '--stage-timeout',str(args.stage_timeout)]
        command+=['--windows-bridge'] if args.windows_bridge else ['--g16-dir',str(args.g16_dir.resolve())]
        with (directory/'owner-console.log').open('xb') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
        result={'status':'submitted','owner_pid':process.pid,'command':command,'started_at_utc':now()}
        save(directory/'launch.json',result)
    else: result=run(args)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
