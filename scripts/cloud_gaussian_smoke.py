"""One real fixed-geometry force job; distinguish native Linux from WSL bridge."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--g16-root',type=Path)
    parser.add_argument('--windows-bridge',action='store_true')
    args=parser.parse_args()
    if bool(args.g16_root)==args.windows_bridge:
        parser.error('Choose native --g16-root or the explicit --windows-bridge')
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    input_path=output/'water_force.gjf'; log=output/'water_force.log'
    input_path.write_text('%chk=water_force.chk\n%mem=1GB\n%nprocshared=2\n'
        '#p wb97x/6-31g(d) force scf=(xqc,maxcycle=128) int=ultrafine nosymm\n\n'
        'AI4Chem infrastructure force probe, not a new scientific label\n\n0 1\n'
        'O 0.000000 0.000000 0.000000\nH 0.758000 0.000000 0.504000\n'
        'H -0.758000 0.000000 0.504000\n\n',encoding='ascii')
    if args.windows_bridge:
        def winpath(path):
            return subprocess.check_output(['wslpath','-w',str(path)],text=True).strip()
        command=['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
                 winpath(ROOT/'run_gaussian_job.ps1'),'-InputFile',winpath(input_path)]
        result=subprocess.run(command,capture_output=True,text=True,timeout=300)
        native=False
    else:
        binary=args.g16_root.resolve()/'g16'
        if not binary.is_file() or not os.access(binary,os.X_OK):
            raise FileNotFoundError(f'Existing executable Gaussian installation required: {binary}')
        scratch=output/'scratch'; scratch.mkdir()
        # A constant shell script with positional path arguments, never string-interpolated commands.
        script='set -e; export g16root="$1"; . "$1/g16/bsd/g16.profile"; export GAUSS_SCRDIR="$2"; exec "$1/g16/g16"'
        with input_path.open('rb') as source,log.open('xb') as target:
            result=subprocess.run(['bash','-c',script,'ai4chem-gaussian',str(binary.parent.parent),
                str(scratch)],cwd=output,stdin=source,stdout=target,stderr=subprocess.STDOUT,timeout=300)
        native=True
    text=log.read_text(errors='replace') if log.exists() else ''
    energies=re.findall(r'SCF Done:.*?=\s*([-\d.]+)',text)
    parts=text.split('Forces (Hartrees/Bohr)')
    forces=[]
    if len(parts)>1:
        for line in parts[-1].splitlines():
            row=re.match(r'^\s*(\d+)\s+(\d+)\s+([-\d.DE+]+)\s+([-\d.DE+]+)\s+([-\d.DE+]+)\s*$',line)
            if row: forces.append([int(row[2]),*[float(row[i].replace('D','E')) for i in (3,4,5)]])
            elif forces: break
    passed=(result.returncode==0 and bool(re.search(r'Normal termination of Gaussian[^\r\n]*\s*\Z',text))
            and len(forces)==3 and [r[0] for r in forces]==[8,1,1] and bool(energies))
    record={'status':'PASS_GAUSSIAN_FORCE' if passed else 'failed','host':socket.gethostname(),
        'observed_at_utc':datetime.now(timezone.utc).isoformat(),'returncode':result.returncode,
        'mode':'native_linux' if native else 'explicit_WSL_to_existing_Windows_Gaussian',
        'native_linux_gaussian_verified':native and passed,'real_cloud_compute_verified':False,
        'energy_hartree':float(energies[-1]) if energies else None,'forces_hartree_bohr':forces,
        'basis':re.findall(r'Standard basis:([^\r\n]+)',text)[-1:] or None,
        'scientific_label_target_alignment_verified':False,'log':str(log)}
    (output/'acceptance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)
    if not passed: raise SystemExit('Gaussian force acceptance failed; retain all raw evidence')


if __name__=='__main__':
    main()
