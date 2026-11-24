#!/usr/bin/env bash
# Run the deployment through a fresh isolated Linux environment on OMEN.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
run_id=${1:?A fresh validation ID is required}
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then exit 2; fi
mode=${2:-ml}
if [[ "$mode" != ml && "$mode" != install ]]; then
    echo "Supported modes: install or ml. Native Linux Gaussian is separately gated." >&2
    exit 2
fi
output="$repo/results/node-readiness/cloud-worker/$run_id"
prefix='/home/meta/.local/opt/miniforge3/envs/ai4chem-cloud-worker'
mkdir -p -- "$repo/results/node-readiness/cloud-worker"
mkdir -- "$output"
trap 'code=$?; printf "%s\n" "$code" > "$output/exit-code.txt"' EXIT
printf 'Installing isolated cloud worker at %s\n' "$prefix"
bash "$repo/scripts/cloud_worker.sh" install --prefix "$prefix"
if [[ "$mode" == install ]]; then exit 0; fi
bash "$repo/scripts/cloud_worker.sh" doctor --prefix "$prefix" -- --device cuda --output "$output/doctor.json"
bash "$repo/scripts/cloud_worker.sh" smoke --prefix "$prefix" -- \
    --foundation "$repo/models/mh1-omol.model" --output "$output/mace-smoke"
"$prefix/bin/python" - "$output" <<'PY_SUMMARY'
from pathlib import Path
from datetime import datetime,timezone
import json,sys,socket
p=Path(sys.argv[1]); mace=json.loads((p/'mace-smoke/acceptance.json').read_text())
if mace['status']!='PASS_MACE_INFERENCE_AND_TWO_EPOCH_TRAINING':
    raise SystemExit('Required actual MACE acceptance did not pass')
record={'status':'PASS_OMEN_ML_DEPLOYMENT_ACCEPTANCE','host':socket.gethostname(),'finished_at_utc':datetime.now(timezone.utc).isoformat(),'mace':mace,'native_linux_gaussian_verified':False,'real_cloud_compute_verified':False,'group_dispatch_authorized_now':False}
(p/'validation_summary.json').write_text(json.dumps(record,indent=2)+'\n'); print(json.dumps(record,indent=2),flush=True)
PY_SUMMARY
