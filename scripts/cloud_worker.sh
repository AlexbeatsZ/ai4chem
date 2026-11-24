#!/usr/bin/env bash
# Project-scoped Linux bootstrap; never changes PATH, base, drivers or proxies.
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
runtime="$repo/environments/cloud-worker/.runtime"
prefix=""
manager=""
action=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) prefix=$2; shift 2 ;;
        --manager) manager=$2; shift 2 ;;
        --) shift; break ;;
        *) break ;;
    esac
done
prefix=${prefix:-"$runtime/env"}

if [[ "$action" == help ]]; then
    cat <<'HELP'
Linux x86_64 commands (all writes stay in the project/dedicated environment):
  bash scripts/cloud_worker.sh install [--prefix /absolute/env] [--manager /path/mamba]
  bash scripts/cloud_worker.sh doctor [--prefix /absolute/env] -- --device cuda
  bash scripts/cloud_worker.sh smoke [--prefix /absolute/env] -- --foundation /absolute/model --output /fresh/output
  bash scripts/cloud_worker.sh train [--prefix /absolute/env] -- --task /bundle/task.json --output /fresh/output
Do not install or run tasks in base. Gaussian binaries are not installed here.
HELP
    exit 0
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'This deployment is for Linux x86_64; use its WSL Linux entry on OMEN.' >&2
    exit 2
fi
if [[ "$prefix" != /* ]]; then
    echo '--prefix must be an absolute dedicated environment path.' >&2
    exit 2
fi

if [[ "$action" == install ]]; then
    command -v python3 >/dev/null
    command -v flock >/dev/null
    mkdir -p -- "$runtime"
    exec 9> "$runtime/installation.lock"
    if ! flock -n 9; then
        echo 'An installer already owns this project environment; inspect its logs.' >&2
        exit 2
    fi
    python3 - "$prefix" "$repo" <<'PY'
import json, shutil, sys
from pathlib import Path
repo=Path(sys.argv[2]); cfg=json.loads((repo/'configs/cloud-worker.json').read_text())
path=Path(sys.argv[1])
while not path.exists(): path=path.parent
free=shutil.disk_usage(path).free/1024**3
if free<cfg['minimum_install_free_disk_GiB']:
    raise SystemExit(f'Require {cfg["minimum_install_free_disk_GiB"]} GiB free; found {free:.1f}')
print(f'PASS disk preflight: {free:.1f} GiB free',flush=True)
PY
    mkdir -p -- "$runtime" "$repo/results/node-readiness/cloud-worker"
    if [[ -z "$manager" ]]; then
        manager="$runtime/bin/micromamba"
        if [[ ! -x "$manager" ]]; then
            command -v curl >/dev/null
            command -v tar >/dev/null
            mkdir -p -- "$runtime/bin"
            archive="$runtime/micromamba-2.6.0.tar.bz2.partial"
            curl --fail --location --retry 2 --connect-timeout 15 --max-time 240 \
                'https://micro.mamba.pm/api/micromamba/linux-64/2.6.0' --output "$archive"
            tar -xjf "$archive" -C "$runtime" bin/micromamba
            rm -- "$archive"
        fi
    fi
    "$manager" --version
    if [[ ! -x "$prefix/bin/python" ]]; then
        "$manager" --root-prefix "$runtime/mamba-root" create --yes \
            --prefix "$prefix" --override-channels --channel conda-forge \
            'python=3.12.16' pip
    fi
    "$prefix/bin/python" - <<'PY'
import sys
if sys.version_info[:3]!=(3,12,16):
    raise SystemExit(f'Dedicated prefix has unexpected Python: {sys.version}')
PY
    "$prefix/bin/python" -m pip install --disable-pip-version-check --timeout 90 --retries 5 --resume-retries 5 \
        --extra-index-url 'https://download.pytorch.org/whl/cu134' \
        --requirement "$repo/requirements.lock.txt"
    "$prefix/bin/python" -m pip check
    "$prefix/bin/python" "$repo/scripts/cloud_worker.py" installation \
        --output "$repo/results/node-readiness/cloud-worker/installation.json"
    exit 0
fi
if [[ ! -x "$prefix/bin/python" ]]; then
    echo 'Dedicated environment missing; run install first.' >&2
    exit 2
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
case "$action" in
    doctor|smoke|train) exec "$prefix/bin/python" "$repo/scripts/cloud_worker.py" "$action" "$@" ;;
    *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
