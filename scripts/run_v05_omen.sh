#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Meta/Project/Workspaces/ai4chem
export OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu TMPDIR=/tmp/.agents
export PYTHONPATH=/mnt/c/Users/Meta/Project/Workspaces/ai4chem
mkdir -p /tmp/.agents
exec /home/meta/.local/opt/miniforge3/bin/mamba run -n ai4chem \
  python -u scripts/continue_v05.py --dft-jobs 2
