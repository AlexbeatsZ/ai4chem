#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Meta/Project/Workspaces/ai4chem
export OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu TMPDIR=/tmp/.agents
mkdir -p /tmp/.agents
/home/meta/.local/opt/miniforge3/bin/mamba run -n ai4chem python -m unittest discover -s tests -v
