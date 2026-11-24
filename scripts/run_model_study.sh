#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TMPDIR=/tmp/.agents
export PYTHONPATH="$project_root"
mkdir -p /tmp/.agents
case "${1:-study}" in
  download) entry=scripts/download_transition1x_train.py ;;
  prepare) entry=scripts/prepare_model_study.py ;;
  study) entry=run_model_study.py ;;
  specialization-prepare) entry=scripts/prepare_specialization.py ;;
  *) exit 2 ;;
esac
exec /home/meta/.local/opt/miniforge3/bin/mamba run -n ai4chem python -u "$entry"
