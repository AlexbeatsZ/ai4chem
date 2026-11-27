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
  optimizer-study) entry=run_optimizer_study.py ;;
  growth-prepare) exec /home/meta/.local/opt/miniforge3/bin/mamba run -n ai4chem python -u run_model_growth.py --prepare-only ;;
  growth) entry=run_model_growth.py ;;
  ts-weight-study) entry=run_ts_weight_study.py ;;
  overnight-acceptance) entry=run_overnight_acceptance.py ;;
  four-hour-compute) exec /home/meta/.local/opt/miniforge3/bin/mamba run -n ai4chem python -u run_four_hour_compute.py --node omen ;;
  ts-weight-prepare) exec /home/meta/.local/opt/miniforge3/bin/mamba run -n ai4chem python -u run_ts_weight_study.py --prepare-only ;;
  *) exit 2 ;;
esac
exec /home/meta/.local/opt/miniforge3/bin/mamba run -n ai4chem python -u "$entry"
