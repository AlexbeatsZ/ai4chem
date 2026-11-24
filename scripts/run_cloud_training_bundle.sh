#!/usr/bin/env bash
# One submission, bounded computation, automatic success/failure evidence return.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
worker=false
if [[ ${1:-} == --worker ]]; then worker=true; shift; fi
bundle=${1:?Usage: bash scripts/run_cloud_training_bundle.sh /absolute/bundle /fresh/absolute/output}
output=${2:?A fresh absolute output directory is required}
prefix=${3:-"$repo/environments/cloud-worker/.runtime/env"}
if [[ "$prefix" != /* ]]; then echo "Dedicated prefix must be absolute." >&2; exit 2; fi
if [[ "$bundle" != /* || "$output" != /* || ! -f "$bundle/task.json" ]]; then
    echo 'Use absolute paths and an existing task.json bundle.' >&2; exit 2
fi
if [[ "$worker" == false && "$(uname -r)" == *microsoft*WSL* ]]; then
    echo "WSL requires scripts/launch_training_bundle.ps1 with a native Windows owner." >&2
    exit 2
fi
if [[ "$worker" == false ]]; then
    mkdir -p -- "$(dirname -- "$output")"
    mkdir -- "$output"
    if [[ -e "$output.tar.gz" ]]; then
        echo 'Return archive already exists; choose a fresh output path.' >&2; exit 2
    fi
    nohup bash "$repo/scripts/run_cloud_training_bundle.sh" --worker "$bundle" "$output" "$prefix" \
        > "$output/launcher.log" 2>&1 < /dev/null &
    owner=$!
    printf '%s\n' "$owner" > "$output/owner.pid"
    printf 'Owner PID: %s\nLog: %s/launcher.log\nReturn: %s.tar.gz\n' "$owner" "$output" "$output"
    exit 0
fi
finish() {
    code=$?
    trap - EXIT
    printf '%s\n' "$code" > "$output/exit-code.txt"
    if [[ "$code" == 0 ]]; then
        printf 'COMPLETED: return this archive to the coordinator.\n' > "$output/RETURN.txt"
    else
        printf 'FAILED (exit %s): return this archive; do not restart training or change parameters.\n' "$code" > "$output/RETURN.txt"
    fi
    # Preserve partial scientific artifacts and diagnostics on either outcome.
    tar --exclude='*/checkpoints' -czf "$output.tar.gz.partial" -C "$(dirname -- "$output")" "$(basename -- "$output")"
    mv -- "$output.tar.gz.partial" "$output.tar.gz"
    exit "$code"
}
trap finish EXIT
printf 'Installing/checking dedicated project environment\n'
bash "$repo/scripts/cloud_worker.sh" install --prefix "$prefix"
bash "$repo/scripts/cloud_worker.sh" doctor --prefix "$prefix" -- --device cuda --output "$output/runtime.json"
foundation=$("$prefix/bin/python" - "$bundle/task.json" <<'PY'
import json,sys
from pathlib import Path
source=Path(sys.argv[1]); task=json.loads(source.read_text())
if 'tasks' in task:
    source=source.parent/task['tasks'][0]; task=json.loads(source.read_text())
path=Path(task['foundation_model'])
print(path if path.is_absolute() else source.parent/path)
PY
)
bash "$repo/scripts/cloud_worker.sh" smoke --prefix "$prefix" -- --foundation "$foundation" --output "$output/smoke"
bash "$repo/scripts/cloud_worker.sh" batch --prefix "$prefix" -- --task "$bundle/task.json" --output "$output/training"
printf 'Training completed; preparing return archive\n'
