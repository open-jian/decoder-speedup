#!/usr/bin/env bash
set -euo pipefail
source /data2/jian/vae-speedup/activate.sh
source /data2/jian/vae-speedup/decoder-speedup/.venv/bin/activate
export TORCH_HOME=/home/jian/.cache/torch
export CUDA_VISIBLE_DEVICES=GPU-d45b0255-e63a-3305-3630-2fb7d3dbb771
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
cd /data2/jian/vae-speedup/decoder-speedup
VAE_RUN=/data2/jian/vae-speedup/decoder-speedup/runs/wan22-amd-width-turbo-10k-20261001
VAE_EXPERIMENT=/data2/jian/vae-speedup/experiments/20261001_wan22_width_turbo
exec 9>"$VAE_EXPERIMENT/train.lock"
flock -n 9 || exit 1
printf '%s\n' "$$" > "$VAE_EXPERIMENT/train.pid"
trap 'code=$?; printf "%s\n" "$code" > "$VAE_EXPERIMENT/train.exitcode"' EXIT
python - <<'PY'
import os, subprocess
uuid = os.environ['CUDA_VISIBLE_DEVICES']
rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
if any(row.split(',')[0].strip() == uuid for row in rows.splitlines()):
    raise SystemExit('Selected GPU is occupied; training was not started')
PY
# Write a recoverable first-update checkpoint, then continue the unchanged budget.
python -m decoder_speedup train "$VAE_EXPERIMENT/train.yaml" --stop-after-updates 1
cp "$VAE_RUN/last.pt" "$VAE_RUN/checkpoint-00000001.pt"
python -m decoder_speedup train "$VAE_RUN/config.resolved.yaml" --resume "$VAE_RUN/checkpoint-00000001.pt"
