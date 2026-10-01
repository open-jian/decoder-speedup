#!/usr/bin/env bash
set -euo pipefail
source /data2/jian/vae-speedup/activate.sh
source /data2/jian/vae-speedup/decoder-speedup/.venv/bin/activate
export TORCH_HOME=/home/jian/.cache/torch
export CUDA_VISIBLE_DEVICES=GPU-d45b0255-e63a-3305-3630-2fb7d3dbb771,GPU-3f2c5b53-3c37-b475-4fd7-de545fb8d0b4,GPU-50a54f51-9dbc-cf89-2d22-c9a38be68198,GPU-0d451142-3bd9-461c-686e-89cfbadc4268
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
cd /data2/jian/vae-speedup/decoder-speedup
VAE_RUN=/data2/jian/vae-speedup/decoder-speedup/runs/wan22-amd-width-turbo-10k-20261001
VAE_EXPERIMENT=/data2/jian/vae-speedup/experiments/20261001_wan22_width_turbo
exec 9>"$VAE_EXPERIMENT/train-4gpu.lock"
flock -n 9 || exit 1
printf '%s\n' "$$" > "$VAE_EXPERIMENT/train-4gpu.pid"
trap 'code=$?; printf "%s\n" "$code" > "$VAE_EXPERIMENT/train-4gpu.exitcode"' EXIT
python - <<'PY'
import os, subprocess
selected = set(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
rows = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
if any(row.split(',')[0].strip() in selected for row in rows.splitlines()):
    raise SystemExit('A selected GPU is occupied; training was not started')
PY
# Verify migration and save a four-rank checkpoint before continuing unchanged.
python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 --master_port=29671 --nproc_per_node=4 -m decoder_speedup train "$VAE_RUN/config.resolved.yaml" \
  --resume "$VAE_RUN/checkpoint-00000001.pt" --allow-framework-change --stop-after-updates 40
cp "$VAE_RUN/last.pt" "$VAE_RUN/checkpoint-00000040.pt"
python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 --master_port=29671 --nproc_per_node=4 -m decoder_speedup train "$VAE_RUN/config.resolved.yaml" \
  --resume "$VAE_RUN/checkpoint-00000040.pt"
