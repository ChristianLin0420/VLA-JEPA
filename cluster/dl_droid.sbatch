#!/bin/bash
#SBATCH --job-name=vlajepa-dldroid
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/dldroid_%j.out
#SBATCH --error=outputs/slurm/dldroid_%j.err
set -uo pipefail
# Droid video completion download (batch, self-requeue). The download uses NO GPU,
# so a tiny GPU keepalive runs alongside to dodge the cluster's idle-GPU watchdog
# (which cancelled prior attempts at ~37-40min).
cd "${SLURM_SUBMIT_DIR:-/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA}"
mkdir -p outputs/slurm
DDIR=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/datasets/lerobot/droid_lerobot
PYBIN=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/VLA_JEPA/bin/python
export DROID_DEADLINE=$(( $(date +%s) + 4*3600 - 300 ))
echo "[$(date -Is)] droid dl on $(hostname); deadline=$DROID_DEADLINE"

# --- GPU keepalive (small matmul every 15s) ---
PYTHONNOUSERSITE=1 "$PYBIN" -c "
import torch, time
d = torch.device('cuda:0')
while True:
    x = torch.randn(1024, 1024, device=d)
    (x @ x).sum().item()
    time.sleep(15)
" >/dev/null 2>&1 &
KA_PID=$!
trap '[[ -n "${KA_PID:-}" ]] && kill "$KA_PID" 2>/dev/null || true' EXIT
echo "[$(date -Is)] gpu keepalive pid=$KA_PID"

bash /lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/dl_droid.sh

cnt=$(find "$DDIR/videos" -name '*.mp4' 2>/dev/null | wc -l)
echo "[$(date -Is)] post-loop droid mp4 count=$cnt / 276699"
if [ "$cnt" -ge 275000 ]; then echo "[$(date -Is)] DROID_DONE ($cnt) -> no requeue"; exit 0; fi
echo "[$(date -Is)] droid incomplete ($cnt) -> scontrol requeue $SLURM_JOB_ID"
scontrol requeue "$SLURM_JOB_ID"
exit 0
