#!/bin/bash
#SBATCH --job-name=vlajepa-evalall
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/vlajepa_evalall_%j.out
#SBATCH --error=outputs/slurm/vlajepa_evalall_%j.err
set -euo pipefail
# Auto-eval the all_robot cotrain (vlajepa_cotrain_allv2) once it completes 100k.
# Submit with: sbatch --dependency=afterok:<allv2_jobid> cluster/eval_after_all.sbatch
REPO="${SLURM_SUBMIT_DIR:-/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA}"
cd "$REPO"; mkdir -p outputs/slurm
source /lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
export HF_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_IB_HCA=mlx5 NCCL_SOCKET_IFNAME='^lo,docker0'

RUN=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/vlajepa_cotrain_allv2
VLA_PY=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/VLA_JEPA/bin/python
STEP=$(cat "$RUN/checkpoints/latest.txt" 2>/dev/null)
echo "[$(date -Is)] all_robot complete; latest=$STEP ; exporting"
OUT_PT="VLA-JEPA-allv2-${STEP}.pt"
"$VLA_PY" cluster/export_vlajepa_ckpt.py "$RUN" "$STEP" "$OUT_PT"
CKPT="$RUN/checkpoints/$OUT_PT"
[[ -f "$CKPT" ]] || { echo "[fatal] export missing $CKPT"; exit 1; }

# all_robot trains stateless (with_state=false); LIBERO tasks use the franka tag in the merged stats.
export CKPT NUM_TRIALS=10 WITH_STATE=false
echo "[$(date -Is)] launching LIBERO eval on $CKPT (with_state=false)"
bash examples/LIBERO/eval_libero_vlajepa.sh

FN=$(basename "$CKPT" .pt)
echo "=============== ALL_ROBOT LIBERO SUCCESS RATES ($FN) ==============="
for s in libero_10 libero_goal libero_object libero_spatial; do
  r=$(grep -E "Total success rate" "results/$s/$FN/eval.log" 2>/dev/null | tail -1)
  echo "  $s : ${r:-N/A}"
done
echo "EVALALL_COMPLETE $FN"
