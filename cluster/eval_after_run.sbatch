#!/bin/bash
#SBATCH --job-name=vlajepa-eval-run
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/vlajepa_evalrun_%j.out
#SBATCH --error=outputs/slurm/vlajepa_evalrun_%j.err
set -euo pipefail

# Generic dependent evaluation for a completed VLA-JEPA run.
# Required: RUN=/absolute/run/path OUT_PREFIX=checkpoint-name-prefix
# Optional: WITH_STATE=false MEMORY_MODE=live NUM_TRIALS=10 UNNORM_KEY=franka

REPO="${SLURM_SUBMIT_DIR:-/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA}"
cd "$REPO"
mkdir -p outputs/slurm
if [[ -n "${EXPECTED_GIT_SHA:-}" && "$(git rev-parse HEAD)" != "$EXPECTED_GIT_SHA" ]]; then
  echo "[fatal] repository HEAD changed after submission; expected $EXPECTED_GIT_SHA" >&2
  exit 2
fi
if [[ -n "${EXPECTED_GIT_SHA:-}" && -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "[fatal] repository became dirty after pipeline submission" >&2
  exit 2
fi

: "${RUN:?RUN must be an absolute completed run directory}"
: "${OUT_PREFIX:?OUT_PREFIX must be a safe checkpoint filename prefix}"
WITH_STATE="${WITH_STATE:-false}"
MEMORY_MODE="${MEMORY_MODE:-live}"
NUM_TRIALS="${NUM_TRIALS:-10}"
UNNORM_KEY="${UNNORM_KEY:-franka}"

[[ "$RUN" = /* ]] || { echo "[fatal] RUN must be absolute: $RUN" >&2; exit 2; }
[[ "$OUT_PREFIX" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "[fatal] unsafe OUT_PREFIX: $OUT_PREFIX" >&2
  exit 2
}
[[ "$MEMORY_MODE" == "live" || "$MEMORY_MODE" == "zero" ]] || {
  echo "[fatal] MEMORY_MODE must be live or zero: $MEMORY_MODE" >&2
  exit 2
}
[[ "$WITH_STATE" == "true" || "$WITH_STATE" == "false" ]] || {
  echo "[fatal] WITH_STATE must be true or false" >&2
  exit 2
}
[[ "$NUM_TRIALS" =~ ^[1-9][0-9]*$ ]] || {
  echo "[fatal] NUM_TRIALS must be a positive integer" >&2
  exit 2
}

RUN="${RUN%/}"
[[ -d "$RUN" ]] || { echo "[fatal] run directory missing: $RUN" >&2; exit 1; }
[[ -f "$RUN/.training_complete" ]] || {
  echo "[fatal] run has no .training_complete marker: $RUN" >&2
  exit 1
}
[[ -s "$RUN/config.yaml" ]] || { echo "[fatal] missing run config" >&2; exit 1; }
[[ -s "$RUN/dataset_statistics.json" ]] || {
  echo "[fatal] missing dataset statistics" >&2
  exit 1
}

LATEST="$RUN/checkpoints/latest.txt"
[[ -s "$LATEST" ]] || { echo "[fatal] missing latest checkpoint pointer" >&2; exit 1; }
STEP="$(tr -d '[:space:]' < "$LATEST")"
[[ "$STEP" =~ ^step_[0-9]+$ ]] || {
  echo "[fatal] invalid latest checkpoint name: $STEP" >&2
  exit 1
}
[[ -s "$RUN/checkpoints/$STEP/model.safetensors" ]] || {
  echo "[fatal] missing model state for $STEP" >&2
  exit 1
}

source /lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
export HF_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME='^lo,docker0'
VLA_PY=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/VLA_JEPA/bin/python

OUT_PT="${OUT_PREFIX}-${STEP}.pt"
CKPT="$RUN/checkpoints/$OUT_PT"
if [[ -s "$CKPT" ]]; then
  echo "[$(date -Is)] reusing existing export $CKPT"
else
  echo "[$(date -Is)] exporting $STEP from $RUN as $OUT_PT"
  "$VLA_PY" cluster/export_vlajepa_ckpt.py "$RUN" "$STEP" "$OUT_PT"
fi
[[ -s "$CKPT" ]] || { echo "[fatal] export missing: $CKPT" >&2; exit 1; }

export CKPT NUM_TRIALS WITH_STATE MEMORY_MODE UNNORM_KEY
echo "[$(date -Is)] LIBERO eval: ckpt=$CKPT with_state=$WITH_STATE memory_mode=$MEMORY_MODE trials=$NUM_TRIALS"
bash examples/LIBERO/eval_libero_vlajepa.sh

FN="$(basename "$CKPT" .pt)"
echo "=============== LIBERO SUCCESS RATES ($FN, memory=$MEMORY_MODE) ==============="
for suite in libero_10 libero_goal libero_object libero_spatial; do
  result="$(grep -E "Total success rate" "results/$suite/$FN/eval.log" 2>/dev/null | tail -1 || true)"
  echo "  $suite : ${result:-N/A}"
done
echo "EVALRUN_COMPLETE checkpoint=$FN memory_mode=$MEMORY_MODE"
