#!/bin/bash
#SBATCH --job-name=vj-memv2-g1
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/vlajepa_memv2_g1_%j.out
#SBATCH --error=outputs/slurm/vlajepa_memv2_g1_%j.err
set -euo pipefail

# Gate G1 (design doc): read unit test + leak audit at stage-1 step 0.  Trains
# only fusion + mem_cond_adapter + wm_mask_token on teacher-forced state traces
# from the memv1 token caches and prints the pre-registered PASS criteria
# (foreign gap > 0 at p < 0.05; bypass arm at the unconditional floor).
REPO="${SLURM_SUBMIT_DIR:?submit from the memexp worktree}"
cd "$REPO"
source /lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1 PYTHONPATH="$REPO"
export HF_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/VLA_JEPA/bin/python

CONFIG="${CONFIG:-scripts/config/vlajepa_memv2_stage1.yaml}"
CACHE_ROOT="${CACHE_ROOT:-results/offline/VLA-JEPA-memv1-live-step_34729/cache}"
SUITE="${SUITE:-libero_object}"
STEPS="${STEPS:-500}"
MAX_EPISODES="${MAX_EPISODES:-24}"
OUT="${OUT:-results/offline/memv2_g1/${SUITE}.json}"

compgen -G "$CACHE_ROOT/$SUITE/*ep*.pt" > /dev/null \
  || { echo "[fatal] no token cache for $SUITE under $CACHE_ROOT (run offline_memexp_pilot first)" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

$PY scripts/offline_eval/g1_read_unit_test.py \
    --config "$CONFIG" \
    ${CKPT:+--ckpt "$CKPT"} \
    --cache-dir "$CACHE_ROOT/$SUITE" \
    --steps "$STEPS" --max-episodes "$MAX_EPISODES" \
    --out "$OUT"

echo "G1_COMPLETE suite=$SUITE steps=$STEPS out=$OUT"
