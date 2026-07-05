#!/bin/bash
#SBATCH --job-name=vj-bench-robomme-smoke
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --output=/lustre/fsw/portfolios/edgeai/users/chrislin/memexp_stage/logs/%x-%j.out
# Track D: RoboMME headless env-creation smoke — one task per suite category
# (Counting=StopCube, Permanence=ButtonUnmask, Reference=PickHighlight,
# Imitation=MoveCube). The Track B gate showed SAPIEN/Vulkan renders headless
# on these nodes with ZERO ICD overrides, so we run the stock environment.
# Submit from the wt-memexp worktree:
#   sbatch cluster/sbatch_vj_bench_robomme_smoke.sh
set -uo pipefail

STAGE=/lustre/fsw/portfolios/edgeai/users/chrislin/memexp_stage
PY=$STAGE/robomme/.venv/bin/python
SMOKE=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA/outputs/wt-memexp/cluster/robomme_headless_smoke.py
OUTDIR=$STAGE/robomme-smoke
mkdir -p "$OUTDIR"

echo "=== node: $(hostname) job: ${SLURM_JOB_ID:-none} ==="
nvidia-smi -L || true

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/cache
export MS_ASSET_DIR=$STAGE/maniskill_assets
export HF_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/hf_cache HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
unset DISPLAY

"$PY" "$SMOKE" \
  --tasks ${SMOKE_TASKS:-StopCube ButtonUnmask PickHighlight MoveCube} \
  --dataset "${SMOKE_DATASET:-test}" \
  --episode-idx "${SMOKE_EPISODE:-0}" \
  --num-steps "${SMOKE_STEPS:-5}" \
  --out-dir "$OUTDIR"
status=$?
echo "VERDICT: $([ "$status" -eq 0 ] && echo PASS || echo FAIL) (exit $status)"
exit "$status"
