#!/bin/bash
#SBATCH --job-name=vj-ft-liberomem-conv
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=cpu_short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --output=/lustre/fsw/portfolios/edgeai/users/chrislin/memexp_stage/logs/%x-%j.out
# Task 2: LIBERO-Mem robosuite HDF5 (961 keyboard-teleop demos, 10 tasks) ->
# LeRobot v2.1 corpus matching our libero_*_no_noops staging conventions.
# CPU-only; parallel episode workers (hdf5 read + x264 encode bound).
# Submit from the wt-memexp worktree:
#   sbatch cluster/sbatch_vj_ft_liberomem_convert.sh
# Verify afterwards:
#   meta/info.json says 961 episodes, splits train 0:761 / val 761:961;
#   ls data/chunk-000 | wc -l == 961; ls videos/chunk-000/*/ | wc -l == 1922.
set -uo pipefail

REPO=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA/outputs/wt-memexp
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/VLA_JEPA/bin/python
STAGE=/lustre/fsw/portfolios/edgeai/users/chrislin/memexp_stage

echo "=== node: $(hostname) job: ${SLURM_JOB_ID:-none} start: $(date -Is) ==="
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

"$PY" "$REPO/scripts/data/libero_mem_to_lerobot.py" \
    --src "$STAGE/libero-mem-data/LIBERO-Mem" \
    --out "$STAGE/libero-mem-lerobot/libero_mem_1.0.0_lerobot" \
    --val-per-task 20 \
    --workers "${WORKERS:-48}" \
    --overwrite
status=$?
echo "=== exit: $status end: $(date -Is) ==="
exit $status
