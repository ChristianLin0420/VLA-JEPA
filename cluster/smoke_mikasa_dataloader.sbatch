#!/bin/bash
#SBATCH --job-name=vj-ft-mikasa-dlsmoke
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/mikasa_dlsmoke_%j.out
#SBATCH --error=outputs/slurm/mikasa_dlsmoke_%j.err
set -euo pipefail
# Standalone dataloader smoke for the all_robot_mikasa mixture: builds the
# exact vla_data dataset of scripts/config/vlajepa_memv1_mikasa_ft.yaml, draws
# >=20 segments incl. >=3 MIKASA ones, checks shapes/stats. First run also
# materializes meta/stats_gr00t.json for every converted MIKASA dataset via
# the loader's own stats path (identical to how existing corpora got theirs).

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
mkdir -p outputs/slurm

source /lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/etc/profile.d/conda.sh
conda activate VLA_JEPA
export PYTHONNOUSERSITE=1
export NO_ALBUMENTATIONS_UPDATE=1

python scripts/data/smoke_mikasa_mixture.py \
  --config scripts/config/vlajepa_memv1_mikasa_ft.yaml \
  --num-samples 20 --min-mikasa 3
