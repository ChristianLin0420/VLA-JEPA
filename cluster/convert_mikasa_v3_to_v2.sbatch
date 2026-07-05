#!/bin/bash
#SBATCH --job-name=vj-ft-mikasa-convert
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/mikasa_convert_%j.out
#SBATCH --error=outputs/slurm/mikasa_convert_%j.err
set -euo pipefail
# CPU-bound LeRobot v3->v2 conversion of the MIKASA anchor subset
# (scripts/data/mikasa_v3_to_v2.py; idempotent, safe to resubmit on timeout).
# Usage: sbatch cluster/convert_mikasa_v3_to_v2.sbatch

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
mkdir -p outputs/slurm

source /lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/etc/profile.d/conda.sh
conda activate VLA_JEPA
export PYTHONNOUSERSITE=1

SRC="${SRC:-/lustre/fsw/portfolios/edgeai/users/chrislin/memexp_stage/mikasa-data}"
DST="${DST:-/lustre/fsw/portfolios/edgeai/users/chrislin/memexp_stage/mikasa-lerobot}"

# The 21-task anchor subset registered as 'mikasa_vla' in
# starVLA/dataloader/gr00t_lerobot/mixtures.py (keep the two lists in sync).
TASKS=(
  shell_game_shuffle_touch_vla_v0
  shell_game_shuffle_touch_long_vla_v0
  shell_game_shuffle_color_lamp_touch_vla_v0
  remember_color_3_long_vla_v0
  remember_color_5_long_vla_v0
  remember_color_9_long_vla_v0
  take_it_back_vla_v0
  chain_of_colors_3_vla_v0
  chain_of_colors_5_vla_v0
  chain_of_colors_7_vla_v0
  seq_of_colors_5_vla_v0
  bunch_of_colors_5_vla_v0
  gather_and_recall_3_vla_v0
  gather_and_recall_5_vla_v0
  remember_shape_5_long_vla_v0
  remember_shape_and_color_3x3_long_vla_v0
  batteries_checker_easy_3_vla_v0
  blink_count_button_press_medium_vla_v0
  timed_transfer_medium_vla_v0
  trace_shape_medium_vla_v0
  intercept_grab_fast_vla_v0
)

python scripts/data/mikasa_v3_to_v2.py \
  --src "$SRC" --dst "$DST" --workers 21 \
  --tasks "${TASKS[@]}"

echo "[convert] done; symlinking into the lerobot data root"
LEROBOT_ROOT=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/datasets/lerobot
for t in "${TASKS[@]}"; do
  ln -sfn "$DST/mikasa_${t}" "$LEROBOT_ROOT/mikasa_${t}"
done
ls -l "$LEROBOT_ROOT" | grep mikasa_ | wc -l
