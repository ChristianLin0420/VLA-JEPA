#!/bin/bash
#SBATCH --job-name=vlajepa-libero-eval
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/vlajepa_libero_eval_%j.out
#SBATCH --error=outputs/slurm/vlajepa_libero_eval_%j.err
set -uo pipefail
# LIBERO sim eval for a VLA-JEPA cotrain checkpoint. Pass CKPT/NUM_TRIALS via --export.
#   sbatch --export=ALL,CKPT=/abs/x.pt,NUM_TRIALS=20 cluster/eval_libero_vlajepa.sbatch
REPO="${SLURM_SUBMIT_DIR:-/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA}"
cd "$REPO"; mkdir -p outputs/slurm
source /lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/etc/profile.d/conda.sh
export HF_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export NCCL_IB_HCA=mlx5 NCCL_SOCKET_IFNAME='^lo,docker0'
bash examples/LIBERO/eval_libero_vlajepa.sh
echo "[$(date -Is)] eval finished; computing success rates"
grep -riE 'success' results/*/"$(basename ${CKPT:-VLA-JEPA-cotrain-step2000.pt} .pt)"/eval.log 2>/dev/null | tail -20 || true
