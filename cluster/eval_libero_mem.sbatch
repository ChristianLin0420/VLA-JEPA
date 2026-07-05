#!/bin/bash
#SBATCH --job-name=vlajepa-libero-mem-eval
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/vlajepa_libero_mem_eval_%j.out
#SBATCH --error=outputs/slurm/vlajepa_libero_mem_eval_%j.err
set -uo pipefail
# LIBERO-Mem (AAAI 2026, github.com/libero-mem/libero-mem) eval for a VLA-JEPA
# checkpoint. Modeled on eval_libero_vlajepa.sbatch, but:
#   - single suite libero_mem (one server on cuda:1 -> 2 GPUs suffice)
#   - LIBERO_HOME_OVERRIDE points at the staged libero-mem fork
#   - MEMORY_MODE & friends pass through to the model server via --export=ALL
# Usage:
#   sbatch --export=ALL,CKPT=/abs/x.pt,NUM_TRIALS=20,MEMORY_MODE=live \
#       cluster/eval_libero_mem.sbatch
# Discriminator arms additionally set e.g.:
#   MEMORY_MODE=foreign,MEMORY_DONOR_DIR=/abs/donor_bank/...   (foreign)
#   MEMORY_MODE=bypass                                          (bypass)
#   MEMORY_STATE_DUMP_DIR=/abs/new_donor_bank                   (live, dumps donors)
REPO="${SLURM_SUBMIT_DIR:-/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA/outputs/wt-memexp}"
cd "$REPO"; mkdir -p outputs/slurm
source /lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/etc/profile.d/conda.sh
export HF_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export NCCL_IB_HCA=mlx5 NCCL_SOCKET_IFNAME='^lo,docker0'

# Benchmark selection: the libero-mem fork tree (staged, never the shared
# anamnesis LIBERO). Init files must exist (examples/LIBERO/make_libero_mem_init_states.py).
export LIBERO_HOME_OVERRIDE="${LIBERO_HOME_OVERRIDE:-/lustre/fsw/portfolios/edgeai/users/chrislin/memexp_stage/libero-mem-home}"
export TASKS="${TASKS:-libero_mem}"
export NUM_TRIALS="${NUM_TRIALS:-20}"
export MEMORY_MODE="${MEMORY_MODE:-live}"

echo "[libero-mem-eval] CKPT=${CKPT:-unset} TASKS=$TASKS NUM_TRIALS=$NUM_TRIALS"
echo "[libero-mem-eval] MEMORY_MODE=$MEMORY_MODE DONOR_DIR=${MEMORY_DONOR_DIR:-} STATE_DUMP_DIR=${MEMORY_STATE_DUMP_DIR:-}"
echo "[libero-mem-eval] LIBERO_HOME_OVERRIDE=$LIBERO_HOME_OVERRIDE"

bash examples/LIBERO/eval_libero_vlajepa.sh
status=$?
echo "[$(date -Is)] eval finished (status=$status); computing success rates"
grep -riE 'success' results/*/"$(basename ${CKPT:-VLA-JEPA-cotrain-step2000.pt} .pt)"/eval.log 2>/dev/null | tail -20 || true
# Push per-arm eval results to wandb (WANDB_EVAL=off to skip); never fails the job.
if [[ "${WANDB_EVAL:-on}" != "off" ]]; then
  PY=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/VLA_JEPA/bin/python
  CA="$($PY -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
  [[ -n "$CA" ]] && export REQUESTS_CA_BUNDLE="$CA" SSL_CERT_FILE="$CA" CURL_CA_BUNDLE="$CA"
  [[ -f "$HOME/.wandb_api_key" ]] && export WANDB_API_KEY="$(tr -d '[:space:]' < "$HOME/.wandb_api_key")"
  $PY scripts/analysis/log_eval_to_wandb.py \
    --ckpt-name "$(basename ${CKPT:-VLA-JEPA-cotrain-step2000.pt} .pt)" \
    || echo "[warn] wandb eval logging failed (job continues)"
fi
exit "$status"
