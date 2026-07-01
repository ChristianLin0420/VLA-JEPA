#!/bin/bash
#SBATCH --job-name=vlajepa-video-8gpu
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --output=outputs/slurm/vlajepa_video_%j.out
#SBATCH --error=outputs/slurm/vlajepa_video_%j.err
set -uo pipefail

# ===================== config (overridable via env) =====================
REPO_ROOT="${SLURM_SUBMIT_DIR:-/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA}"
cd "$REPO_ROOT"
mkdir -p outputs/slurm
CONFIG="${CONFIG:-./scripts/config/vlajepa_video_pretrain.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-./starVLA/training/train_vlajepa_video.py}"
EXP_ID="${EXP_ID:-E20260625-video}"
MICRO="${MICRO:-8}"
NPROC="${NPROC:-8}"
RUN_ROOT="${RUN_ROOT:-/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs}"
RUN_ID="${RUN_ID:-vlajepa_video}"
OUTPUT_DIR="$RUN_ROOT/$RUN_ID"
GRACE="${GRACE:-300}"
STAGE=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage

# --- HuggingFace offline: all weights are staged locally at absolute paths ---
export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/edgeai/users/chrislin/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- conda env ---------------------------------------------------------------
source /lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/etc/profile.d/conda.sh
conda activate VLA_JEPA

# --- isolate from the (full + polluted) $HOME / user-site --------------------
# Ignore ~/.local (corrupt user-site) and push all default-home caches to lustre,
# since /home is full. matplotlib (JEPA figures) and triton write to ~/.cache by default.
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/cache
export MPLCONFIGDIR="$XDG_CACHE_HOME/mpl"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TMPDIR="${TMPDIR:-/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/tmp}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$TMPDIR"

# --- W&B online: certifi CA bundle (root cause of offline fallback) ----------
CA="$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
[[ -n "$CA" ]] && export REQUESTS_CA_BUNDLE="$CA" SSL_CERT_FILE="$CA" CURL_CA_BUNDLE="$CA"
[[ -f "$HOME/.wandb_api_key" ]] && export WANDB_API_KEY="$(tr -d '[:space:]' < "$HOME/.wandb_api_key")"
export WANDB_MODE=online
export WANDB_ENTITY="${WANDB_ENTITY:-crlc112358}"
export WANDB_PROJECT="${WANDB_PROJECT:-vla-jepa}"
export WANDB_DIR="$OUTPUT_DIR/wandb"
export WANDB_CONFIG_DIR="$XDG_CACHE_HOME/wandb-config"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb-cache"
mkdir -p "$WANDB_DIR" "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR"

# --- NCCL / IB (same fabric as the ANAMNESIS H100/H200 cluster) --------------
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME='^lo,docker0'
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

# --- wall-clock deadline: PRIMARY graceful-stop signal -----------------------
# Each worker reads TRAIN_DEADLINE_EPOCH and self-checkpoints `grace` seconds
# before it; we leave a further 120s pad before the SLURM hard kill at 4h.
LIMIT_SECONDS=$((4*3600))
export TRAIN_DEADLINE_EPOCH=$(( $(date +%s) + LIMIT_SECONDS - 120 ))
# Optional short deadline for validating the requeue/resume path quickly.
[[ -n "${DEADLINE_OVERRIDE_SECONDS:-}" ]] && export TRAIN_DEADLINE_EPOCH=$(( $(date +%s) + DEADLINE_OVERRIDE_SECONDS ))
echo "[$(date -Is)] deadline=$TRAIN_DEADLINE_EPOCH (stop ~grace=$GRACE s before it)"

# --- USR1 trap (backup; deadline is primary). Forward to the process group. --
GOT_SIG=0
trap 'GOT_SIG=1; echo "[trap] USR1 received"; pkill -USR1 -g 0 -f train_vlajepa 2>/dev/null || true' USR1

# --- background GPU-util sampler ---------------------------------------------
UTIL_LOG="outputs/slurm/vlajepa_util_${SLURM_JOB_ID}.csv"
( nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits -l 15 > "$UTIL_LOG" 2>/dev/null ) &
SMI_PID=$!
trap '[[ -n "${SMI_PID:-}" ]] && kill "$SMI_PID" 2>/dev/null || true' EXIT

# --- launch (accelerate spawns NPROC local workers; no srun needed for 1 node)
echo "[$(date -Is)] launching accelerate ($NPROC procs) :: $TRAIN_SCRIPT $CONFIG"
accelerate launch \
  --config_file ./starVLA/config/accelerate/ddp_bf16.yaml \
  --num_processes "$NPROC" --num_machines 1 --main_process_port "$MASTER_PORT" \
  "$TRAIN_SCRIPT" \
  --config_yaml "$CONFIG" \
  --run_root_dir "$RUN_ROOT" --run_id "$RUN_ID" \
  --experiment_id "$EXP_ID" \
  --trainer.resume_from_latest true \
  --trainer.grace_seconds "$GRACE" \
  --wandb.mode online --wandb.entity "$WANDB_ENTITY" --wandb.project "$WANDB_PROJECT" \
  --datasets.video_data.per_device_batch_size "$MICRO" &
PYPID=$!
set +e
wait "$PYPID"; EXIT=$?
while kill -0 "$PYPID" 2>/dev/null; do wait "$PYPID"; EXIT=$?; done
set -e
echo "[$(date -Is)] accelerate returned $EXIT (got_sig=$GOT_SIG)"

# --- requeue decision --------------------------------------------------------
if [[ -f "$OUTPUT_DIR/.training_complete" ]]; then
  echo "[$(date -Is)] training complete -> no requeue"
  exit 0
fi
# Incomplete: requeue on graceful deadline stop (exit 0), SLURM kill (137/143/124),
# or USR1. A genuine crash (other non-zero exit) is NOT auto-requeued to avoid loops.
if [[ "$GOT_SIG" == "1" || $EXIT -eq 0 || $EXIT -eq 124 || $EXIT -eq 137 || $EXIT -eq 143 ]]; then
  echo "[$(date -Is)] incomplete + graceful/timed-out -> scontrol requeue $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
echo "[$(date -Is)] incomplete + nonzero exit ($EXIT) -> treating as crash, NOT requeueing"
exit "$EXIT"
