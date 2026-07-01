#!/bin/bash
#SBATCH --job-name=vlajepa-cotrain-8gpu
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
#SBATCH --output=outputs/slurm/vlajepa_cotrain_%j.out
#SBATCH --error=outputs/slurm/vlajepa_cotrain_%j.err
set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA}"
cd "$REPO_ROOT"
mkdir -p outputs/slurm
if [[ -n "${EXPECTED_GIT_SHA:-}" && "$(git rev-parse HEAD)" != "$EXPECTED_GIT_SHA" ]]; then
  echo "[fatal] repository HEAD changed after submission; expected $EXPECTED_GIT_SHA" >&2
  exit 2
fi
if [[ -n "${EXPECTED_GIT_SHA:-}" && -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "[fatal] repository became dirty after pipeline submission" >&2
  exit 2
fi
CONFIG="${CONFIG:-./scripts/config/vlajepa_cotrain_cluster.yaml}"
if [[ -n "${EXPECTED_CONFIG_SHA:-}" && "$(sha256sum "$CONFIG" | awk '{print $1}')" != "$EXPECTED_CONFIG_SHA" ]]; then
  echo "[fatal] training config changed after submission: $CONFIG" >&2
  exit 2
fi
TRAIN_SCRIPT="${TRAIN_SCRIPT:-./starVLA/training/train_vlajepa_cotrain.py}"
EXP_ID="${EXP_ID:-E20260625-cotrain}"
MICRO="${MICRO:-8}"
NPROC="${NPROC:-8}"
RUN_ROOT="${RUN_ROOT:-/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs}"
RUN_ID="${RUN_ID:-vlajepa_cotrain}"
OUTPUT_DIR="$RUN_ROOT/$RUN_ID"
GRACE="${GRACE:-300}"

export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/edgeai/users/chrislin/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source /lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/etc/profile.d/conda.sh
conda activate VLA_JEPA

# --- isolate from the (full + polluted) $HOME / user-site --------------------
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/cache
export MPLCONFIGDIR="$XDG_CACHE_HOME/mpl"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TMPDIR="${TMPDIR:-/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/tmp}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$TMPDIR"

CA="$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
[[ -n "$CA" ]] && export REQUESTS_CA_BUNDLE="$CA" SSL_CERT_FILE="$CA" CURL_CA_BUNDLE="$CA"
[[ -f "$HOME/.wandb_api_key" ]] && export WANDB_API_KEY="$(tr -d '[:space:]' < "$HOME/.wandb_api_key")"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-crlc112358}"
export WANDB_PROJECT="${WANDB_PROJECT:-vla-jepa}"
export WANDB_DIR="$OUTPUT_DIR/wandb"
export WANDB_CONFIG_DIR="$XDG_CACHE_HOME/wandb-config"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb-cache"
mkdir -p "$WANDB_DIR" "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR"

export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME='^lo,docker0'
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

LIMIT_SECONDS=$((4*3600))
export TRAIN_DEADLINE_EPOCH=$(( $(date +%s) + LIMIT_SECONDS - 120 ))
[[ -n "${DEADLINE_OVERRIDE_SECONDS:-}" ]] && export TRAIN_DEADLINE_EPOCH=$(( $(date +%s) + DEADLINE_OVERRIDE_SECONDS ))
echo "[$(date -Is)] deadline=$TRAIN_DEADLINE_EPOCH"

GOT_SIG=0
trap 'GOT_SIG=1; echo "[trap] USR1 received"; pkill -USR1 -g 0 -f train_vlajepa 2>/dev/null || true' USR1

UTIL_LOG="outputs/slurm/vlajepa_util_${SLURM_JOB_ID}.csv"
( nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits -l 15 > "$UTIL_LOG" 2>/dev/null ) &
SMI_PID=$!
trap '[[ -n "${SMI_PID:-}" ]] && kill "$SMI_PID" 2>/dev/null || true' EXIT

echo "[$(date -Is)] launching accelerate ($NPROC procs) :: $TRAIN_SCRIPT $CONFIG"
EXTRA_OVERRIDES=()
[[ -n "${PRETRAINED_CHECKPOINT:-}" ]] && EXTRA_OVERRIDES+=(--trainer.pretrained_checkpoint "$PRETRAINED_CHECKPOINT")
[[ -n "${MAX_TRAIN_STEPS:-}" ]] && EXTRA_OVERRIDES+=(--trainer.max_train_steps "$MAX_TRAIN_STEPS")
[[ -n "${CKPT_INTERVAL:-}" ]] && EXTRA_OVERRIDES+=(--trainer.ckpt_interval "$CKPT_INTERVAL" --trainer.save_interval "$CKPT_INTERVAL")
[[ -n "${EVAL_INTERVAL:-}" ]] && EXTRA_OVERRIDES+=(--trainer.eval_interval "$EVAL_INTERVAL")
[[ -n "${SEGMENT_LENGTH:-}" ]] && EXTRA_OVERRIDES+=(--datasets.vla_data.segment_length "$SEGMENT_LENGTH")
[[ -n "${BURN_IN_MAX_DECISIONS:-}" ]] && EXTRA_OVERRIDES+=(--datasets.vla_data.burn_in_max_decisions "$BURN_IN_MAX_DECISIONS")
accelerate launch \
  --config_file ./starVLA/config/accelerate/ddp_bf16.yaml \
  --num_processes "$NPROC" --num_machines 1 --main_process_port "$MASTER_PORT" \
  "$TRAIN_SCRIPT" \
  --config_yaml "$CONFIG" \
  --run_root_dir "$RUN_ROOT" --run_id "$RUN_ID" \
  --experiment_id "$EXP_ID" \
  --trainer.resume_from_latest true \
  --trainer.grace_seconds "$GRACE" \
  --wandb.mode "$WANDB_MODE" --wandb.entity "$WANDB_ENTITY" --wandb.project "$WANDB_PROJECT" \
  --datasets.vla_data.per_device_batch_size "$MICRO" \
  --datasets.video_data.per_device_batch_size "$MICRO" \
  "${EXTRA_OVERRIDES[@]}" &
PYPID=$!
set +e
wait "$PYPID"; EXIT=$?
while kill -0 "$PYPID" 2>/dev/null; do wait "$PYPID"; EXIT=$?; done
set -e
echo "[$(date -Is)] accelerate returned $EXIT (got_sig=$GOT_SIG)"

if [[ -f "$OUTPUT_DIR/.training_complete" ]]; then
  echo "[$(date -Is)] training complete -> no requeue"
  exit 0
fi
if [[ "$GOT_SIG" == "1" || $EXIT -eq 0 || $EXIT -eq 124 || $EXIT -eq 137 || $EXIT -eq 143 ]]; then
  echo "[$(date -Is)] incomplete + graceful/timed-out -> scontrol requeue $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
echo "[$(date -Is)] incomplete + nonzero exit ($EXIT) -> crash, NOT requeueing"
exit "$EXIT"
