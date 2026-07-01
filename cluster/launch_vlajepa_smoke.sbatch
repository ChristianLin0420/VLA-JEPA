#!/bin/bash
#SBATCH --job-name=vlajepa-smoke
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/vlajepa_smoke_%j.out
#SBATCH --error=outputs/slurm/vlajepa_smoke_%j.err
set -uo pipefail
# Functional smoke: short video JEPA pretrain run to completion on 8 GPUs.
# Validates env + SSv2 data + Qwen3VL/V-JEPA2 forward + DeepSpeed + W&B online +
# JEPA representation analysis panels + full-state checkpoint + completion marker.

REPO_ROOT="${SLURM_SUBMIT_DIR:-/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA}"
cd "$REPO_ROOT"; mkdir -p outputs/slurm
RUN_ROOT=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs
RUN_ID=vlajepa_smoke
OUTPUT_DIR="$RUN_ROOT/$RUN_ID"

export HF_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source /lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/etc/profile.d/conda.sh
conda activate VLA_JEPA
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/cache
export MPLCONFIGDIR="$XDG_CACHE_HOME/mpl"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TMPDIR="${TMPDIR:-/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/tmp}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$TMPDIR"
CA="$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
[[ -n "$CA" ]] && export REQUESTS_CA_BUNDLE="$CA" SSL_CERT_FILE="$CA" CURL_CA_BUNDLE="$CA"
[[ -f "$HOME/.wandb_api_key" ]] && export WANDB_API_KEY="$(tr -d '[:space:]' < "$HOME/.wandb_api_key")"
export WANDB_MODE=online WANDB_ENTITY=crlc112358 WANDB_PROJECT=vla-jepa WANDB_DIR="$OUTPUT_DIR/wandb"
export WANDB_CONFIG_DIR="$XDG_CACHE_HOME/wandb-config" WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb-cache"
mkdir -p "$WANDB_DIR" "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR"
export NCCL_IB_HCA=mlx5 NCCL_SOCKET_IFNAME='^lo,docker0'
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

UTIL_LOG="outputs/slurm/vlajepa_util_${SLURM_JOB_ID}.csv"
( nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -l 15 > "$UTIL_LOG" 2>/dev/null ) &
SMI_PID=$!; trap '[[ -n "${SMI_PID:-}" ]] && kill "$SMI_PID" 2>/dev/null || true' EXIT

echo "[$(date -Is)] SMOKE: 30-step video JEPA pretrain on 8 GPUs"
accelerate launch \
  --config_file ./starVLA/config/accelerate/ddp_bf16.yaml \
  --num_processes 8 --num_machines 1 --main_process_port "$MASTER_PORT" \
  ./starVLA/training/train_vlajepa_video.py \
  --config_yaml ./scripts/config/vlajepa_video_pretrain.yaml \
  --run_root_dir "$RUN_ROOT" --run_id "$RUN_ID" --experiment_id E-smoke \
  --trainer.resume_from_latest true \
  --trainer.max_train_steps 30 --trainer.num_warmup_steps 3 \
  --trainer.ckpt_interval 10 --trainer.save_interval 10 \
  --trainer.logging_frequency 2 --trainer.jepa_log_interval 5 --trainer.jepa_figure_interval 10 \
  --trainer.grace_seconds 60 \
  --wandb.mode online --wandb.entity crlc112358 --wandb.project vla-jepa \
  --datasets.video_data.per_device_batch_size 2
EXIT=$?
echo "[$(date -Is)] smoke accelerate exit=$EXIT; complete_marker=$([[ -f "$OUTPUT_DIR/.training_complete" ]] && echo yes || echo no)"
exit $EXIT
