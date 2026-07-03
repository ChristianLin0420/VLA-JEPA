#!/bin/bash
#SBATCH --job-name=vj-m1-offline-pilot
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/vlajepa_offline_pilot_%j.out
#SBATCH --error=outputs/slurm/vlajepa_offline_pilot_%j.err
set -euo pipefail

# T0.2 offline teacher-forced pilot: cache Qwen tokens for the four LIBERO
# suites (one GPU each, in parallel), then replay the mode battery + burn-in
# and lambda sweeps with probe-state dumps (T0.6 input).
REPO="${SLURM_SUBMIT_DIR:?submit from the memexp worktree}"
cd "$REPO"
source /lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1 PYTHONPATH="$REPO"
export HF_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/VLA_JEPA/bin/python

CKPT="${CKPT:?}"
DATA_ROOT=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/datasets/lerobot
OUT_ROOT="${OUT_ROOT:-results/offline/$(basename "$CKPT" .pt)}"
MAX_EPISODES="${MAX_EPISODES:-60}"
mkdir -p "$OUT_ROOT"

# cache_tokens writes episode files under <out-dir>/<suite>/; skip suites
# whose cache already exists so replay-only reruns reuse the Qwen encodes.
pids=()
gpu=0
for suite in libero_10 libero_goal libero_object libero_spatial; do
  if compgen -G "$OUT_ROOT/cache/$suite/*ep*.pt" > /dev/null; then
    echo "[skip] cache exists for $suite"
    continue
  fi
  $PY scripts/offline_eval/cache_tokens.py \
      --ckpt "$CKPT" --dataset "${suite}_no_noops_1.0.0_lerobot" --suite "$suite" \
      --data-root "$DATA_ROOT" --out-dir "$OUT_ROOT/cache" \
      --max-episodes "$MAX_EPISODES" --device "cuda:$gpu" \
      > "$OUT_ROOT/cache_$suite.log" 2>&1 &
  pids+=($!); gpu=$((gpu+1))
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ $status -eq 0 ]] || { echo "[fatal] caching failed; see $OUT_ROOT/cache_*.log" >&2; exit 1; }

pids=(); gpu=0
for suite in libero_10 libero_goal libero_object libero_spatial; do
  $PY scripts/offline_eval/replay_engine.py \
      --ckpt "$CKPT" --cache-dir "$OUT_ROOT/cache/$suite" \
      --out "$OUT_ROOT/replay_$suite" \
      --conditions live,bypass,prior,shuffled,frozen_after=8 \
      --burnin 0,1,2,4,8,16,32 --burnin-orders forward,shuffled,reversed \
      --lam 0,0.25,0.5,1,2,4,8 --lam-conditions live,shuffled \
      --state-dump "$OUT_ROOT/states/$suite" --device "cuda:$gpu" \
      > "$OUT_ROOT/replay_$suite.log" 2>&1 &
  pids+=($!); gpu=$((gpu+1))
done
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ $status -eq 0 ]] || { echo "[fatal] replay failed; see $OUT_ROOT/replay_*.log" >&2; exit 1; }

# Push offline pilot results to wandb (WANDB_EVAL=off to skip); never fails the job.
if [[ "${WANDB_EVAL:-on}" != "off" ]]; then
  CA="$($PY -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
  [[ -n "$CA" ]] && export REQUESTS_CA_BUNDLE="$CA" SSL_CERT_FILE="$CA" CURL_CA_BUNDLE="$CA"
  [[ -f "$HOME/.wandb_api_key" ]] && export WANDB_API_KEY="$(tr -d '[:space:]' < "$HOME/.wandb_api_key")"
  $PY scripts/analysis/log_eval_to_wandb.py \
    --ckpt-name "$(basename "$CKPT" .pt)" \
    --results-root "$(dirname "$(dirname "$OUT_ROOT")")" \
    || echo "[warn] wandb eval logging failed (job continues)"
fi
echo "OFFLINE_PILOT_COMPLETE ckpt=$(basename "$CKPT") out=$OUT_ROOT"
