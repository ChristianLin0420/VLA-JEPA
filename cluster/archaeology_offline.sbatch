#!/bin/bash
#SBATCH --job-name=vj-m1-archaeology
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/vlajepa_archaeology_%j.out
#SBATCH --error=outputs/slurm/vlajepa_archaeology_%j.err
set -euo pipefail

# T0.7 checkpoint archaeology: export archived safetensors anchors to .pt,
# replay the T0.2 mode battery (live/bypass/prior/shuffled only, no sweeps)
# on the checkpoint-portable step_34729 Qwen caches for every anchor x LIBERO
# suite (8-GPU grid), then summarize paired dMSE + tanh(policy gate) emergence.
REPO="${SLURM_SUBMIT_DIR:?submit from the memexp worktree}"
cd "$REPO"
source /lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1 PYTHONPATH="$REPO"
export HF_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/VLA_JEPA/bin/python

STEPS="${STEPS:-stage1_5000,30000,40000,41686,48668}"
CACHE_ROOT="${CACHE_ROOT:-results/offline/VLA-JEPA-memv1-live-step_34729/cache}"
ARCHIVE=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/memv1_ckpt_archive
COTRAIN_RUN=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/vlajepa_memv1_cotrain
STAGE1_RUN=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/vlajepa_memv1_stage1
OUT_ROOT=results/offline/archaeology
SUITES=(libero_10 libero_goal libero_object libero_spatial)
mkdir -p "$OUT_ROOT"

for suite in "${SUITES[@]}"; do
  compgen -G "$CACHE_ROOT/$suite/*ep*.pt" > /dev/null \
    || { echo "[fatal] no token cache for $suite under $CACHE_ROOT (run offline_memexp_pilot first)" >&2; exit 1; }
done

# Export each anchor into a pseudo run dir under exports/: the archived cotrain
# steps have no sibling config.yaml, so they borrow the live run's (frozen-Qwen
# caches make the anchors config-compatible); stage1 borrows the stage1 run's,
# correct for that checkpoint.  The config.yaml + dataset_statistics.json
# symlinks satisfy read_mode_config both for export_vlajepa_ckpt.py and for
# the replay's from_pretrained, so the .pt is exported in place and kept.
declare -A CKPT
for tag in ${STEPS//,/ }; do
  if [[ "$tag" =~ ^stage1_[0-9]+$ ]]; then
    step="step_${tag#stage1_}"; src_run="$STAGE1_RUN"; src_step="$STAGE1_RUN/checkpoints/$step"
    pt_name="VLA-JEPA-memv1-stage1-$step.pt"
  elif [[ "$tag" =~ ^[0-9]+$ ]]; then
    step="step_$tag"; src_run="$COTRAIN_RUN"; src_step="$ARCHIVE/cotrain/$step"
    pt_name="VLA-JEPA-memv1-$step.pt"
  else
    echo "[fatal] bad step tag '$tag' in STEPS (want N or stage1_N)" >&2; exit 1
  fi
  run_dir="$OUT_ROOT/exports/$tag"
  CKPT[$tag]="$run_dir/checkpoints/$pt_name"
  if [[ -f "${CKPT[$tag]}" ]]; then echo "[skip] export exists for $tag"; continue; fi
  [[ -f "$src_step/model.safetensors" ]] \
    || { echo "[fatal] missing $src_step/model.safetensors" >&2; exit 1; }
  mkdir -p "$run_dir/checkpoints"
  ln -sfn "$src_run/config.yaml" "$run_dir/config.yaml"
  ln -sfn "$src_run/dataset_statistics.json" "$run_dir/dataset_statistics.json"
  ln -sfn "$src_step" "$run_dir/checkpoints/$step"
  $PY cluster/export_vlajepa_ckpt.py "$run_dir" "$step" "$pt_name"
done

# Replay grid (checkpoint x suite) round-robined over the node's 8 GPUs; one
# sequential queue per GPU bounds concurrency at 8.  Battery conditions only:
# no burn-in/lambda sweeps, no state dumps (T0.2's step_34729 run covers those).
jobs=()
for tag in ${STEPS//,/ }; do
  for suite in "${SUITES[@]}"; do
    if [[ -f "$OUT_ROOT/$tag/replay_$suite.parquet" ]]; then
      echo "[skip] replay exists for $tag/$suite"
    else
      jobs+=("$tag $suite")
    fi
  done
done

run_queue() {
  local gpu=$1 rc=0 tag suite out_dir; shift
  for job in "$@"; do
    read -r tag suite <<< "$job"
    out_dir="$OUT_ROOT/$tag"; mkdir -p "$out_dir"
    echo "[replay] $tag $suite on cuda:$gpu"
    $PY scripts/offline_eval/replay_engine.py \
        --ckpt "${CKPT[$tag]}" --cache-dir "$CACHE_ROOT/$suite" \
        --out "$out_dir/replay_$suite" \
        --conditions live,bypass,prior,shuffled --burnin "" --lam "" \
        --device "cuda:$gpu" \
        > "$out_dir/replay_$suite.log" 2>&1 \
      || { echo "[error] replay $tag $suite failed (see $out_dir/replay_$suite.log)" >&2; rc=1; }
  done
  return $rc
}

status=0; pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  queue=()
  for ((i=gpu; i<${#jobs[@]}; i+=8)); do queue+=("${jobs[$i]}"); done
  [[ ${#queue[@]} -eq 0 ]] && continue
  run_queue "$gpu" "${queue[@]}" &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ $status -eq 0 ]] || { echo "[fatal] replay failed; see $OUT_ROOT/*/replay_*.log" >&2; exit 1; }

# Summary (csv + printed table) + wandb (WANDB_EVAL=off to skip); never fails the job.
SUMMARY_ARGS=(--root "$OUT_ROOT" --steps "$STEPS" --archive-root "$ARCHIVE")
if [[ "${WANDB_EVAL:-on}" != "off" ]]; then
  CA="$($PY -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
  [[ -n "$CA" ]] && export REQUESTS_CA_BUNDLE="$CA" SSL_CERT_FILE="$CA" CURL_CA_BUNDLE="$CA"
  [[ -f "$HOME/.wandb_api_key" ]] && export WANDB_API_KEY="$(tr -d '[:space:]' < "$HOME/.wandb_api_key")"
else
  SUMMARY_ARGS+=(--no-wandb)
fi
$PY scripts/analysis/archaeology_summary.py "${SUMMARY_ARGS[@]}" \
  || echo "[warn] archaeology summary failed (job continues)"
echo "ARCHAEOLOGY_COMPLETE steps=$STEPS out=$OUT_ROOT"
