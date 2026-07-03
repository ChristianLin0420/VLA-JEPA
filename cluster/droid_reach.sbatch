#!/bin/bash
#SBATCH --job-name=vj-m1-droid-reach
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/vlajepa_droid_reach_%j.out
#SBATCH --error=outputs/slurm/vlajepa_droid_reach_%j.err
set -euo pipefail

# T0.2d DROID temporal-reach driver: enumerate training-unseen DROID episodes,
# cache the N deepest with >=MIN_DECISIONS decisions, replay the
# {live,bypass,shuffled} battery, and summarize paired dMSE + state stability
# by decision depth (100-150 decisions = 8-12x the d=12 training horizon).
# Overrides via --export: CKPT, COMPLETED_STEPS, N_EPISODES, WANDB_EVAL=off.
REPO="${SLURM_SUBMIT_DIR:?submit from the memexp worktree}"
cd "$REPO"
source /lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1 PYTHONPATH="$REPO"
export HF_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/VLA_JEPA/bin/python

CKPT="${CKPT:-/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/vlajepa_memv1_cotrain/checkpoints/VLA-JEPA-memv1-live-step_34729.pt}"
RUN_CONFIG="${RUN_CONFIG:-/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/vlajepa_memv1_cotrain/config.yaml}"
COMPLETED_STEPS="${COMPLETED_STEPS:-48668}"
N_EPISODES="${N_EPISODES:-150}"
MIN_DECISIONS="${MIN_DECISIONS:-100}"
DATA_ROOT=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/datasets/lerobot
OUT_ROOT="${OUT_ROOT:-results/offline/droid_reach/$(basename "$CKPT" .pt)}"
NGPU="${SLURM_GPUS_ON_NODE:-8}"
[[ -s "$CKPT" ]] || { echo "[fatal] checkpoint missing: $CKPT" >&2; exit 1; }
mkdir -p "$OUT_ROOT"

# Steps 1+2 are skipped when the cache already exists (replay-only reruns).
if compgen -G "$OUT_ROOT/cache/droid/*ep*.pt" > /dev/null; then
  echo "[skip] DROID token cache exists under $OUT_ROOT/cache/droid"
else
  # ---- Step 1 (CPU): sampler-replay enumerator -> top-N unseen episode list.
  ENUM_DIR="$OUT_ROOT/consumed_step${COMPLETED_STEPS}"
  if [[ ! -s "$ENUM_DIR/unseen_episodes.jsonl" ]]; then
    $PY scripts/data/enumerate_consumed_episodes.py \
        --config "$RUN_CONFIG" --completed-steps "$COMPLETED_STEPS" \
        --world-size 8 --output-dir "$ENUM_DIR"
  fi
  # Deepest-first selection; if fewer than N exist, take all and say so.
  $PY - "$ENUM_DIR/unseen_episodes.jsonl" "$OUT_ROOT/episodes.txt" \
      "$N_EPISODES" "$MIN_DECISIONS" <<'SELECT'
import json, sys
unseen, out, n, min_dec = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
records = [r for r in map(json.loads, open(unseen))
           if r["dataset"] == "droid_lerobot" and r["num_decisions"] >= min_dec]
records.sort(key=lambda r: (-r["num_decisions"], r["episode_index"]))
if len(records) < n:
    print(f"[select] only {len(records)} unseen droid_lerobot episodes with "
          f">={min_dec} decisions exist (requested {n}); taking all of them")
records = records[:n]
if not records:
    sys.exit(f"[fatal] no unseen droid_lerobot episodes with >={min_dec} decisions")
with open(out, "w") as fh:
    fh.writelines(f"{r['episode_index']}\n" for r in records)
depths = [r["num_decisions"] for r in records]
print(f"[select] {len(records)} episodes, decision depth {min(depths)}..{max(depths)}")
SELECT

  # ---- Step 2 (GPU): cache Qwen tokens, one round-robin episode shard per GPU
  # (droid_lerobot's robot type in the training mixture is droid_libero).
  rm -rf "$OUT_ROOT/cache_shards"
  mkdir -p "$OUT_ROOT/cache/droid" "$OUT_ROOT/cache_shards"
  awk -v n="$NGPU" -v dir="$OUT_ROOT/cache_shards" \
      '{ shard = sprintf("%s/episodes_%d.txt", dir, NR % n); print > shard }' \
      "$OUT_ROOT/episodes.txt"
  pids=()
  for gpu in $(seq 0 $((NGPU - 1))); do
    shard="$OUT_ROOT/cache_shards/episodes_$gpu.txt"
    [[ -s "$shard" ]] || continue
    $PY scripts/offline_eval/cache_tokens.py \
        --ckpt "$CKPT" --dataset droid_lerobot --suite droid \
        --robot-type droid_libero --data-root "$DATA_ROOT" \
        --episodes "$shard" --out-dir "$OUT_ROOT/cache_shards/$gpu" \
        --device "cuda:$gpu" > "$OUT_ROOT/cache_shards/cache_$gpu.log" 2>&1 &
    pids+=($!)
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  [[ $status -eq 0 ]] || { echo "[fatal] caching failed; see $OUT_ROOT/cache_shards/cache_*.log" >&2; exit 1; }
  for gpu in $(seq 0 $((NGPU - 1))); do
    [[ -d "$OUT_ROOT/cache_shards/$gpu/droid" ]] || continue
    mv "$OUT_ROOT/cache_shards/$gpu/droid/"*.pt "$OUT_ROOT/cache/droid/"
    cat "$OUT_ROOT/cache_shards/$gpu/droid/index.jsonl" >> "$OUT_ROOT/cache/droid/index.jsonl"
  done
  cached=$(compgen -G "$OUT_ROOT/cache/droid/*ep*.pt" | wc -l)
  wanted=$(wc -l < "$OUT_ROOT/episodes.txt")
  [[ "$cached" -eq "$wanted" ]] || { echo "[fatal] cached $cached of $wanted selected episodes" >&2; exit 1; }
  echo "[cache] $cached episodes under $OUT_ROOT/cache/droid"
fi

# ---- Step 3 (GPU): mode battery only — no burn-in or lambda sweeps (T0.2d).
$PY scripts/offline_eval/replay_engine.py \
    --ckpt "$CKPT" --cache-dir "$OUT_ROOT/cache/droid" \
    --out "$OUT_ROOT/replay_droid" \
    --conditions live,bypass,shuffled --burnin "" --lam "" \
    --device cuda:0 > "$OUT_ROOT/replay_droid.log" 2>&1 \
  || { echo "[fatal] replay failed; see $OUT_ROOT/replay_droid.log" >&2; exit 1; }

# ---- Step 4 (CPU): depth-binned paired dMSE + state-stability summary.
# CSV first (must succeed), then the guarded wandb hook (WANDB_EVAL=off to
# skip); the recompute is seconds of CPU and never fails the job.
$PY scripts/analysis/droid_reach_summary.py \
    --replay "$OUT_ROOT/replay_droid.parquet" \
    --out-csv "$OUT_ROOT/droid_reach_summary.csv" --no-wandb
if [[ "${WANDB_EVAL:-on}" != "off" ]]; then
  CA="$($PY -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
  [[ -n "$CA" ]] && export REQUESTS_CA_BUNDLE="$CA" SSL_CERT_FILE="$CA" CURL_CA_BUNDLE="$CA"
  [[ -f "$HOME/.wandb_api_key" ]] && export WANDB_API_KEY="$(tr -d '[:space:]' < "$HOME/.wandb_api_key")"
  $PY scripts/analysis/droid_reach_summary.py \
      --replay "$OUT_ROOT/replay_droid.parquet" \
      --out-csv "$OUT_ROOT/droid_reach_summary.csv" \
    || echo "[warn] wandb summary logging failed (job continues)"
fi
echo "DROID_REACH_COMPLETE ckpt=$(basename "$CKPT") out=$OUT_ROOT"
