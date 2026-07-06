#!/bin/bash
#SBATCH --job-name=vj-mikasa-offline-disc
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/vlajepa_mikasa_disc_%j.out
#SBATCH --error=outputs/slurm/vlajepa_mikasa_disc_%j.err
set -euo pipefail

# Offline content-read discriminator on MIKASA anchors, before vs after
# content-demand fine-tuning. Qwen is frozen in both checkpoints, so one
# token cache serves both: paired live/bypass/prior/shuffled dMSE measures
# content-reading far below the closed-loop success floor.
REPO="${SLURM_SUBMIT_DIR:?submit from the memexp worktree}"
cd "$REPO"
source /lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1 PYTHONPATH="$REPO"
export HF_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/VLA_JEPA/bin/python

FT_RUN=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/vlajepa_memv1_mikasa_ft
PRE_CKPT=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/vlajepa_memv1_cotrain/checkpoints/VLA-JEPA-memv1-live-step_100000.pt
DATA_ROOT=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/datasets/lerobot
ANCHORS=(mikasa_shell_game_touch_vla_v0 mikasa_remember_color9_long_vla_v0 mikasa_take_it_back_vla_v0 mikasa_chain_of_colors7_vla_v0)
OUT_ROOT=results/offline/mikasa_disc
MAX_EPISODES="${MAX_EPISODES:-40}"
mkdir -p "$OUT_ROOT"

# Export the FT checkpoint (latest durable; final when run post-completion).
FT_STEP="$(tr -d '[:space:]' < "$FT_RUN/checkpoints/latest.txt")"
FT_CKPT="$FT_RUN/checkpoints/VLA-JEPA-mikasaft-$FT_STEP.pt"
[[ -f "$FT_CKPT" ]] || $PY cluster/export_vlajepa_ckpt.py "$FT_RUN" "$FT_STEP" "$(basename "$FT_CKPT")"

# Cache anchors once (Qwen frozen in both checkpoints -> caches shared).
pids=(); gpu=0
for ds in "${ANCHORS[@]}"; do
  if ! compgen -G "$OUT_ROOT/cache/$ds/*ep*.pt" > /dev/null; then
    $PY scripts/offline_eval/cache_tokens.py \
        --ckpt "$PRE_CKPT" --dataset "$ds" --suite "$ds" --robot-type mikasa_robo \
        --data-root "$DATA_ROOT" --out-dir "$OUT_ROOT/cache" \
        --max-episodes "$MAX_EPISODES" --device "cuda:$gpu" \
        > "$OUT_ROOT/cache_$ds.log" 2>&1 &
    pids+=($!); gpu=$((gpu+1))
  fi
done
status=0; for pid in ${pids[@]+"${pids[@]}"}; do wait "$pid" || status=1; done
[[ $status -eq 0 ]] || { echo "[fatal] caching failed; see $OUT_ROOT/cache_*.log" >&2; exit 1; }

# Replay both checkpoints x 4 anchors on the shared caches (8 replays, 8 GPUs).
pids=(); gpu=0
for tag in pre100k ft; do
  ckpt="$PRE_CKPT"; [[ "$tag" == ft ]] && ckpt="$FT_CKPT"
  for ds in "${ANCHORS[@]}"; do
    out="$OUT_ROOT/$tag"; mkdir -p "$out"
    $PY scripts/offline_eval/replay_engine.py \
        --ckpt "$ckpt" --cache-dir "$OUT_ROOT/cache/$ds" \
        --out "$out/replay_$ds" \
        --conditions live,bypass,prior,shuffled --burnin "" --lam "" \
        --device "cuda:$gpu" > "$out/replay_$ds.log" 2>&1 &
    pids+=($!); gpu=$(( (gpu+1) % 8 ))
  done
done
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ $status -eq 0 ]] || { echo "[fatal] replay failed; see $OUT_ROOT/*/replay_*.log" >&2; exit 1; }

$PY - <<'EOF'
import pandas as pd, glob, os
for tag in ["pre100k", "ft"]:
    dfs = [pd.read_parquet(p).assign(suite=os.path.basename(p)[7:-8])
           for p in glob.glob(f"results/offline/mikasa_disc/{tag}/replay_*.parquet")]
    df = pd.concat(dfs)
    base = df[(df.condition=="live")&(df.J==-1)&(df["lam"]==1.0)].set_index(["suite","episode","decision"]).mse
    row = []
    for c in ["bypass","prior","shuffled"]:
        m = df[(df.condition==c)&(df.J==-1)&(df["lam"]==1.0)].set_index(["suite","episode","decision"]).mse
        d = (pd.concat([base,m],axis=1,keys=["l","a"]).dropna())
        row.append(f"{c}={float((d.a-d.l).mean()):+.6f}")
    print(f"{tag:8s} n={len(base)}  " + "  ".join(row))
EOF
echo "MIKASA_DISC_COMPLETE"
