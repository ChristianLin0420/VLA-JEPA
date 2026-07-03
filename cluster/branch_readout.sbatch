#!/bin/bash
#SBATCH --job-name=vj-m1-branch-readout
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/vlajepa_branch_readout_%j.out
#SBATCH --error=outputs/slurm/vlajepa_branch_readout_%j.err
set -euo pipefail

# T2.3 readout: does the ctx16 credit window create memory-content use?
# Exports each branch's latest checkpoint, replays the offline mode battery
# on the step_34729 LIBERO caches, dumps states, fits probes.  Decisive
# metric: the live-shuffled (foreign content) gap in ctx16 vs ref.
REPO="${SLURM_SUBMIT_DIR:?submit from the memexp worktree}"
cd "$REPO"
source /lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1 PYTHONPATH="$REPO"
export HF_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/VLA_JEPA/bin/python

RUN_ROOT=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs
CACHE_ROOT="${CACHE_ROOT:-results/offline/VLA-JEPA-memv1-live-step_34729/cache}"
OUT_ROOT=results/offline/branch_readout
SUITES=(libero_10 libero_goal libero_object libero_spatial)
declare -A CKPT

for b in ref ctx16; do
  run="$RUN_ROOT/vlajepa_memv1_branch_$b"
  step="$(tr -d '[:space:]' < "$run/checkpoints/latest.txt")"
  pt="VLA-JEPA-memv1-branch-$b-$step.pt"
  CKPT[$b]="$run/checkpoints/$pt"
  [[ -f "${CKPT[$b]}" ]] || $PY cluster/export_vlajepa_ckpt.py "$run" "$step" "$pt"
done

pids=(); gpu=0; status=0
for b in ref ctx16; do
  for suite in "${SUITES[@]}"; do
    out="$OUT_ROOT/$b"; mkdir -p "$out"
    $PY scripts/offline_eval/replay_engine.py \
        --ckpt "${CKPT[$b]}" --cache-dir "$CACHE_ROOT/$suite" \
        --out "$out/replay_$suite" \
        --conditions live,bypass,prior,shuffled --burnin "" --lam "" \
        --state-dump "$out/states/$suite" --device "cuda:$gpu" \
        > "$out/replay_$suite.log" 2>&1 &
    pids+=($!); gpu=$(( (gpu+1) % 8 ))
  done
done
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ $status -eq 0 ]] || { echo "[fatal] replay failed; see $OUT_ROOT/*/replay_*.log" >&2; exit 1; }

export OMP_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 MKL_NUM_THREADS=20
pids=()
for b in ref ctx16; do
  for suite in "${SUITES[@]}"; do
    $PY scripts/probe/fit_probes.py --dump-dir "$OUT_ROOT/$b/states/$suite" \
        --out-dir "$OUT_ROOT/$b/probes/$suite" --bootstrap 300 \
        > "$OUT_ROOT/$b/probes_$suite.log" 2>&1 &
    pids+=($!)
  done
done
for pid in "${pids[@]}"; do wait "$pid" || echo "[warn] a probe fit failed"; done

$PY - <<'EOF'
import pandas as pd
for b in ["ref", "ctx16"]:
    dfs = [pd.read_parquet(f"results/offline/branch_readout/{b}/replay_{s}.parquet").assign(suite=s)
           for s in ["libero_10","libero_goal","libero_object","libero_spatial"]]
    df = pd.concat(dfs)
    base = df[(df.condition=="live")&(df.J==-1)&(df["lam"]==1.0)].set_index(["suite","episode","decision"]).mse
    row = {}
    for c in ["bypass","prior","shuffled"]:
        m = df[(df.condition==c)&(df.J==-1)&(df["lam"]==1.0)].set_index(["suite","episode","decision"]).mse
        d = (pd.concat([base,m],axis=1,keys=["l","a"]).dropna())
        row[c] = (d.a-d.l).mean()
    print(f"{b:6s} dMSE bypass={row['bypass']:+.5f} prior={row['prior']:+.5f} shuffled={row['shuffled']:+.6f}")
EOF
echo "BRANCH_READOUT_COMPLETE"
