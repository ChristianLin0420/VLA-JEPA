#!/usr/bin/env bash
# memv2.4 gate watcher: at each pre-registered checkpoint step, export the
# trainer checkpoint (schema-2 exporter) and submit the fwdseq endpoint
# discriminator (n=32, held-out) — the tracked go/no-go metric — plus a
# LIBERO-goal mini-regression (main arm only) as the competence guardrail.
#
# Run detached from a login node (archive_watch.sh precedent):
#   nohup bash scripts/analysis/memv2p4_gate_watch.sh > /path/watch.log 2>&1 &
#
# keep_last_checkpoints=2 with ckpt_interval=500 gives each gate step a
# ~50-minute survival window, so the poll interval must stay well under it.
set -u

RUNS_ROOT=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs
ARMS=("vlajepa_memv2p4_stage1" "vlajepa_memv2p4_stage1_privdec")
TAGS=("main" "privdec")
GATE_STEPS=(2500 5000 7500 10000)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/VLA_JEPA/bin/python
ACCOUNT=$(sacctmgr -nP show assoc user="$USER" format=account | grep -m1 edgeai_tao)
POLL_SECONDS=300
DEADLINE=$(( $(date +%s) + 16*3600 ))

submit_fwdseq() {
    local run_dir=$1 tag=$2 step=$3 ckpt=$4
    mkdir -p "$run_dir/gates"
    sbatch --account="$ACCOUNT" --partition=batch --gres=gpu:1 \
        --job-name="vj-p4-gate-${tag}-${step}" --time=01:30:00 --cpus-per-task=16 \
        --output="$run_dir/gates/fwdseq_step${step}_%j.out" \
        --wrap="cd $REPO && PYTHONPATH=$REPO $PY scripts/analysis/memv2_fwdseq_disc.py \
            --config $run_dir/config.yaml --ckpt $ckpt --num-segments 32 \
            --out $run_dir/gates/fwdseq_step${step}.json"
}

submit_guardrail() {
    local step=$1 ckpt=$2
    cd "$REPO"
    export CKPT="$ckpt" NUM_TRIALS=20 SUITES=libero_goal
    export LIBERO_HOME_OVERRIDE=/lustre/fsw/portfolios/edgeai/users/chrislin/memexp_stage/libero-mem-home
    sbatch --export=ALL --job-name="vj-p4-guard-${step}" cluster/eval_libero_vlajepa.sbatch
}

for step in "${GATE_STEPS[@]}"; do
    for i in 0 1; do
        run_dir="$RUNS_ROOT/${ARMS[$i]}"; tag="${TAGS[$i]}"
        marker="$run_dir/gates/.exported_step${step}"
        [ -f "$marker" ] && continue
        while [ ! -f "$run_dir/checkpoints/step_${step}/model.safetensors" ]; do
            if [ "$(date +%s)" -gt "$DEADLINE" ]; then
                echo "[$(date -Is)] deadline reached waiting for ${tag} step_${step}; exiting"
                exit 1
            fi
            # A later checkpoint means this step was pruned before we saw it.
            newer=$(ls -d "$run_dir"/checkpoints/step_* 2>/dev/null \
                | grep -oE '[0-9]+$' | sort -n | awk -v s="$step" '$1 > s' | head -1)
            if [ -n "$newer" ]; then
                echo "[$(date -Is)] ${tag} step_${step} pruned (saw step_${newer}); skipping gate"
                mkdir -p "$run_dir/gates"; touch "$marker"
                continue 2
            fi
            sleep "$POLL_SECONDS"
        done
        ckpt="$run_dir/checkpoints/VLA-JEPA-memv2p4-${tag}-step_${step}.pt"
        echo "[$(date -Is)] exporting ${tag} step_${step}"
        "$PY" "$REPO/scripts/analysis/memv2_export.py" "$run_dir" "step_${step}" "$(basename "$ckpt")" \
            || { echo "[$(date -Is)] export failed for ${tag} step_${step}"; continue; }
        mkdir -p "$run_dir/gates"; touch "$marker"
        submit_fwdseq "$run_dir" "$tag" "$step" "$ckpt"
        if [ "$tag" = "main" ] && { [ "$step" = "2500" ] || [ "$step" = "5000" ]; }; then
            submit_guardrail "$step" "$ckpt"
        fi
    done
done
echo "[$(date -Is)] all gate steps handled"
