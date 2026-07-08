#!/usr/bin/env bash
# memv3 M2 gate watcher: at each pre-registered checkpoint step, export the
# trainer checkpoint and submit the fwdseq endpoint discriminator (n=32,
# certified in-mixture anchors) for both arms plus a LIBERO-goal guardrail
# (real arm only).  Run detached: nohup bash m3_gate_watch.sh > watch.log 2>&1 &
set -u

RUNS_ROOT=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs
ARMS=("vlajepa_m3_retro_m2" "vlajepa_m3_retro_m2_priorread")
TAGS=("live" "priorread")
GATE_STEPS=(2500 5000 7500 10000)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/VLA_JEPA/bin/python
ACCOUNT=$(sacctmgr -nP show assoc user="$USER" format=account | grep -m1 edgeai_tao)
POLL_SECONDS=300
DEADLINE=$(( $(date +%s) + 20*3600 ))
GATE_DATASETS="mikasa_shell_game_shuffle_color_lamp_touch_vla_v0,mikasa_gather_and_recall_5_vla_v0,mikasa_batteries_checker_easy_3_vla_v0,mikasa_chain_of_colors_7_vla_v0"

submit_fwdseq() {
    local run_dir=$1 tag=$2 step=$3 ckpt=$4
    mkdir -p "$run_dir/gates"
    sbatch --account="$ACCOUNT" --partition=batch --gres=gpu:1 \
        --job-name="vj-m3-gate-${tag}-${step}" --time=01:30:00 --cpus-per-task=16 \
        --output="$run_dir/gates/fwdseq_step${step}_%j.out" \
        --wrap="cd $REPO && PYTHONPATH=$REPO $PY scripts/analysis/memv2_fwdseq_disc.py \
            --config $run_dir/config.yaml --ckpt $ckpt --num-segments 32 \
            --datasets $GATE_DATASETS \
            --out $run_dir/gates/fwdseq_step${step}.json"
}

submit_guardrail() {
    local step=$1 ckpt=$2
    cd "$REPO"
    export CKPT="$ckpt" NUM_TRIALS=20 SUITES=libero_goal
    export LIBERO_HOME_OVERRIDE=/lustre/fsw/portfolios/edgeai/users/chrislin/memexp_stage/libero-mem-home
    sbatch --export=ALL --job-name="vj-m3-guard-${step}" cluster/eval_libero_vlajepa.sbatch
}

for step in "${GATE_STEPS[@]}"; do
    for i in 0 1; do
        run_dir="$RUNS_ROOT/${ARMS[$i]}"; tag="${TAGS[$i]}"
        marker="$run_dir/gates/.exported_step${step}"
        [ -f "$marker" ] && continue
        while [ ! -f "$run_dir/checkpoints/step_${step}/model.safetensors" ]; do
            if [ "$(date +%s)" -gt "$DEADLINE" ]; then
                echo "[$(date -Is)] deadline waiting for ${tag} step_${step}; exiting"
                exit 1
            fi
            newer=$(ls -d "$run_dir"/checkpoints/step_* 2>/dev/null \
                | grep -oE '[0-9]+$' | sort -n | awk -v s="$step" '$1 > s' | head -1)
            if [ -n "$newer" ]; then
                echo "[$(date -Is)] ${tag} step_${step} pruned; skipping gate"
                mkdir -p "$run_dir/gates"; touch "$marker"
                continue 2
            fi
            sleep "$POLL_SECONDS"
        done
        ckpt="$run_dir/checkpoints/VLA-JEPA-m3-${tag}-step_${step}.pt"
        echo "[$(date -Is)] exporting ${tag} step_${step}"
        "$PY" "$REPO/scripts/analysis/memv2_export.py" "$run_dir" "step_${step}" "$(basename "$ckpt")" \
            || { echo "[$(date -Is)] export failed for ${tag} step_${step}"; continue; }
        mkdir -p "$run_dir/gates"; touch "$marker"
        submit_fwdseq "$run_dir" "$tag" "$step" "$ckpt"
        if [ "$tag" = "live" ] && { [ "$step" = "2500" ] || [ "$step" = "5000" ]; }; then
            submit_guardrail "$step" "$ckpt"
        fi
    done
done
echo "[$(date -Is)] all gate steps handled"
