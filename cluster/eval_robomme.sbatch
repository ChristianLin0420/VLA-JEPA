#!/bin/bash
#SBATCH --job-name=vlajepa-robomme-eval
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/vlajepa_robomme_eval_%j.out
#SBATCH --error=outputs/slurm/vlajepa_robomme_eval_%j.err
set -uo pipefail
# RoboMME (arXiv 2603.04639, ICML 2026 Spotlight) eval for a VLA-JEPA checkpoint.
# Modeled on cluster/eval_libero_mem.sbatch:
#   - model server (VLA_JEPA conda env) on cuda:1; RoboMME/ManiSkill3 sim client
#     (robomme uv venv) renders via Vulkan on GPU 0 -> 2 GPUs.
#   - MEMORY_MODE & friends pass through to the model server via --export=ALL.
#   - SAPIEN/Vulkan needs NO ICD overrides on these nodes (Track B gate,
#     jobs 6068718/6069759); do not add VK_ICD_FILENAMES here.
# Benchmark protocol (fixed seeds baked into the repo's env_metadata):
#   splits train=100 / val=50 / test=50 episodes per task; official eval =
#   DATASET=test, NUM_EPISODES=50 per task, MAX_STEPS=1300 (MME-VLA setting).
# Usage:
#   sbatch --export=ALL,CKPT=/abs/x.pt,TASKS=all,MEMORY_MODE=live cluster/eval_robomme.sbatch
# Smoke (<30 min, lead-free):
#   sbatch --job-name=vj-bench-robomme-eval-smoke --time=00:30:00 \
#     --export=ALL,CKPT=/abs/x.pt,TASKS=PickHighlight,NUM_EPISODES=2,MAX_STEPS=200 \
#     cluster/eval_robomme.sbatch
# Discriminator arms additionally set e.g.:
#   MEMORY_MODE=foreign,MEMORY_DONOR_DIR=/abs/donor_bank/...   (foreign)
#   MEMORY_MODE=bypass                                          (bypass)
REPO="${SLURM_SUBMIT_DIR:-/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA/outputs/wt-memexp}"
cd "$REPO"; mkdir -p outputs/slurm
STAGE=/lustre/fsw/portfolios/edgeai/users/chrislin/memexp_stage
CONDA=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs
server_python=${SERVER_PY:-$CONDA/VLA_JEPA/bin/python}         # serves the VLA-JEPA model
sim_python=${SIM_PY:-$STAGE/robomme/.venv/bin/python}          # runs RoboMME/ManiSkill3

export HF_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/cache
export MS_ASSET_DIR=$STAGE/maniskill_assets
export PYTHONPATH=$REPO
unset DISPLAY

CKPT=${CKPT:?set CKPT=/abs/path.pt (export a checkpoint via cluster/export_vlajepa_ckpt.py)}
TASKS=${TASKS:-all}                      # comma-separated RoboMME task ids or "all" (16 tasks)
DATASET=${DATASET:-test}                 # train / val / test (fixed 100/50/50 seed splits)
NUM_EPISODES=${NUM_EPISODES:-50}         # official test protocol: 50 episodes per task
EPISODE_START=${EPISODE_START:-0}
MAX_STEPS=${MAX_STEPS:-1300}             # MME-VLA experiments cap
ACTION_MODE=${ACTION_MODE:-delta_to_abs} # delta_to_abs (LIBERO ckpts) | absolute (robomme-FT ckpts)
POS_SCALE=${POS_SCALE:-0.02}             # m per unit delta; >0.03 makes their IK (1mm tol) fail
ROT_SCALE=${ROT_SCALE:-0.1}              # rad per unit delta
UNNORM_KEY=${UNNORM_KEY:-franka}
WITH_STATE=${WITH_STATE:-true}
INGEST_CONDITIONING=${INGEST_CONDITIONING:-true}
export MEMORY_MODE="${MEMORY_MODE:-live}"

host=127.0.0.1; port=${BASE_PORT:-15184}
folder_name=$(basename "$CKPT" .pt)
video_out="results/robomme_${DATASET}/${folder_name}"; mkdir -p "$video_out"

echo "[robomme-eval] CKPT=$CKPT TASKS=$TASKS DATASET=$DATASET eps/task=$NUM_EPISODES max_steps=$MAX_STEPS"
echo "[robomme-eval] MEMORY_MODE=$MEMORY_MODE DONOR_DIR=${MEMORY_DONOR_DIR:-} STATE_DUMP_DIR=${MEMORY_STATE_DUMP_DIR:-}"
echo "[robomme-eval] action_mode=$ACTION_MODE ingest_conditioning=$INGEST_CONDITIONING -> $video_out"

server_pid=""; sim_pid=""
cleanup() {
  [[ -n "$sim_pid" ]] && kill "$sim_pid" 2>/dev/null || true
  [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

$server_python ./deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT" --port "$port" --use_bf16 --cuda 1 \
    > "$video_out/server.log" 2>&1 &
server_pid=$!
sleep 30   # give the server time to load the 7.5GB model before the sim connects

ingest_flag="--args.ingest-conditioning"
[[ "$INGEST_CONDITIONING" != "true" ]] && ingest_flag="--args.no-ingest-conditioning"

$sim_python ./examples/ROBOMME/eval_robomme.py \
    --args.pretrained-path "$CKPT" --args.host "$host" --args.port "$port" \
    --args.tasks "$TASKS" --args.dataset "$DATASET" \
    --args.num-episodes "$NUM_EPISODES" --args.episode-start "$EPISODE_START" \
    --args.max-steps "$MAX_STEPS" --args.action-mode "$ACTION_MODE" \
    --args.pos-scale "$POS_SCALE" --args.rot-scale "$ROT_SCALE" \
    --args.unnorm-key "$UNNORM_KEY" --args.with-state "$WITH_STATE" \
    "$ingest_flag" \
    --args.video-out-path "$video_out" \
    > "$video_out/eval.log" 2>&1 &
sim_pid=$!
wait "$sim_pid"; status=$?
sim_pid=""
kill "$server_pid" 2>/dev/null || true; server_pid=""

echo "[$(date -Is)] eval finished (status=$status)"
grep -E "success rate|Total episodes" "$video_out/eval.log" | tail -20 || true
tail -5 "$video_out/server.log" || true
exit "$status"
