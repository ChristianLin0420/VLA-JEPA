#!/bin/bash
#SBATCH --job-name=vlajepa-mikasa-eval
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/vlajepa_mikasa_eval_%j.out
#SBATCH --error=outputs/slurm/vlajepa_mikasa_eval_%j.err
set -uo pipefail
# MIKASA-Robo-VLA (arXiv 2502.10550, github.com/CognitiveAISystems/MIKASA-Robo)
# eval for a VLA-JEPA checkpoint. Modeled on cluster/eval_libero_mem.sbatch:
#   - model server (VLA_JEPA conda env) on cuda:1; ManiSkill3/SAPIEN sim +
#     Vulkan render (memexp_stage/envs/maniskill uv env) on GPU 0
#   - Vulkan needs ZERO ICD overrides on our H100 nodes (Track B PASS config
#     A_default, jobs 6068718/6069759); do NOT set VK_ICD_FILENAMES here
#   - MEMORY_MODE & friends pass through to the model server via --export=ALL
# Usage:
#   sbatch --export=ALL,CKPT=/abs/x.pt,NUM_TRIALS=50,MEMORY_MODE=live \
#       cluster/eval_mikasa.sbatch
# NOTE: sbatch --export splits on commas, so a multi-task ENV_IDS list must be
# exported from the shell instead:
#   export ENV_IDS="ShellGameTouch-VLA-v0,RememberColor3-VLA-v0"
#   sbatch --export=ALL,CKPT=/abs/x.pt cluster/eval_mikasa.sbatch
# Smoke (2 eps, 1 task, <30 min; job names must be vj-bench-*):
#   sbatch --job-name=vj-bench-mikasa-smoke --time=00:30:00 \
#       --export=ALL,ENV_IDS=ShellGameTouch-VLA-v0,NUM_TRIALS=2,MEMORY_MODE=live,CKPT=/abs/x.pt \
#       cluster/eval_mikasa.sbatch
# Discriminator arms additionally set e.g.:
#   MEMORY_MODE=foreign,MEMORY_DONOR_DIR=/abs/donor_bank/...   (foreign)
#   MEMORY_MODE=bypass                                          (bypass)
REPO="${SLURM_SUBMIT_DIR:-/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA/outputs/wt-memexp}"
cd "$REPO"; mkdir -p outputs/slurm

STAGE=/lustre/fsw/portfolios/edgeai/users/chrislin/memexp_stage
CONDA=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs
SERVER_PY=${SERVER_PY:-$CONDA/VLA_JEPA/bin/python}          # serves the VLA-JEPA model
SIM_PY=${SIM_PY:-$STAGE/envs/maniskill/bin/python}          # ManiSkill3 + mikasa_robo_suite (editable from $STAGE/mikasa)

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/cache
export HF_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# ManiSkill asset dir staged by the Track B smoke (compute nodes are offline).
export MS_ASSET_DIR=$STAGE/maniskill_assets
# Headless SAPIEN/Vulkan: works with no ICD overrides on these nodes (Track B).
unset DISPLAY

CKPT=${CKPT:-/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/vlajepa_memv1_cotrain/checkpoints/VLA-JEPA-memv1-live-step_100000.pt}
ENV_IDS=${ENV_IDS:-ShellGameTouch-VLA-v0,RememberColor3-VLA-v0,TakeItBack-VLA-v0,ChainOfColors3-VLA-v0}
NUM_TRIALS=${NUM_TRIALS:-50}
START_SEED=${START_SEED:-4242424242}      # canonical MIKASA benchmark START_SEED
SIM_BACKEND=${SIM_BACKEND:-auto}          # 'auto' = CPU sim + Vulkan GPU render at num_envs=1
RENDER_MODE=${RENDER_MODE:-none}          # 'none' keeps the human render camera off
MAX_STEPS_CAP=${MAX_STEPS_CAP:-0}
WITH_STATE=${WITH_STATE:-true}
UNNORM_KEY=${UNNORM_KEY:-new_embodiment}
# Job-unique port: co-scheduled eval jobs on one node must not collide
# (the client-server mode handshake aborts on a mismatched server).
BASE_PORT=${BASE_PORT:-$((15183 + SLURM_JOB_ID % 997))}
export MEMORY_MODE="${MEMORY_MODE:-live}"

host=127.0.0.1
port=$BASE_PORT
folder_name=$(basename "$CKPT" .pt)
video_out="results/mikasa/${folder_name}"; mkdir -p "$video_out"

echo "[mikasa-eval] CKPT=$CKPT ENV_IDS=$ENV_IDS NUM_TRIALS=$NUM_TRIALS"
echo "[mikasa-eval] MEMORY_MODE=$MEMORY_MODE DONOR_DIR=${MEMORY_DONOR_DIR:-} STATE_DUMP_DIR=${MEMORY_STATE_DUMP_DIR:-}"
echo "[mikasa-eval] sim_backend=$SIM_BACKEND render_mode=$RENDER_MODE start_seed=$START_SEED -> $video_out"

server_pid=""
cleanup() { [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

export PYTHONPATH=$REPO

# Preflight (compute node; login-node imports are too slow to be useful):
# verify the 90 *-VLA-v0 envs register and the anchor tasks are present.
if [[ "${PREFLIGHT:-1}" == "1" ]]; then
  $SIM_PY - <<'PYEOF' 2>&1 | tee "$video_out/preflight.log"
import time; t0 = time.time()
import mikasa_robo_suite.vla.memory_envs  # registers VLA env IDs
import gymnasium as gym
ids = sorted(i for i in gym.registry.keys() if i.endswith("-VLA-v0"))
anchors = [i for i in ids if any(k in i for k in ("ShellGame", "RememberColor", "TakeItBack", "ChainOfColors"))]
print(f"[preflight] {len(ids)} VLA envs registered after {time.time()-t0:.0f}s")
print(f"[preflight] anchors ({len(anchors)}):", *anchors, sep="\n  ")
from mikasa_robo_suite.vla.utils.apply_wrappers import MIKASA_VLA_ENV_IDS
from mikasa_robo_suite.vla.benchmarking import load_benchmark_tasks
assert len(MIKASA_VLA_ENV_IDS) == 90, len(MIKASA_VLA_ENV_IDS)
assert len(load_benchmark_tasks()) == 90
print("[preflight] REGISTRATION_OK")
PYEOF
  grep -q "REGISTRATION_OK" "$video_out/preflight.log" || { echo "[mikasa-eval] FATAL: env registration preflight failed"; exit 1; }
fi

$SERVER_PY ./deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT" --port "$port" --use_bf16 --cuda 1 \
    > "$video_out/server.log" 2>&1 &
server_pid=$!
sleep 30   # let the 7.5GB model start loading; the client retries for up to 600s

if ! kill -0 "$server_pid" 2>/dev/null; then
  echo "[mikasa-eval] FATAL: model server died during startup"; tail -30 "$video_out/server.log"; exit 1
fi

$SIM_PY ./examples/MIKASA/eval_mikasa.py \
    --args.pretrained-path "$CKPT" --args.host "$host" --args.port "$port" \
    --args.env-ids "$ENV_IDS" --args.num-trials-per-task "$NUM_TRIALS" \
    --args.start-seed "$START_SEED" --args.sim-backend "$SIM_BACKEND" \
    --args.render-mode "$RENDER_MODE" --args.max-steps-cap "$MAX_STEPS_CAP" \
    --args.video-out-path "$video_out" --args.with-state "$WITH_STATE" \
    --args.unnorm-key "$UNNORM_KEY" \
    2>&1 | tee "$video_out/eval.log"
status=${PIPESTATUS[0]}

echo "[$(date -Is)] eval finished (status=$status); success rates:"
grep -iE "success rate" "$video_out/eval.log" | tail -10 || true
kill "$server_pid" 2>/dev/null || true
server_pid=""
exit "$status"
