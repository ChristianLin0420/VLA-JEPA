#!/bin/bash
# LIBERO evaluation for VLA-JEPA cotrain checkpoints on THIS cluster.
# Model server runs in the VLA_JEPA env; the LIBERO simulator runs in the
# dedicated vlajepa_eval env (clone of anamnesis: has libero+robosuite+mujoco).
# Export a checkpoint first:
#   python cluster/export_vlajepa_ckpt.py <run_dir> <step_dir> <name>.pt
# Then:  CKPT=/abs/path.pt NUM_TRIALS=10 bash examples/LIBERO/eval_libero_vlajepa.sh
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/cache
# Headless GL backend (override with MUJOCO_GL=glx if EGL fails on the node).
export MUJOCO_GL=${MUJOCO_GL:-egl}
if [[ "$MUJOCO_GL" == "egl" ]]; then
  # The cluster ships ONLY mesa's EGL vendor json; mesa EGL fails headless on GPU.
  # Provide a custom NVIDIA EGL ICD and force glvnd to use it (verified via gl_probe4).
  _nv_egl=$(ls /usr/share/glvnd/egl_vendor.d/*nvidia*.json 2>/dev/null | head -1)
  if [[ -z "$_nv_egl" ]]; then
    _nv_egl=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/egl_vendor/10_nvidia.json
    mkdir -p "$(dirname "$_nv_egl")"
    [[ -f "$_nv_egl" ]] || printf '%s\n' '{ "file_format_version":"1.0.0", "ICD":{"library_path":"libEGL_nvidia.so.0"} }' > "$_nv_egl"
  fi
  export __EGL_VENDOR_LIBRARY_FILENAMES="$_nv_egl"
  export PYOPENGL_PLATFORM=egl
fi

# Resolve the repo from this script's location so worktree checkouts eval their own code.
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
export LIBERO_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/anamnesis_stage/LIBERO
# LIBERO reads $LIBERO_CONFIG_PATH/config.yaml; the valid (ANAMNESIS-staged) config
# lives at ~/.libero/config.yaml. Pointing here avoids LIBERO's first-run input() prompt
# (which EOFErrors under batch). Seed it if somehow absent.
export LIBERO_CONFIG_PATH="${HOME}/.libero"
if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  mkdir -p "${LIBERO_CONFIG_PATH}"
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<YAML
benchmark_root: ${LIBERO_HOME}/libero/libero
bddl_files: ${LIBERO_HOME}/libero/libero/bddl_files
init_states: ${LIBERO_HOME}/libero/libero/init_files
datasets: ${LIBERO_HOME}/libero/datasets
assets: ${LIBERO_HOME}/libero/libero/assets
YAML
fi
# Do not append the login shell's stale /tmp/LIBERO path: that regular package
# shadows the staged namespace and makes `from libero.libero ...` fail.
export PYTHONPATH=${REPO}:${LIBERO_HOME}

CONDA=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs
server_python=${SERVER_PY:-$CONDA/VLA_JEPA/bin/python}     # serves the VLA-JEPA model
sim_python=${SIM_PY:-$CONDA/vlajepa_eval/bin/python}        # runs LIBERO simulator

CKPT=${CKPT:-/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/vlajepa_cotrain/checkpoints/VLA-JEPA-cotrain-step2000.pt}
folder_name=$(basename "$CKPT" .pt)
read -r -a items <<< "${TASKS:-libero_10 libero_goal libero_object libero_spatial}"
host=127.0.0.1; base_port=${BASE_PORT:-15083}
unnorm_key=${UNNORM_KEY:-franka}; with_state=${WITH_STATE:-true}
num_trials_per_task=${NUM_TRIALS:-10}

echo "[eval] ckpt=$CKPT suites=${items[*]} trials/task=$num_trials_per_task unnorm_key=$unnorm_key"
server_pids=(); sim_pids=()
cleanup() {
  if ((${#sim_pids[@]})); then kill "${sim_pids[@]}" 2>/dev/null || true; fi
  if ((${#server_pids[@]})); then kill "${server_pids[@]}" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM
index=0
for task in "${items[@]}"; do
  index=$((index+1)); port=$((base_port+index))
  video_out="results/${task}/${folder_name}"; mkdir -p "$video_out"
  echo "[eval] $task -> GPU $index port $port -> $video_out"
  $server_python ./deployment/model_server/server_policy.py \
      --ckpt_path "$CKPT" --port "$port" --use_bf16 --cuda "$index" \
      > "$video_out/server.log" 2>&1 &
  server_pids+=($!)
  sleep 30   # give the server time to load the 7.5GB model before the sim connects
  $sim_python ./examples/LIBERO/eval_libero.py \
      --args.pretrained-path "$CKPT" --args.host "$host" --args.port "$port" \
      --args.task-suite-name "$task" --args.num-trials-per-task "$num_trials_per_task" \
      --args.video-out-path "$video_out" --args.with_state "$with_state" \
      --args.unnorm-key "$unnorm_key" \
      > "$video_out/eval.log" 2>&1 &
  sim_pids+=($!)
done
echo "[eval] launched ${#items[@]} suites; waiting on sims..."
sim_status=0
for pid in "${sim_pids[@]}"; do
  if ! wait "$pid"; then sim_status=1; fi
done
sim_pids=()
echo "[eval] sims done; stopping model servers"
kill "${server_pids[@]}" 2>/dev/null || true
server_pids=()
if ((sim_status != 0)); then
  echo "[eval] one or more LIBERO suites failed" >&2
  exit "$sim_status"
fi
echo "[eval] all suites done; success rates in results/*/${folder_name}/eval.log"
