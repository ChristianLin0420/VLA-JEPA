#!/bin/bash
#SBATCH --job-name=vj-bench-maniskill-smoke
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --output=/lustre/fsw/portfolios/edgeai/users/chrislin/memexp_stage/logs/%x-%j.out
# Track B: SAPIEN/Vulkan headless render gate for ManiSkill3 (MIKASA-Robo-VLA + RoboMME).
# Tries several Vulkan ICD configurations; PASS = any attempt renders a non-black frame.
set -uo pipefail

STAGE=/lustre/fsw/portfolios/edgeai/users/chrislin/memexp_stage
PY=$STAGE/envs/maniskill/bin/python
SMOKE=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA/outputs/wt-memexp/cluster/maniskill_vulkan_smoke.py
OUTDIR=$STAGE/maniskill-smoke
mkdir -p "$OUTDIR"

echo "=== node: $(hostname) job: ${SLURM_JOB_ID:-none} ==="
nvidia-smi -L || true
echo "--- Vulkan ICD inventory ---"
ls -la /usr/share/vulkan/icd.d/ 2>&1 || true
ls -la /etc/vulkan/icd.d/ 2>&1 || true
ls -la /usr/share/glvnd/egl_vendor.d/ 2>&1 || true
for f in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
  [ -f "$f" ] && { echo "--- $f ---"; cat "$f"; }
done
echo "--- nvidia vulkan libs ---"
ldconfig -p | grep -iE "vulkan|libGLX_nvidia|nvidia_gpu" || true
command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>&1 | head -40 || echo "(no vulkaninfo binary)"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/cache
export MS_ASSET_DIR=$STAGE/maniskill_assets
export CUDA_VISIBLE_DEVICES=0    # 8-GPU node, use 1 GPU
unset DISPLAY

# Fallback ICD in case the node only ships a mesa one (mirrors the eval.sh EGL trick).
CUSTOM_ICD=$STAGE/vulkan_icd/nvidia_icd.json
mkdir -p "$(dirname "$CUSTOM_ICD")"
cat > "$CUSTOM_ICD" <<'JSON'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.194"
    }
}
JSON

run_attempt () {
  local name=$1; shift
  local simbk=$1; shift
  echo ""
  echo "=========== ATTEMPT $name (sim_backend=$simbk) ==========="
  echo "extra env: $*"
  if env "$@" "$PY" "$SMOKE" --out "$OUTDIR/frame_$name.png" --sim-backend "$simbk" > "$OUTDIR/attempt_$name.log" 2>&1; then
    echo "ATTEMPT $name: PASS"
    sed -n '1,60p' "$OUTDIR/attempt_$name.log"
    cp -f "$OUTDIR/frame_$name.png" "$OUTDIR/frame.png" 2>/dev/null || true
    { echo "$name"; echo "sim_backend=$simbk"; printf 'env: %s\n' "$*"; } > "$OUTDIR/PASS_CONFIG.txt"
    return 0
  else
    echo "ATTEMPT $name: FAIL (exit $?)"
    tail -25 "$OUTDIR/attempt_$name.log"
    return 1
  fi
}

PASS=0
SYS_ICD=""
for f in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
  [ -f "$f" ] && SYS_ICD=$f && break
done

# A: stock environment, no Vulkan overrides
run_attempt A_default auto && PASS=1

# B: pin the system NVIDIA ICD (both old and new env-var names)
if [ "$PASS" -eq 0 ] && [ -n "$SYS_ICD" ]; then
  run_attempt B_system_icd auto VK_ICD_FILENAMES="$SYS_ICD" VK_DRIVER_FILES="$SYS_ICD" && PASS=1
fi

# C: our custom ICD json (libGLX_nvidia.so.0)
if [ "$PASS" -eq 0 ]; then
  run_attempt C_custom_icd auto VK_ICD_FILENAMES="$CUSTOM_ICD" VK_DRIVER_FILES="$CUSTOM_ICD" && PASS=1
fi

# D: custom ICD + explicit CPU sim backend (isolates Vulkan render from PhysX-GPU issues)
if [ "$PASS" -eq 0 ]; then
  run_attempt D_cpu_sim cpu VK_ICD_FILENAMES="$CUSTOM_ICD" VK_DRIVER_FILES="$CUSTOM_ICD" && PASS=1
fi

echo ""
if [ "$PASS" -eq 1 ]; then
  echo "VERDICT: PASS ($(head -1 "$OUTDIR/PASS_CONFIG.txt"))"
  exit 0
else
  echo "VERDICT: FAIL (all attempts)"
  exit 1
fi
