#!/bin/bash
#SBATCH --job-name=vlajepa-glprobe3
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=outputs/slurm/glprobe3_%j.out
#SBATCH --error=outputs/slurm/glprobe3_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA}"
mkdir -p outputs/slurm
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/cache
export LIBERO_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/anamnesis_stage/LIBERO
export LIBERO_CONFIG_PATH="${HOME}/.libero"
export PYTHONPATH="$PWD:$LIBERO_HOME:${PYTHONPATH:-}"
SIM=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/vlajepa_eval/bin/python

# Create a custom NVIDIA EGL vendor ICD json (system only ships mesa's).
NVDIR=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/egl_vendor
mkdir -p "$NVDIR"
cat > "$NVDIR/10_nvidia.json" <<'JSON'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
JSON
export __EGL_VENDOR_LIBRARY_FILENAMES="$NVDIR/10_nvidia.json"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
echo "=== node $(hostname); vendor json forced: $__EGL_VENDOR_LIBRARY_FILENAMES ==="
cat "$NVDIR/10_nvidia.json"

echo "=== (A) EGL+nvidia-json : mujoco.GLContext ==="
"$SIM" -c "import mujoco; c=mujoco.GLContext(128,128); c.make_current(); print('GLCTX_OK egl-nv')" 2>&1 | tail -4

echo "=== (B) EGL+nvidia-json : robosuite render ==="
"$SIM" -c "
import robosuite as suite
e=suite.make('Lift',robots='Panda',has_renderer=False,has_offscreen_renderer=True,use_camera_obs=True,camera_names='agentview',camera_heights=128,camera_widths=128)
o=e.reset();print('EGL_NV_ROBOSUITE_OK',o['agentview_image'].shape)
" 2>&1 | tail -5

echo "=== (C) EGL+nvidia-json : LIBERO env render ==="
"$SIM" -c "
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os
bd=benchmark.get_benchmark_dict()['libero_object']()
td=bd.get_task(0); bp=get_libero_path('bddl_files')
ef=os.path.join(bp, td.problem_folder, td.bddl_file)
env=OffScreenRenderEnv(bddl_file_name=ef, camera_heights=128, camera_widths=128)
env.seed(0); o=env.reset(); print('LIBERO_EGL_NV_OK', o['agentview_image'].shape)
" 2>&1 | tail -8
echo "GLPROBE3_DONE"
