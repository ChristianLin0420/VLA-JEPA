#!/bin/bash
#SBATCH --job-name=vlajepa-glprobe2
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=outputs/slurm/glprobe2_%j.out
#SBATCH --error=outputs/slurm/glprobe2_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA}"
mkdir -p outputs/slurm
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/cache
export LIBERO_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/anamnesis_stage/LIBERO
export LIBERO_CONFIG_PATH="${HOME}/.libero"
export PYTHONPATH="$PWD:$LIBERO_HOME:${PYTHONPATH:-}"
SIM=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/vlajepa_eval/bin/python

echo "=== node $(hostname) ==="
echo "=== glvnd egl vendor jsons ==="
ls -1 /usr/share/glvnd/egl_vendor.d/ 2>/dev/null
NV_JSON=$(ls /usr/share/glvnd/egl_vendor.d/*nvidia*.json 2>/dev/null | head -1)
echo "nvidia egl json: ${NV_JSON:-NONE}"

echo "=== (A) EGL forced to NVIDIA vendor : robosuite render ==="
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES="$NV_JSON" "$SIM" -c "
import robosuite as suite
e=suite.make('Lift',robots='Panda',has_renderer=False,has_offscreen_renderer=True,use_camera_obs=True,camera_names='agentview',camera_heights=128,camera_widths=128)
o=e.reset();print('EGL_NV_ROBOSUITE_OK',o['agentview_image'].shape)
" 2>&1 | tail -5

echo "=== (B) GLX : robosuite render ==="
MUJOCO_GL=glx "$SIM" -c "
import robosuite as suite
e=suite.make('Lift',robots='Panda',has_renderer=False,has_offscreen_renderer=True,use_camera_obs=True,camera_names='agentview',camera_heights=128,camera_widths=128)
o=e.reset();print('GLX_ROBOSUITE_OK',o['agentview_image'].shape)
" 2>&1 | tail -5

echo "=== (C) GLX : actual LIBERO env render ==="
MUJOCO_GL=glx "$SIM" -c "
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
bd=benchmark.get_benchmark_dict()['libero_object']()
tid=0; td=bd.get_task(tid)
bp=get_libero_path('bddl_files')
import os
ef=os.path.join(bp, td.problem_folder, td.bddl_file)
env=OffScreenRenderEnv(bddl_file_name=ef, camera_heights=128, camera_widths=128)
env.seed(0); o=env.reset(); print('LIBERO_GLX_OK', o['agentview_image'].shape)
" 2>&1 | tail -8
echo "GLPROBE2_DONE"
