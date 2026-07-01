#!/bin/bash
#SBATCH --job-name=vlajepa-glprobe
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=outputs/slurm/glprobe_%j.out
#SBATCH --error=outputs/slurm/glprobe_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA}"
mkdir -p outputs/slurm
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/cache
SIM=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/vlajepa_eval/bin/python

echo "=== node $(hostname) ==="
nvidia-smi -L 2>&1 | head
echo "=== compute-node GL libs ==="
ls -1 /usr/lib/x86_64-linux-gnu/libEGL*  2>/dev/null | head
ls -1 /usr/lib/x86_64-linux-gnu/libGLX*  2>/dev/null | head
ls -1 /lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/vlajepa_eval/lib/libOSMesa* 2>/dev/null | head

for gl in egl osmesa glx; do
  echo "=== MUJOCO_GL=$gl : mujoco.GLContext(128,128) ==="
  MUJOCO_GL=$gl PYOPENGL_PLATFORM=$gl "$SIM" -c "import mujoco; c=mujoco.GLContext(128,128); c.make_current(); print('GLCTX_OK', '$gl')" 2>&1 | tail -4
done

echo "=== robosuite offscreen render (MUJOCO_GL=egl) ==="
MUJOCO_GL=egl "$SIM" -c "
import robosuite as suite
env=suite.make('Lift', robots='Panda', has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, camera_names='agentview', camera_heights=128, camera_widths=128)
o=env.reset(); print('ROBOSUITE_RENDER_OK', o['agentview_image'].shape, o['agentview_image'].dtype)
" 2>&1 | tail -8
echo "GLPROBE_DONE"
