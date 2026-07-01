#!/bin/bash
#SBATCH --job-name=vlajepa-glprobe4
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=outputs/slurm/glprobe4_%j.out
#SBATCH --error=outputs/slurm/glprobe4_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA}"
mkdir -p outputs/slurm
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/cache
export LIBERO_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/anamnesis_stage/LIBERO
export LIBERO_CONFIG_PATH="${HOME}/.libero"
export PYTHONPATH="$PWD:$LIBERO_HOME:${PYTHONPATH:-}"
SIM=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/vlajepa_eval/bin/python
NVDIR=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/egl_vendor
mkdir -p "$NVDIR"
printf '%s\n' '{ "file_format_version":"1.0.0", "ICD":{"library_path":"libEGL_nvidia.so.0"} }' > "$NVDIR/10_nvidia.json"

run_test () {  # $1=label  $2=extra-env-prefix
  echo "########## $1 ##########"
  env $2 "$SIM" - <<'PY' 2>err.tmp
import os, traceback
print("env MUJOCO_GL=", os.environ.get("MUJOCO_GL"), "VENDOR=", os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"))
def robos():
    import robosuite as suite
    e=suite.make('Lift',robots='Panda',has_renderer=False,has_offscreen_renderer=True,
                 use_camera_obs=True,camera_names='agentview',camera_heights=128,camera_widths=128)
    o=e.reset(); img=o['agentview_image']
    print(f"RESULT robosuite PASS shape={img.shape} sum={int(img.sum())}")
def libero():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bd=benchmark.get_benchmark_dict()['libero_object']()
    td=bd.get_task(0); ef=os.path.join(get_libero_path('bddl_files'), td.problem_folder, td.bddl_file)
    env=OffScreenRenderEnv(bddl_file_name=ef, camera_heights=128, camera_widths=128)
    env.seed(0); o=env.reset(); img=o['agentview_image']
    print(f"RESULT libero PASS shape={img.shape} sum={int(img.sum())}")
for name,fn in [("robosuite",robos),("libero",libero)]:
    try: fn()
    except BaseException as ex:
        print(f"RESULT {name} FAIL {type(ex).__name__}: {str(ex)[:200]}")
        traceback.print_exc()
PY
  echo "--- stderr tail ---"; tail -3 err.tmp 2>/dev/null; rm -f err.tmp
}

run_test "EGL + nvidia vendor json"  "MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=$NVDIR/10_nvidia.json"
run_test "OSMESA (LD_LIBRARY_PATH=env/lib)" "MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=/lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/vlajepa_eval/lib"
echo "GLPROBE4_DONE"
