#!/bin/bash
set -eEuo pipefail

# Submit, exactly once, the memv1 training/evaluation dependency chain.
# This script intentionally has no resume/reuse mode: any pre-existing run
# directory or queued job name requires explicit operator reconciliation.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RUN_ROOT=/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs
BASELINE="$RUN_ROOT/vlajepa_cotrain_allv2/checkpoints/VLA-JEPA-allv2-step_100000.pt"

VIDEO_RUN=vlajepa_memv1_video
STAGE1_RUN=vlajepa_memv1_stage1
COTRAIN_RUN=vlajepa_memv1_cotrain

VIDEO_EXP=E20260630-memv1-video
STAGE1_EXP=E20260630-memv1-stage1
COTRAIN_EXP=E20260630-memv1-cotrain

VIDEO_CONFIG=./scripts/config/vlajepa_memv1_video.yaml
STAGE1_CONFIG=./scripts/config/vlajepa_memv1_stage1.yaml
COTRAIN_CONFIG=./scripts/config/vlajepa_memv1_cotrain.yaml

VIDEO_JOB_NAME=vj-memv1-video
STAGE1_JOB_NAME=vj-memv1-stage1
COTRAIN_JOB_NAME=vj-memv1-cotrain
EVAL_JOB_NAME=vj-memv1-eval

die() {
  echo "[fatal] $*" >&2
  exit 1
}

cd "$REPO_ROOT"
command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable"
command -v squeue >/dev/null 2>&1 || die "squeue is unavailable"

for path in \
  "$VIDEO_CONFIG" \
  "$STAGE1_CONFIG" \
  "$COTRAIN_CONFIG" \
  cluster/launch_vlajepa_video_8gpu.sbatch \
  cluster/launch_vlajepa_cotrain_8gpu.sbatch \
  cluster/eval_after_run.sbatch; do
  [[ -s "$path" ]] || die "required file is missing or empty: $path"
done
[[ -s "$BASELINE" ]] || die "allv2 100K parent checkpoint is missing: $BASELINE"
[[ -s "$RUN_ROOT/vlajepa_cotrain_allv2/config.yaml" ]] || die "baseline config is missing"
[[ -s "$RUN_ROOT/vlajepa_cotrain_allv2/dataset_statistics.json" ]] || {
  die "baseline dataset statistics are missing"
}

# The launch must describe committed code that is already on GitHub.
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || {
  die "worktree is not clean; commit before submitting"
}
GIT_SHA="$(git rev-parse --verify HEAD)"
ORIGIN_SHA="$(git rev-parse --verify origin/main)"
[[ "$GIT_SHA" == "$ORIGIN_SHA" ]] || {
  die "HEAD ($GIT_SHA) is not the locally known origin/main ($ORIGIN_SHA); push first"
}

for run_id in "$VIDEO_RUN" "$STAGE1_RUN" "$COTRAIN_RUN"; do
  [[ ! -e "$RUN_ROOT/$run_id" ]] || {
    die "run directory already exists and will not be reused: $RUN_ROOT/$run_id"
  }
done

for job_name in "$VIDEO_JOB_NAME" "$STAGE1_JOB_NAME" "$COTRAIN_JOB_NAME" "$EVAL_JOB_NAME"; do
  [[ -z "$(squeue -h --name="$job_name" -o '%i')" ]] || {
    die "a Slurm job named $job_name already exists"
  }
done

SHORT_SHA="${GIT_SHA:0:12}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PIPELINE_ID="memv1-${SHORT_SHA}-${STAMP}"
MANIFEST="outputs/slurm/vlajepa_${PIPELINE_ID}.manifest"
mkdir -p outputs/slurm
[[ ! -e "$MANIFEST" ]] || die "manifest already exists: $MANIFEST"

VIDEO_JOB=""
STAGE1_JOB=""
COTRAIN_JOB=""
EVAL_JOB=""
STATUS=intent
SUBMITTED_AT=""

VIDEO_CONFIG_SHA="$(sha256sum "$VIDEO_CONFIG" | awk '{print $1}')"
STAGE1_CONFIG_SHA="$(sha256sum "$STAGE1_CONFIG" | awk '{print $1}')"
COTRAIN_CONFIG_SHA="$(sha256sum "$COTRAIN_CONFIG" | awk '{print $1}')"

write_manifest() {
  local tmp="${MANIFEST}.tmp.$$"
  umask 077
  {
    echo "schema_version=1"
    echo "pipeline_id=$PIPELINE_ID"
    echo "status=$STATUS"
    echo "created_at=$STAMP"
    echo "submitted_at=$SUBMITTED_AT"
    echo "repo_root=$REPO_ROOT"
    echo "git_sha=$GIT_SHA"
    echo "origin_main_sha=$ORIGIN_SHA"
    echo "run_root=$RUN_ROOT"
    echo "baseline_checkpoint=$BASELINE"
    echo "video_run=$VIDEO_RUN"
    echo "video_config=$VIDEO_CONFIG"
    echo "video_config_sha256=$VIDEO_CONFIG_SHA"
    echo "video_job=$VIDEO_JOB"
    echo "stage1_run=$STAGE1_RUN"
    echo "stage1_config=$STAGE1_CONFIG"
    echo "stage1_config_sha256=$STAGE1_CONFIG_SHA"
    echo "stage1_parent=$RUN_ROOT/$VIDEO_RUN/final_model/pytorch_model.pt"
    echo "stage1_job=$STAGE1_JOB"
    echo "stage1_dependency=${VIDEO_JOB:+afterok:$VIDEO_JOB}"
    echo "cotrain_run=$COTRAIN_RUN"
    echo "cotrain_config=$COTRAIN_CONFIG"
    echo "cotrain_config_sha256=$COTRAIN_CONFIG_SHA"
    echo "cotrain_parent=$RUN_ROOT/$STAGE1_RUN/final_model/pytorch_model.pt"
    echo "cotrain_job=$COTRAIN_JOB"
    echo "cotrain_dependency=${STAGE1_JOB:+afterok:$STAGE1_JOB}"
    echo "eval_run=$RUN_ROOT/$COTRAIN_RUN"
    echo "eval_out_prefix=VLA-JEPA-memv1-live"
    echo "eval_with_state=false"
    echo "eval_memory_mode=live"
    echo "eval_job=$EVAL_JOB"
    echo "eval_dependency=${COTRAIN_JOB:+afterok:$COTRAIN_JOB}"
  } > "$tmp"
  mv "$tmp" "$MANIFEST"
}

on_error() {
  local rc=$?
  STATUS=failed
  write_manifest || true
  echo "[fatal] pipeline submission failed (exit=$rc); partial intent: $MANIFEST" >&2
  exit "$rc"
}
on_interrupt() {
  STATUS=interrupted
  write_manifest || true
  echo "[fatal] pipeline submission interrupted; partial intent: $MANIFEST" >&2
  exit 130
}
trap on_error ERR
trap on_interrupt INT TERM

parse_job_id() {
  local raw="$1"
  local job_id="${raw%%;*}"
  [[ "$job_id" =~ ^[0-9]+$ ]] || die "unexpected sbatch response: $raw"
  printf '%s\n' "$job_id"
}

write_manifest
STATUS=submitting
write_manifest

VIDEO_RAW="$(sbatch --parsable \
  --comment="vla-jepa-memv1:${SHORT_SHA}:video" \
  --job-name="$VIDEO_JOB_NAME" \
  --export="ALL,EXPECTED_GIT_SHA=$GIT_SHA,EXPECTED_CONFIG_SHA=$VIDEO_CONFIG_SHA,CONFIG=$VIDEO_CONFIG,RUN_ROOT=$RUN_ROOT,RUN_ID=$VIDEO_RUN,EXP_ID=$VIDEO_EXP,MICRO=8" \
  cluster/launch_vlajepa_video_8gpu.sbatch)"
VIDEO_JOB="$(parse_job_id "$VIDEO_RAW")"
write_manifest
echo "[submit] video=$VIDEO_JOB"

STAGE1_RAW="$(sbatch --parsable \
  --dependency="afterok:$VIDEO_JOB" \
  --comment="vla-jepa-memv1:${SHORT_SHA}:stage1" \
  --job-name="$STAGE1_JOB_NAME" \
  --export="ALL,EXPECTED_GIT_SHA=$GIT_SHA,EXPECTED_CONFIG_SHA=$STAGE1_CONFIG_SHA,CONFIG=$STAGE1_CONFIG,RUN_ROOT=$RUN_ROOT,RUN_ID=$STAGE1_RUN,EXP_ID=$STAGE1_EXP,MICRO=1" \
  cluster/launch_vlajepa_cotrain_8gpu.sbatch)"
STAGE1_JOB="$(parse_job_id "$STAGE1_RAW")"
write_manifest
echo "[submit] stage1=$STAGE1_JOB dependency=afterok:$VIDEO_JOB"

COTRAIN_RAW="$(sbatch --parsable \
  --dependency="afterok:$STAGE1_JOB" \
  --comment="vla-jepa-memv1:${SHORT_SHA}:cotrain" \
  --job-name="$COTRAIN_JOB_NAME" \
  --export="ALL,EXPECTED_GIT_SHA=$GIT_SHA,EXPECTED_CONFIG_SHA=$COTRAIN_CONFIG_SHA,CONFIG=$COTRAIN_CONFIG,RUN_ROOT=$RUN_ROOT,RUN_ID=$COTRAIN_RUN,EXP_ID=$COTRAIN_EXP,MICRO=1" \
  cluster/launch_vlajepa_cotrain_8gpu.sbatch)"
COTRAIN_JOB="$(parse_job_id "$COTRAIN_RAW")"
write_manifest
echo "[submit] cotrain=$COTRAIN_JOB dependency=afterok:$STAGE1_JOB"

EVAL_RAW="$(sbatch --parsable \
  --dependency="afterok:$COTRAIN_JOB" \
  --comment="vla-jepa-memv1:${SHORT_SHA}:eval" \
  --job-name="$EVAL_JOB_NAME" \
  --export="ALL,EXPECTED_GIT_SHA=$GIT_SHA,RUN=$RUN_ROOT/$COTRAIN_RUN,OUT_PREFIX=VLA-JEPA-memv1-live,WITH_STATE=false,MEMORY_MODE=live,NUM_TRIALS=10" \
  cluster/eval_after_run.sbatch)"
EVAL_JOB="$(parse_job_id "$EVAL_RAW")"

STATUS=submitted
SUBMITTED_AT="$(date -Is)"
write_manifest
trap - ERR INT TERM

echo "[submit] eval=$EVAL_JOB dependency=afterok:$COTRAIN_JOB"
echo "[submit] manifest=$MANIFEST"
printf 'video=%s stage1=%s cotrain=%s eval=%s\n' \
  "$VIDEO_JOB" "$STAGE1_JOB" "$COTRAIN_JOB" "$EVAL_JOB"
