#!/usr/bin/env bash
set -o pipefail

export PATH="$HOME/.local/bin:$PATH"

repo_dir=$(cd "$(dirname "$0")/../../.." && pwd)
stamp=${1:-$(date +%Y%m%d-%H%M%S)}
run_dir="$repo_dir/out/conical_journal/studies/paper-reproduction/overnight-$stamp"

[[ ! -e "$run_dir" ]] || { printf 'error: output exists: %s\n' "$run_dir" >&2; exit 2; }
mkdir -p "$run_dir"
exec > >(tee -a "$run_dir/overnight.log") 2>&1

source /opt/openfoam14/etc/bashrc

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$repo_dir"
printf 'run directory: %s\n' "$run_dir"
printf 'commit: %s\n' "$(git rev-parse HEAD)"
printf 'OpenFOAM: %s-%s\n' "$WM_PROJECT" "$WM_PROJECT_VERSION"
printf 'logical CPUs: %s\n' "$(getconf _NPROCESSORS_ONLN)"

three_d_status=0
uv run --frozen python -m studies.conical_journal.paper_reproduction.three_d \
  --outdir "$run_dir/three-d" \
  --jobs 4 \
  --mpi-ranks 8 \
  2>&1 | tee "$run_dir/three-d.log" || three_d_status=${PIPESTATUS[0]}

(
  uv run --frozen python studies/conical_journal/paper_reproduction/run.py \
    figure6 \
    --outdir "$run_dir/fixed-eccentricity-sensitivity" \
    --n-theta 512 \
    --n-axial 160 \
    --max-revolutions 24 \
    --jobs 12 \
    --eccentricity-ratios 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 \
    2>&1 | tee "$run_dir/fixed-eccentricity-sensitivity.log"
) &
eccentricity_pid=$!

(
  uv run --frozen python -m studies.conical_journal.paper_reproduction.section4 \
    --outdir "$run_dir/section4" \
    --models reynolds \
    --seed-jobs 8 \
    --jobs 28 \
    --max-revolutions 24 \
    2>&1 | tee "$run_dir/section4.log"
) &
section4_pid=$!

wait "$eccentricity_pid"
eccentricity_status=$?
wait "$section4_pid"
section4_status=$?

printf 'three_d_status=%s\neccentricity_status=%s\nsection4_status=%s\n' \
  "$three_d_status" "$eccentricity_status" "$section4_status" \
  | tee "$run_dir/status.txt"

(( three_d_status == 0 && eccentricity_status == 0 && section4_status == 0 ))
