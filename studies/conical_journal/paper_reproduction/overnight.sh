#!/usr/bin/env bash
set -uo pipefail

repo_dir=$(cd "$(dirname "$0")/../../.." && pwd)
stamp=${1:-$(date +%Y%m%d-%H%M%S)}
run_dir="$repo_dir/out/conical_journal/studies/paper-reproduction/overnight-$stamp"

[[ ! -e "$run_dir" ]] || { printf 'error: output exists: %s\n' "$run_dir" >&2; exit 2; }
mkdir -p "$run_dir"
exec > >(tee -a "$run_dir/overnight.log") 2>&1

set +u
source /opt/openfoam14/etc/bashrc
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$repo_dir"
printf 'run directory: %s\n' "$run_dir"
printf 'commit: %s\n' "$(git rev-parse HEAD)"
printf 'OpenFOAM: %s\n' "$(foamVersion)"
printf 'logical CPUs: %s\n' "$(getconf _NPROCESSORS_ONLN)"

three_d_status=0
uv run --frozen python studies/conical_journal/paper_reproduction/three_d.py \
  --outdir "$run_dir/three-d" \
  --jobs 4 \
  --mpi-ranks 8 \
  2>&1 | tee "$run_dir/three-d.log" || three_d_status=${PIPESTATUS[0]}

section4_status=0
uv run --frozen python studies/conical_journal/paper_reproduction/section4.py \
  --outdir "$run_dir/section4" \
  --seed-jobs 8 \
  --jobs 28 \
  --max-revolutions 24 \
  2>&1 | tee "$run_dir/section4.log" || section4_status=${PIPESTATUS[0]}

printf 'three_d_status=%s\nsection4_status=%s\n' \
  "$three_d_status" "$section4_status" | tee "$run_dir/status.txt"

(( three_d_status == 0 && section4_status == 0 ))
