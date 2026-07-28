#!/usr/bin/env bash
set -eo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
openfoam_dir="$repo_dir/../../.openfoam/OpenFOAM-14"
case_dir=$(mktemp -d "${TMPDIR:-/tmp}/jfoBearingFoam-check.XXXXXX")

cleanup()
{
    if [[ -n "${case_dir:-}" && -d "$case_dir" ]]; then
        rm -rf -- "$case_dir"
    fi
}
trap cleanup EXIT

source "$openfoam_dir/etc/bashrc"
set -u
export FOAM_USER_APPBIN="$repo_dir/build/openfoam/bin"

(
    cd "$repo_dir/openfoam/jfoBearingFoam"
    wmake
)

cp -a "$repo_dir/openfoam/cases/jfoPaperExact/." "$case_dir/"
foamDictionary "$case_dir/constant/jfoProperties" -entry rpm -set 0
blockMesh -case "$case_dir" >/dev/null
solver_output=$("$FOAM_USER_APPBIN/jfoBearingFoam" -case "$case_dir")

if [[ "$solver_output" != *"JFO_RESULT converged=true accepted=true rpm=0"* ]]; then
    printf '%s\n' "$solver_output"
    exit 1
fi

printf '%s\n' "$solver_output" | sed -n '/JFO_RESULT/p'
