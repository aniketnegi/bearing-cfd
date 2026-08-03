#!/usr/bin/env bash
set -euo pipefail

template_dir=$(cd "$(dirname "$0")" && pwd)
repo_dir=$(cd "$template_dir/../../../.." && pwd)
default_case="$repo_dir/out/conical_journal/simulation/openfoam-single-phase-oq90"
expected_mesh_sha="b75c2fdb0201de822018760d156c6c2d0aeac335a040841d6b37fcaa37923384"

usage()
{
    cat <<EOF
Usage:
  $0 prepare FLUENT_MESH [WORK_CASE]
  $0 atmospheric [WORK_CASE]
  $0 pressure-fed [WORK_CASE]
  $0 496p563rpm [WORK_CASE]
  $0 2000rpm [WORK_CASE]

Default WORK_CASE: $default_case
Run one stage at a time and inspect it before starting the next.
EOF
}

die()
{
    printf 'error: %s\n' "$*" >&2
    exit 1
}

require_command()
{
    command -v "$1" >/dev/null || die "required command not found: $1"
}

require_openfoam()
{
    local command_name
    for command_name in foamVersion fluentMeshToFoam checkMesh decomposePar \
        foamDictionary foamListTimes foamRun reconstructPar mpirun; do
        require_command "$command_name"
    done
    [[ $(foamVersion) == OpenFOAM-14* ]] || die "OpenFOAM Foundation v14 is required"
}

read_state()
{
    local case_dir=$1
    [[ -f "$case_dir/.bearing-cfd-stage" ]] || die "case was not prepared by this runner: $case_dir"
    cat "$case_dir/.bearing-cfd-stage"
}

expect_state()
{
    local case_dir=$1
    local expected=$2
    local actual
    actual=$(read_state "$case_dir")
    [[ "$actual" == "$expected" ]] || die "expected stage $expected, found $actual"
}

latest_time()
{
    foamListTimes -case "$1" -latestTime
}

checkpoint()
{
    local case_dir=$1
    local label=$2
    local time
    local destination
    time=$(latest_time "$case_dir")
    destination="$case_dir/checkpoints/$label"
    [[ ! -e "$destination" ]] || die "checkpoint already exists: $destination"
    mkdir -p "$destination"
    cp --reflink=auto -a "$case_dir/$time" "$destination/"
    printf '%s\n' "$time" >"$destination/terminal-time"
}

prepare_case()
{
    local mesh=$1
    local case_dir=$2
    local mesh_sha
    local extended_status

    require_command sha256sum
    [[ -f "$mesh" ]] || die "Fluent mesh not found: $mesh"
    [[ ! -e "$case_dir" ]] || die "work case already exists: $case_dir"
    read -r mesh_sha _ < <(sha256sum "$mesh")
    [[ "$mesh_sha" == "$expected_mesh_sha" ]] || die "mesh SHA-256 $mesh_sha does not match $expected_mesh_sha"

    mkdir -p "$(dirname "$case_dir")"
    mkdir "$case_dir"
    cp -a "$template_dir/0" "$template_dir/constant" "$template_dir/system" "$case_dir/"

    fluentMeshToFoam -case "$case_dir" "$mesh" 2>&1 | tee "$case_dir/log.fluentMeshToFoam"
    if [[ -d "$case_dir/0/polyMesh" ]]; then
        mv "$case_dir/0/polyMesh" "$case_dir/constant/polyMesh"
    fi
    [[ -f "$case_dir/constant/polyMesh/boundary" ]] || die "mesh conversion did not create constant/polyMesh"

    checkMesh -case "$case_dir" 2>&1 | tee "$case_dir/log.checkMesh.standard"
    grep -q 'Mesh OK' "$case_dir/log.checkMesh.standard" || die "standard checkMesh did not pass"

    set +e
    checkMesh -case "$case_dir" -allGeometry -allTopology 2>&1 | tee "$case_dir/log.checkMesh.extended"
    extended_status=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$extended_status" >"$case_dir/log.checkMesh.extended.status"

    decomposePar -case "$case_dir" -force 2>&1 | tee "$case_dir/log.decomposePar.initial"
    printf '%s\n' "$(realpath "$mesh")" >"$case_dir/.bearing-cfd-source-mesh"
    printf '%s\n' prepared >"$case_dir/.bearing-cfd-stage"
    printf 'prepared %s\nextended checkMesh exit status: %s\n' "$case_dir" "$extended_status"
}

set_inputs()
{
    local case_dir=$1
    local rpm=$2
    local pressure=$3
    local time
    time=$(latest_time "$case_dir")
    foamDictionary -writePrecision 15 "$case_dir/$time/U" \
        -entry boundaryField/journal_wall/omega -set "$rpm [rpm]"
    foamDictionary -writePrecision 15 "$case_dir/$time/p" \
        -entry boundaryField/pressure_feed/value -set "uniform $pressure"
    decomposePar -case "$case_dir" -fields -latestTime 2>&1 \
        | tee "$case_dir/log.decomposeFields.$rpm-rpm"
}

run_stage()
{
    local case_dir=$1
    local label=$2
    local expected=$3
    local rpm=$4
    local pressure=$5
    local log_path="$case_dir/log.$label"

    expect_state "$case_dir" "$expected"
    if [[ "$label" != atmospheric ]]; then
        set_inputs "$case_dir" "$rpm" "$pressure"
    fi
    mpirun -np 4 foamRun -parallel -case "$case_dir" 2>&1 | tee "$log_path"
    grep -q 'SIMPLE solution converged' "$log_path" || die "$label did not meet SIMPLE residual control"
    reconstructPar -case "$case_dir" -latestTime 2>&1 | tee "$case_dir/log.reconstruct.$label"
    checkpoint "$case_dir" "$label"
    printf '%s\n' "$label" >"$case_dir/.bearing-cfd-stage"
    printf 'completed %s at OpenFOAM time %s\n' "$label" "$(latest_time "$case_dir")"
}

operation=${1:---help}
case "$operation" in
    -h|--help)
        usage
        ;;
    prepare)
        [[ $# -ge 2 && $# -le 3 ]] || { usage >&2; exit 2; }
        require_openfoam
        prepare_case "$2" "${3:-$default_case}"
        ;;
    atmospheric)
        [[ $# -le 2 ]] || { usage >&2; exit 2; }
        require_openfoam
        run_stage "${2:-$default_case}" atmospheric prepared 0 0
        ;;
    pressure-fed)
        [[ $# -le 2 ]] || { usage >&2; exit 2; }
        require_openfoam
        run_stage "${2:-$default_case}" pressure-fed atmospheric 0 581.395348837
        ;;
    496p563rpm)
        [[ $# -le 2 ]] || { usage >&2; exit 2; }
        require_openfoam
        run_stage "${2:-$default_case}" 496p563rpm pressure-fed 496.563 581.395348837
        ;;
    2000rpm)
        [[ $# -le 2 ]] || { usage >&2; exit 2; }
        require_openfoam
        run_stage "${2:-$default_case}" 2000rpm 496p563rpm 2000 581.395348837
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
