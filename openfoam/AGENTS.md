# OpenFOAM Instructions

Use OpenFOAM Foundation v14 for tracked cases and scripts. Keep dictionaries
under `cases/conical_journal/openfoam/`; keep native solver code and build
checks under `openfoam/`.

## Versioned and generated data

- Commit compact dictionaries, solver sources, exact run procedures, and small
  acceptance fixtures.
- Keep `constant/polyMesh`, numeric time directories, `processor*`,
  `postProcessing`, VTK exports, ordinary logs, and reconstructed checkpoints
  under ignored `out/` paths.
- Record the Git commit, dirty state, OpenFOAM build, MPI version, mesh hash,
  dictionary hashes, commands, and output hashes for a retained run.

## Single-phase OQ90 baseline

- Run in this order: zero-rpm atmospheric equilibrium, zero-rpm pressure-fed
  equilibrium, 496.563 rpm, then 2000 rpm. Stop after each stage to inspect and
  report its acceptance checks before continuing.
- Use `rho = 860 kg/m3`, `mu = 0.0277 Pa s`, and
  `nu = 3.22093023256e-5 m2/s`. A 500 kPa gauge feed is
  `581.395348837 m2/s2` in kinematic pressure. Convert with
  `p_abs = 101325 + 860 p`.
- At each rotating stage, record residual convergence, global absolute-pressure
  extrema, patch flows, net mass balance, force, torque, and velocity extrema.
- Standard `checkMesh` acceptance and extended quality diagnostics are separate
  results. Record both; do not hide a failed extended determinant threshold
  behind `Mesh OK`.
- A converged single-phase field with negative absolute pressure is a numerical
  baseline, not a physically admissible oil-film prediction.

## Execution

- Run long solver commands in the foreground. Do not background them or launch
  all stages as an unattended batch.
- On `coe9`, pull the exact pushed commit before generating inputs. Confirm
  `foamVersion`, `mpicc`, and `mpirun` before the run. The user performs any
  command requiring `sudo`.
- Do not start or extend cavitation work unless explicitly requested. The
  current priority is reproducing and presenting the single-phase OpenFOAM
  result.
- Do not run licensed Fluent. A Fluent mesh export or static audit is not a
  live Fluent acceptance result.
