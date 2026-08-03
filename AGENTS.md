# Repository Instructions

## Read before acting

- Read [ENGINEERING.md](ENGINEERING.md) in full before any change; it governs
  engineering and prose.
- Before documentation work, read [docs/AGENTS.md](docs/AGENTS.md). Before
  OpenFOAM case work or execution, read
  [openfoam/AGENTS.md](openfoam/AGENTS.md).
- Inspect relevant code, tests, manifests, and Git status; verify current
  behavior.

## Scope and ownership

- Add bearings only with real implementations and model-specific names.
- Bearing-specific work belongs under
  `bearing_cfd/bearings/conical_journal/`. Share only genuinely cross-bearing
  format adapters and artifact-publication policy.
- Do not split sequential numerical kernels when that obscures ordering or
  state.
- Use `argparse` with explicit dispatch. Do not add bearing base classes,
  managers, plugin registries, factories, or placeholder extension points.
- Backward compatibility is not a goal. Update tracked callers, tests, and
  documentation, then remove obsolete paths. Add no compatibility wrappers or
  deprecation layers unless requested.

## Numerical contract

- CAD geometry uses millimetres and the bearing axis is global `+Z`; solver
  meshes use metres. Preserve the documented clearance and pressure
  conventions.
- Record units, tolerances, convergence gates, tool versions, revision, and
  hashes with numerical claims.
- Keep Python Reynolds--JFO and native OpenFOAM implementations independent.
- Distinguish geometry validation, mesh acceptance, solver convergence,
  conservation, and physical validation. Never promote one as evidence of the
  next.

## Artifacts and execution

- Track compact inputs, scripts, manifests, small fixtures, and selected
  evidence. Ignore meshes, OpenFOAM times, processor partitions,
  post-processing data, logs, caches, archives, and handoff ZIPs.
- New outputs use `out/conical_journal/<stage>/...`. Keep the retained current
  case at `out/openfoam_oq90_single_phase/` and prior generations under
  `out/archive/conical_journal/`; do not relocate or delete either implicitly.
- Python generators publish complete generations atomically with `run.json`;
  failed generations must not replace accepted artifacts. OpenFOAM work cases
  are staged in place, and retained evidence receives `run.json` only after
  acceptance.
- Run simulations in the foreground, one stage at a time. Report exact checks
  and obtain a checkpoint before advancing. Do not run licensed Fluent or
  production cavitation studies unless requested.
- Run focused checks, then the default suite, Ruff, and `compileall`. State only
  commands actually executed.
- Keep commits logically narrow, inspect the staged diff, and never include
  unrelated dirty files.
