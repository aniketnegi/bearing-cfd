# Solver-neutral mesh interchange

`interchange/export_solver_neutral.py` reads only the validated
`mesh_arrays.npz` and its JSON manifests. It does not read STEP/BREP, rebuild
CAD, remesh, initialize a CFD solution, or select OpenFOAM or Fluent as the
final solver.

The exporter registers one clean discrete `fluid` volume containing linear
Prism6 cells and six external surface groups. The shared feed-film interface
is reconstructed as an internal owner-neighbour face with two incident cells;
it is never registered as a surface entity or boundary condition.

## Export

```bash
uv run python interchange/export_solver_neutral.py \
  --case-dir out/ported_prism_default/nGap_04 \
  --outdir out/interchange_default/nGap_04 \
  --fluent auto
```

Other validated inputs are selected only by path, for example:

```bash
uv run python interchange/export_solver_neutral.py \
  --case-dir out/ported_prism_default/nGap_08 \
  --outdir out/interchange_default/nGap_08 \
  --fluent skip

uv run python interchange/export_solver_neutral.py \
  --case-dir out/ported_prism_e03_g20/nGap_04 \
  --outdir out/interchange_e03_g20/nGap_04 \
  --fluent skip
```

Use `--overwrite` only to replace an existing interchange output atomically.
`--fluent skip` never probes or launches Ansys. `auto` attempts a real audit
only when PyFluent is installed and treats a missing installation/license as
`STATIC_PASS_FLUENT_NOT_RUN`. `required` exits nonzero unless a real import
passes. `--gui` applies only to a real Fluent audit.

## Output contract

- `bearing_prism_gmsh41.msh`: clean Gmsh MSH 4.1 binary, no views.
- `bearing_prism_gmsh22_ascii.msh`: clean Gmsh MSH 2.2 ASCII, no views.
- `bearing_prism.cgns`: official Gmsh CGNS writer output with double-precision
  metre coordinates, `PENTA_6`, `TRI_3`, `QUAD_4`, `fluid`, and all six named
  boundary groups; no solution fields.
- `interchange_report.json`: source and per-format round-trip audits.
- `interchange_manifest.json`: concise status and zone/file contract.
- `zones.csv`: exact zone types, roles, and counts.
- `file_hashes.json`: SHA-256 provenance for the NPZ, source manifest/report,
  and all three exported meshes.
- `README_OPEN_ME_FIRST.txt`: exact operator handoff.

The exporter rejects a CGNS file whose actual boundary definitions lose names
or memberships. A sidecar cannot repair that failure. Such output is treated
as `GEOMETRY_ONLY_NOT_FLUENT_READY` and the overall run fails rather than being
published.

## Static validation

The source and reopened exports must retain exact node, Prism6, patch, total
boundary, and internal-mouth counts; metre coordinates and bounding box;
positive signed volumes/Jacobians; total volume and established BREP-relative
error; one connected cell graph; unique cells/faces; a closed manifold
external shell; one/two incidence for external/internal faces; local outward
orientation for every boundary face; pressure-feed connectivity through the
feed and mouth to the film; and paths to both axial ends.

Round-trip signatures are independent of node and element renumbering. The
CGNS audit additionally requires one 3D unstructured `fluid` zone, only
`PENTA_6` volume elements, exact boundary memberships, and coordinate error
below the configured tolerance (default `1e-14 m`, far below the 20 micrometre
minimum gap).

Allowed overall statuses are:

- `STATIC_PASS_FLUENT_NOT_RUN`
- `FLUENT_IMPORT_PASS`
- `FAIL`

A written CGNS file or Gmsh round trip never produces `FLUENT_IMPORT_PASS`.
Without a live proprietary import the readiness field is exactly
`STATICALLY_VALIDATED_NOT_IMPORTED`.

## Fluent and Workbench

Ansys documents CGNS mesh import at **File > Import > CGNS > Mesh...** and
recommends checking boundary and cell zones after third-party import:
<https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/flu_ug/flu_ug_FileImport.html>.
PyFluent documents 3D/double launch options and its CGNS volume-mesh import and
mesh-check interfaces:
<https://fluent.docs.pyansys.com/version/stable/user_guide/session/launching_ansys_fluent.html>.

Gmsh `.msh` and Fluent `.msh` are different formats; renaming the extension is
not conversion. Do not expect the Workbench Meshing editor to open or edit the
Gmsh volume mesh. The supported route is:

1. import `bearing_prism.cgns` in standalone Fluent or a Fluent Setup session;
2. run the live import audit in `interchange/FLUENT_IMPORT.md`;
3. save `bearing_prism_imported.msh.h5` (or an explicitly import-only
   `.cas.h5`) only after every check passes;
4. use that native Fluent file for later Workbench integration, including
   **Import Fluent Case** on releases that expose it on the Setup cell.

Keep the BREP/CAD route separate if Ansys Meshing is later asked to generate a
different mesh. No pressure, speed, material, turbulence, cavitation, thermal,
initialization, iteration, or result quantity belongs to this interchange
stage.
