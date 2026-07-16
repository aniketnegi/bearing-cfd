# Fluent import-only audit

This stage imports and inspects `bearing_prism.cgns`. It does not initialize a
flow field, run iterations, select a solver model, or assign physical values.
CGNS import can require the Ansys VKI license in addition to an available
Fluent installation.

## Automated PyFluent route

```bash
uv run python interchange/fluent_import_check.py \
  --interchange-dir out/interchange_default/nGap_04
```

The script starts Fluent in 3D double precision, imports the CGNS volume mesh,
runs mesh and quality checks, records `fluent_import_transcript.txt`, queries
the cell and face zones, and compares live Fluent results with the canonical
NPZ counts. It writes `bearing_prism_imported.msh.h5` only if every mandatory
live check passes. A missing PyFluent installation or license is `NOT_RUN`,
never an import pass. `--gui` keeps the Fluent UI available during the audit.

## Manual Fluent GUI route

1. Start standalone Fluent in **3D** and **Double Precision**.
2. Choose **File > Import > CGNS > Mesh...**.
3. Open `bearing_prism.cgns`; do not use Mesh & Data.
4. Run **Mesh > Check** at maximum verbosity.
5. Confirm one `fluid` cell zone containing only wedge/Prism6 cells.
6. Confirm exactly these external face zones and their counts against
   `interchange_report.json`: `journal_wall`, `bushing_bore_wall`,
   `feed_tube_wall`, `pressure_feed`, `axial_end_z0`, and `axial_end_zL`.
7. Confirm the internal face-zone list has no feed-mouth boundary. Reject
   `feed_mouth`, `mouth_cap`, `internal_feed`, `defaultFaces`, or any unknown
   merged/default zone.
8. Compare nodes, faces, cells, bounding box, total volume, minimum cell
   volume, maximum skewness, and disconnected-region count with the report.
9. Display the full mesh; then display only `pressure_feed` plus
   `feed_tube_wall`; then only `bushing_bore_wall` to see the circular opening.
10. Create an `x=0` section and a clipped cell view through the feed-film
    junction. The shared feed-film interface must not appear in the boundary
    zone list.
11. Only after all checks pass, save `bearing_prism_imported.msh.h5`. An
    optional import-only `.cas.h5` must be labeled as having no physics,
    initialization, or solution.

Generic zone types may then be assigned without values: the three wall zones
as `wall`, `pressure_feed` as `pressure-inlet`, and the two axial ends as
`pressure-outlet`. This is optional and is not part of the static export.

The supplied `fluent_import_check.jou` performs a console import/check only.
It intentionally does not save because a static journal cannot safely decide
that every version-specific transcript check passed.
