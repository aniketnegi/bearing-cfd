# Conformal central-feed layered Prism6 mesh

`meshing/layered_prism_central_feed.py` builds the full-360 lubricant-fluid
mesh containing the eccentric conical film and its open radial feed passage.
It is a mesh-generation and validation stage only: it does not assign CFD
physics or run a flow solver.

## Why this is a separate mesh

`meshing/structured_hex_no_port.py` remains the frozen all-Hex8 regression
baseline. Its analytic topology is useful for convergence studies of the film
without a port, but a circular feed joining a 20--80 micrometre film is not a
natural single-block Hex8 topology. The ported pipeline therefore uses a
first-order triangular master surface and explicitly sweeps its Tri3 cells to
Prism6 cells. This retains several controlled layers through the film while
allowing a robust, conformal circular feed connection.

Gmsh generates only the two-dimensional master Tri3 mesh. NumPy constructs all
three-dimensional Prism6 nodes and connectivity; Gmsh does not generate the
volume mesh. The native OCCT BREP is used as an independent validation
reference, not as the meshing source.

## Geometry and construction

Source dimensions are read from `params.json` in millimetres, including the
resolved `y_feed_end`. Coordinates are converted exactly once to metres for
solver exports. No inlet position is hardcoded.

The unwrapped master plane uses

```text
u = Rm * theta,  0 <= theta < 2*pi,  0 <= z <= L
```

and is partitioned at `z1` and `z2`. These partitions preserve `ring_A`,
`hole_band`, and `ring_B` as cell metadata; they are conformal internal lines,
not CFD patches. The periodic `u=0` and `u=2*pi*Rm` representations are
collapsed to shared nodes, so no circumferential seam is exposed.

The feed disk remains part of the master mesh. Every master triangle is swept
from the journal to the bore through each requested gap interval. Consequently
the journal wall contains one triangle for every master triangle and remains
continuous beneath the feed. At the bore, triangles outside the disk form
`bushing_bore_wall`; disk triangles are reused directly as the first layer of
the feed-tube sweep. Tube nodes at the mouth are therefore shared by identity,
not created coincident and merged later.

The open mouth is proven topologically. A canonical Prism6 face census requires
every mouth triangle to have exactly two incident cells: one film prism and one
feed prism. It is a shared feed-film interface and an internal owner-neighbour
face, absent from every external boundary group. Thus there is no internal cap,
wall patch, gap, overlap, or hanging-node interface at the feed mouth.

## Clearance convention

Film layers use radial clearance at the same axial coordinate: points are
interpolated along bearing-centred rays between the eccentric journal and the
coaxial bore. The extreme radial gaps are `c-e` and `c+e`.

This is not the shortest surface-normal clearance. For parallel cones at the
two extreme meridians, the corresponding normal distances are the radial gaps
multiplied by `cos(gamma)`. The mesh construction and its gap fields use the
radial definition, consistently with the validated CAD and no-port baseline.

## Circular-rim faceting

All rim nodes are analytic cylinder/bore intersection points. A first-order
Tri3 mesh joins adjacent nodes with straight chords, so the surface between
nodes is a faceted approximation rather than an exact curved cylinder. For
hole radius `r_h` and `N` rim segments, the reported maximum sagitta is

```text
r_h * (1 - cos(pi/N))
```

and the regular-polygon area is

```text
0.5 * N * r_h^2 * sin(2*pi/N).
```

For the default `r_h=2 mm` and `N=128`, the sagitta is about `0.000602 mm`;
the program computes and records the actual value and polygon-versus-circle
area error. Nodal residuals against the analytic surfaces and native BREP are
reported separately from this between-node chord error.

## Physical boundaries

| ID | Name | Role |
|---:|---|---|
| 101 | `journal_wall` | Moving journal wall; continuous under the feed |
| 102 | `bushing_bore_wall` | Stationary bore wall outside the feed mouth |
| 103 | `axial_end_z0` | Axial end at `z=0` |
| 104 | `axial_end_zL` | Axial end at `z=L` |
| 105 | `feed_tube_wall` | Stationary cylindrical feed wall |
| 106 | `pressure_feed` | External inlet disk, outward normal `+Y` |
| 201 | `fluid` | One connected volume region |

The mouth is internal and has no physical boundary name. In particular, the
solver mesh contains no `feed_mouth`, `mouth_cap`, `internal_feed`,
`defaultFaces`, symmetry, or circumferential seam patch.

## Validation and artifacts

Before publication, each gap-level mesh must pass analytic cone and inlet
checks, the complete face census, local outward orientation for every external
face, exact per-patch counts, one-region connectivity, film-to-inlet path,
strictly positive Prism6 Jacobians and volumes, Gmsh quality checks, native
BREP volume and closest-point checks, and MSH 4.1, MSH 2.2, VTU, and NPZ
round trips. Publication is atomic. Failure removes solver-eligible artifacts
and retains a failure report rather than weakening a tolerance.

The native-BREP surface classifier currently rejects `eccentricity_ratio=0`
explicitly because concentric journal and bore cone centres are ambiguous to
that classifier. Optional GUI launch happens only after atomic publication; a
Wayland/FLTK failure is reported as a warning and cannot replace valid meshes.

Each `nGap_XX/` directory contains the true, complete mesh in:

- `ported_prism.msh`: Gmsh 4.1 binary with quality fields;
- `ported_prism_openfoam.msh`: Gmsh 2.2 ASCII conversion input;
- `volume_prism.vtu`: ParaView volume mesh;
- `boundary_faces.vtu`, `mesh_arrays.npz`, reports, manifests, and physical
  group metadata.

Files under `viz/` are inspection artifacts, not solver meshes. The exact
cutaway and `x0` section retain unchanged coordinates but contain only subsets
of the true mesh. `mouth_interface_DIAGNOSTIC_ONLY.vtu` displays the internal
shared triangles and is permanently marked `solve_eligible=0` and
`diagnostic_only=1` and is shown as a translucent wireframe by default so it
cannot resemble a physical cap. No exaggerated mesh is produced.

## Generate the meshes

Default case:

```bash
uv run python meshing/layered_prism_central_feed.py --openfoam skip
```

Second validated parameter case:

```bash
uv run python meshing/layered_prism_central_feed.py \
  --params out/strict_case_e03_g20/params.json \
  --brep out/strict_case_e03_g20/film_unsplit.brep \
  --preflight out/gmsh_preflight_e03_g20/preflight_report.json \
  --outdir out/ported_prism_e03_g20 \
  --openfoam skip
```

Run all regressions, including the frozen no-port baseline:

```bash
uv run pytest -q
```

Use `--openfoam auto` to audit conversion when `gmshToFoam` and `checkMesh`
are installed. Missing tools are a recorded skip in auto mode and an error in
required mode. A successful conversion audit is still not a CFD solution.

## Inspect in Gmsh

```bash
uv run gmsh out/ported_prism_default/nGap_08/ported_prism.msh
uv run gmsh out/ported_prism_default/nGap_08/viz/feed_cutaway_exact.msh
uv run gmsh out/ported_prism_default/nGap_08/viz/feed_boundary_only.msh
```

The full MSH contains ElementData views such as `region_id`,
`gap_layer_index`, `gap_um`, `minSICN`, and `minDetJac`. The cutaway exposes
all through-gap layers, continuous film beneath the feed, and the tube-to-film
connection without altering coordinates.

## Inspect in ParaView

On Arch Linux/Wayland, use the supplied XCB launcher or run:

```bash
QT_QPA_PLATFORM=xcb paraview out/ported_prism_default/nGap_08/volume_prism.vtu
out/ported_prism_default/nGap_08/viz/launch_paraview_xcb.sh
```

Then select `volume_prism.vtu` in the Pipeline Browser, click **Apply**, press
**R**, choose **Surface With Edges**, and color by `region_id`,
`gap_layer_index`, `gap_um`, or `minSICN`. To inspect the feed axis, apply
**Filters > Clip** with origin `x=0` and normal `(1,0,0)`, or open:

```bash
QT_QPA_PLATFORM=xcb paraview \
  out/ported_prism_default/nGap_08/viz/feed_cutaway_exact.vtu
QT_QPA_PLATFORM=xcb paraview \
  out/ported_prism_default/nGap_08/viz/x0_section.vtu
```

When `pvpython` is available, the relocatable helper creates the same clip:

```bash
pvpython out/ported_prism_default/nGap_08/viz/open_in_paraview.py
```

## Limitations and next extension

The circular wall and inlet use linear faceting; increasing `--rim-segments`
reduces chord error at increased cost. The mesh resolves the complete
three-dimensional feed passage but does not yet prescribe rotation, pressure,
viscosity, cavitation, turbulence, or any solver boundary condition. Very thin
film prisms naturally have high aspect ratio, so quality acceptance requires
positive Jacobians and reports `minSICN` rather than imposing an inappropriate
isotropic-quality threshold.

A later optional branch may recombine suitable master triangles into quads to
produce a quad-dominant Hex8/Prism6 hybrid. That extension must preserve the
same shared mouth nodes, two-incident-cell interface proof, and physical boundaries;
it is not required for the robust ported Prism6 mesh.

The conical-bearing paper is only visual motivation for layered thin-film
discretization. Its principal numerical model is a developed-surface
Reynolds-equation FEM model; it does not resolve this three-dimensional feed
passage, and no such claim is made here.
