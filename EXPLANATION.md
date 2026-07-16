# Eccentric Conical Journal-Bearing Fluid Volume

## Engineering, mathematics, physics, code, and output walkthrough

This document explains `bearing_film.py` and the two required parameter cases. It also records the corrected interchange status after enforcing the requested STEP round-trip tolerance.

## Current status

The most accurate summary is:

- **In-memory OCCT geometry: PASS.** The fluid domain, feed passage, axial zones, topology, exact volumes, clearances, inlet face, and context solids all pass.
- **Native OCCT BREP interchange: PASS.** The full fluid and three-zone compound round-trip with relative volume errors around `10^-13`.
- **STEP interchange at the required `1e-6` relative volume tolerance: FAIL on this software stack.** The error is confined to shapes containing the radial feed/bore intersection.
- **Overall program result: FAIL**, deliberately. The code no longer calls these STEP files acceptable.

Therefore, use `film_unsplit.brep` or `film_zones.brep` for a future Gmsh/OpenCASCADE workflow. Do not begin the physical-port mesh from the current STEP translations. STEP remains necessary for SOLIDWORKS and ANSYS, but that path needs further interchange work before acceptance.

The earlier directories `out/default` and `out/case_e03_g20` contain STEP files produced when the code used a looser acceptance threshold. They are useful for visual inspection only. The corrected runs are in:

- `out/strict_default`
- `out/strict_case_e03_g20`

Because those corrected runs fail STEP validation, they retain only the validated BREP fallback, the thickness plot, `requirements.txt`, and the failure-report `params.json`. The staged STEP files are not published, and any stale program-generated STEP files already in that output directory are removed.

## 1. What the program represents

The production body is the lubricant-fluid region, not the journal or bushing metal. It contains:

1. the complete 360-degree thin annular film between the conical journal and conical bearing bore;
2. the fluid in a radial oil-feed drilling along global `+Y`;
3. a short inlet extension outside the illustrative bushing wall.

The journal and bushing are exported only as visual context. They do not belong to the CFD fluid domain.

The coordinate system is:

```text
                         +Y: feed direction
                          ^
                          |       circular inlet disk
                          |=================|
                          |  radial passage |
                          |                 |
              -X <--------+--------> +X
                         O bearing axis
                         |
                         | default journal-axis offset e
                         v
                        -Y: eccentricity/minimum-clearance line

Global Z is the bearing axis, with 0 <= z <= L.
```

The default journal axis is displaced by `0.03 mm` toward `-Y`. Consequently, the minimum same-`z` radial film lies on `-Y`, while the feed is placed at the nominal maximum-clearance direction on `+Y`.

The phrase **eccentricity/minimum-clearance line** is intentional. It is not automatically the external load line. In a hydrodynamic bearing, the integrated pressure load is generally phase-shifted from the displacement vector; its direction must come from the later fluid solution.

## 2. Input and derived parameters

`InputParams` is a frozen dataclass containing only independent inputs. `ResolvedParams` is a separate frozen dataclass containing the resolved feed position and every derived value used by construction or validation.

The two required cases are:

| Quantity | Default | Second case |
|---|---:|---:|
| Length `L` | 60 mm | 60 mm |
| Mean journal radius `Rm` | 50 mm | 50 mm |
| Cone semi-angle `gamma` | 10 deg | 20 deg |
| Radial clearance `c` | 0.050 mm | 0.050 mm |
| Eccentricity ratio `epsilon` | 0.6 | 0.3 |
| Eccentricity `e = epsilon c` | 0.030 mm | 0.015 mm |
| Eccentricity angle `phi` | -90 deg | -90 deg |
| Feed diameter | 4 mm | 4 mm |
| Resolved feed position `zh` | 30 mm | 30 mm |
| Split planes | `z=26`, `z=34` mm | `z=26`, `z=34` mm |

All geometric calculations are performed in millimetres.

## 3. Cone geometry

Let

```math
m = \tan(\gamma).
```

The journal, bore, and illustrative outer-wall radii are

```math
R_j(z) = R_m + \left(\frac{L}{2}-z\right)m,
```

```math
R_b(z) = R_j(z)+c,
```

```math
R_o(z) = R_b(z)+t_w.
```

Because `Rj` is linear in `z`, its value at midspan is exactly `Rm`; its average over the bearing length is also exactly `Rm`.

### Default radii

| `z` | `Rj` | `Rb` | `Ro` |
|---:|---:|---:|---:|
| 0 mm | 55.289809421 mm | 55.339809421 mm | 65.339809421 mm |
| 30 mm | 50.000000000 mm | 50.050000000 mm | 60.050000000 mm |
| 60 mm | 44.710190579 mm | 44.760190579 mm | 54.760190579 mm |

### 20-degree radii

| `z` | `Rj` | `Rb` | `Ro` |
|---:|---:|---:|---:|
| 0 mm | 60.919107028 mm | 60.969107028 mm | 70.969107028 mm |
| 30 mm | 50.000000000 mm | 50.050000000 mm | 60.050000000 mm |
| 60 mm | 39.080892972 mm | 39.130892972 mm | 49.130892972 mm |

The large end is at `z=0`; radius decreases as `z` increases.

## 4. Eccentric journal placement

The journal axis remains parallel to global `Z`, but its transverse displacement is

```math
e = \epsilon c,
```

```math
e_x=e\cos\phi, \qquad e_y=e\sin\phi.
```

For `phi=-90 deg`, floating-point evaluation gives an effectively zero `ex` and a negative `ey`:

```text
default: ex = 1.84e-18 mm, ey = -0.030 mm
second:  ex = 9.18e-19 mm, ey = -0.015 mm
```

The program first revolves the journal about global `Z`, then translates the completed solid by `(ex, ey, 0)`. It does not revolve around an offset axis. This produces a cone whose axis is parallel to the bore axis, exactly as required.

## 5. Exact same-z radial clearance

Radial clearance is measured in a plane of constant `z`, outward from the global bearing axis. It is not the shortest three-dimensional distance between the conical surfaces.

At a fixed `z`, the journal cross-section is a circle of radius `Rj(z)` centred at the eccentricity vector. Let a bearing-frame ray have polar angle `theta`, and define

```math
\alpha=\theta-\phi.
```

If `r` is the distance from the bearing axis to the journal intersection along that ray, the circle equation is

```math
\left|r\,\hat{u}_{\theta}-e\,\hat{u}_{\phi}\right|^2=R_j(z)^2.
```

Expanding gives

```math
r^2-2er\cos\alpha+e^2-R_j(z)^2=0.
```

Taking the outward, positive root:

```math
r_j(\theta,z)=e\cos\alpha+
\sqrt{R_j(z)^2-e^2\sin^2\alpha}.
```

Therefore the exact same-`z` radial thickness is

```math
h_{\mathrm{radial}}(\theta,z)
=R_b(z)-r_j(\theta,z).
```

At the displacement meridian, `alpha=0`, so

```math
h_{\mathrm{radial,min}}=c-e=c(1-\epsilon).
```

At the opposite meridian, `alpha=pi`, so

```math
h_{\mathrm{radial,max}}=c+e=c(1+\epsilon).
```

This produces:

| Case | Radial minimum | Radial maximum |
|---|---:|---:|
| Default | 0.020000 mm | 0.080000 mm |
| `epsilon=0.3`, `gamma=20 deg` | 0.035000 mm | 0.065000 mm |

The code sections both bore and journal with planes at five interior axial stations. It rotates each section by `-phi`, so the measurement works for arbitrary eccentricity direction rather than assuming `+/-Y`. Bounding extrema in the rotated frame then directly measure the two radial gaps. All ten measurements in each case match their targets to numerical precision.

### Why the common approximation is not exact

A frequently used thin-clearance expression is

```math
h_{\mathrm{first\ order}}=c\left(1-\epsilon\cos\alpha\right).
```

It follows from expanding the square root for `e/Rj << 1`. It is exact at `alpha=0` and `alpha=pi`, but away from those meridians the exact square-root term contains an `e^2/Rj` correction. Since `Rj` changes along a cone, the exact radial thickness also has a very small `z` dependence.

The plotted map uses the exact expression, not the first-order approximation.

## 6. Radial clearance versus shortest surface-normal distance

The two conical surfaces are parallel because their radii differ by the same radial amount at every `z`. In an axial meridian, their generator lines have slope magnitude `tan(gamma)`.

For two parallel lines separated horizontally by a same-`z` radial amount `Delta r`, the perpendicular distance is

```math
\frac{\Delta r}{\sqrt{1+\tan^2\gamma}}
=\Delta r\cos\gamma.
```

Hence

```math
h_{\mathrm{normal,min}}=(c-e)\cos\gamma,
```

```math
h_{\mathrm{normal,max}}=(c+e)\cos\gamma.
```

The numerical results are:

| Case | Normal minimum target | OCCT lateral-face distance | Normal maximum target |
|---|---:|---:|---:|
| Default | 0.019696155060 mm | 0.019696155060 mm | 0.078784620241 mm |
| Second | 0.032889241728 mm | 0.032889241727 mm | 0.061080020351 mm |

This distinction matters physically and geometrically:

- radial clearance is the bearing-design input at equal axial coordinate;
- shortest normal distance is the actual minimum Euclidean separation between sloping surfaces;
- an unrestricted OCCT `distance_to()` query returns the latter, so comparing it with `c-e` would be dimensionally plausible but geometrically wrong.

The reported closest points also have slightly different `z` coordinates, directly showing that the shortest connecting segment is not radial at the same `z`.

## 7. Exact base-film volume

At each axial station, the coaxial annular cross-sectional area before adding the feed is

```math
A(z)=\pi\left(R_b(z)^2-R_j(z)^2\right).
```

Using `Rb=Rj+c`:

```math
A(z)=\pi\left(2cR_j(z)+c^2\right).
```

Integrating over the length:

```math
V_{\mathrm{base}}
=\pi\int_0^L\left(2cR_j(z)+c^2\right)\,dz.
```

The mean value of the linear radius law is `Rm`, so

```math
V_{\mathrm{base,exact}}
=\pi L\left(2R_m c+c^2\right).
```

With `L=60 mm`, `Rm=50 mm`, and `c=0.05 mm`:

```text
Vbase = 942.949034974976 mm^3.
```

The default Boolean result is `942.949034974845 mm^3`; the second case is `942.949034974809 mm^3`. Both agree with the analytic value far inside the mandatory `1e-8` relative tolerance.

Two useful consequences follow:

1. Transverse journal translation does not change its volume, provided it remains fully contained in the bore.
2. Changing `gamma` does not change the integrated base volume when `L`, `Rm`, and `c` are fixed, because the linear cone-radius increase on one side of midspan cancels the decrease on the other.

## 8. Feed passage geometry

The feed cylinder has radius

```math
r_h=\frac{d_h}{2}=2\ \mathrm{mm},
```

and its axis passes through `(0, 0, zh)` in direction `+Y`.

### Why it begins at `y=0`

The cylinder deliberately starts deep inside the journal cutter. The program proves that the whole circular starting disk is contained in the journal, rather than relying on a default-only coordinate.

For non-negative cone angle, the smallest journal radius touched by the disk occurs at

```math
z_{\mathrm{hole,max}}=z_h+r_h.
```

A conservative maximum distance from any starting-disk point to the displaced journal axis is

```math
\sqrt{(|e_x|+r_h)^2+e_y^2}.
```

The remaining containment margins are very large:

```text
default: 47.647121 mm
second:  47.272003 mm
```

After the bore and feed are fused, subtracting the journal last trims this deliberately overlong cylinder exactly to the journal surface. This is more robust than trying to construct a cylinder that begins on a complicated journal/feed intersection curve.

### Why the remote endpoint uses the lower axial edge

The remote inlet disk spans `zh-rh <= z <= zh+rh`. Radius is largest at the lower axial edge for the permitted non-negative cone angle, so

```math
z_{\mathrm{hole,min}}=z_h-r_h,
```

```math
y_{\mathrm{feed,end}}
=R_o(z_{\mathrm{hole,min}})+s_u.
```

This gives:

```text
default: 63.402653961 mm
second:  63.777940469 mm
```

Every point of the remote disk is therefore beyond the largest outer-wall radius it touches, with at least the requested axial-tube extension in the conservative `Y` sense.

The surviving inlet-face measurements are:

| Quantity | Default | Second case |
|---|---:|---:|
| Area | 12.566370614 mm^2 | 12.566370614 mm^2 |
| Centre | `(0, 63.402653961, 30)` mm | `(0, 63.777940469, 30)` mm |
| Normal | `(0, +1, 0)` | `(0, +1, 0)` |
| Boundary | one circle | one circle |

The area equals `pi * 2^2`, and the disk is a real planar B-rep face suitable for later boundary-condition selection.

## 9. Boolean set logic

Let `B` be the bore-volume blank, `J` the extended eccentric journal cutter, and `F` the feed cylinder. The exact construction order is:

```text
base_film = B - J
wet       = B + F
film      = wet - J
```

or, in set notation,

```math
\mathrm{base}=B\setminus J,
```

```math
\mathrm{wet}=B\cup F,
```

```math
\mathrm{film}=(B\cup F)\setminus J.
```

This order is an engineering robustness choice:

- the feed deliberately overlaps both bore and journal;
- the bore/feed union must first become one connected valid solid;
- subtracting the journal last opens the passage into the film and trims the internal overlap;
- no fuzzy Boolean or loosened OCCT tolerance is used.

Since the journal is contained in the bore, the exact newly added fluid volume obeys

```math
V_{\mathrm{film}}-V_{\mathrm{base}}
=\mathrm{Vol}(F\setminus B).
```

The measured values are:

| Quantity | Default | Second case |
|---|---:|---:|
| Added volume from `film-base` | 167.922786887 mm^3 | 172.640532984 mm^3 |
| Added volume from `feed-bore` | 167.919875681 mm^3 | 172.636051105 mm^3 |
| Simple scale estimate | 163.362817987 mm^3 | 163.362817987 mm^3 |

The small discrepancy between the two exact Boolean routes is about `1.7e-5` and `2.6e-5` relative. It comes from OCCT representations of the shared bore/feed intersection under default modelling tolerances; the code does not hide it with fuzzy operations. The mandatory identity tolerance is `5e-5`.

The scale estimate

```math
\pi r_h^2(t_w+s_u)
```

is only diagnostic. A conical wall and a finite-radius hole do not have one universal passage length, so the estimate is not an acceptance bound.

## 10. Why the journal cutter extends beyond both ends

The production fluid ends at `z=0` and `z=L`, but the subtraction cutter spans

```text
-delta <= z <= L+delta.
```

With `delta=1 mm`, the journal end caps cannot be coincident with the bore end caps. This avoids a fragile coplanar-face Boolean and leaves clean annular fluid end faces exactly at the bore blank's limits.

The journal context copy is separate and spans exactly `0 <= z <= L`.

## 11. Axial zoning

The valid full film is split at

```math
z_1=z_h-w=26\ \mathrm{mm},
```

```math
z_2=z_h+w=34\ \mathrm{mm}.
```

The three solids are labelled by their geometric centre and bounding range, not by trusting the order returned by OCCT:

- `ring_A`: `0 <= z <= 26 mm`
- `hole_band`: `26 <= z <= 34 mm`
- `ring_B`: `34 <= z <= 60 mm`

They are not re-fused. Their coincident split faces remain available for downstream conformal partitioning.

### Zone volumes

| Zone | Default | Second case |
|---|---:|---:|
| `ring_A` | 433.095730076 mm^3 | 459.151571152 mm^3 |
| `hole_band` | 293.649324883 mm^3 | 298.367070981 mm^3 |
| `ring_B` | 384.126766902 mm^3 | 358.070925826 mm^3 |
| Sum | 1110.871821862 mm^3 | 1115.589567959 mm^3 |

The relative split-volume conservation errors are `5.12e-15` and `3.14e-14`, far below `1e-8`.

The larger `ring_A` volume follows from the larger cone radius near `z=0`. The feed-containing middle band includes both annular film and external passage.

These three axial bodies are useful meshing partitions, but they do not imply an all-hexahedral mesh. Circumferential sectoring and meshing strategy remain future work.

## 12. Bearing physics represented by this geometry

### 12.1 Hydrodynamic film formation

When the journal rotates, viscous no-slip at the journal surface drags lubricant tangentially. The eccentric geometry creates a converging and diverging film thickness around the circumference. In the converging region, viscous transport drives a pressure rise; that pressure integrated over the journal surface produces a load-carrying force.

The local velocity profile can be understood as the sum of:

- **Couette flow**, driven by the moving journal wall;
- **Poiseuille flow**, driven by pressure gradients.

The balance between those mechanisms under the thin-film approximation leads to the Reynolds lubrication equation. In a simple cylindrical, steady, incompressible form it is commonly written as

```math
\frac{1}{R^2}\frac{\partial}{\partial\theta}
\left(h^3\frac{\partial p}{\partial\theta}\right)
+\frac{\partial}{\partial z}
\left(h^3\frac{\partial p}{\partial z}\right)
=6\mu\Omega\frac{\partial h}{\partial\theta},
```

subject to the chosen pressure, flow, and cavitation boundary conditions. Sign and factor conventions vary with coordinate and wall-velocity definitions.

For a cone, the surface metric varies with radius. Parameterising the bore surface by `(theta,z)` gives orthogonal scale factors approximately

```math
h_\theta=R_b(z), \qquad h_z=\sec\gamma.
```

Thus a cone-aware reduced-order Reynolds model must include the varying circumferential scale and generator length. A full three-dimensional CFD solver instead receives the exact B-rep volume and resolves the local port disturbance directly.

This CAD program does not solve the Reynolds equation. It has no viscosity, density, rotational speed, supply pressure, cavitation model, or temperature model, so it cannot yet predict pressure, torque, flow rate, attitude angle, or load capacity.

### 12.2 Why a complete 360-degree domain is necessary

For the default placement, the static geometry is mirror-symmetric about `x=0`. The rotating-wall velocity is not a scalar mirror-symmetric boundary condition.

For rotation `Omega` about the displaced journal axis,

```math
\mathbf{u}_w
=\Omega\hat{\mathbf{z}}\times
\left[(x-e_x)\hat{\mathbf{x}}+(y-e_y)\hat{\mathbf{y}}\right].
```

Therefore

```math
\mathbf{u}_w
=\left[-\Omega(y-e_y),\ \Omega(x-e_x),\ 0\right].
```

Reflection across a diametral plane does not turn this rotational vector field into an ordinary CFD symmetry condition. A half model could reproduce the static shape but would impose the wrong moving-wall kinematics. For arbitrary `phi`, even the static geometric symmetry may disappear.

That is why the production model is a complete 360-degree fluid volume. The optional half export is labelled only as geometry debug and is forbidden for rotating-journal CFD.

### 12.3 Why eccentricity is not the load direction

The journal centre displacement determines the geometric minimum-gap line. The pressure peak is produced downstream within the converging wedge, and its integrated force depends on speed, viscosity, feed conditions, axial leakage, cavitation, and cone geometry. The force vector and displacement vector therefore need not be collinear.

The later CFD solution will determine the hydrodynamic load and attitude relation. The CAD correctly reports only the geometry-defined eccentricity line.

### 12.4 Why the external inlet extension exists

Ending the feed exactly on the conical metal wall would leave a less convenient boundary patch at a geometric transition. The short extension creates an unambiguous circular planar inlet outside the context bushing. A later solver can apply a supply pressure, mass-flow rate, or velocity condition to that disk without guessing which face represents the port.

## 13. Code walkthrough

### 13.1 Errors and immutable data

The script defines specific exceptions for parameter errors, geometry construction, Booleans, topology, validation, export, and interchange round trips. Mandatory checks do not rely on bare `assert`, so running Python with optimisation cannot disable them.

`InputParams` and `ResolvedParams` are frozen. This prevents accidental mutation after validation and keeps `params.json` reproducible.

### 13.2 Parameter resolution

`resolve_params()`:

- resolves an omitted `zh` to the current `L/2`, including when length is overridden;
- converts degrees to radians;
- computes `tan(gamma)`, `e`, `ex`, and `ey`;
- computes the feed disk's axial range and conservative containment margin;
- computes radial and normal clearance targets;
- computes the analytic base volume and feed scale estimate.

`validate_params()` checks every prohibited combination before any B-rep construction. It accumulates all parameter problems into one explicit exception rather than failing at the first opaque OCCT operation.

### 13.3 Revolved solids

`_revolved_frustum()` builds a closed polygon in the `XZ` plane:

```text
(0,z0) -> (R(z0),z0) -> (R(z1),z1) -> (0,z1) -> close
```

It converts that wire to a planar face and revolves it 360 degrees about global `Z`. The result is a closed exact solid, not a shell, STL, tessellation, or offset mesh.

`make_bore_blank()` uses the bore radius law over `[0,L]`. `make_journal()` uses the journal law over `[-delta,L+delta]` for cutting, or `[0,L]` for context, then translates the solid.

### 13.4 Feed construction

`make_feed_cylinder()` defines a local plane whose local cylinder axis maps to global `+Y`. It constructs the exact cylinder and immediately searches its faces for one planar circular disk with the expected area, centre, and `+Y` normal. Failure to find exactly one disk is a construction error.

### 13.5 Boolean construction

`make_base_film()` performs `bore_blank - journal`.

`make_full_film()` first performs `bore_blank + feed_cylinder`, then requires that intermediate `wet` shape to be one valid manifold solid. Only then does it subtract the journal.

`make_context_bushing()` constructs `outer_frustum - bore_blank - feed_cylinder`.

No alternative approximate geometry is substituted if an exact operation fails.

### 13.6 Splitting and labelling

`split_axial_zones()` applies two exact plane splits, sorts pieces by centre-of-mass `Z`, and verifies each bounding interval. This avoids depending on undocumented return order.

### 13.7 Geometry validation

`validate_geometry()` creates structured PASS/FAIL records for:

- full-film and intermediate topology;
- validity and manifoldness of every solid;
- connectedness and face-count limits;
- analytic base volume;
- positive and Boolean-consistent feed-added volume;
- five section-based radial minimum and maximum measurements;
- lateral-face normal minimum distance and closest points;
- split-volume conservation;
- full-domain bounding box;
- unique inlet identity, area, centre, normal, and circular edge.

The geometry stage contributes 42 records, all of which pass in both required cases.

### 13.8 STEP and BREP export

Every STEP export now explicitly uses:

```python
export_step(
    shape,
    path,
    unit=Unit.MM,
    write_pcurves=True,
    precision_mode=PrecisionMode.GREATEST,
)
```

`film_unsplit.brep` and `film_zones.brep` are written with `export_brep()`.

The native BREP files are re-imported first. Only after they pass validity, manifoldness, solid count, and a `1e-12` relative volume tolerance are they copied to the requested output directory as a trusted fallback.

STEP files are re-imported separately and must satisfy:

```python
STEP_ROUNDTRIP_REL_TOL = 1.0e-6
```

If any STEP file exceeds this threshold, `RoundTripValidationError` is raised, the process exits nonzero, and the staged STEP batch is discarded. Previously generated STEP files in the same output directory are also removed so they cannot be mistaken for products of the new run. The validated BREP files and non-CAD diagnostics remain available.

### 13.9 Plot and live preview

Matplotlib generates `film_thickness_map.png` from the exact square-root thickness equation. It is a geometry diagnostic, not a pressure map.

With `--preview`, `preview_geometry()` sends the fluid, journal context, and transparent bushing context to an `ocp_vscode` viewer on port `3939`. Preview occurs after geometry validation and before interchange validation, so the geometry can still be inspected even when strict STEP checking subsequently rejects the translated files.

### 13.10 Transactional output behaviour

The complete export is first written beside the requested output directory in a temporary staging directory, not in `/tmp`. A fully passing run atomically publishes the staged files. A STEP-failing run publishes only the already validated BREP fallback, plot, and dependency record.

`params.json` records `overall: FAIL`, the exact error, every validation record, both round-trip result sets, dependency versions, diagnostics, and SHA-256 hashes of the trusted fallback files.

## 14. Topology results

For both parameter cases:

| Shape | Solids | Faces | Valid | Manifold |
|---|---:|---:|---|---|
| Bore blank | 1 | 3 | yes | yes |
| Extended journal cutter | 1 | 3 | yes | yes |
| Feed cylinder | 1 | 3 | yes | yes |
| Base film | 1 | 4 | yes | yes |
| Wet bore/feed union | 1 | 5 | yes | yes |
| Full film | 1 | 6 | yes | yes |
| `ring_A` | 1 | 4 | yes | yes |
| `hole_band` | 1 | 6 | yes | yes |
| `ring_B` | 1 | 4 | yes | yes |
| Journal context | 1 | 3 | yes | yes |
| Bushing context | 1 | 5 | yes | yes |

The full film being one solid proves that the feed passage is connected to the annular film rather than surviving as a detached tube.

## 15. Full-fluid and feed volumes

| Quantity | Default | Second case |
|---|---:|---:|
| Base annular film | 942.949034975 mm^3 | 942.949034975 mm^3 |
| Feed-added fluid | 167.922786887 mm^3 | 172.640532984 mm^3 |
| Total fluid | 1110.871821862 mm^3 | 1115.589567959 mm^3 |

The steeper 20-degree cone changes the port's intersection with the bore and wall, increasing the added feed volume. It does not change the base-film integral because the mean radius remains fixed.

## 16. Bounding-box interpretation

The default final fluid bounds are approximately:

```text
x: -55.3398095 to +55.3398095 mm
y: -55.3398095 to +63.4026541 mm
z: -0.0000001 to 60.0000001 mm
```

The `x` and negative-`y` limits are set by the large-end bore. The positive-`y` limit is set by the feed extension. The tiny `1e-7 mm` envelope beyond nominal coordinates is OCCT's default topological tolerance and remains far inside the mandatory `0.001 mm` bounding-box tolerance. Modelling tolerances were not loosened.

## 17. Corrected interchange results

### 17.1 STEP

All STEP files are written with `PrecisionMode.GREATEST`. On build123d `0.11.1` with `cadquery-ocp-novtk 7.9.3.1.1`, the measured relative volume errors are:

| File | Default | Second case | Required |
|---|---:|---:|---:|
| `film_unsplit.step` | `2.800e-5` | `5.962e-5` | `<=1e-6` |
| `hole_band.step` | `1.059e-4` | `2.229e-4` | `<=1e-6` |
| `film_zones.step` | `2.800e-5` | `5.962e-5` | `<=1e-6` |
| `ring_A.step` | about `1e-12` | about `1e-12` | `<=1e-6` |
| `ring_B.step` | about `1e-12` | about `1e-12` | `<=1e-6` |
| `context_assembly.step` | `3.219e-8` | `1.228e-7` | `<=1e-6` |

The absolute full-fluid changes are approximately:

```text
default: 0.0311089 mm^3
second:  0.0665082 mm^3
```

The clean annular lands and context assembly pass, while every failing body contains the radial feed intersection. This strongly localises the translation loss to the cone/cylinder intersection curves and their p-curves.

An explicit local probe compared `PrecisionMode.AVERAGE` and `PrecisionMode.GREATEST`; both produced the same measured volume errors on this stack. `GREATEST` is still the correct requested setting, but it does not by itself remove this OCCT translation effect. The code therefore fails honestly instead of assuming that selecting the enum guarantees the target.

### 17.2 Native BREP

| File | Default | Second case | Required |
|---|---:|---:|---:|
| `film_unsplit.brep` | `1.373e-13` | `6.563e-14` | `<=1e-12` |
| `film_zones.brep` | `2.419e-13` | `9.844e-14` | `<=1e-12` |

Both native files re-import with the expected solid counts, and all solids remain valid and manifold. This demonstrates that the geometry itself is not losing the feed volume; the measurable change appears during STEP translation/re-import.

### 17.3 Validation totals

Each strict case now has 80 records:

- 42 geometry records: all pass;
- 8 native BREP records: all pass;
- 30 STEP records: 27 pass and 3 volume records fail.

Thus the truthful summary is `77/80 PASS`, followed by `OVERALL: FAIL`.

## 18. Output files

### Corrected failing runs

The strict output directories contain:

| File | Purpose |
|---|---|
| `film_unsplit.brep` | One native OCCT solid containing the complete 360-degree fluid domain |
| `film_zones.brep` | Native OCCT compound containing the three coincident axial solids |
| `film_thickness_map.png` | Exact same-`z` radial thickness diagnostic |
| `params.json` | Complete FAIL record, measurements, dependency versions, and hashes |
| `requirements.txt` | Exact tested Python dependency pins |

No STEP files are published in these strict directories because their mandatory volume checks fail.

### Files produced after a future fully passing STEP round trip

A completely successful run will additionally publish:

- `film_unsplit.step`
- `film_zones.step`
- `ring_A.step`
- `hole_band.step`
- `ring_B.step`
- `context_assembly.step`
- optional `geometry_half_debug.step`

The individual zone STEP files remain necessary as label-independent fallbacks. BREP is native topology and should not be expected to carry the same XCAF labels as STEP assemblies.

## 19. Reading the thickness map

The horizontal axis is bearing-frame angle `theta` from 0 to 360 degrees. The vertical axis is axial coordinate `z`. Colour is the exact same-`z` radial thickness in millimetres.

For the default case:

- the dashed eccentricity line at 270 degrees marks the `0.020 mm` radial minimum;
- the `+Y` feed direction at 90 degrees coincides with the `0.080 mm` radial maximum;
- the tiny axial colour variation away from the extrema comes from the exact square-root geometry and changing conical radius.

This image does not show pressure, velocity, temperature, cavitation, or load.

## 20. Fontconfig warnings

The repeated Fontconfig messages are unrelated to CAD validity. The Python OCP wheel loads a bundled older Fontconfig library, while the Artix system configuration uses newer XML attributes such as `xsi:nil` and newer generic-family syntax. The bundled parser prints warnings when it reads the host configuration.

The evidence that these warnings are non-geometric is direct:

- B-rep construction completes;
- all in-memory solids are valid and manifold;
- exact geometry checks pass;
- native BREP round trips pass near machine precision;
- the Matplotlib PNG is created.

Do not edit `/etc/fonts` merely to silence this CAD process. A future OCP wheel with a compatible Fontconfig dependency is the cleaner fix.

## 21. Reproducible commands

From the project directory:

```bash
cd /home/aniketnegi/code/ankt/btp/code/eccentric-conical-bearing
uv sync
```

Run the two strict cases:

```bash
uv run python bearing_film.py --outdir out/strict_default
```

```bash
uv run python bearing_film.py \
  --eccentricity-ratio 0.3 \
  --semicone-angle-deg 20 \
  --outdir out/strict_case_e03_g20
```

On the currently tested stack, both commands are expected to exit nonzero with `OVERALL: FAIL` because STEP exceeds `1e-6`. The BREP fallback and plot should still be present in each output directory.

### Live viewer

Start the standalone viewer:

```bash
uv run python -m ocp_vscode --host 127.0.0.1 --port 3939
```

Open `http://127.0.0.1:3939`, then run in another terminal:

```bash
uv run python bearing_film.py --outdir out/preview_default --preview
```

The validated in-memory fluid and transparent context solids are sent to the viewer before the later strict STEP rejection.

### FreeCAD inspection

Open either native fallback directly in FreeCAD:

```text
out/strict_default/film_unsplit.brep
out/strict_default/film_zones.brep
```

`film_unsplit.brep` should appear as one solid. `film_zones.brep` should contain three solids with coincident split interfaces.

## 22. What remains before CFD

This phase intentionally stops at CAD. Before solving a rotating-journal case, later work must still define:

- a meshing strategy and circumferential partitions;
- journal angular velocity and exact moving-wall condition;
- lubricant density and viscosity, possibly temperature dependence;
- inlet pressure or mass-flow condition;
- outlet/end-face pressure conditions;
- cavitation treatment;
- turbulence or laminar assumptions based on operating scales;
- mesh-independence and conservation checks;
- load, torque, leakage, and pressure post-processing.

For the OSS meshing route, native BREP is the current accepted handoff because Gmsh's OpenCASCADE kernel can import and serialise `.brep` geometry without a STEP translation. For SOLIDWORKS and ANSYS, retain STEP as the required exchange format but keep the strict `1e-6` check active until a writer/importer combination actually satisfies it.

## 23. Primary references

- [build123d import/export documentation](https://build123d.readthedocs.io/en/latest/import_export.html) — documented STEP precision mode and BREP import/export APIs.
- [Gmsh reference manual](https://gmsh.info/doc/texinfo/) — OpenCASCADE geometry kernel and native `.brep` serialisation/import.
- [OCP CAD Viewer for VS Code](https://github.com/bernhard-42/vscode-cadquery-viewer) — viewer setup and `show()` API.
- [NASA hydrodynamic lubrication report](https://ntrs.nasa.gov/api/citations/19910021217/downloads/19910021217.pdf) — background on Reynolds-equation bearing analysis.

## Final engineering judgement

The CAD construction is sound and genuinely parametric. The exact radial/normal gap distinction, analytic volume agreement, topology, feed connectivity, inlet identity, and native BREP round trips provide strong independent evidence that the intended lubricant domain has been built correctly.

The STEP translation is not yet accepted for the thin-film physical-port workflow. The code now says so explicitly, exits nonzero, preserves only the validated native fallback, and records the measured failure rather than weakening the tolerance.
