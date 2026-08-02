#let navy = rgb("#17324D")
#let blue = rgb("#176B87")
#let grey = rgb("#53636D")
#let rule = rgb("#CBD5DA")

#set document(
  title: "OpenFOAM results — eccentric conical journal bearing",
  author: "BTP project",
  date: datetime(year: 2026, month: 8, day: 3),
)
#set page(
  paper: "a4",
  margin: (top: 19mm, bottom: 18mm, left: 21mm, right: 21mm),
  header: grid(
    columns: (1fr, auto),
    text(font: "Inter", size: 7pt, fill: grey)[OPENFOAM RESULTS],
    text(font: "Inter", size: 7pt, fill: grey)[ECCENTRIC CONICAL JOURNAL BEARING],
  ),
  footer: align(center)[
    #text(font: "Inter", size: 7pt, fill: grey)[#context counter(page).display("1")]
  ],
)
#set text(font: "Libertinus Serif", size: 10.2pt, fill: rgb("#17242C"), lang: "en")
#set par(justify: true, leading: 0.68em)
#set list(indent: 15pt, body-indent: 5pt, spacing: 3.5pt)
#set heading(numbering: none)
#show heading.where(level: 1): it => block(
  above: 13pt,
  below: 6pt,
  stroke: (bottom: 0.7pt + rule),
  inset: (bottom: 4pt),
)[
  #text(font: "Inter", size: 19pt, weight: "bold", fill: navy)[#it.body]
]
#show heading.where(level: 2): it => block(above: 10pt, below: 4pt)[
  #text(font: "Inter", size: 12.5pt, weight: "semibold", fill: blue)[#it.body]
]
#show raw: set text(font: "JetBrains Mono", size: 8pt)
#show figure.caption: set text(size: 8.3pt, fill: grey)
#show figure.caption: set par(justify: false)
#set figure(gap: 5pt)

#align(center)[
  #v(8mm)
  #text(font: "Inter", size: 27pt, weight: "bold", fill: navy)[
    OpenFOAM results
  ]
  #v(2mm)
  #text(font: "Inter", size: 16pt, weight: "medium", fill: blue)[
    Eccentric conical journal bearing
  ]
  #v(3mm)
  #text(size: 10pt, fill: grey)[OQ90 mesh · 0.5 MPa gauge feed · 2000 rpm]
  #v(2mm)
  #text(size: 9pt, fill: grey)[Discussion notes · 3 August 2026]
]

#v(8mm)

= Summary

The OpenFOAM Foundation v14 calculation reached the declared steady SIMPLE
residual limits on the full 1,252,800-cell mesh. The final 2000 rpm branch
required 135 iterations and ended at OpenFOAM time 402.

The principal computed quantities are:

- radial load: *116.081 kN*;
- axial load magnitude: *292.03 N*;
- shaft-axis torque magnitude: *13.871 N·m*;
- mechanical power associated with that torque: *2.905 kW*;
- feed volume flow: *1.215 mL/s*;
- net boundary-flow imbalance: *0.0229% of feed*;
- absolute-pressure range: *−17.160 to 17.413 MPa*.

The pressure, load, torque, and feed-flow scales are close to the teammate's
Fluent run. The axial load differs by 7.46%; the other four compared scalars
differ by less than 0.6%. This comparison is not strict input parity because
the OpenFOAM viscosity is 0.02770 Pa·s and the Fluent viscosity is
0.02777 Pa·s.

The report covers the converged fully filled single-phase calculation. The
negative minimum absolute pressure is retained as a limitation of that model;
phase-change modelling is outside the present scope.

= Case definition

The fluid region is the clearance between an eccentric conical journal and
its stationary bearing surface. The bearing axis is the global $z$-axis.
Coordinates are generated in millimetres and converted to metres for the
solver mesh.

- bearing length: 100 mm;
- mean journal radius: 50 mm;
- radial clearance: 50 μm;
- semi-cone angle: 10°;
- eccentricity ratio: 0.6;
- central surface-pressure inlet diameter: 4 mm;
- feed pressure: 500 kPa gauge;
- both axial ends: zero gauge pressure;
- density: ρ = 860 kg/m³;
- dynamic viscosity: μ = 0.0277 Pa·s;
- kinematic viscosity: ν = 3.22093023256 × 10⁻⁵ m²/s.

The incompressible OpenFOAM pressure is kinematic. The conversion used for
all reported pressure values is

$
  p_"abs" ["Pa"] = 101325 + 860 p_"kinematic".
$

The journal-wall angular velocity at 2000 rpm is 209.44 rad/s. The calculation
uses the incompressible laminar SIMPLE solver on four MPI ranks.

#pagebreak()

= Mesh assessment

The imported mesh contains 1,252,800 hexahedral cells, 1,364,688 points, and
the five required boundary patches: `journal_wall`, `stationary_wall`,
`pressure_feed`, `axial_end_z0`, and `axial_end_zl`.

#figure(
  image("../evidence/conical_journal/oq90_3d/full_3d_fluent_oq_overview.png", width: 100%),
  caption: [
    Full three-dimensional mesh-quality projection. The orthogonal-quality
    metric follows the Fluent definition, but the same mesh is used here for
    OpenFOAM.
  ],
)

The standard OpenFOAM mesh check returned `Mesh OK`. Maximum aspect ratio is
124.19, maximum and mean non-orthogonality are 13.45° and 8.11°, and maximum
skewness is 0.4408. The independently calculated Fluent-equivalent orthogonal
quality has minimum 0.922660 and mean 0.970117; no cell falls below 0.9.

The extended mesh check flags the normalized determinant for 1,234,496
thin-film cells. Its minimum is $5.27 times 10^(-8)$, below the default
0.001 threshold. All other extended topology and geometry checks pass. The
standard check therefore passes, while the extended determinant remains a
thin-film conditioning caveat.

= Run progression

The solution was continued through four states:

- zero speed with all boundaries at atmospheric pressure;
- zero speed with the 0.5 MPa gauge feed;
- pressure-fed operation at 496.563 rpm;
- pressure-fed operation at 2000 rpm.

Each state started from the preceding converged field. The four branches added
1, 127, 138, and 135 SIMPLE iterations, respectively.

#figure(
  image("../evidence/conical_journal/openfoam_single_phase_oq90/run_summary.png", width: 100%),
  caption: [Pressure, force, torque, and boundary-flow progression across the four retained states.],
)

At zero speed, the pressure-fed case remains between 0.101 and 0.602 MPa
absolute and carries only 0.806 kN of radial load. At 496.563 rpm, the radial
load is 28.830 kN and the torque magnitude is 3.443 N·m. At 2000 rpm, these
increase to 116.081 kN and 13.871 N·m. Feed flow changes by less than 1% across
the speed continuation.

#pagebreak()

= Convergence at 2000 rpm

The 2000 rpm branch reached the declared initial-residual limits of
$p < 5 times 10^(-5)$ and $U < 1 times 10^(-5)$. At the terminal iteration,
the pressure initial residual was $4.52 times 10^(-5)$, and the three velocity
initial residuals were $2.31 times 10^(-6)$, $1.42 times 10^(-6)$, and
$9.75 times 10^(-6)$.

#figure(
  image("../evidence/conical_journal/openfoam_single_phase_oq90/convergence_2000rpm.png", width: 100%),
  caption: [
    Residuals, pressure extrema, integrated load and torque, and boundary-flow
    histories for the final 135-iteration branch.
  ],
)

The final log records:

#text(font: "JetBrains Mono", size: 8pt)[
  SIMPLE solution converged in 402 iterations\
  U initial residuals = 2.31e-06, 1.42e-06, 9.75e-06\
  p initial residual = 4.52e-05; final residual = 9.94e-06\
  sum(boundary phi) = 2.78514627202e-10 m3/s\
  maxMag(U) = 11.8030261167 m/s
]

The net boundary flow is $2.785 times 10^(-10)$ m³/s, or 0.0229% of the
feed flow. The maximum internal-cell speed is 11.803 m/s.

#pagebreak()

= Pressure and velocity fields

#figure(
  image("../evidence/conical_journal/openfoam_single_phase_oq90/pressure_3d_2000rpm.png", width: 100%),
  caption: [Absolute-pressure field on the three-dimensional conical film at 2000 rpm.],
)

The hydrodynamic high-pressure lobe reaches 17.413 MPa absolute, equivalent to
17.312 MPa gauge. The opposite lobe reaches −17.160 MPa absolute in the fully
filled single-phase model. The peak gauge-pressure ratio is

$
  overline(p)_"max"
  = frac(p_"abs,max" - p_"atm", p_s)
  = frac(17.4131039 - 0.101325, 0.5)
  = 34.6236.
$

For the paper's 2000 rpm point at the same cone angle and eccentricity ratio,
the digitized peak ratios are approximately 35.6 for FEA and 32.0 for Fluent.
The present peak is 2.74% below the paper FEA value and 8.20% above the paper
Fluent value. This is a comparison of one scalar, not the complete pressure
distribution.

#figure(
  image("../evidence/conical_journal/openfoam_single_phase_oq90/unwrapped_rotating_fields.png", width: 100%),
  caption: [
    Unwrapped absolute pressure and velocity magnitude at 496.563 and
    2000 rpm. The central feed meridian is shown at 180°.
  ],
)

The pressure-lobe amplitude and maximum velocity both increase with journal
speed. The feed affects the field locally near the central meridian but does
not control the main circumferential pressure pattern.

#pagebreak()

= Bearing quantities

OpenFOAM reports the total force at 2000 rpm as

$
  (F_x, F_y, F_z)
  = (116077.9501, -844.1044, -292.0295) " N".
$

The radial and axial load magnitudes are

$
  W_r = sqrt(F_x^2 + F_y^2) = 116081.019 " N" = 116.081 " kN",
$

$
  W_a = abs(F_z) = 292.03 " N".
$

With $p_s = 500000$ Pa and $R_j = 0.05$ m, the corresponding
nondimensional loads are

$
  overline(W)_r = frac(W_r, p_s R_j^2) = 92.8648,
  quad
  overline(W)_a = frac(W_a, p_s R_j^2) = 0.2336.
$

The shaft-axis torque magnitude is 13.871 N·m. Its equivalent tangential force
at the 50 mm journal radius is

$
  F_t = frac(abs(M_z), R_j) = frac(13.8711044, 0.05) = 277.42 " N".
$

At 2000 rpm, the corresponding mechanical power is

$
  P = abs(M_z) omega = 13.8711044 times 209.44 = 2905.16 " W"
  = 2.905 " kW".
$

The minimum film thickness follows directly from the geometry:

$
  h_"min" = (1 - epsilon) cos(gamma) c
  = (1 - 0.6) cos(10 degree) times 50 " μm"
  = 19.70 " μm".
$

This thickness is a geometric value, not a minimum inferred from the CFD
field.

The pressure-feed flow magnitude is 1.215 mL/s. The two axial-end discharges
are 0.6741 and 0.5413 mL/s. Flow signs in the raw output use OpenFOAM's
outward-normal convention: feed inflow is negative and axial discharge is
positive.

#pagebreak()

= Teammate Fluent comparison

The teammate's Fluent calculation uses the same geometry and mesh but a
slightly different dynamic viscosity. The comparison below is included to
check the scale and sign of the integrated OpenFOAM quantities, not to present
the Fluent work as part of this report.

- *Radial load:* OpenFOAM 116.081 kN; Fluent 116.394 kN; difference −0.269%.
- *Axial load:* OpenFOAM 292.03 N; Fluent 271.75 N; difference +7.46%.
- *Torque:* OpenFOAM 13.871 N·m; Fluent 13.905 N·m; difference −0.246%.
- *Peak pressure ratio:* OpenFOAM 34.6236; Fluent 34.6910; difference −0.194%.
- *Feed volume flow:* OpenFOAM 1.215 mL/s; Fluent 1.208 mL/s; difference +0.583%.

The radial load, torque, peak pressure, and feed flow agree within 0.6%. The
axial load requires a closer comparison of pressure convention, force-surface
selection, and report definitions before assigning the difference to either
solver.

= Interpretation and next work

The calculation establishes that the full OQ90 mesh imports into OpenFOAM,
passes the standard mesh check, and supports a converged steady single-phase
solution at 2000 rpm. It also provides reproducible pressure, force, torque,
velocity, and boundary-flow outputs for comparison with other implementations.

The following work remains:

- align viscosity, pressure convention, force surfaces, and exported
  quantities between OpenFOAM and Fluent;
- resolve the axial-load difference;
- define and enforce a tighter boundary-flow acceptance target;
- run at least three mesh resolutions;
- compare the complete circumferential pressure distribution or experimental
  pressure taps rather than only the peak value.

The current evidence does not establish grid independence or experimental
validation. Those are the main limits on interpreting the computed loads and
pressures as bearing-performance predictions.

= Evidence

The calculation and its retained logs are under
`out/openfoam_oq90_single_phase`. The report figures are copied into
`evidence/conical_journal/openfoam_single_phase_oq90` so that the document does
not depend on the ignored simulation tree.

#text(font: "JetBrains Mono", size: 7.5pt)[
  RUN_RESULTS.md\
  1071989ae4c0a87d04240a2d3ba5a0d56f4c8fb977261b506c75a14237aabf17\
  log.2000rpm\
  8cf4b9220e929777d299522f54da02a459e00f9d2b0c650e5a198fe8ba5d53a7\
  log.checkMesh.standard\
  746ecafccbed77b4c127766f6be76aa3005ae46d0fd44b63bf7a2d7e18f5721d\
  evidence tree\
  d2d7a003a3e00a8109b12680d02b01d0b600f4c423f3d51ad18f8546db2d4f61
]
