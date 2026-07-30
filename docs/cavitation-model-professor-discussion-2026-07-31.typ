#let navy = rgb("#17324D")
#let blue = rgb("#1F6E8C")
#let cyan = rgb("#DDEFF4")
#let pale = rgb("#F3F7F9")
#let amber = rgb("#FFF1CC")
#let red = rgb("#B33A3A")
#let redpale = rgb("#FBE8E8")
#let green = rgb("#2E6B57")
#let greenpale = rgb("#E7F2ED")
#let grey = rgb("#55626D")

#set document(
  title: "Cavitation Model Decision Brief — Eccentric Conical Bearing",
  author: "BTP project",
  date: datetime(year: 2026, month: 7, day: 31),
)
#set page(
  paper: "a4",
  margin: (top: 18mm, bottom: 18mm, left: 18mm, right: 18mm),
  header: align(right)[
    #text(size: 7.5pt, fill: grey)[CAVITATION MODEL DECISION BRIEF · 31 JULY 2026]
  ],
  footer: align(center)[
    #text(size: 7.5pt, fill: grey)[#context counter(page).display("1")]
  ],
)
#set text(
  font: "IBM Plex Sans",
  size: 9.25pt,
  fill: rgb("#1C2833"),
  lang: "en",
)
#set par(justify: true, leading: 0.58em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => block(
  above: 10pt,
  below: 6pt,
  stroke: (bottom: 1pt + blue),
  inset: (bottom: 4pt),
)[
  #text(font: "IBM Plex Serif", size: 17pt, weight: "semibold", fill: navy)[#it.body]
]
#show heading.where(level: 2): it => block(
  above: 8pt,
  below: 4pt,
)[
  #text(size: 11.5pt, weight: "semibold", fill: blue)[#it.body]
]
#show link: set text(fill: blue)
#show table.cell: set text(size: 7.8pt)
#set table(
  stroke: 0.45pt + rgb("#B8C5CC"),
  inset: 4pt,
)
#set list(indent: 15pt, body-indent: 5pt, spacing: 3pt)
#set enum(indent: 15pt, body-indent: 5pt, spacing: 3pt)

#let tag(body, fill: cyan, color: navy) = box(
  fill: fill,
  radius: 3pt,
  inset: (x: 5pt, y: 2pt),
)[#text(size: 7.3pt, weight: "semibold", fill: color)[#body]]

#let callout(title, body, fill: pale, stroke: blue) = block(
  width: 100%,
  fill: fill,
  stroke: 0.8pt + stroke,
  radius: 4pt,
  inset: 9pt,
)[
  #text(weight: "bold", fill: stroke)[#title]
  #v(3pt)
  #body
]

#let source(label, url) = link(url)[#text(size: 7.5pt, weight: "semibold")[#label]]

#align(center)[
  #v(17mm)
  #text(font: "IBM Plex Serif", size: 29pt, weight: "bold", fill: navy)[
    Cavitation Model\
    Decision Brief
  ]
  #v(5mm)
  #text(size: 14pt, weight: "medium", fill: blue)[
    Eccentric conical hydrodynamic journal bearing
  ]
  #v(3mm)
  #text(size: 10.5pt, fill: grey)[Professor discussion · 31 July 2026]
  #v(12mm)
]

#callout(
  "The point of tomorrow’s meeting",
  [
    Review the four executed mechanism screens, select the *primary physical
    claim*, and approve the next validation evidence. The professor is not
    being asked to invent missing oil properties. Every input below is tied to
    the conical paper, a named published surrogate, a calculation from that
    source, or an explicitly labelled simplification.
  ],
  fill: amber,
  stroke: rgb("#A16B00"),
)

#v(8mm)

#grid(
  columns: (1fr, 1fr),
  gutter: 8mm,
  [
    #text(size: 8pt, weight: "semibold", fill: grey)[RECOMMENDED PRIMARY MODEL]
    #v(2pt)
    #text(size: 15pt, weight: "bold", fill: navy)[Reynolds–JFO / Elrod–Adams]
    #v(3pt)
    Use a mass-conserving film-rupture model for the paper-aligned bearing
    performance study. It matches the thin-film governing framework far better
    than importing arbitrary vapour data into a 3-D phase-change model.
  ],
  [
    #text(size: 8pt, weight: "semibold", fill: grey)[SEPARATE RESEARCH BRANCH]
    #v(2pt)
    #text(size: 15pt, weight: "bold", fill: navy)[3-D phase-resolved CFD]
    #v(3pt)
    Oil-vapour, non-condensable-gas, and air-ingestion screens have now been
    run separately. None passed its final physical/convergence gate, so one
    branch should be selected and repaired rather than blending the models.
  ],
)

#v(9mm)

#table(
  columns: (1fr, 1fr, 1fr),
  fill: (_, y) => if y == 0 { cyan } else { white },
  table.header(
    [*Evidence already available*],
    [*Evidence still missing*],
    [*Decision needed*],
  ),
  [
    Exact 3-D Hex8 mesh; sourced A–D mechanism screens; JFO through 4000 rpm;
    oil-vapour onset; gas and ventilation diagnostics; paper scalar check.
  ],
  [
    Exact oil identity and temperature; vapour/gas data; end flooding or
    ventilation condition; matching cavity experiment.
  ],
  [
    JFO, vapour, gas, or ventilation claim; data-acquisition route; pressure
    convention; validation and RPM matrix.
  ],
)

#v(8mm)
#text(size: 8pt, fill: grey)[
  Prepared from project results generated 30 July 2026. Green means a declared
  numerical gate passed—not experimental validation. Unsettled and rejected
  states are retained as evidence and are not promoted to performance results.
]

#pagebreak()

= Executed four-track outcome

#callout(
  "Bottom line",
  [
    All four interpretations were implemented and exercised. *Only
    Reynolds–JFO passed its declared numerical gates.* The oil-vapour field
    reached phase onset but did not plateau, the gas case failed its zero-speed
    mass-balance gate, and ventilation reached negative absolute pressure
    before air propagated into the low-pressure region.
  ],
  fill: amber,
  stroke: rgb("#A16B00"),
)

#v(5pt)
#image("handoff/media/cavitation_four_track/four_track_status.png", width: 100%)

#v(5pt)
#table(
  columns: (0.48fr, 1.3fr, 0.88fr, 1.02fr, 1.6fr),
  fill: (_, y) => if y == 0 { cyan } else if y == 1 { greenpale } else if y == 4 { redpale } else { pale },
  table.header([*Track*], [*Mechanism*], [*State*], [*Key value*], [*Decision*]),
  [A], [Reynolds–JFO], [4000 rpm], [$p_"g,max"=16.495$ MPa], [Accepted numerical sensitivity],
  [B], [oil vapour], [28 rpm], [$alpha_"oil,min"=0.1112$], [Unsettled phase field],
  [C], [non-condensable gas], [0 rpm], [2.035% imbalance], [Zero-speed gate failed],
  [D], [atmospheric ventilation], [3500 rpm], [$p_"abs,min"=-31.76$ MPa], [Physically reject],
)

#v(5pt)
Track A remains an *unvalidated numerical candidate*. Native OpenFOAM and an
independent Python implementation agree closely, but both share the same
physics and input assumptions. Tracks B–D are diagnostic evidence only.

#pagebreak()

= Geometry and boundary values used

#table(
  columns: (1.35fr, 1.02fr, 2.25fr),
  fill: (_, y) => if y == 0 { cyan } else if calc.odd(y) { pale } else { white },
  table.header([*Quantity*], [*Value*], [*Meaning/source*]),
  [Bearing length], [100 mm], [Direct conical-bearing case],
  [Mean radius], [50 mm], [Direct conical-bearing case],
  [Semi-cone angle], [10°], [Selected direct-paper geometry],
  [Radial clearance $c$], [0.05 mm], [50 micrometres],
  [Eccentricity ratio], [0.6], [$e=0.03$ mm],
  [Film-thickness range], [0.02–0.08 mm], [$c(1-epsilon)$ to $c(1+epsilon)$],
  [Pressure feed], [4 mm diameter], [Circular surface patch at $z=50$ mm],
  [Feed pressure], [601325 Pa absolute], [Declared assumption: 0.5 MPa gauge],
  [Axial openings], [101325 Pa absolute], [Atmospheric boundary scenario],
  [Gravity], [zero], [Therefore VOF $p_"rgh"=p$],
)

#v(5pt)
#figure(
  image("handoff/media/cavitation_four_track/geometry_mesh/film_thickness_map.png", width: 100%),
  caption: [
    Exact same-z radial film thickness. This is a geometry diagnostic—not a
    CFD pressure result. The feed direction and radial minimum are distinct.
  ],
)

#callout(
  "Pressure and end-condition boundary",
  [
    The direct paper does not conclusively state whether 0.5 MPa is gauge or
    absolute or whether the axial ends are flooded or air-accessible. The
    values above define the scenario that was run; they are not silently
    promoted to missing experimental facts.
  ],
  fill: amber,
  stroke: rgb("#A16B00"),
)

#pagebreak()

= Mesh used in each model

#table(
  columns: (1.25fr, 1.55fr, 1.75fr),
  fill: (_, y) => if y == 0 { cyan } else if calc.odd(y) { pale } else { white },
  table.header([*Check*], [*Track A · Reynolds–JFO*], [*Tracks B–D · 3-D VOF*]),
  [Representation], [Reduced conical midsurface], [Full conical film volume],
  [Resolution], [256 × 80 background], [1,252,800 Hex8; 12 gap layers],
  [Points], [midsurface mesh], [1,364,688],
  [Pressure feed], [96 cells; 32 rim faces], [396 boundary Quad4 faces],
  [Minimum Fluent-equivalent OQ], [0.956754], [0.922660],
  [Mean Fluent-equivalent OQ], [—], [0.970117],
  [Cells below OQ 0.9], [0], [0],
  [OpenFOAM check], [max non-orthogonality 7.64865°], [standard `checkMesh`: Mesh OK],
)

#v(5pt)
#figure(
  image("handoff/media/cavitation_four_track/geometry_mesh/accepted_geometry_and_mesh.png", width: 100%),
  caption: [Track A reduced body-fitted geometry and conforming 4 mm feed.],
)

#v(3pt)
#figure(
  image("handoff/media/oq90_3d/full_3d_fluent_oq_overview.png", width: 68%),
  caption: [
    Full 3-D mesh used by B–D. Quality is projected conservatively through all
    12 physical gap layers; no cell is below the 0.9 OQ gate.
  ],
)

#pagebreak()

= Lubricants and model values

== Tracks A and B · published SAE 10W-40 surrogate

Muchammad et al. (2024) provide an internally named simulation set. They do
not identify a commercial product or operating temperature, so these values
are a published surrogate—not a property claim for the paper’s V-32 oil.

#table(
  columns: (1.75fr, 1.05fr, 1.6fr),
  fill: (_, y) => if y == 0 { cyan } else if calc.odd(y) { pale } else { white },
  table.header([*Input*], [*Value*], [*Used by*]),
  [Liquid density], [850 kg/m³], [A and B],
  [Liquid dynamic viscosity], [0.0125 Pa·s], [A and B],
  [Saturation/rupture pressure], [29185 Pa absolute], [A pressure floor; B $p_"sat"$],
  [Vapour density], [10.95 kg/m³], [B only],
  [Vapour dynamic viscosity], [$2.0 times 10^(-5)$ Pa·s], [B only],
)

#v(5pt)
Track B uses OpenFOAM 14 `incompressibleVoF` + `VoFCavitation` +
Schnerr–Sauer:

#table(
  columns: (1.45fr, 1.3fr, 2.05fr),
  fill: (_, y) => if y == 0 { cyan } else if calc.odd(y) { pale } else { white },
  table.header([*Input*], [*Value*], [*Origin/status*]),
  [Nuclei diameter $d_"nuc"$], [$2 times 10^(-6)$ m], [Twice the source’s 1 µm Zwart radius],
  [Number density $n$], [$1.1942592 times 10^14$ m⁻³], [Calculated to reproduce $alpha_"nuc"=5 times 10^(-4)$],
  [$C_c,\ C_v$], [1, 1], [OpenFOAM Schnerr–Sauer screen],
  [Surface tension], [0 N/m], [Declared homogeneous-mixture simplification; source omitted it],
)

#callout(
  "Why this is not a Zwart reproduction",
  [
    The source reports Zwart $R_b=10^(-6)$ m,
    $alpha_"nuc"=5 times 10^(-4)$, $F_"vap"=50$, and
    $F_"cond"=0.01$. OpenFOAM Foundation 14 has no native Zwart model in this
    installation. The explicit Schnerr–Sauer translation is a compatibility
    screen, not fitted oil data.
  ],
  fill: amber,
  stroke: rgb("#A16B00"),
)

#pagebreak()

== Track C · ISO VG32 turbine oil and non-condensable air

Values are taken or evaluated from Wettmarshausen et al. (2025).

#table(
  columns: (1.75fr, 1.25fr, 1.55fr),
  fill: (_, y) => if y == 0 { cyan } else if calc.odd(y) { pale } else { white },
  table.header([*Input*], [*Value*], [*Interpretation*]),
  [Oil temperature], [50°C], [Isothermal adaptation],
  [Oil density], [837.1 kg/m³], [Paper Eq. 12],
  [Oil dynamic viscosity], [0.0236624469 Pa·s], [Evaluated from paper Eq. 10],
  [Oil $c_p$; conductivity], [2140.2 J/(kg·K); 0.134 W/(m·K)], [Source values],
  [Air reference density], [1.185 kg/m³], [25°C and 101325 Pa],
  [Air viscosity], [$1.83 times 10^(-5)$ Pa·s], [Source value],
  [Air $c_p$; conductivity], [1004.4 J/(kg·K); 0.0261 W/(m·K)], [Source values],
  [Bunsen coefficient], [0.09], [Pressure-dependent free-gas initialization],
  [Initial/feed fractions], [$alpha_"gas"=0.0161711$; $alpha_"oil"=0.9838289$], [Evaluated at 601325 Pa],
  [Phase change; surface tension], [none; 0 N/m], [Homogeneous free-gas model],
)

This is compressible free gas/pseudo-cavitation. It does not implement
dissolved-air desorption.

== Track D · VG22 oil and atmospheric air

#table(
  columns: (1.75fr, 1.15fr, 1.65fr),
  fill: (_, y) => if y == 0 { cyan } else if calc.odd(y) { pale } else { white },
  table.header([*Input*], [*Value*], [*Source/condition*]),
  [Oil density; viscosity], [860 kg/m³; 0.019 Pa·s], [VG22 at 40°C],
  [Air density; viscosity], [1.23 kg/m³; $1.75 times 10^(-5)$ Pa·s], [Air at 28°C],
  [Oil–air surface tension], [0.04 N/m], [Measured by source],
  [Feed / axial reverse flow], [pure oil / pure air], [Declared boundary mechanism],
  [Initial film; phase change], [oil-filled; none], [Ventilation, not vaporisation],
  [Wall contact angle], [not supplied], [`zeroGradient`; no invented value],
)

The film-only mesh has no external reservoir or meniscus, so Track D is a
boundary-ingestion screen rather than a resolved ventilation experiment.

== Numerical controls actually run

#table(
  columns: (0.42fr, 1.18fr, 2.75fr),
  fill: (_, y) => if y == 0 { cyan } else if calc.odd(y) { pale } else { white },
  table.header([*Track*], [*Time treatment*], [*Executed controls*]),
  [A], [`jfoBearingFoam`; pseudo-step], [
    256 circumferential divisions; max 8 revolutions and 200 active-set
    iterations; convergence $10^(-8)$, policy tolerance $10^(-7)$.
  ],
  [B], [`incompressibleVoF`; `localEuler`], [
    1 outer / 2 pressure correctors; 2 alpha correctors and 2 MULES iterations;
    max Co 1, max alpha Co 0.5. Indices 0–145 are not seconds.
  ],
  [C], [`compressibleVoF`; `localEuler`], [
    1 outer / 3 pressure correctors; 2 alpha correctors and 2 MULES iterations;
    max Co 1, max alpha Co 0.5. Indices 0–40 are not seconds.
  ],
  [D], [`incompressibleVoF`; physical `Euler`], [
    Initial $Delta t=10^(-6)$ s; adaptive maximum $10^(-5)$ s; max Co 0.3,
    max alpha Co 0.2; 1 outer / 3 pressure correctors. Stopped at 4.368 µs.
  ],
)

#v(3pt)
#text(size: 8pt, fill: grey)[
  Each 3-D case used four Scotch partitions. B/C frames are pseudo-time
  convergence states; D frames are physical transient states.
]

#pagebreak()

= Quantitative results and higher RPM

#image("handoff/media/cavitation_four_track/paper_comparison_and_higher_rpm.png", width: 100%)

#v(5pt)
#table(
  columns: (0.8fr, 1.25fr, 1.05fr, 1.05fr),
  fill: (_, y) => if y == 0 { cyan } else { white },
  table.header([*Speed*], [*$p_"g,max"$*], [*$theta_"min"$*], [*Ruptured area*]),
  [496.563 rpm], [2.2063 MPa], [0.27368], [32.913%],
  [1000 rpm], [4.4001 MPa], [0.26862], [39.622%],
  [2000 rpm], [8.6075 MPa], [0.26557], [46.931%],
  [3000 rpm], [12.6268 MPa], [0.26394], [51.257%],
  [4000 rpm], [16.4950 MPa], [0.26315], [54.163%],
)

#v(5pt)
At 4000 rpm, feed flow is $1.00125029 times 10^(-5)$ m³/s. Native OpenFOAM
and independent Python peak pressures differ by at most 0.0257% over the
sweep. Speeds of 3000 and 4000 rpm are explicitly extrapolative sensitivities.

#figure(
  image("handoff/media/jfo_sae10w40/final_fields_4000rpm.png", width: 87%),
  caption: [
    Track A pressure and fill fields at the final accepted numerical
    sensitivity point. JFO $theta$ is not VOF vapour fraction.
  ],
)

The earlier paper-viscosity JFO run gives
$p_"max,gauge"/p_s=34.8411$ at 2000 rpm, versus graph-read values of about
35.6 for FEM and 32.0 for Fluent: −2.13% and +8.88%. This checks one scalar
only. The SAE 10W-40 result is not called a paper match because its viscosity
is different.

#pagebreak()

= Three-dimensional screening evidence

#image("handoff/media/cavitation_four_track/screening_diagnostics.png", width: 100%)

#v(5pt)
#table(
  columns: (0.42fr, 1.25fr, 2.95fr),
  fill: (_, y) => if y == 0 { cyan } else if y == 3 { redpale } else { pale },
  table.header([*Track*], [*Final retained metric*], [*Why it stopped*]),
  [B], [$alpha_"oil,min"=0.11117$ at 28 rpm], [
    Onset near 24.4 rpm; mean oil fraction 0.998999 and mass imbalance
    0.1321%, but the phase field did not plateau.
  ],
  [C], [2.035% imbalance at 0 rpm], [
    $alpha_"oil,min"=0.90709$ and mean 0.94015 at step 40; failed the declared
    0.5% conservation gate, so rotation was not attempted.
  ],
  [D], [$p_"abs,min"=-31.7571$ MPa], [
    At 3500 rpm startup, air had not propagated before pressure became
    inadmissible; the standalone film-only ventilation model was rejected.
  ],
)

#v(5pt)
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 4mm,
  [
    #image("handoff/media/cavitation_four_track/track_b_vapour/alpha_oil_003.png", width: 100%)
    #align(center)[#text(size: 7.2pt, weight: "semibold")[B · oil vapour]]
  ],
  [
    #image("handoff/media/cavitation_four_track/track_c_gas/alpha_oil_002.png", width: 100%)
    #align(center)[#text(size: 7.2pt, weight: "semibold")[C · free gas]]
  ],
  [
    #image("handoff/media/cavitation_four_track/track_d_ventilation/alpha_oil_002.png", width: 100%)
    #align(center)[#text(size: 7.2pt, weight: "semibold")[D · ventilation]]
  ],
)

#v(5pt)
#text(size: 8pt, fill: grey)[
  Full-resolution frames and MP4/GIF checkpoint animations are stored under
  `docs/handoff/media/cavitation_four_track/`; the accepted JFO animation is
  under `docs/handoff/media/jfo_sae10w40/`. Animation shows saved solver
  states, but does not by itself establish physical convergence.
]

#pagebreak()

= What the present OpenFOAM runs establish

#table(
  columns: (1.32fr, 0.75fr, 0.86fr, 0.86fr, 1.15fr),
  fill: (_, y) => if y == 0 { cyan } else if calc.odd(y) { pale } else { white },
  table.header(
    [*Case*], [*$U_"max"$*], [*$p_"abs,min"$*], [*$p_"abs,max"$*], [*Meaning*],
  ),
  [0 rpm, atmospheric], [0 m/s], [0.101 MPa], [0.101 MPa], [Physical identity pass],
  [0 rpm, 0.5 MPa feed], [2.128 m/s], [0.101 MPa], [0.602 MPa], [Physical pressure-fed pass],
  [496.563 rpm, feed], [2.931 m/s], [−4.166 MPa], [4.418 MPa], [Converged; physically reject],
  [2000 rpm, feed], [11.803 m/s], [−17.160 MPa], [17.413 MPa], [Converged; physically reject],
)

#v(5pt)

The rotating single-phase solver converged numerically, but a liquid cannot
sustain those large negative absolute pressures. More SIMPLE iterations will
not supply the missing rupture or phase physics. At 2000 rpm, the gauge peak
ratio is $P_"max"/P_s = 34.6236$, compared with approximately 35.6 in the
paper’s FEM graph and 32.0 in its Fluent graph. That is a useful one-scalar
check—2.74% below the FEM value and 8.20% above the Fluent value—but it does
*not* validate the negative-pressure half, pressure curve, load, torque,
leakage, or cavitation model.

#figure(
  image("../out/openfoam_oq90_single_phase/visualization/run_summary.png", width: 100%),
  caption: [
    Current single-phase run summary. Red rotating cases are retained only as
    numerical initial states and diagnostics.
  ],
)

== What $p$, $U$, $p_"rgh"$, and $alpha$ mean

#table(
  columns: (0.72fr, 1.15fr, 2.6fr),
  fill: (_, y) => if y == 0 { cyan } else if calc.odd(y) { pale } else { white },
  table.header([*Field*], [*Units here*], [*Interpretation*]),
  [$p$], [$"m"^2/"s"^2$], [
    Kinematic gauge pressure in the current incompressible single-phase case.
    Convert with $p_"abs" = 101325 + rho p$, using $rho=860 "kg/m"^3$.
  ],
  [$U$], [m/s], [
    Oil velocity vector. It includes journal-driven circumferential motion,
    feed inflow, and axial discharge; it is not journal speed alone.
  ],
  [$p_"rgh"$], [Pa], [
    Pressure field in the VOF case. With gravity set to zero, $p_"rgh"=p$;
    use absolute pressure consistently for cavitation.
  ],
  [$alpha_"oil"$], [0–1], [
    Local oil volume fraction in a VOF model. It is not the same as JFO fill
    fraction $theta$, and it cannot change if a guard stops before phase update.
  ],
)

#pagebreak()

= First decide what “cavitation” means

The four options below are not interchangeable. Choosing the solver before
choosing the mechanism is how unsupported material values enter a model.

#table(
  columns: (1.05fr, 1.42fr, 1.35fr, 1.22fr),
  fill: (_, y) => if y == 0 { cyan } else if calc.odd(y) { pale } else { white },
  table.header([*Mechanism*], [*Physical statement*], [*Implementation*], [*Main missing evidence*]),
  [
    *A · Film rupture*\
    #tag([recommended], fill: greenpale, color: green)
  ],
  [
    The film becomes partially filled below a rupture boundary while liquid
    mass is conserved through rupture and reformation.
  ],
  [
    Reynolds equation with JFO / Elrod–Adams complementarity; fields $p$ and
    fill fraction $theta$.
  ],
  [
    Effective $p_c$, flooded/vented end condition, conical/feed discretization,
    and external validation.
  ],
  [
    *B · Oil vapour*
  ],
  [
    Liquid vaporizes and condenses around the saturation pressure of the exact
    oil mixture at the operating temperature.
  ],
  [
    OpenFOAM `incompressibleVoF` + `VoFCavitation` + Schnerr–Sauer; fields
    $U$, $p_"rgh"$, $alpha_"oil"$.
  ],
  [
    Exact product, temperature, $p_"sat"(T)$, vapour density/viscosity,
    surface tension, nuclei, and calibration.
  ],
  [
    *C · Gas / pseudo-cavitation*
  ],
  [
    Entrained air expands, or dissolved air is released, as pressure falls.
    For low-volatility oils this can dominate the visible cavity.
  ],
  [
    Compressible non-condensable-gas VOF, or a custom dissolved-air
    mass-transfer model coupled to solubility.
  ],
  [
    Initial free-gas fraction, gas solubility versus pressure/temperature,
    desorption kinetics, and experiment.
  ],
  [
    *D · Atmospheric ventilation*
  ],
  [
    Air enters from an axial end or reservoir and displaces oil; no
    oil-to-air phase-change source exists.
  ],
  [
    Transient oil–air VOF with a real air-access domain, correct backflow phase
    fractions, interface tension, and wetting.
  ],
  [
    Rig/end geometry, reservoir/meniscus, oil–air surface tension and contact
    angle. The present film-only mesh is incomplete for a resolved meniscus.
  ],
)

#v(7pt)

#callout(
  "Recommended claim for this BTP",
  [
    Use *A* as the primary performance model because Gangrade et al. use a
    modified Reynolds thin-film formulation and do not disclose a complete
    liquid–vapour or air model. Treat *B*, *C*, or *D* as a separate,
    mechanism-specific CFD study. Do not call a JFO cavity “vapour volume,” and
    do not call a VOF vapour fraction a JFO fill fraction.
  ],
  fill: greenpale,
  stroke: green,
)

== Why pure-vapour values cannot be guessed

The target paper labels the liquid as V-32 and provides
$rho_l=860 "kg/m"^3$ and $mu_l=0.0277 "Pa s"$, but no exact product, operating
temperature, vapour pressure, vapour phase properties, dissolved-air content,
or nuclei calibration. A mineral hydraulic oil is a mixture, not a single
molecular species with one obvious “oil-vapour” material card.

For scale only, an official Shell Tellus S2 MX 32 SDS reports the product as a
mixture of highly refined mineral oils and additives, an *estimated*
vapour-pressure bound below 0.5 Pa at 20 °C, and density 854 kg/m³ at 15 °C.
That makes a useful sourced substitute/sensitivity case; it does not make
those numbers properties of the paper’s unidentified V-32 oil.
#source(
  "[D1] Official Shell Tellus S2 MX 32 SDS",
  "https://www.epc.shell.com/DocumentManagement/BlobDocumentDownload?DocId=127795886",
)

#pagebreak()

= Source-or-measure matrix

This is the answer to “where will the value come from?” The professor chooses
one defensible route; the professor is not the source.

#table(
  columns: (1.02fr, 1.42fr, 1.47fr, 1.2fr),
  fill: (_, y) => if y == 0 { cyan } else if calc.odd(y) { pale } else { white },
  table.header([*Required input*], [*Available source*], [*If unavailable*], [*Claim allowed*]),
  [
    Geometry: $R,L,c,gamma,epsilon$; liquid $rho,mu$; nominal $P_s$
  ],
  [
    Gangrade et al. plus the exact local CAD/mesh. Use the paper values only
    for reproducing the paper baseline.
  ],
  [
    None needed for those disclosed quantities; retain paper ambiguities
    explicitly.
  ],
  [
    Paper-aligned single-phase or JFO comparison.
  ],
  [
    Exact lubricant identity, batch, and operating temperature
  ],
  [
    Bottle label, purchase record, rig log, vendor certificate, and measured
    sump/film temperatures.
  ],
  [
    Either identify and measure it; adopt a *named substitute*; or keep a
    bounded sensitivity study.
  ],
  [
    Quantitative exact-oil claim only after identification.
  ],
  [
    Liquid density and viscosity versus temperature
  ],
  [
    Paper baseline at its unstated temperature; otherwise exact-product TDS
    and ASTM measurement at the chosen temperature.
  ],
  [
    Measure the oil. Do not combine the paper’s density with another product’s
    vapour properties and call the blend “V-32.”
  ],
  [
    Substitute-oil or sensitivity claim when product differs.
  ],
  [
    Feed pressure and gauge/absolute reference
  ],
  [
    Rig transducer calibration and test procedure. The paper says 0.5 MPa but
    does not resolve the reference.
  ],
  [
    Run both explicit interpretations: 500,000 Pa absolute, or 601,325 Pa
    absolute for 0.5 MPa gauge at standard atmosphere.
  ],
  [
    Boundary-condition sensitivity, not paper-confirmed convention.
  ],
  [
    Axial ends: flooded or air-accessible
  ],
  [
    Test-rig drawing, photograph, reservoir level, seal layout, or experiment.
  ],
  [
    Run flooded and vented boundary scenarios separately. Never use an
    atmospheric pressure floor while claiming a sealed flooded film.
  ],
  [
    Scenario study only until the rig condition is known.
  ],
  [
    Oil saturation pressure $p_"sat"(T)$
  ],
  [
    Exact-product SDS/TDS, supplier data, or vapour-pressure measurement. The
    Shell value above is only a sourced substitute bound.
  ],
  [
    Use a stated range and report onset sensitivity, or omit quantitative
    vapour CFD.
  ],
  [
    Qualitative/sensitivity vapour result only.
  ],
  [
    Vapour density and viscosity
  ],
  [
    Valid property model for the actual oil mixture at chosen temperature and
    pressure, or measurement/supplier data.
  ],
  [
    Define an openly named surrogate phase; do not derive a fictitious exact
    vapour from liquid density.
  ],
  [
    Surrogate-model result only.
  ],
  [
    Dissolved/free air and solubility
  ],
  [
    Oil sample preparation plus measurement/estimate; ASTM D3827 provides a
    standard estimation route for gas solubility in petroleum liquids.
  ],
  [
    Use literature ranges only as a sensitivity envelope, with no exact-oil
    claim.
  ],
  [
    Gas-cavitation sensitivity only.
  ],
  [
    Nuclei $n,d_"nuc"$ and Schnerr–Sauer $C_c,C_v$
  ],
  [
    Calibrated benchmark or target experiment. OpenFOAM requires them, but
    they are not supplied by the conical paper.
  ],
  [
    Sweep them and report model-form uncertainty. A water-tutorial value is
    not an oil property.
  ],
  [
    Exploratory phase-transfer result only.
  ],
  [
    Surface tension and contact angle
  ],
  [
    Exact oil–air–solid system at operating temperature: supplier data or
    tensiometer/goniometer measurement.
  ],
  [
    Explicit sensitivity range.
  ],
  [
    Needed for resolved oil–air interface claims.
  ],
  [
    JFO effective cavitation pressure $p_c$
  ],
  [
    Match the declared scenario: vapour bound; measured/calibrated
    gas-influenced value; or ambient pressure only for a vent-connected film.
  ],
  [
    Run the three scenarios separately. Do not tune $p_c$ only to match peak
    load.
  ],
  [
    Scenario-dependent film-rupture prediction.
  ],
)

#v(6pt)
#source(
  "[D2] ASTM D3827—gas solubility in petroleum and other organic liquids",
  "https://store.astm.org/d3827-92r20.html",
)

#callout(
  "If the original oil cannot be identified",
  [
    There are only three honest options: *(1)* measure the actual oil;
    *(2)* choose and name a substitute product, then relabel the work as a
    substitute-oil study; or *(3)* publish bounded sensitivities and stop short
    of quantitative cavitation validation. There is no fourth option in which
    missing phase properties become known because a solver requests them.
  ],
  fill: redpale,
  stroke: red,
)

#pagebreak()

= Model definitions and implementation details

== Track A — mass-conserving Reynolds–JFO

Use pressure $p$ and fill fraction $theta$ on the developed conical film. The
rupture/reformation condition is

#align(center)[
  #box(fill: pale, inset: 8pt, radius: 3pt)[
    $p >= p_c, quad 0 <= theta <= 1, quad (p-p_c)(1-theta)=0.$
  ]
]

In a full-film cell, $theta=1$ and pressure can exceed $p_c$. In a ruptured
cell, $p=p_c$ and $theta<1$. This conserves lubricant through rupture and
reformation, unlike a simple pressure clip or Half-Sommerfeld deletion.

The repository already contains `openfoam/jfoBearingFoam` and the
`jfoPaperExact` case. Treat its 2000 rpm conical result as a *candidate*, not a
validated answer: the feed footprint remains grid-dependent, the conical
assembly has not passed an external manufactured solution, and the meaning of
the imposed 101,325 Pa floor depends on whether air can actually reach the
film.

Minimum implementation gates:

- preserve the physical 4 mm feed footprint on every grid;
- verify the shared complementarity kernel and then the complete conical
  assembly;
- report pressure, $theta$, complementarity residual, local/global mass
  balance, load vector and attitude, torque, and leakage;
- use at least three systematically refined grids after feed consistency;
- compare the complete circumferential pressure curve/taps, not just
  $P_"max"$.

== Track B — 3-D oil-vapour CFD in OpenFOAM 14

#table(
  columns: (1.1fr, 2.8fr),
  fill: (_, y) => if y == 0 { cyan } else if calc.odd(y) { pale } else { white },
  table.header([*Item*], [*Implementation*]),
  [Solver/model], [
    `incompressibleVoF` with the native `VoFCavitation` source and
    Schnerr–Sauer mass transfer. The official class requires the model inputs
    $p_"sat",n,d_"nuc",C_c,C_v$.
  ],
  [Fields], [
    $U$ in m/s, $p_"rgh"$ in Pa, and $alpha_"oil"$ from 0 to 1. With $g=0$,
    $p_"rgh"$ is the absolute static pressure used by the phase-change law.
  ],
  [Initialization], [
    Start from the accepted 0 rpm pressure-fed solution; convert kinematic
    pressure to absolute Pa; initialize $alpha_"oil"=1$; establish a
    cavitation-enabled zero-speed equilibrium before adding rotation.
  ],
  [Boundaries], [
    If 0.5 MPa is gauge, feed $p_"abs"=601325$ Pa and atmospheric ends
    $p_"abs"=101325$ Pa. If it is absolute, use 500000 Pa. Set backflow phase
    fractions according to flooded versus air-accessible ends.
  ],
  [Run type], [
    Use physical transient time for cavity evolution; limit both flow and
    interface Courant numbers, and demonstrate timestep independence. A
    pseudo-time counter must not be presented as seconds or an animation of
    real cavity motion.
  ],
  [Monitors], [
    Residuals, total and per-phase mass balance, $p_"min/max"$, phase volume,
    boundary fluxes, force/load attitude, torque, leakage, and periodic
    statistics if the cavity oscillates.
  ],
)

#v(5pt)
#source(
  "[O1] OpenFOAM 14 VoFCavitation class and required inputs",
  "https://cpp.openfoam.org/v14/classFoam_1_1fv_1_1compressible_1_1VoFCavitation.html",
)

The old exploratory case used $p_"sat"=0.5$ Pa, $sigma=0.03$ N/m, assumed
vapour properties, and water-tutorial nuclei. Those are documented screening
placeholders. They may be used only as a named sensitivity set—not as
paper-derived V-32 data.

#pagebreak()

== Track C — gaseous or pseudo-cavitation

Li et al. formulated oil-film gaseous cavitation from air solubility, and Shen
and Khonsari showed that effective cavitation pressure varies with dissolved
gas and operating conditions. Song, Gu, and Ren validated a gaseous
journal-bearing model without relying on one arbitrarily assigned cavitation
pressure. More recently, Wettmarshausen et al. experimentally validated a 3-D
VOF model in which compressible non-condensable gas explains fractional film
content.

The native OpenFOAM `VoFCavitation` branch is a liquid–vapour mass-transfer
model; it does *not* become a dissolved-air model by renaming the vapour phase
“air.” Two defensible gas implementations exist:

1. *Pseudo-cavitation:* initialize a measured free-gas fraction and solve an
   oil + compressible non-condensable-gas VOF model without oil-to-gas phase
   change.
2. *Dissolved-air release:* add a dissolved-gas variable and a pressure- and
   temperature-dependent solubility/desorption source. This is custom-model
   work and requires measurement or calibration.

#callout(
  "Scope warning",
  [
    Track C is likely more physically relevant than pure oil vapour for a
    low-volatility lubricant, but it is also the least defensible branch
    without air-content data. Recommend it only if the project can obtain an
    oil sample, define preparation/degassing, and validate cavity fraction or
    pressure against an experiment.
  ],
  fill: amber,
  stroke: rgb("#A16B00"),
)

== Track D — atmospheric ventilation

Use transient oil–air VOF with no oil-to-air cavitation mass transfer. The
model needs a real pathway from atmosphere to film, correct pressure and phase
backflow conditions, and ideally an external reservoir/end region so the
meniscus is inside the computational domain. The current film-only mesh can
test an imposed air-ingestion boundary, but it cannot independently predict
the external meniscus or prove that the cavity is connected to ambient air.

== Recommended RPM campaign

Do not extrapolate the rejected full-liquid field. Once one cavitation model
passes its zero-speed and benchmark gates, continue from the previous
converged state:

#table(
  columns: (0.65fr, 1.35fr, 2.4fr),
  fill: (_, y) => if y == 0 { cyan } else if calc.odd(y) { pale } else { white },
  table.header([*Stage*], [*Speed*], [*Purpose*]),
  [0], [0 rpm], [Atmospheric identity, then pressure-fed equilibrium],
  [1], [496.6 rpm], [
    Matches the stated 2.6 m/s surface speed at $R=50$ mm:
    $N=60V/(2 pi R)=496.6$ rpm.
  ],
  [2], [1000 rpm], [Intermediate continuation and cavity-onset/evolution check],
  [3], [2000 rpm], [Paper’s separately labelled operating point],
  [4], [3000 and 4000 rpm], [
    Higher-speed trend only after physical admissibility, grid/timestep
    independence, and thermal/inertial assumptions are checked.
  ],
)

For an isoviscous, fixed-clearance Reynolds model, hydrodynamic pressure tends
to scale approximately with speed. At higher RPM, cavitation extent, heating,
viscosity loss, inertia, turbulence, and deformation can break that scaling.
Therefore the higher-RPM deliverable is a trend with uncertainty—not a linear
extrapolation from the present ±17 MPa fully filled solution.

#pagebreak()

= Validation ladder and acceptance gates

#table(
  columns: (0.55fr, 1.34fr, 2.45fr),
  fill: (_, y) => if y == 0 { cyan } else if calc.odd(y) { pale } else { white },
  table.header([*Gate*], [*Question*], [*Evidence required before advancing*]),
  [1], [Code verification], [
    Does the implementation solve its equations? Use Elrod/Giacopini benchmark
    behavior and the Gravenkamp–Pfeil–Codina manufactured Reynolds–Elrod
    solution; show observed grid order and complementarity/conservation errors.
  ],
  [2], [Independent cavitation benchmark], [
    For OpenFOAM VOF, reproduce a fully disclosed cylindrical journal-bearing
    case such as Concli’s Schnerr–Sauer/Kunz study, or a Song–Gu experimental
    case. Use that paper’s own geometry, oil, pressure, and boundaries; do not
    transfer its properties to the conical case.
  ],
  [3], [Conical liquid baseline], [
    Reproduce Gangrade’s normalized maximum-pressure trend and available
    experimental circumferential pressure taps at the disclosed operating
    conditions. Report the 496.6-versus-2000-rpm inconsistency.
  ],
  [4], [Target application], [
    Apply the checked method to the exact conical mesh. Demonstrate mesh,
    timestep, model-parameter, pressure-reference, and end-boundary
    sensitivity.
  ],
  [5], [Direct validation], [
    Call the combined conical-cavitation result directly validated only after
    matching cavity/pressure/load data exist for the same geometry, oil,
    temperature, feed, end condition, and speed.
  ],
)

== Minimum acceptance checklist

#grid(
  columns: (1fr, 1fr),
  gutter: 8mm,
  [
    *Numerical*
    - no negative absolute pressure below the selected physical floor;
    - residual histories and stable load/torque/flow monitors;
    - total mass imbalance target declared in advance;
    - per-phase balance for VOF;
    - three-grid study and timestep study;
    - results insensitive to further iteration/statistical window.
  ],
  [
    *Physical and comparative*
    - complete pressure curve and tap values;
    - load magnitude, components, and attitude angle;
    - torque/friction and axial leakage;
    - cavity/fill map with an unambiguous definition;
    - pressure convention and flooded/vented condition;
    - uncertainty bars and no transferred-oil relabelling.
  ],
)

#callout(
  "What the existing 2000 rpm image can and cannot show",
  [
    It is suitable for explaining where a fully filled model demands tension
    and why cavitation physics is necessary. It is not a cavitation image.
    Likewise, a rotating camera or interpolated frame sequence is a
    visualization; it is not evidence of transient cavity dynamics unless the
    solver advanced in physical time.
  ],
  fill: redpale,
  stroke: red,
)

#figure(
  image("../out/openfoam_oq90_single_phase/visualization/pressure_3d_2000rpm.png", width: 82%),
  caption: [
    2000 rpm single-phase pressure field: a diagnostic precursor with
    nonphysical tensile pressure, not a cavitating solution.
  ],
)

#pagebreak()

= Papers to cite—and exactly what each supports

#table(
  columns: (0.38fr, 1.52fr, 2.45fr),
  fill: (_, y) => if y == 0 { cyan } else if calc.odd(y) { pale } else { white },
  table.header([*ID*], [*Reference*], [*Use in discussion / boundary*]),
  [[R1]], [
    Gangrade, Phalle & Mantha (2019),\
    #source("DOI 10.1007/s40997-018-0217-2", "https://doi.org/10.1007/s40997-018-0217-2")
  ], [
    Closest target geometry and liquid baseline; modified Reynolds/FEM,
    Fluent comparison, and pressure taps. It does not supply a reproducible
    3-D cavitation property set.
  ],
  [[R2]], [
    Elrod (1981), “A Cavitation Algorithm,”\
    #source("DOI 10.1115/1.3251669", "https://doi.org/10.1115/1.3251669")
  ], [
    Foundational mass-conserving rupture/reformation algorithm for
    lubrication. Cite for why pressure clipping or Half-Sommerfeld deletion is
    insufficient.
  ],
  [[R3]], [
    Giacopini et al. (2010),\
    #source("DOI 10.1115/1.4002215", "https://doi.org/10.1115/1.4002215")
  ], [
    Mass-conserving complementarity formulation. Cite for the $p$–$theta$
    constraints and numerical verification route.
  ],
  [[R4]], [
    Concli (2016),\
    #source("DOI 10.1002/ls.1334", "https://doi.org/10.1002/ls.1334")
  ], [
    OpenFOAM 3-D VOF journal-bearing implementation comparing Kunz and
    Schnerr–Sauer. Its cylindrical cases and their $p_"sat"$ values are
    benchmarks, not properties for this oil.
  ],
  [[R5]], [
    Li et al. (2012),\
    #source("DOI 10.1115/1.4006702", "https://doi.org/10.1115/1.4006702")
  ], [
    Gaseous cavitation based on air solubility. Cite for distinguishing
    dissolved-air release from pure oil-vapour cavitation.
  ],
  [[R6]], [
    Shen & Khonsari (2013),\
    #source("DOI 10.1007/s11249-013-0158-2", "https://doi.org/10.1007/s11249-013-0158-2")
  ], [
    Effective cavitation pressure depends on dissolved gas and operating
    conditions; it is not a universal constant to copy from another oil.
  ],
  [[R7]], [
    Song, Gu & Ren (2015),\
    #source("DOI 10.1177/1350650115576247", "https://doi.org/10.1177/1350650115576247")
  ], [
    Development and experimental validation of a gaseous cavitation model for
    hydrodynamic lubrication.
  ],
  [[R8]], [
    Song & Gu (2015),\
    #source("DOI 10.1115/1.4030633", "https://doi.org/10.1115/1.4030633")
  ], [
    3-D journal-bearing CFD comparing Half-Sommerfeld, vapourous, and gaseous
    treatments against experiment; strong method benchmark, but cylindrical
    and grooved.
  ],
  [[R9]], [
    Wettmarshausen et al. (2025),\
    #source("DOI 10.3390/lubricants13040140", "https://doi.org/10.3390/lubricants13040140")
  ], [
    Experimentally validated 3-D VOF treatment using compressible
    non-condensable gas; supports pseudo-cavitation as a distinct hypothesis.
  ],
  [[R10]], [
    Chen et al. (2022),\
    #source("DOI 10.1016/j.rineng.2022.100582", "https://doi.org/10.1016/j.rineng.2022.100582")
  ], [
    Closest conical multiphase topic, but a spiral-groove bearing. Use as
    supporting evidence, not direct validation of the smooth bearing.
  ],
  [[R11]], [
    Gravenkamp, Pfeil & Codina (2024),\
    #source("DOI 10.1016/j.cma.2023.116488", "https://doi.org/10.1016/j.cma.2023.116488")
  ], [
    Manufactured Reynolds–Elrod solution and convergence evidence. Use for
    code verification before interpreting the conical JFO field.
  ],
)

== The one-sentence literature position

#callout(
  "Use this wording",
  [
    “Gangrade validates the conical thin-film pressure framework; Elrod and
    Giacopini supply mass-conserving rupture mathematics; Concli and Song–Gu
    supply independent 3-D cavitation benchmarks; Li, Shen–Khonsari, and
    Wettmarshausen show why air/gas physics must be separated from pure oil
    vapour. No one paper supplies the complete combined conical-cavitation
    case.”
  ],
  fill: greenpale,
  stroke: green,
)

#pagebreak()

= Exact decisions to request from the professor

#enum(
  [*Primary claim:* approve Reynolds–JFO as the paper-aligned bearing
   performance model, or explicitly choose a phase-resolved mechanism as the
   main research question.],
  [*Oil-data route:* choose one—identify and measure the actual oil; adopt a
   named commercial substitute at a defined temperature; or approve a bounded
   sensitivity study with no exact-oil validation claim.],
  [*Pressure convention:* approve 0.5 MPa gauge ($601325$ Pa absolute at the
   feed) versus 0.5 MPa absolute. Report both if the source ambiguity cannot be
   resolved.],
  [*End condition:* provide rig evidence for flooded versus air-accessible
   ends, or approve both as explicit boundary scenarios.],
  [*Validation stack:* approve separate conical single-phase/JFO and
   cylindrical cavitation benchmarks before the combined application.],
  [*RPM matrix:* resolve the paper’s 2.6 m/s = 496.6 rpm versus its separately
   labelled 2000 rpm point, then approve 0, 496.6, 1000, 2000, 3000, and
   4000 rpm continuation after each physical gate passes.],
)

== Sixty-second opening script

#callout(
  "Say this",
  [
    “Sir, we implemented four distinct interpretations with source-backed
    substitute values. Reynolds–JFO passed its numerical gates through
    4000 rpm; its paper-viscosity peak ratio at 2000 rpm lies between the
    paper’s graph-read FEA and Fluent values, but that is only a scalar check.
    The oil-vapour case reached onset near 24.4 rpm but remained unsettled at
    28 rpm. The non-condensable-gas case failed its zero-speed mass gate, and
    the ventilation startup reached negative absolute pressure before air
    propagated. We therefore recommend JFO as the primary film-rupture model
    and ask you to choose which 3-D hypothesis, if any, should receive the next
    validation effort. We still need the exact V-32 product and temperature,
    gauge-versus-absolute feed pressure, and flooded-versus-air-accessible end
    condition.”
  ],
  fill: cyan,
  stroke: blue,
)

== If asked “why not just use typical values?”

Typical values are useful for *sensitivity*, not identity. $p_"sat"$,
dissolved-air fraction, nuclei, surface tension, and vapour properties control
cavity onset and size. Copying them from water or a different oil can produce
a smooth, colourful solution whose quantitative result belongs to neither the
paper nor the test rig. The honest label is “scenario using source X,” followed
by a parameter sweep.

== Red lines for the presentation

- Do not call convergence a physical pass when $p_"abs"<0$.
- Do not call peak-pressure agreement alone “validation.”
- Do not say the Shell SDS describes the paper’s V-32; it describes a named
  substitute product.
- Do not transfer Concli’s oil or water phase properties to this bearing.
- Do not describe JFO $theta$ as vapour volume fraction.
- Do not describe `alpha.oil = 1` at a pre-update stopping guard as evidence
  that cavitation is absent.
- Do not present pseudo-time frames or camera motion as physical transient
  cavity dynamics.
- Do not use 101,325 Pa as a cavity floor unless the model’s vented/air-access
  interpretation is stated.

== Proposed meeting outcome

#table(
  columns: (1.05fr, 2.8fr),
  fill: (_, y) => if y == 0 { cyan } else if calc.odd(y) { pale } else { white },
  table.header([*Decision*], [*Recommended entry in meeting notes*]),
  [Main model], [Reynolds–JFO / Elrod–Adams for paper-aligned performance],
  [3-D branch], [Choose one: oil vapour, non-condensable gas, or ventilation],
  [Oil source], [Actual oil measurement / named substitute / sensitivity only],
  [Pressure], [0.5 MPa gauge or absolute, explicitly recorded],
  [Ends], [Flooded or air-accessible, supported by rig evidence or A/B cases],
  [Validation label], [Hierarchical method validation; combined case not directly validated yet],
  [Next simulation], [Three-grid JFO/paper-curve study; resume only the selected 3-D branch after fixing its failed gate],
)

#v(6pt)
#text(size: 7.5pt, fill: grey)[
  Project evidence: `docs/handoff/16-four-track-cavitation-results.org`,
  `docs/handoff/media/cavitation_four_track/`,
  `out/openfoam_oq90_*/*RESULTS.md`, and the case-specific input ledgers.
  Literature links above are the citable external sources; project-generated
  figures are not external validation data.
]
