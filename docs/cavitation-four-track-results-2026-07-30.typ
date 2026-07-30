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
  title: "Four-Track Cavitation OpenFOAM Results",
  author: "BTP project",
  date: datetime(year: 2026, month: 7, day: 30),
)
#set page(
  paper: "a4",
  margin: (top: 17mm, bottom: 17mm, left: 17mm, right: 17mm),
  header: align(right)[
    #text(size: 7.3pt, fill: grey)[FOUR-TRACK CAVITATION RESULTS · 30 JULY 2026]
  ],
  footer: align(center)[
    #text(size: 7.3pt, fill: grey)[#context counter(page).display("1")]
  ],
)
#set text(font: "IBM Plex Sans", size: 9pt, fill: rgb("#1C2833"), lang: "en")
#set par(justify: true, leading: 0.56em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => block(
  above: 9pt,
  below: 6pt,
  stroke: (bottom: 1pt + blue),
  inset: (bottom: 4pt),
)[#text(font: "IBM Plex Serif", size: 17pt, weight: "semibold", fill: navy)[#it.body]]
#show heading.where(level: 2): it => block(
  above: 7pt,
  below: 4pt,
)[#text(size: 11.3pt, weight: "semibold", fill: blue)[#it.body]]
#show link: set text(fill: blue)
#show table.cell: set text(size: 7.7pt)
#set table(stroke: 0.45pt + rgb("#B8C5CC"), inset: 4pt)
#set list(indent: 15pt, body-indent: 5pt, spacing: 2.5pt)

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

#align(center)[
  #v(15mm)
  #text(font: "IBM Plex Serif", size: 28pt, weight: "bold", fill: navy)[
    Four-Track Cavitation\
    OpenFOAM Results
  ]
  #v(4mm)
  #text(size: 13pt, weight: "medium", fill: blue)[
    Eccentric conical hydrodynamic bearing
  ]
  #v(3mm)
  #text(size: 10pt, fill: grey)[Professor discussion results brief · 30 July 2026]
]

#v(10mm)

#callout(
  "Outcome",
  [
    All four mechanisms were implemented and exercised with published
    surrogate inputs. Only the Reynolds–JFO branch passed its numerical gates.
    The three 3-D branches produced useful onset or failure evidence, but none
    produced a defensible steady quantitative solution. They are preserved and
    labelled; no failed branch was pushed to a cosmetic high-speed result.
  ],
  fill: amber,
  stroke: rgb("#A16B00"),
)

#v(7mm)
#image("handoff/media/cavitation_four_track/four_track_status.png", width: 100%)

#v(5mm)

#table(
  columns: (0.7fr, 1.25fr, 0.85fr, 1fr, 1.5fr),
  fill: (_, y) => if y == 0 { cyan } else if y == 1 { greenpale } else if y == 4 { redpale } else { pale },
  table.header([*Track*], [*Model*], [*Highest state*], [*Key number*], [*Decision*]),
  [A], [Reynolds–JFO], [4000 rpm], [$p_"g,max" = 16.495$ MPa], [Accepted numerical sensitivity],
  [B], [oil vapour], [28 rpm], [$alpha_"oil,min" = 0.1112$], [Unsettled phase field],
  [C], [non-condensable gas], [0 rpm], [2.035% imbalance], [Zero-speed gate failed],
  [D], [atmospheric ventilation], [3500 rpm startup], [$p_"abs,min" = -31.76$ MPa], [Physically reject],
)

#pagebreak()

= Track A — Reynolds–JFO

#callout(
  "Accepted numerical candidate, not experimental validation",
  [
    The mass-conserving pressure–fill solver converged at every retained speed.
    Native OpenFOAM and independent Python peak pressures agree within 0.0257%.
    This establishes implementation parity, not correctness of the shared
    physics, feed model, or cavity front.
  ],
  fill: greenpale,
  stroke: green,
)

#v(5pt)
#image("handoff/media/cavitation_four_track/paper_comparison_and_higher_rpm.png", width: 100%)

#v(5pt)

At 2000 rpm, the earlier run using the conical paper's viscosity gives
$p_"max,gauge" / p_s = 34.8411$. Graph reading from Gangrade et al. gives
approximately 35.6 for FEA and 32.0 for Fluent: a narrow one-scalar difference
of −2.13% and +8.88%, respectively. This does not validate load, torque,
leakage, fill fraction, or cavity geometry.

The named SAE 10W-40 surrogate gives 17.215 at 2000 rpm because its viscosity
is 0.0125 Pa s rather than the direct paper's 0.0277 Pa s. It is therefore not
presented as a match. At 3000 and 4000 rpm the named-oil study gives:

#table(
  columns: (1fr, 1.2fr, 1.1fr, 1fr),
  fill: (_, y) => if y == 0 { cyan } else { white },
  table.header([*Speed*], [*$p_"g,max"$*], [*Ruptured area*], [*$alpha_"min"$*]),
  [3000 rpm], [12.6268 MPa], [51.257%], [0.26394],
  [4000 rpm], [16.4950 MPa], [54.163%], [0.26315],
)

#v(4pt)
These two speeds are explicitly extrapolative sensitivity cases beyond the
direct paper's quoted 2000 rpm point.

#pagebreak()

= Tracks B–D — 3-D screening evidence

#image("handoff/media/cavitation_four_track/screening_diagnostics.png", width: 100%)

== B · Oil-vapour phase change

- Native OpenFOAM Schnerr–Sauer, homogeneous VOF, SAE 10W-40 published
  simulation surrogate, and $p_"sat" = 29185$ Pa.
- Stable liquid states were retained at 0 and 20 rpm.
- Phase onset occurred near 24.4 rpm. At step 145 of the 28 rpm hold,
  $alpha_"oil,min" = 0.11117$, mean oil fraction was 0.998999, and boundary
  mass imbalance was 0.1321%.
- The minimum fraction was recovering while the mean fraction continued to
  fall. The phase field had not plateaued; higher speed was blocked.
- The source reports Zwart coefficients. This OpenFOAM installation has no
  native Zwart model, so the documented Schnerr–Sauer translation is a screen,
  not a reproduction.

#v(3pt)
#grid(
  columns: (1fr, 1fr),
  gutter: 5mm,
  image("handoff/media/cavitation_four_track/track_b_vapour/alpha_oil_003.png", width: 100%),
  image("handoff/media/cavitation_four_track/track_b_vapour/p_rgh_003.png", width: 100%),
)

#pagebreak()

== C · Non-condensable gas / pseudo-cavitation

- Compressible VOF with ISO VG32 oil and ideal-gas air; no phase-change mass
  transfer.
- Oil, air, temperature, and Bunsen coefficient came from Wettmarshausen et
  al. The initial gas fraction was evaluated from their pressure-dependent
  equation, not guessed.
- At zero speed and step 40, $alpha_"oil,min" = 0.90709$, mean oil fraction
  was 0.94015, and boundary mass imbalance was 2.035%.
- The declared mass gate was below 0.5%; the field was still drifting.
  Rotation was therefore not attempted.

#v(3pt)
#grid(
  columns: (1fr, 1fr),
  gutter: 5mm,
  image("handoff/media/cavitation_four_track/track_c_gas/alpha_oil_002.png", width: 100%),
  image("handoff/media/cavitation_four_track/track_c_gas/p_rgh_002.png", width: 100%),
)

== D · Atmospheric ventilation

- Native incompressible VOF; pure-oil pressure feed; pure-air backflow at both
  axial openings; no phase-change source.
- VG22 oil/air data and measured 0.04 N/m surface tension came from Sakai,
  Ochiai, and Hashimoto.
- At 20 rpm over 10 microseconds, interior $alpha_"oil,min"$ was still
  0.999939: the air boundary activated, but no resolved interior tongue formed.
- At 3500 rpm, the worst startup pressures were −31.76 and +21.91 MPa
  absolute. The run was stopped because negative absolute pressure is
  inadmissible; vapour or released-gas physics would intervene first.
- The film-only mesh has no external reservoir, and no measured wall contact
  angle is available. This is not a resolved meniscus model.

#pagebreak()

= Discussion decisions

#callout(
  "Recommended statement",
  [
    Reynolds–JFO is the primary model justified by the present data for
    mass-conserving film rupture. Oil vapour, released gas, and atmospheric
    ventilation remain separate hypotheses. Each needs its own material data,
    boundary evidence, and validation experiment before it can replace JFO or
    be used for quantitative conical-bearing predictions.
  ],
  fill: greenpale,
  stroke: green,
)

== Questions requiring the professor's decision

1. Is the intended observable pressure/load, a visible gas cavity, oil vapour,
   dissolved-air release, or ambient-air ingestion?
2. What exact V-32 product and operating temperature should be used?
3. Is the 0.5 MPa supply pressure gauge or absolute?
4. Are both axial ends flooded, sealed, or air-accessible?
5. Can the rig drawing, reservoir level, seal arrangement, and pressure-tap
   data be obtained?
6. For ventilation, can oil-metal contact angle be measured?

== Validation route

1. Reproduce Gangrade et al.'s full normalized-pressure curve and five
   experimental pressure taps, retaining 496.6 and 2000 rpm as separate points.
2. Perform at least a three-grid study for peak pressure, load, feed flow,
   rupture boundary, and liquid deficit.
3. Validate any selected 3-D cavitation mechanism on its own fully specified
   experimental benchmark before applying it to the conical mesh.
4. Reserve “validated conical cavitation” for matching conical cavity data.

== Sources and generated artifacts

- Gangrade et al.: #link("https://doi.org/10.1007/s40997-018-0217-2")[conical bearing comparison].
- Muchammad et al.: #link("https://jurnaltribologi.mytribos.org/v40/JT-40-39-60.pdf")[SAE 10W-40 surrogate].
- Wettmarshausen et al.: #link("https://doi.org/10.3390/lubricants13040140")[non-condensable-gas model].
- Sakai et al.: #link("https://doi.org/10.3390/lubricants7090074")[visualized VG22 oil-air VOF].
- Giacopini et al.: #link("https://doi.org/10.1115/1.4002215")[JFO complementarity benchmark].
- Miraskari et al.: #link("https://doi.org/10.1115/1.4034244")[JFO numerical benchmark].

#v(5pt)

The complete machine-readable ledger, full-resolution figures, and animations
are in `docs/handoff/media/cavitation_four_track/`. Track A media are in
`docs/handoff/media/jfo_sae10w40/`. Case-specific source and result ledgers are
stored beside each OpenFOAM case.
