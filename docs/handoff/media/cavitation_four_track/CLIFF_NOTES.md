# Cavitation study cliff notes

## Bottom line

We implemented and ran four **different physical models**. They are not four
parameter settings of one cavitation model.

| Track | Model | Highest state | Result |
|---|---|---:|---|
| A | Reynolds-JFO film rupture | 4000 rpm | Accepted numerical sensitivity study; not experimental validation |
| B | Oil-vapour phase change, Schnerr-Sauer | 28 rpm hold | Unsettled; pressure reached `pSat`, phase field did not plateau |
| C | Compressible non-condensable gas | 0 rpm, step 40 | Unsettled; 2.035% boundary mass imbalance failed the 0.5% gate |
| D | Atmospheric oil-air ventilation | 3500 rpm startup, 4.368 us | Rejected; absolute pressure fell as low as -31.76 MPa before air propagated |

Only Track A is presently suitable for quantitative plots, and even it must be
called an **unvalidated numerical candidate/sensitivity study**.

## What was actually implemented

### A - Reynolds-JFO

- Mass-conserving Reynolds equation with Elrod-Adams/JFO pressure-fill
  complementarity.
- Fields: absolute pressure `p` and film fill fraction `thetaFill`.
- Published SAE 10W-40 surrogate: `rho = 850 kg/m3`,
  `mu = 0.0125 Pa s`, and rupture pressure `29185 Pa`.
- Native OpenFOAM and independent Python implementations agree within
  `0.0257%` in peak pressure over the complete speed sweep.
- At 4000 rpm: maximum gauge pressure `16.4950 MPa`, minimum fill fraction
  `0.26315`, ruptured area `54.163%`, and feed flow `1.00125e-5 m3/s`.
- The 3000 and 4000 rpm points are extrapolative sensitivity cases beyond the
  direct paper's quoted 2000 rpm point.

### B - oil vapour

- Homogeneous VOF plus native OpenFOAM Schnerr-Sauer liquid-vapour mass
  transfer.
- Same published SAE 10W-40 surrogate, including `pSat = 29185 Pa`.
- The source paper reports Zwart inputs; OpenFOAM Foundation 14 has no native
  Zwart model here, so the documented Schnerr-Sauer translation is not an
  exact reproduction.
- Vapour onset occurred near 24.4 rpm. At the final 28 rpm hold,
  `alpha.oil_min = 0.11117` and boundary mass imbalance was `0.1321%`, but the
  phase field was still changing. No higher-rpm result was accepted.

### C - gas/pseudo-cavitation

- Compressible VOF with ISO VG32 oil and ideal-gas air; no phase-change mass
  transfer.
- Properties and Bunsen solubility coefficient came from Wettmarshausen et
  al. (2025).
- The initial gas fraction was evaluated from their pressure-dependent Eq.
  (14), not guessed.
- At zero speed and step 40, `alpha.oil_min = 0.90709`,
  `alpha.oil_mean = 0.94015`, and boundary mass imbalance was `2.035%`.
- Because the zero-speed gate failed, applying journal rotation would only
  compound an unsettled initialization/boundary problem.

### D - atmospheric ventilation

- Incompressible oil-air VOF with no phase-change source.
- VG22 oil/air properties and measured `sigma = 0.04 N/m` came from Sakai,
  Ochiai, and Hashimoto (2019).
- Pure oil enters through the pressure feed; reverse flow at either axial end
  imports pure air.
- The current film-only mesh has no external reservoir. Pure-air backflow was
  activated, but a resolved interior air tongue had not formed.
- An impulsive 3500 rpm startup produced `p_abs_min = -31.76 MPa`.
  Negative absolute pressure makes this standalone ventilation result
  physically inadmissible; vapour/released-gas physics would intervene first.

## Does it match the conical-bearing paper?

There is one useful but narrow scalar comparison at 2000 rpm:

| Source/model | Approx. `pmax,gauge / ps` |
|---|---:|
| Gangrade et al. FEA curve, graph read | 35.6 |
| Gangrade et al. Fluent curve, graph read | 32.0 |
| Earlier JFO run using the paper's `mu = 0.0277 Pa s` | 34.8411 |

The paper-input JFO value is about `2.13%` below the graph-read FEA value and
`8.88%` above the graph-read Fluent value. This checks only one pressure
scalar. It does **not** validate load, torque, leakage, cavity shape, fill
fraction, or the feed model.

The new named SAE 10W-40 run gives `pmax,gauge / ps = 17.215` at 2000 rpm.
That is deliberately **not compared as a match** because its viscosity
`0.0125 Pa s` differs from the conical paper's `0.0277 Pa s`.

The direct paper is also internally inconsistent: `2.6 m/s` at `R = 50 mm`
is `496.6 rpm`, not 2000 rpm. Both points were retained separately.

## What to tell the professor

1. We now have four source-backed implementations and did not force failed
   branches to produce high-speed pictures.
2. Reynolds-JFO is the defensible primary model for mass-conserving film
   rupture with the data currently available.
3. Vapour, released gas, and atmospheric ventilation require different
   equations and different missing evidence; “cavitation” alone does not pick
   one.
4. The first decision needed is the intended observable: pressure/load only,
   visible gas cavity, oil vapour, dissolved-air release, or ambient-air
   ingestion.
5. For a direct application result we still need:
   - exact V-32 product and temperature;
   - whether 0.5 MPa is gauge or absolute;
   - flooded versus air-accessible axial ends;
   - rig/end/reservoir geometry;
   - oil vapour pressure or dissolved-air data as applicable;
   - measured oil-metal contact angle for a resolved ventilation meniscus.
6. For paper validation, reproduce the full normalized-pressure curve and the
   five experimental pressure taps, then run a grid/uncertainty study. A
   single peak-pressure match is not enough.

## Source papers

- Gangrade et al., conical-bearing comparison:
  https://doi.org/10.1007/s40997-018-0217-2
- Muchammad et al., SAE 10W-40 simulation surrogate:
  https://jurnaltribologi.mytribos.org/v40/JT-40-39-60.pdf
- Wettmarshausen et al., experimentally validated non-condensable-gas model:
  https://doi.org/10.3390/lubricants13040140
- Sakai, Ochiai, and Hashimoto, visualized VG22 oil-air VOF study:
  https://doi.org/10.3390/lubricants7090074
- Giacopini et al., mass-conserving complementarity benchmark:
  https://doi.org/10.1115/1.4002215
- Miraskari et al., JFO complementarity benchmark:
  https://doi.org/10.1115/1.4034244

## Artifact map

- `four_track_status.png/pdf`: decision table.
- `screening_diagnostics.png/pdf`: time/pseudo-time diagnostics for B-D.
- `paper_comparison_and_higher_rpm.png/pdf`: paper scalar comparison and
  0-4000 rpm JFO sensitivity.
- `mechanism_summary.csv`: compact result ledger.
- `track_b_vapour/`, `track_c_gas/`, and `track_d_ventilation/`: phase/pressure
  frames and MP4/GIF animations.
- `../jfo_sae10w40/`: accepted Track A plots and checkpoint animation.
