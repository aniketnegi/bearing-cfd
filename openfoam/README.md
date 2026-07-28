# Native OpenFOAM Reynolds/JFO solver

`jfoBearingFoam` solves the conservative Reynolds/JFO thin-film equation
directly with OpenFOAM finite-volume fields and matrices. The supplied case
uses the paper geometry, 0.5 MPa gauge feed, atmospheric axial edges, and a
256 x 80 unwrapped conical-film mesh.

Run the build-and-zero-speed acceptance check:

```bash
bash openfoam/check_jfoBearingFoam.sh
```

The production case is
`out/openfoam_jfo_native_256x80`. Its latest time is the converged 2000 rpm
field, and `VTK/openfoam_jfo_native_256x80_4932.vtk` contains pressure, fill
fraction, film thickness, surface radius, and metric fields.

Regenerate the tracked diagnostic figures, CSV, MP4, and GIF with:

```bash
MPLCONFIGDIR=/tmp/jfo-matplotlib \
  .venv/bin/python openfoam/visualize_jfo.py
```

The generated media are stored under
`docs/handoff/media/jfo_candidate`. They describe the current unvalidated
candidate; the animation frames are converged checkpoints rather than a
physical acceleration history.

For a fresh manual run, copy `openfoam/cases/jfoPaperExact` to a new output
directory, run `blockMesh`, build `openfoam/jfoBearingFoam` with `wmake`, and
invoke `jfoBearingFoam -case <case>`. Change `rpm` with `foamDictionary`; the
solver resumes from `latestTime` and exits nonzero unless convergence,
pressure-floor, and 0.5% flow-balance gates all pass.
